"""
regenerate_test_ranks.py

Re-runs filtered-rank inference on the provenance-enhanced NBFNet and
saves test_ranks.pt consistent with predictions.pt for the same model.

Format matches what evaluate_system.py expects:
  test_ranks.pt  — (N, 2) LongTensor [query_idx, filtered_rank]

Also updates nbfnet_test_metrics.json with correct filtered metrics.

Usage:
  cd ~/A600/Untitled
  python regenerate_test_ranks.py \
      --dataset fb15k237 \
      --checkpoint checkpoints/fb15k237/nbfnet_best.pt \
      --output_dir experiments/results/fb15k237
"""

import sys
import json
import argparse
import logging
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from config import CHECKPOINT_DIR, NBFNET_CONFIG, GPU_CONFIG
from src.data.dataset import build_dataloaders
from src.models.nbfnet.model import NBFNet
from src.models.nbfnet.trainer import FullGraphNBFNetTrainer
from src.utils.gpu_utils import get_device, autocast_ctx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",    required=True, choices=["fb15k237", "wn18rr", "hetionet"])
    p.add_argument("--checkpoint", default=None,
                   help="Path to NBFNet checkpoint. Defaults to checkpoints/{dataset}/nbfnet_best.pt")
    p.add_argument("--output_dir", default=None,
                   help="Where to write test_ranks.pt and nbfnet_test_metrics.json. "
                        "Defaults to experiments/results/{dataset}/")
    p.add_argument("--gpu",        type=int, default=None)
    return p.parse_args()


def compute_filtered_ranks(trainer, test_loader, device):
    """
    Run inference over test_loader using the same filtered-rank logic as
    FullGraphNBFNetTrainer._save_predictions().

    Returns:
        rows: list of (query_idx, filtered_rank)  — 1-indexed
        hits: per-query hit vectors for @1 @3 @10
    """
    trainer.model.eval()
    rows = []
    query_idx = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Filtered ranking"):
            from src.utils.gpu_utils import move_batch_to_device
            batch = move_batch_to_device(batch, device)
            heads = batch["head"]
            rels  = batch["relation"]
            tails = batch["tail"]

            for i in range(len(heads)):
                h, r, t = heads[i].item(), rels[i].item(), tails[i].item()

                subgraph = trainer._extract_subgraph(h, r, t, eval_mode=True)
                if subgraph is None:
                    rows.append((query_idx, -1))
                    query_idx += 1
                    continue

                edge_index  = subgraph["edge_index"].to(device)
                edge_type   = subgraph["edge_type"].to(device)
                edge_prov   = subgraph["edge_prov"].to(device)
                local_head  = subgraph["local_head"]
                local_tail  = subgraph["local_tail"]
                num_nodes   = subgraph["num_nodes"]
                g2l         = subgraph["global_to_local"]
                l2g         = {v: k for k, v in g2l.items()}

                with autocast_ctx(device, trainer.use_amp, trainer.amp_dtype):
                    scores, _ = trainer.model(
                        edge_index=edge_index,
                        edge_type=edge_type,
                        edge_prov=edge_prov,
                        query_head=local_head,
                        query_relation=r,
                        num_nodes=num_nodes,
                    )

                # Filtered rank: skip other known true tails for (h, r)
                true_score = scores[local_tail].item()
                rank = 1
                for node_local, node_score in enumerate(scores):
                    if node_local == local_tail:
                        continue
                    global_id = l2g.get(node_local, -1)
                    if global_id == -1:
                        continue
                    if (h, r, global_id) in trainer.all_triples and global_id != t:
                        continue
                    if node_score.item() > true_score:
                        rank += 1

                rows.append((query_idx, rank))
                query_idx += 1

    return rows


def main():
    args = parse_args()

    gpu_id = args.gpu if args.gpu is not None else GPU_CONFIG.get(args.dataset, 0)
    device = get_device(gpu_id)
    logger.info(f"Device: {device}")

    ckpt_path = Path(args.checkpoint) if args.checkpoint else \
                Path(CHECKPOINT_DIR) / args.dataset / "nbfnet_best.pt"
    out_dir   = Path(args.output_dir) if args.output_dir else \
                Path("experiments/results") / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load dataset ──────────────────────────────────────────────────────────
    logger.info(f"Loading dataset: {args.dataset}")
    _, _, test_loader, train_ds = build_dataloaders(
        dataset_name=args.dataset,
        batch_size=NBFNET_CONFIG["batch_size"],
        num_workers=0,
        device="cpu",
        use_provenance=True,
    )
    num_entities  = train_ds.num_entities
    num_relations = train_ds.num_relations

    # ── Load model ────────────────────────────────────────────────────────────
    logger.info(f"Loading checkpoint: {ckpt_path}")
    model = NBFNet.load(str(ckpt_path), device=device)
    model.eval()

    # ── Build trainer (needed for _extract_subgraph and all_triples filter) ──
    all_triples   = train_ds.get_all_triples_tensor()
    all_edge_index = torch.stack([all_triples[:, 0], all_triples[:, 2]], dim=0)
    trainer = FullGraphNBFNetTrainer(
        model=model,
        device=device,
        dataset_name=args.dataset,
        all_triples=train_ds.true_triples,
        num_entities=num_entities,
        all_edge_index=all_edge_index,
        all_edge_type=all_triples[:, 1],
        all_edge_prov=train_ds.get_all_provenance(),
    )

    # ── Compute filtered ranks ────────────────────────────────────────────────
    rows = compute_filtered_ranks(trainer, test_loader, device)

    ranks_tensor = torch.tensor(rows, dtype=torch.long)  # (N, 2)
    save_path = out_dir / "test_ranks.pt"
    torch.save(ranks_tensor, save_path)
    logger.info(f"Saved test_ranks.pt → {save_path}  shape={list(ranks_tensor.shape)}")

    # ── Compute metrics from valid ranks (rank > 0) ───────────────────────────
    valid_ranks = ranks_tensor[:, 1]
    valid_ranks = valid_ranks[valid_ranks > 0].float()
    N_total = len(ranks_tensor)
    N_valid = len(valid_ranks)

    mrr   = (1.0 / valid_ranks).mean().item()
    mr    = valid_ranks.mean().item()
    h1    = (valid_ranks <= 1).float().mean().item()
    h3    = (valid_ranks <= 3).float().mean().item()
    h10   = (valid_ranks <= 10).float().mean().item()

    metrics = {
        "mrr":     round(mrr, 6),
        "mr":      round(mr,  4),
        "hits@1":  round(h1,  6),
        "hits@3":  round(h3,  6),
        "hits@10": round(h10, 6),
        "n_queries": N_total,
        "n_valid":   N_valid,
    }
    metrics_path = out_dir / "nbfnet_test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved nbfnet_test_metrics.json → {metrics_path}")

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n── Provenance-enhanced NBFNet — filtered test metrics ──")
    print(f"  MRR     : {mrr:.4f}")
    print(f"  MR      : {mr:.1f}")
    print(f"  Hits@1  : {h1:.4f}")
    print(f"  Hits@3  : {h3:.4f}")
    print(f"  Hits@10 : {h10:.4f}")
    print(f"  N total : {N_total}   N valid: {N_valid}")
    print(f"\n── Rank distribution ──")
    r = ranks_tensor[:, 1]
    print(f"  Rank == 1  : {(r==1).sum().item():>6d} / {N_total}")
    print(f"  Rank <= 3  : {(r<=3).sum().item():>6d} / {N_total}")
    print(f"  Rank <= 10 : {(r<=10).sum().item():>6d} / {N_total}")
    print(f"  Rank >  100: {(r>100).sum().item():>6d} / {N_total}")
    print(f"\n  ✓ Correct selective-prediction baseline H@1 = {h1:.4f}")


if __name__ == "__main__":
    main()
