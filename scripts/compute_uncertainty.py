"""
Compute uncertainty scores for all test queries using three measures:

  U_conformal  — prediction-set size (original, low variance when quantile=1.0)
  U_margin     — 1 - (rank1_score - rank2_score) / rank1_score  [PRIMARY]
  U_entropy    — normalised entropy of softmax over top-100 scores

Usage:
    python scripts/compute_uncertainty.py --dataset fb15k237 --gpu 0
"""

import sys
import json
import math
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    UNCERTAINTY_CONFIG, NBFNET_CONFIG, CHECKPOINT_DIR,
    PROCESSED_DATA, RESULTS_DIR, GPU_CONFIG
)
from src.data.dataset import build_dataloaders
from src.models.nbfnet.model import NBFNet
from src.models.nbfnet.trainer import FullGraphNBFNetTrainer
from src.models.trust.uncertainty import ConformalUncertaintyQuantifier
from src.utils.gpu_utils import get_device, get_amp_dtype, autocast_ctx
from src.utils.logging_utils import setup_logging, format_metrics_table

logger = logging.getLogger(__name__)

_LOG100 = math.log(100)   # normalisation constant for entropy


def _margin_uncertainty(scores: torch.Tensor) -> float:
    """
    U_margin = 1 - (top1_score - top2_score) / top1_score

    High margin between rank-1 and rank-2 → model is confident → low U.
    Scores are assumed to be in [0, 1] (sigmoid outputs).
    Returns 0.0 if fewer than 2 nodes in subgraph.
    """
    if scores.numel() < 2:
        return 0.0
    top2 = scores.topk(min(2, scores.numel())).values
    s1, s2 = top2[0].item(), top2[1].item()
    if s1 < 1e-9:
        return 1.0
    return 1.0 - (s1 - s2) / s1


def _entropy_uncertainty(scores: torch.Tensor, top_k: int = 100) -> float:
    """
    U_entropy = H(softmax(top_k scores)) / log(top_k)

    Softmax over the top-k scores, then Shannon entropy normalised to [0, 1].
    When all mass is on one entity → entropy ≈ 0 (certain).
    When mass is spread uniformly → entropy = 1 (maximally uncertain).
    """
    k = min(top_k, scores.numel())
    top_scores = scores.topk(k).values          # (k,)
    probs = F.softmax(top_scores.float(), dim=0)
    # clamp to avoid log(0)
    entropy = -(probs * probs.clamp(min=1e-12).log()).sum().item()
    log_k = math.log(k) if k > 1 else 1.0
    return float(entropy / log_k)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute uncertainty scores")
    parser.add_argument("--dataset", required=True, choices=["fb15k237", "wn18rr", "hetionet"])
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--debug", action="store_true", help="Process only first 100 queries")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging()

    gpu_id = args.gpu if args.gpu is not None else GPU_CONFIG.get(args.dataset, 0)
    device = get_device(gpu_id)
    logger.info(f"Device: {device}")

    # ─── Load dataset ───────────────────────────────────────────────────────────
    train_loader, valid_loader, test_loader, train_ds = build_dataloaders(
        args.dataset, batch_size=64, num_workers=0, device="cpu"
    )
    num_entities = train_ds.num_entities
    num_relations = train_ds.num_relations

    # ─── Load NBFNet ────────────────────────────────────────────────────────────
    best_ckpt = Path(CHECKPOINT_DIR) / args.dataset / "nbfnet_best.pt"
    if not best_ckpt.exists():
        logger.warning(f"No checkpoint found at {best_ckpt}. Creating fresh model.")
        model = NBFNet(num_relations=num_relations)
    else:
        model = NBFNet.load(str(best_ckpt), device=device)
    model.to(device)
    model.eval()

    # ─── Build trainer for subgraph extraction ──────────────────────────────────
    all_triples = train_ds.get_all_triples_tensor()
    all_edge_index = torch.stack([all_triples[:, 0], all_triples[:, 2]], dim=0)
    all_edge_type = all_triples[:, 1]
    all_edge_prov = train_ds.get_all_provenance()

    trainer = FullGraphNBFNetTrainer(
        model=model,
        device=device,
        dataset_name=args.dataset,
        all_triples=train_ds.true_triples,
        num_entities=num_entities,
        all_edge_index=all_edge_index,
        all_edge_type=all_edge_type,
        all_edge_prov=all_edge_prov,
    )

    # ─── Calibrate conformal quantifier ─────────────────────────────────────────
    quantifier = ConformalUncertaintyQuantifier(
        alpha=UNCERTAINTY_CONFIG["alpha"],
        num_entities=num_entities,
    )
    logger.info("Calibrating conformal uncertainty quantifier on validation set...")
    quantifier.calibrate(
        model=model,
        calibration_loader=valid_loader,
        device=device,
        trainer=trainer,
        num_entities=num_entities,
    )

    quant_path = Path(CHECKPOINT_DIR) / args.dataset / "conformal_quantile.pkl"
    quantifier.save(str(quant_path))

    # ─── Compute all three U measures for test queries ───────────────────────────
    logger.info("Computing U_conformal / U_margin / U_entropy for test queries...")
    use_amp = NBFNET_CONFIG["use_amp"] and device.type == "cuda"
    amp_dtype = get_amp_dtype(NBFNET_CONFIG["amp_dtype"])

    q_ids        = []
    U_conformal  = []
    U_margin     = []
    U_entropy    = []
    pred_set_sizes = []
    coverage_hits  = 0
    total          = 0

    for batch in tqdm(test_loader, desc="Test queries"):
        heads  = batch["head"]
        rels   = batch["relation"]
        tails  = batch["tail"]
        q_idxs = batch["query_idx"]

        for i in range(len(heads)):
            if args.debug and total >= 100:
                break

            h, r, t = heads[i].item(), rels[i].item(), tails[i].item()
            q_idx   = q_idxs[i].item()

            subgraph = trainer._extract_subgraph(h, r, t)
            if subgraph is None:
                U_conformal.append(1.0)
                U_margin.append(1.0)
                U_entropy.append(1.0)
                pred_set_sizes.append(0)
                q_ids.append(q_idx)
                total += 1
                continue

            edge_index = subgraph["edge_index"].to(device)
            edge_type  = subgraph["edge_type"].to(device)
            edge_prov  = subgraph["edge_prov"].to(device)
            num_nodes  = subgraph["num_nodes"]

            with torch.no_grad():
                with autocast_ctx(device, use_amp, amp_dtype):
                    scores, _ = model(
                        edge_index=edge_index,
                        edge_type=edge_type,
                        edge_prov=edge_prov,
                        query_head=subgraph["local_head"],
                        query_relation=r,
                        num_nodes=num_nodes,
                    )

            # ── U_conformal ────────────────────────────────────────────────────
            U_c, pred_set = quantifier.compute_uncertainty_from_scores(
                scores, num_entities=num_entities
            )
            local_tail = subgraph["local_tail"]
            if local_tail in pred_set:
                coverage_hits += 1
            U_conformal.append(U_c)
            pred_set_sizes.append(len(pred_set))

            # ── U_margin (primary) ─────────────────────────────────────────────
            U_margin.append(_margin_uncertainty(scores))

            # ── U_entropy ─────────────────────────────────────────────────────
            U_entropy.append(_entropy_uncertainty(scores, top_k=100))

            q_ids.append(q_idx)
            total += 1

        if args.debug and total >= 100:
            break

    # ─── Save structured output ──────────────────────────────────────────────────
    results_dir = Path(RESULTS_DIR) / args.dataset
    results_dir.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "U_conformal": torch.tensor(U_conformal, dtype=torch.float32),
        "U_margin":    torch.tensor(U_margin,    dtype=torch.float32),
        "U_entropy":   torch.tensor(U_entropy,   dtype=torch.float32),
        "query_ids":   torch.tensor(q_ids,       dtype=torch.long),
    }
    torch.save(save_dict, results_dir / "uncertainty_scores.pt")
    logger.info(f"Saved uncertainty_scores.pt  ({total} queries, 3 measures)")

    # ─── Print statistics for each measure ───────────────────────────────────────
    def _stats(name: str, vals: list) -> dict:
        arr = np.array(vals)
        return {
            "measure":      name,
            "mean":         float(arr.mean()),
            "std":          float(arr.std()),
            "min":          float(arr.min()),
            "max":          float(arr.max()),
            "pct10":        float(np.percentile(arr, 10)),
            "pct90":        float(np.percentile(arr, 90)),
        }

    measures_stats = [
        _stats("U_conformal", U_conformal),
        _stats("U_margin",    U_margin),
        _stats("U_entropy",   U_entropy),
    ]

    print(f"\n{'='*60}")
    print(f"  Uncertainty Statistics — {args.dataset}  ({total} queries)")
    print(f"{'='*60}")
    for s in measures_stats:
        print(f"\n  {s['measure']}")
        print(f"    mean={s['mean']:.4f}  std={s['std']:.4f}  "
              f"[{s['min']:.4f}, {s['max']:.4f}]")
        print(f"    p10={s['pct10']:.4f}  p90={s['pct90']:.4f}")

    coverage = coverage_hits / max(total, 1)
    print(f"\n  Conformal coverage : {coverage:.4f}  "
          f"(target {1.0 - UNCERTAINTY_CONFIG['alpha']:.2f})")
    print(f"  Conformal quantile : {quantifier.quantile:.4f}")
    print(f"{'='*60}\n")

    stats_out = {
        "num_queries":        total,
        "conformal_quantile": quantifier.quantile,
        "empirical_coverage": coverage,
        "target_coverage":    1.0 - UNCERTAINTY_CONFIG["alpha"],
        "measures":           measures_stats,
    }
    with open(results_dir / "uncertainty_stats.json", "w") as f:
        json.dump(stats_out, f, indent=2)
    logger.info("Saved uncertainty_stats.json")


if __name__ == "__main__":
    main()
