"""
Train NBFNet on a knowledge graph dataset.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/train_nbfnet.py --dataset fb15k237 --gpu 0
    CUDA_VISIBLE_DEVICES=0 python scripts/train_nbfnet.py --dataset wn18rr --gpu 0
    CUDA_VISIBLE_DEVICES=1 python scripts/train_nbfnet.py --dataset hetionet --gpu 1
"""

import sys
import argparse
import json
import logging
from pathlib import Path

import torch
from tqdm import tqdm

# Project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    NBFNET_CONFIG, GPU_CONFIG, CHECKPOINT_DIR, RANDOM_SEED, PROCESSED_DATA
)
from src.data.dataset import build_dataloaders
from src.models.nbfnet.model import NBFNet
from src.models.nbfnet.trainer import FullGraphNBFNetTrainer
from src.utils.gpu_utils import get_device, log_gpu_memory
from src.utils.logging_utils import setup_logging, ExperimentLogger, format_metrics_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train NBFNet on a KG dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["fb15k237", "wn18rr", "hetionet"],
        help="Dataset to train on",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU ID to use (0 or 1). Defaults to config.GPU_CONFIG.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs from config.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size from config.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run with small data subset for debugging.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the best checkpoint (checkpoints/{dataset}/nbfnet_best.pt).",
    )
    parser.add_argument(
        "--precompute",
        action="store_true",
        help=(
            "Pre-compute and cache all training subgraphs before the first epoch. "
            "Cache is stored in data/processed/{dataset}/subgraph_cache_train.pt and "
            "reused on subsequent runs. Raises GPU utilization from ~17%% to ~80-95%%."
        ),
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=1,
        dest="start_epoch",
        help="Epoch number to start from (for log display). Use with --resume.",
    )
    parser.add_argument(
        "--no_provenance",
        action="store_true",
        help="Disable provenance features (vanilla NBFNet baseline, all edge_prov = 1.0).",
    )
    parser.add_argument(
        "--save_as",
        type=str,
        default=None,
        help="Save checkpoints and results under this name instead of the default dataset name. "
             "Checkpoint goes to checkpoints/{dataset}/{save_as}.pt; results to "
             "experiments/results/{dataset}/{save_as}/.",
    )
    return parser.parse_args()


def _save_predictions(trainer, test_loader, results_dir: Path, device) -> None:
    """
    Run the trained model over the test set and save predictions with filtered rank.

    Uses the same filtered ranking as evaluate(): other known true answers for the
    same (head, relation) pair are excluded when computing the rank of the true tail.
    This matches the reported Hits@10 metric.

    Produces:
        results_dir/predictions.pt  — (N, 4) LongTensor
            col 0: query_idx
            col 1: true_tail_id
            col 2: predicted_tail_id   (top-1, global entity id)
            col 3: filtered_rank       (1-indexed filtered rank of true tail)

        results_dir/test_ranks.pt   — (N, 2) LongTensor
            col 0: query_idx
            col 1: filtered_rank

    filtered_rank=-1 when the subgraph is None (no valid subgraph extracted).
    These files are required by scripts/build_trust_calibration_v2.py.
    """
    trainer.model.eval()
    rows = []
    query_idx = 0

    from src.utils.gpu_utils import autocast_ctx

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Saving predictions", leave=False):
            from src.utils.gpu_utils import move_batch_to_device
            batch = move_batch_to_device(batch, device)
            heads = batch["head"]
            rels  = batch["relation"]
            tails = batch["tail"]

            for i in range(len(heads)):
                h, r, t = heads[i].item(), rels[i].item(), tails[i].item()

                subgraph = trainer._extract_subgraph(h, r, t, eval_mode=True)
                if subgraph is None:
                    rows.append((query_idx, t, -1, -1))
                    query_idx += 1
                    continue

                edge_index      = subgraph["edge_index"].to(device)
                edge_type       = subgraph["edge_type"].to(device)
                edge_prov       = subgraph["edge_prov"].to(device)
                local_head      = subgraph["local_head"]
                local_tail      = subgraph["local_tail"]
                num_nodes       = subgraph["num_nodes"]
                g2l             = subgraph["global_to_local"]
                local_to_global = {v: k for k, v in g2l.items()}

                with autocast_ctx(device, trainer.use_amp, trainer.amp_dtype):
                    scores, _ = trainer.model(
                        edge_index=edge_index,
                        edge_type=edge_type,
                        edge_prov=edge_prov,
                        query_head=local_head,
                        query_relation=r,
                        num_nodes=num_nodes,
                    )

                # Top-1 local index → global entity id
                pred_local  = int(scores.argmax().item())
                pred_global = local_to_global.get(pred_local, -1)

                # Filtered rank: same logic as evaluate() —
                # skip nodes whose global id is another known true answer for (h, r).
                true_score = scores[local_tail].item()
                rank = 1
                for node_local_idx, node_score in enumerate(scores):
                    if node_local_idx == local_tail:
                        continue
                    global_id = local_to_global.get(node_local_idx, -1)
                    if global_id == -1:
                        continue
                    if (h, r, global_id) in trainer.all_triples and global_id != t:
                        continue  # filtered: don't penalise for other correct answers
                    if node_score.item() > true_score:
                        rank += 1

                rows.append((query_idx, t, pred_global, rank))
                query_idx += 1

    predictions = torch.tensor(rows, dtype=torch.long)  # (N, 4)
    torch.save(predictions, str(results_dir / "predictions.pt"))

    # Also persist filtered ranks as a standalone file for build_trust_calibration_v2.py
    test_ranks = predictions[:, [0, 3]]  # (N, 2): [query_idx, filtered_rank]
    torch.save(test_ranks, str(results_dir / "test_ranks.pt"))

    n_correct = int((predictions[:, 1] == predictions[:, 2]).sum().item())
    n_hits10  = int((predictions[:, 3].clamp(min=1) <= 10).sum().item())
    N = len(predictions)
    logger.info(
        "predictions.pt saved: N=%d, top-1=%.4f, filtered hits@10=%.4f",
        N, n_correct / max(N, 1), n_hits10 / max(N, 1),
    )
    logger.info("test_ranks.pt saved: N=%d (filtered ranks)", N)


def main():
    args = parse_args()
    setup_logging()
    torch.manual_seed(RANDOM_SEED)

    # ─── Device ────────────────────────────────────────────────────────────────
    gpu_id = args.gpu if args.gpu is not None else GPU_CONFIG.get(args.dataset, 0)
    device = get_device(gpu_id)
    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        log_gpu_memory(device, prefix="Initial ")

    # ─── DataLoaders ────────────────────────────────────────────────────────────
    batch_size = args.batch_size or NBFNET_CONFIG["batch_size"]
    logger.info(f"Loading dataset: {args.dataset}")

    use_provenance = not args.no_provenance
    train_loader, valid_loader, test_loader, train_ds = build_dataloaders(
        dataset_name=args.dataset,
        batch_size=batch_size,
        num_workers=NBFNET_CONFIG["num_workers"] if not args.debug else 0,
        device="cpu",
        use_provenance=use_provenance,
    )

    num_entities = train_ds.num_entities
    num_relations = train_ds.num_relations
    logger.info(
        f"Dataset: {args.dataset} | "
        f"Entities={num_entities} | Relations={num_relations} | "
        f"Train triples={len(train_ds)}"
    )

    # ─── Build full graph tensors for FullGraphNBFNetTrainer ───────────────────
    all_triples = train_ds.get_all_triples_tensor()  # (N, 3)
    all_edge_index = torch.stack([all_triples[:, 0], all_triples[:, 2]], dim=0)
    all_edge_type = all_triples[:, 1]
    all_edge_prov = train_ds.get_all_provenance()

    # ─── Model ─────────────────────────────────────────────────────────────────
    model = NBFNet(
        num_relations=num_relations,
        hidden_dim=NBFNET_CONFIG["hidden_dim"],
        num_layers=NBFNET_CONFIG["num_layers"],
        dropout=NBFNET_CONFIG["dropout"],
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"NBFNet parameters: {total_params:,}")

    # ─── Checkpoint / results paths (respects --save_as) ───────────────────────
    ckpt_name = args.save_as if args.save_as else "nbfnet_best"
    ckpt_dir = Path(CHECKPOINT_DIR) / args.dataset
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / f"{ckpt_name}.pt"

    results_dir = (
        Path("experiments/results") / args.dataset / args.save_as
        if args.save_as
        else Path("experiments/results") / args.dataset
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.no_provenance:
        logger.info("Provenance DISABLED — training vanilla NBFNet baseline")

    # ─── Trainer ───────────────────────────────────────────────────────────────
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

    # Point trainer to the correct best-checkpoint filename
    trainer.best_ckpt_name = f"{ckpt_name}.pt"

    # ─── Subgraph Precomputation (optional) ────────────────────────────────────
    if args.precompute:
        cache_path = Path(PROCESSED_DATA[args.dataset]) / "subgraph_cache_train.pt"
        logger.info("Phase 1: Pre-computing subgraphs (runs once, then cached) ...")
        subgraph_cache = trainer.precompute_subgraphs(all_triples, cache_path)
        trainer.subgraph_cache = subgraph_cache
        logger.info("Phase 2: Training with cached subgraphs (GPU-bound) ...")

    # ─── Training ──────────────────────────────────────────────────────────────
    num_epochs_cfg = NBFNET_CONFIG["num_epochs"]
    if isinstance(num_epochs_cfg, dict):
        num_epochs = num_epochs_cfg.get(args.dataset, 30)
    else:
        num_epochs = num_epochs_cfg
    num_epochs = args.epochs or num_epochs

    if args.debug:
        num_epochs = 2
        logger.info("DEBUG mode: running only 2 epochs")

    # ─── Resume from checkpoint ─────────────────────────────────────────────────
    start_epoch = args.start_epoch
    initial_best_mrr = 0.0
    if args.resume:
        best_ckpt = best_ckpt_path
        if best_ckpt.exists():
            logger.info(f"Resuming from {best_ckpt} (start_epoch={start_epoch})")
            model = NBFNet.load(str(best_ckpt), device=device)
            trainer.model = model
            # Rebind optimizer and scaler to the new model's parameters.
            # The original optimizer still references the pre-load model's params,
            # which breaks AMP's inf-check tracking after trainer.model is replaced.
            from src.utils.gpu_utils import GradScalerWrapper
            trainer.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=NBFNET_CONFIG["learning_rate"],
                weight_decay=NBFNET_CONFIG["weight_decay"],
            )
            trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                trainer.optimizer, T_max=30, eta_min=1e-5
            )
            trainer.scaler = GradScalerWrapper(use_amp=trainer.use_amp, device=device)
            # Remaining epochs = total - already done
            num_epochs = max(num_epochs - (start_epoch - 1), 1)
        else:
            logger.warning(f"--resume set but {best_ckpt} not found — starting from scratch.")

    exp_logger = ExperimentLogger(
        run_name=f"nbfnet_{args.dataset}",
        config={"dataset": args.dataset, "num_epochs": num_epochs, **NBFNET_CONFIG},
    )

    logger.info(f"Starting training for {num_epochs} epochs (from epoch {start_epoch})...")
    history = trainer.train(
        train_loader, valid_loader,
        num_epochs=num_epochs,
        start_epoch=start_epoch,
        initial_best_mrr=initial_best_mrr,
    )

    # Log history
    for entry in history:
        exp_logger.log(entry, step=entry["step"])

    # ─── Final Evaluation ──────────────────────────────────────────────────────
    logger.info("Loading best checkpoint for final evaluation...")
    if best_ckpt_path.exists():
        model = NBFNet.load(str(best_ckpt_path), device=device)
        trainer.model = model

    test_metrics = trainer.evaluate(test_loader)
    title = f"Test Results — {args.dataset}" + (f" ({args.save_as})" if args.save_as else "")
    print("\n" + format_metrics_table(test_metrics, title=title))

    exp_logger.log_summary(test_metrics)
    history_path = ckpt_dir / f"{ckpt_name}_history.json"
    exp_logger.save_history(str(history_path))
    exp_logger.finish()

    # Save test metrics
    metrics_path = results_dir / "nbfnet_test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    logger.info(f"Test metrics saved to {metrics_path}")

    # ─── Save predictions.pt for build_trust_calibration_v2.py ─────────────────
    logger.info("Saving predictions.pt for trust calibration pipeline ...")
    _save_predictions(trainer, test_loader, results_dir, device)

    if device.type == "cuda":
        log_gpu_memory(device, prefix="Final ")

    # Check targets
    mrr = test_metrics["mrr"]
    h10 = test_metrics["hits@10"]
    targets = {"fb15k237": (0.30, 0.50), "wn18rr": (0.40, 0.60), "hetionet": (0.20, 0.40)}
    target_mrr, target_h10 = targets.get(args.dataset, (0.25, 0.45))
    print(f"\nTarget MRR: {target_mrr:.2f} | Achieved: {mrr:.4f} {'✓' if mrr >= target_mrr else '✗'}")
    print(f"Target H@10: {target_h10:.2f} | Achieved: {h10:.4f} {'✓' if h10 >= target_h10 else '✗'}")


if __name__ == "__main__":
    main()
