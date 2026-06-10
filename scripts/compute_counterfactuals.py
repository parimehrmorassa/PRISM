"""
Compute counterfactual sensitivity (Delta_CF) for all test queries.

Runs on GPU 1 (COUNTERFACTUAL_CONFIG["gpu"]).

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/compute_counterfactuals.py --dataset fb15k237 --gpu 1
"""

import sys
import json
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    COUNTERFACTUAL_CONFIG, NBFNET_CONFIG, CHECKPOINT_DIR,
    RESULTS_DIR, GPU_CONFIG
)
from src.data.dataset import build_dataloaders
from src.models.nbfnet.model import NBFNet
from src.models.nbfnet.trainer import FullGraphNBFNetTrainer
from src.models.trust.counterfactual import CounterfactualAnalyzer
from src.utils.gpu_utils import get_device
from src.utils.logging_utils import setup_logging, format_metrics_table

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute counterfactual scores")
    parser.add_argument("--dataset", required=True, choices=["fb15k237", "wn18rr", "hetionet"])
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--debug", action="store_true", help="Process only first 100 queries")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging()

    gpu_id = args.gpu if args.gpu is not None else COUNTERFACTUAL_CONFIG.get("gpu", 1)
    device = get_device(gpu_id)
    logger.info(f"Counterfactual analysis on device: {device}")

    # ─── Load dataset ───────────────────────────────────────────────────────────
    train_loader, valid_loader, test_loader, train_ds = build_dataloaders(
        args.dataset, batch_size=64, num_workers=0, device="cpu"
    )
    num_entities = train_ds.num_entities
    num_relations = train_ds.num_relations

    # ─── Load NBFNet ────────────────────────────────────────────────────────────
    best_ckpt = Path(CHECKPOINT_DIR) / args.dataset / "nbfnet_best.pt"
    if not best_ckpt.exists():
        logger.warning(f"No checkpoint at {best_ckpt}, using fresh model")
        model = NBFNet(num_relations=num_relations)
    else:
        model = NBFNet.load(str(best_ckpt), device=device)
    model.to(device)
    model.eval()

    # ─── Build trainer ──────────────────────────────────────────────────────────
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

    # ─── Counterfactual analyzer ────────────────────────────────────────────────
    analyzer = CounterfactualAnalyzer(
        model=model,
        device=device,
        top_k=COUNTERFACTUAL_CONFIG["top_k_critical"],
    )

    # ─── Collect all test queries ────────────────────────────────────────────────
    queries = []
    for batch in test_loader:
        for i in range(len(batch["head"])):
            queries.append({
                "head": batch["head"][i].item(),
                "relation": batch["relation"][i].item(),
                "tail": batch["tail"][i].item(),
                "query_idx": batch["query_idx"][i].item(),
            })
        if args.debug and len(queries) >= 100:
            break

    logger.info(f"Computing counterfactuals for {len(queries)} test queries...")
    cf_results = analyzer.compute_sensitivity_batch(queries, trainer)

    # ─── Build result tensors ────────────────────────────────────────────────────
    # Tensor columns: [query_idx, Delta_CF, max_critical_impact, mean_random_impact]
    rows = []
    all_critical = []
    all_random = []
    edge_impact_details = {}

    for res in cf_results:
        q_idx = res["query_idx"]
        delta_cf = res["Delta_CF"]
        crits = res["critical_impacts"]
        randoms = res["random_impacts"]

        max_crit = max(crits) if crits else 0.0
        mean_rand = float(np.mean(randoms)) if randoms else 0.0

        rows.append([q_idx, delta_cf, max_crit, mean_rand])
        all_critical.extend(crits)
        all_random.extend(randoms)

        edge_impact_details[str(q_idx)] = {
            "Delta_CF": delta_cf,
            "critical_impacts": crits,
            "random_impacts": randoms,
            "baseline_score": res["baseline_score"],
            "critical_edges": res["critical_edges"],
        }

    # ─── Save results ────────────────────────────────────────────────────────────
    results_dir = Path(RESULTS_DIR) / args.dataset
    results_dir.mkdir(parents=True, exist_ok=True)

    result_tensor = torch.tensor(rows, dtype=torch.float)
    torch.save(result_tensor, results_dir / "counterfactual_scores.pt")

    with open(results_dir / "edge_impacts.json", "w") as f:
        json.dump(edge_impact_details, f, indent=2)

    logger.info(f"Saved counterfactual_scores.pt and edge_impacts.json")

    # ─── Statistics and significance test ────────────────────────────────────────
    all_delta_cf = [r["Delta_CF"] for r in cf_results]

    t_stat, p_value = 0.0, 1.0
    if all_critical and all_random:
        t_stat, p_value = stats.ttest_ind(all_critical, all_random, alternative="greater")

    stats_dict = {
        "num_queries": len(cf_results),
        "mean_delta_cf": float(np.mean(all_delta_cf)),
        "std_delta_cf": float(np.std(all_delta_cf)),
        "mean_critical_impact": float(np.mean(all_critical)) if all_critical else 0.0,
        "mean_random_impact": float(np.mean(all_random)) if all_random else 0.0,
        "ttest_statistic": float(t_stat),
        "pvalue_critical_gt_random": float(p_value),
    }

    print("\n" + format_metrics_table(stats_dict, title=f"Counterfactual Statistics — {args.dataset}"))

    if p_value < 0.001:
        print("Critical edges are significantly more impactful than random (p < 0.001) ✓")
    elif p_value < 0.05:
        print(f"Critical > random is significant at p={p_value:.4f}")
    else:
        print(f"Warning: critical vs random not significant (p={p_value:.4f})")

    with open(results_dir / "counterfactual_stats.json", "w") as f:
        json.dump(stats_dict, f, indent=2)


if __name__ == "__main__":
    main()
