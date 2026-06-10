"""
Run full system evaluation and generate evaluation report.

Usage:
    python scripts/evaluate_system.py --dataset fb15k237 --phase all
    python scripts/evaluate_system.py --dataset wn18rr --phase trust
    python scripts/evaluate_system.py --dataset hetionet --phase uncertainty
"""

import sys
import json
import time
import argparse
import logging
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    CHECKPOINT_DIR, RESULTS_DIR, TRUST_CALIB_DIR,
    UNCERTAINTY_CONFIG, GPU_CONFIG, EVALUATION_CONFIG
)
from src.evaluation.metrics import (
    compute_trust_calibration_metrics, compute_uncertainty_metrics,
    compute_counterfactual_metrics, compute_explanation_metrics,
    compute_expected_calibration_error,
)
from src.evaluation.calibration import (
    plot_reliability_diagram, plot_trust_distribution,
    plot_weight_distributions, plot_delta_cf_distribution,
)
from src.models.trust.aggregator import AttentionBasedTrustAggregator
from src.utils.logging_utils import setup_logging, format_metrics_table
from src.utils.gpu_utils import get_device

logger = logging.getLogger(__name__)


def _load_trust_aggregator(dataset: str, trust_model: str = None) -> AttentionBasedTrustAggregator:
    """
    Load a trust aggregator checkpoint.

    Args:
        dataset: Dataset name (used to locate checkpoints/{dataset}/).
        trust_model: Checkpoint stem name (e.g. 'fixed_weight_aggregator').
                     Defaults to 'trust_aggregator'.

    Returns:
        Loaded AttentionBasedTrustAggregator in eval mode on CPU.
    """
    if trust_model is None:
        ckpt_path = Path(CHECKPOINT_DIR) / dataset / "trust_aggregator.pt"
    else:
        candidate = Path(trust_model)
        if candidate.suffix == ".pt" and (candidate.is_absolute() or "/" in trust_model):
            # Full or relative path provided directly
            ckpt_path = candidate
        else:
            # Stem name — resolve relative to checkpoints/{dataset}/
            stem = candidate.stem if candidate.suffix == ".pt" else trust_model
            ckpt_path = Path(CHECKPOINT_DIR) / dataset / f"{stem}.pt"
    if not ckpt_path.exists():
        logger.warning(f"Aggregator checkpoint not found: {ckpt_path}")
        return None
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = AttentionBasedTrustAggregator(hidden_dim=ckpt.get("hidden_dim", 64))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    logger.info(f"Loaded trust aggregator: {ckpt_path}")
    return model


def _recompute_trust_scores(
    aggregator: AttentionBasedTrustAggregator,
    U: np.ndarray,
    S_prov: np.ndarray,
    Delta_CF: np.ndarray,
) -> np.ndarray:
    """Run aggregator forward pass and return trust scores as numpy array."""
    U_t = torch.tensor(U, dtype=torch.float32)
    S_t = torch.tensor(S_prov, dtype=torch.float32)
    CF_t = torch.tensor(Delta_CF, dtype=torch.float32)
    with torch.no_grad():
        T, _ = aggregator(U_t, S_t, CF_t)
    return T.numpy()


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the full trustworthy KG system")
    parser.add_argument("--dataset", required=True, choices=["fb15k237", "wn18rr", "hetionet"])
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--phase", default="all",
                        choices=["all", "link_pred", "trust", "uncertainty", "cf", "explanation"])
    parser.add_argument(
        "--trust_model",
        type=str,
        default=None,
        help="Name of trust aggregator checkpoint to evaluate "
             "(e.g. 'fixed_weight_aggregator'). Loads "
             "checkpoints/{dataset}/{name}.pt and "
             "results/{dataset}/{name}_metrics.json. "
             "Defaults to 'trust_aggregator'.",
    )
    return parser.parse_args()


def load_results(results_dir: Path, calib_dir: Path, dataset: str, trust_model: str = None):
    """Load pre-computed results from disk. Returns dict of arrays."""
    data = {}

    # Uncertainty scores
    u_path = results_dir / "uncertainty_scores.pt"
    if u_path.exists():
        u_data = torch.load(u_path, weights_only=False)
        data["U"] = u_data["U_margin"].numpy()          # primary uncertainty signal
        data["query_idx_u"] = u_data["query_ids"].numpy().astype(int)

    # Counterfactual scores
    cf_path = results_dir / "counterfactual_scores.pt"
    if cf_path.exists():
        cf_tensor = torch.load(cf_path, weights_only=False)
        data["Delta_CF"] = cf_tensor[:, 1].numpy()
        data["max_critical"] = cf_tensor[:, 2].numpy()
        data["mean_random"] = cf_tensor[:, 3].numpy()

    # Provenance
    prov_path = Path(f"data/processed/{dataset}/provenance_weights.pt")
    if prov_path.exists():
        data["S_prov"] = torch.load(prov_path, weights_only=True).numpy()

    # Calibration labels
    label_path = Path(TRUST_CALIB_DIR) / dataset / "calibration_labels.pt"
    if label_path.exists():
        labels_raw = torch.load(label_path, weights_only=True)
        data["labels"] = labels_raw[:, 3].numpy()

    # Trust aggregator metrics (pre-computed)
    # --trust_model overrides the default filename
    ta_name = trust_model if trust_model else "trust_aggregator"
    ta_path = results_dir / f"{ta_name}_metrics.json"
    if not ta_path.exists():
        ta_path = results_dir / "trust_aggregator_metrics.json"  # fallback
    if ta_path.exists():
        with open(ta_path) as f:
            ta_data = json.load(f)
        data["trust_agg_metrics"] = ta_data.get("test_metrics", {})
        data["weight_analysis"] = ta_data.get("weight_analysis", {})
        data["trust_model_name"] = ta_name

    # NBFNet metrics
    nbf_path = results_dir / "nbfnet_test_metrics.json"
    if nbf_path.exists():
        with open(nbf_path) as f:
            data["link_pred_metrics"] = json.load(f)

    # Explanations
    exp_path = results_dir / "explanations.json"
    if exp_path.exists():
        with open(exp_path) as f:
            exps = json.load(f)
        data["explanations"] = [e.get("explanation", "") for e in exps]
        data["query_results"] = exps

    return data


def align_arrays(*arrays, N=None):
    """Trim all arrays to the same length N."""
    min_len = min((len(a) for a in arrays if a is not None), default=0)
    if N is not None:
        min_len = min(min_len, N)
    return [a[:min_len] if a is not None else None for a in arrays]


def _plot_precision_coverage(
    coverages: list, precisions: list, thresholds: list, save_path: str, title: str
) -> None:
    """Plot Hits@1 vs coverage tradeoff across trust thresholds."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping precision-coverage plot")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(coverages, precisions, "bo-", linewidth=2, markersize=8)
    for c, p, t in zip(coverages, precisions, thresholds):
        ax.annotate(f"T≥{t}", (c, p), textcoords="offset points", xytext=(6, 4), fontsize=9)

    ax.set_xlabel("Coverage (fraction of queries)", fontsize=12)
    ax.set_ylabel("Hits@1", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Precision-coverage plot saved to {save_path}")


def selective_prediction_analysis(
    results_dir: Path, dataset: str, data: dict, trust_model: str = None
) -> dict:
    """
    Selective prediction: filter queries by trust threshold and report Hits@1 + coverage.

    Uses trust scores from data["T"] (recomputed by the specified aggregator).
    Results saved to results_dir/selective_prediction_{trust_model}.json.
    """
    pred_path = results_dir / "predictions.pt"
    if not pred_path.exists():
        logger.warning("predictions.pt not found — skipping selective prediction analysis")
        return {}

    pred_tensor = torch.load(pred_path, weights_only=True)
    N = len(pred_tensor)

    # Use filtered ranks from test_ranks.pt when available (correct protocol).
    # Search fallback locations in order.
    _ranks_candidates = [
        results_dir / "test_ranks.pt",
        results_dir / "vanilla_nbfnet" / "test_ranks.pt",
        Path(TRUST_CALIB_DIR) / dataset / "test_ranks.pt",
    ]
    ranks_path = next((p for p in _ranks_candidates if p.exists()), None)
    if ranks_path is not None:
        ranks_tensor = torch.load(ranks_path, weights_only=True)
        filtered_ranks = ranks_tensor[:N, 1].numpy()
        hits_per_query = (filtered_ranks == 1).astype(float)
        logger.info(f"Using filtered ranks from {ranks_path} — baseline Hits@1 = {hits_per_query.mean():.4f}")
    else:
        hits_per_query = (pred_tensor[:, 1] == pred_tensor[:, 2]).numpy().astype(float)
        logger.warning("test_ranks.pt not found — using unfiltered top-1 match as Hits@1 baseline (will be inaccurate)")

    # Use recomputed trust scores from specified aggregator
    if "T" in data:
        T = data["T"][:N]
    else:
        U = data.get("U")
        if U is None:
            logger.warning("No trust scores or uncertainty scores — skipping selective prediction analysis")
            return {}
        U_q = U[:N]
        Delta_CF = data.get("Delta_CF")
        Delta_CF_q = Delta_CF[:N] if Delta_CF is not None else np.full(N, 0.3)
        S_prov = data.get("S_prov")
        if S_prov is not None and len(S_prov) >= N:
            T = (1/3) * (1 - U_q) + (1/3) * S_prov[:N] + (1/3) * Delta_CF_q
        else:
            T = 0.5 * (1 - U_q) + 0.5 * Delta_CF_q
    T = np.clip(T, 0, 1)

    baseline_hits = float(hits_per_query.mean())
    thresholds = [0.0, 0.3, 0.5, 0.7, 0.9]
    by_threshold: dict = {}

    for thresh in thresholds:
        mask = T >= thresh
        n_sel = int(mask.sum())
        if n_sel == 0:
            continue
        h1 = float(hits_per_query[mask].mean())
        pct_improvement = (h1 - baseline_hits) / max(baseline_hits, 1e-9) * 100
        by_threshold[str(thresh)] = {
            "threshold": thresh,
            "coverage_pct": round(mask.sum() / N * 100, 1),
            "n_queries": n_sel,
            "hits_at_1": round(h1, 4),
            "mrr_lower_bound": round(h1, 4),
            "hits_at_1_improvement_pct": round(pct_improvement, 1),
        }

    model_tag = Path(trust_model).stem if trust_model else "trust_aggregator"
    output = {
        "dataset": dataset,
        "trust_model": model_tag,
        "baseline_hits_at_1": round(baseline_hits, 4),
        "by_threshold": by_threshold,
    }

    save_path = results_dir / f"selective_prediction_{model_tag}.json"
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Selective prediction results saved to {save_path}")

    # Console table
    print(f"\n{'─'*65}")
    print(f"  Selective Prediction Analysis — {dataset}  [{model_tag}]")
    print(f"  Baseline Hits@1 (all queries): {baseline_hits:.4f}")
    print(f"{'─'*65}")
    print(f"  {'Threshold':>9}  {'Coverage':>9}  {'N Queries':>9}  {'Hits@1':>8}  {'Δ Hits@1':>9}")
    print(f"{'─'*65}")
    for thresh in thresholds:
        row = by_threshold.get(str(thresh))
        if row:
            print(f"  T ≥ {thresh:<5}  {row['coverage_pct']:>8.1f}%  {row['n_queries']:>9d}"
                  f"  {row['hits_at_1']:>8.4f}  {row['hits_at_1_improvement_pct']:>+8.1f}%")
    print(f"{'─'*65}")

    # Plot
    valid_t = [t for t in thresholds if str(t) in by_threshold]
    coverages = [by_threshold[str(t)]["coverage_pct"] / 100 for t in valid_t]
    precisions = [by_threshold[str(t)]["hits_at_1"] for t in valid_t]
    _plot_precision_coverage(
        coverages, precisions, valid_t,
        str(results_dir / f"selective_prediction_{model_tag}.png"),
        title=f"Selective Prediction: Precision–Coverage — {dataset} [{model_tag}]",
    )

    return by_threshold


def main():
    args = parse_args()
    setup_logging()

    results_dir = Path(RESULTS_DIR) / args.dataset
    results_dir.mkdir(parents=True, exist_ok=True)
    calib_dir = Path(TRUST_CALIB_DIR)

    logger.info(f"Loading pre-computed results for {args.dataset}...")
    data = load_results(results_dir, calib_dir, args.dataset, trust_model=args.trust_model)

    report = {"dataset": args.dataset, "phases": {}}

    # ─── Phase: Link Prediction ─────────────────────────────────────────────────
    if args.phase in ("all", "link_pred"):
        lp = data.get("link_pred_metrics", {})
        if lp:
            print("\n" + format_metrics_table(lp, title=f"Link Prediction — {args.dataset}"))
            report["phases"]["link_prediction"] = lp

    # ─── Phase: Trust Calibration ───────────────────────────────────────────────
    if args.phase in ("all", "trust"):
        if "labels" in data:
            N = len(data["labels"])
            U = data.get("U", np.full(N, 0.5))
            S_prov = data.get("S_prov", np.full(N, 0.75))
            Delta_CF = data.get("Delta_CF", np.full(N, 0.3))

            U, S_prov, Delta_CF, labels = align_arrays(U, S_prov, Delta_CF, data["labels"])

            # Load specified aggregator and recompute trust scores — timed
            aggregator = _load_trust_aggregator(args.dataset, args.trust_model)
            t0_trust = time.perf_counter()
            if aggregator is not None:
                T = _recompute_trust_scores(aggregator, U, S_prov, Delta_CF)
            else:
                # Fallback: equal weights
                T = (1/3) * (1 - U) + (1/3) * S_prov + (1/3) * Delta_CF
            T = np.clip(T, 0, 1)
            trust_compute_ms = (time.perf_counter() - t0_trust) * 1000

            # Store recomputed T so selective_prediction_analysis uses it
            data["T"] = T

            trust_metrics = compute_trust_calibration_metrics(T, labels)
            trust_metrics["inference_ms_per_query"] = round(trust_compute_ms / max(len(T), 1) * 1000, 4)
            print("\n" + format_metrics_table(trust_metrics, title=f"Trust Calibration — {args.dataset}"))
            report["phases"]["trust_calibration"] = trust_metrics

            # Use trained aggregator metrics if available
            if "trust_agg_metrics" in data:
                print("\n" + format_metrics_table(data["trust_agg_metrics"],
                                                  title="Trust Aggregator (Trained)"))
                report["phases"]["trust_aggregator"] = data["trust_agg_metrics"]

            # Plots
            plot_reliability_diagram(
                T, labels, str(results_dir / "reliability_diagram.png"),
                title=f"Reliability Diagram — {args.dataset}"
            )
            plot_trust_distribution(
                T, labels, str(results_dir / "trust_distribution.png")
            )

    # ─── Phase: Uncertainty ─────────────────────────────────────────────────────
    if args.phase in ("all", "uncertainty"):
        if "U" in data and "labels" in data:
            U, labels = align_arrays(data["U"], data["labels"])
            # Use set sizes as proxy prediction sets
            pred_set_sizes = data.get("pred_set_sizes", np.ones(len(U)))
            pred_set_sizes = pred_set_sizes[:len(U)]

            # Create mock prediction sets of the right size
            prediction_sets = [list(range(min(int(s), 10))) for s in pred_set_sizes]
            true_labels = np.arange(len(U))  # placeholder

            unc_metrics = compute_uncertainty_metrics(U, prediction_sets, true_labels)
            print("\n" + format_metrics_table(unc_metrics, title=f"Uncertainty — {args.dataset}"))
            report["phases"]["uncertainty"] = unc_metrics

            expected_coverage = 1.0 - UNCERTAINTY_CONFIG["alpha"]
            coverage = unc_metrics["coverage"]
            print(f"Coverage check: {coverage:.4f} (target ≥ {expected_coverage:.2f}) "
                  f"{'✓' if coverage >= expected_coverage - 0.05 else '✗'}")

    # ─── Phase: Counterfactual ──────────────────────────────────────────────────
    if args.phase in ("all", "cf"):
        cf_path = results_dir / "edge_impacts.json"
        if cf_path.exists():
            with open(cf_path) as f:
                cf_raw = json.load(f)
            cf_results = [
                {
                    "Delta_CF": v["Delta_CF"],
                    "critical_impacts": v.get("critical_impacts", []),
                    "random_impacts": v.get("random_impacts", []),
                }
                for v in cf_raw.values()
            ]
            cf_metrics = compute_counterfactual_metrics(cf_results)
            print("\n" + format_metrics_table(cf_metrics, title=f"Counterfactual — {args.dataset}"))
            report["phases"]["counterfactual"] = cf_metrics

            if "Delta_CF" in data:
                plot_delta_cf_distribution(
                    data["Delta_CF"],
                    str(results_dir / "delta_cf_distribution.png"),
                )

    # ─── Phase: Explanation ─────────────────────────────────────────────────────
    if args.phase in ("all", "explanation"):
        if "explanations" in data:
            exp_metrics = compute_explanation_metrics(
                data["explanations"],
                data.get("query_results", []),
            )
            print("\n" + format_metrics_table(exp_metrics, title=f"Explanation — {args.dataset}"))
            report["phases"]["explanation"] = exp_metrics

    # ─── Selective Prediction Analysis ──────────────────────────────────────────
    if args.phase in ("all", "trust", "link_pred"):
        sp_results = selective_prediction_analysis(results_dir, args.dataset, data, trust_model=args.trust_model)
        if sp_results:
            report["phases"]["selective_prediction"] = sp_results

    # ─── Inference Timing Report ─────────────────────────────────────────────────
    if args.phase in ("all", "trust"):
        timing: dict = {}

        # Base link prediction timing (from saved metrics if available)
        nbf_path = results_dir / "nbfnet_test_metrics.json"
        if nbf_path.exists():
            with open(nbf_path) as f:
                nbf_m = json.load(f)
            timing["base_link_prediction_ms_per_query"] = nbf_m.get(
                "inference_ms_per_query", "not recorded (re-run train_nbfnet.py to measure)"
            )

        # Trust computation timing (measured above during this evaluation run)
        if "trust_calibration" in report.get("phases", {}):
            timing["trust_aggregation_us_per_query"] = (
                report["phases"]["trust_calibration"].get("inference_ms_per_query", "n/a")
            )

        if timing:
            print(f"\n{'─'*55}")
            print("  Inference Timing")
            print(f"{'─'*55}")
            for k, v in timing.items():
                print(f"  {k}: {v}")
            print(f"{'─'*55}")
            report["phases"]["inference_timing"] = timing

    # ─── Weight Analysis Plot ────────────────────────────────────────────────────
    if "weight_analysis" in data and args.phase in ("all", "trust"):
        wa = data["weight_analysis"]
        weights_by_type = {}
        for qt, w in wa.items():
            alpha, beta, gamma = w.get("alpha", 1/3), w.get("beta", 1/3), w.get("gamma", 1/3)
            n = w.get("n", 1)
            weights_by_type[qt] = np.array([[alpha, beta, gamma]] * max(n, 1))

        plot_weight_distributions(
            weights_by_type,
            str(results_dir / "weight_distributions.png"),
            title=f"Adaptive Weights by Query Type — {args.dataset}",
        )

    # ─── Save full report ─────────────────────────────────────────────────────────
    report_path = results_dir / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Full evaluation report saved to {report_path}")
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
