"""
Baseline comparison and ablation study for the trustworthy KG framework.

Baselines:
  1. vanilla_nbfnet:       Standard NBFNet, S_prov = 1.0 everywhere
  2. nbfnet_uncertainty:   NBFNet + conformal only (β=γ=0, α=1)
  3. nbfnet_provenance:    NBFNet + provenance only (α=γ=0, β=1)
  4. fixed_weight_trust:   T = 0.33*(1-U) + 0.33*S_prov + 0.33*Delta_CF
  5. full_system:          Complete framework with learned α,β,γ

Ablations:
  - remove_uncertainty:    T = β·S_prov + γ·Δ_CF  (renormalized)
  - remove_provenance:     T = α(1-U) + γ·Δ_CF    (renormalized)
  - remove_counterfactual: T = α(1-U) + β·S_prov   (renormalized)
  - fixed_vs_learned:      Compare fixed 0.33/0.33/0.33 vs learned
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy import stats

logger = logging.getLogger(__name__)


class BaselineComparison:
    """
    Run all baselines and the full system for comparison.

    Args:
        trust_aggregator: Trained AttentionBasedTrustAggregator (for full system)
        device: Torch device
    """

    def __init__(self, trust_aggregator=None, device: torch.device = None):
        self.trust_aggregator = trust_aggregator
        self.device = device or torch.device("cpu")

    def run_all_baselines(
        self,
        dataset_name: str,
        trust_data: Dict,
    ) -> Dict:
        """
        Evaluate all 5 baselines on pre-computed trust components.

        Args:
            dataset_name: Dataset identifier
            trust_data: Dict with keys:
                U, S_prov, Delta_CF: (N,) arrays of trust components
                labels: (N,) binary labels
                link_pred_metrics: dict of MRR/Hits from trained NBFNet

        Returns:
            results_dict mapping baseline_name → metric dict
        """
        U = np.array(trust_data["U"], dtype=np.float64)
        S_prov = np.array(trust_data["S_prov"], dtype=np.float64)
        Delta_CF = np.array(trust_data["Delta_CF"], dtype=np.float64)
        labels = np.array(trust_data["labels"], dtype=np.float64)

        results = {}

        # 1. Vanilla NBFNet: ignore uncertainty, provenance, CF → use raw score
        results["vanilla_nbfnet"] = self._eval_fixed_weights(
            U, S_prov, Delta_CF, labels,
            alpha=0.0, beta=0.0, gamma=0.0,
            fallback_score=trust_data.get("confidence", np.zeros_like(U)),
        )
        results["vanilla_nbfnet"]["description"] = "Standard NBFNet (no trust components)"

        # 2. NBFNet + Uncertainty only (α=1, β=0, γ=0)
        results["nbfnet_uncertainty"] = self._eval_fixed_weights(
            U, S_prov, Delta_CF, labels, alpha=1.0, beta=0.0, gamma=0.0
        )
        results["nbfnet_uncertainty"]["description"] = "NBFNet + Conformal Uncertainty only"

        # 3. NBFNet + Provenance only (α=0, β=1, γ=0)
        results["nbfnet_provenance"] = self._eval_fixed_weights(
            U, S_prov, Delta_CF, labels, alpha=0.0, beta=1.0, gamma=0.0
        )
        results["nbfnet_provenance"]["description"] = "NBFNet + Provenance only"

        # 4. Fixed equal weights (0.33, 0.33, 0.33)
        results["fixed_weight_trust"] = self._eval_fixed_weights(
            U, S_prov, Delta_CF, labels, alpha=1/3, beta=1/3, gamma=1/3
        )
        results["fixed_weight_trust"]["description"] = "Fixed equal weights (1/3, 1/3, 1/3)"

        # 5. Full system: learned α,β,γ from trust aggregator
        results["full_system"] = self._eval_full_system(U, S_prov, Delta_CF, labels)
        results["full_system"]["description"] = "Full system: learned adaptive weights"

        # Add link prediction metrics to all (from NBFNet — same for all)
        lp = trust_data.get("link_pred_metrics", {})
        for name in results:
            results[name].update({k: lp.get(k, 0.0) for k in ["mrr", "hits@1", "hits@3", "hits@10"]})

        return results

    def _eval_fixed_weights(
        self,
        U: np.ndarray,
        S_prov: np.ndarray,
        Delta_CF: np.ndarray,
        labels: np.ndarray,
        alpha: float,
        beta: float,
        gamma: float,
        fallback_score: Optional[np.ndarray] = None,
    ) -> Dict:
        """Compute trust scores with fixed weights and evaluate."""
        certainty = 1.0 - U
        w_sum = alpha + beta + gamma

        if w_sum < 1e-8:
            # Vanilla: use raw confidence if available
            T = fallback_score if fallback_score is not None else certainty
            T = np.array(T, dtype=np.float64).clip(0, 1)
        else:
            T = (alpha * certainty + beta * S_prov + gamma * Delta_CF) / w_sum
            T = np.clip(T, 0, 1)

        from src.evaluation.metrics import compute_trust_calibration_metrics
        metrics = compute_trust_calibration_metrics(T, labels)
        metrics["weights"] = {"alpha": alpha, "beta": beta, "gamma": gamma}
        return metrics

    def _eval_full_system(
        self,
        U: np.ndarray,
        S_prov: np.ndarray,
        Delta_CF: np.ndarray,
        labels: np.ndarray,
    ) -> Dict:
        """Evaluate the full system with learned weights."""
        if self.trust_aggregator is None:
            # Fallback to equal weights
            return self._eval_fixed_weights(U, S_prov, Delta_CF, labels, 1/3, 1/3, 1/3)

        self.trust_aggregator.eval()
        U_t = torch.tensor(U, dtype=torch.float)
        S_t = torch.tensor(S_prov, dtype=torch.float)
        CF_t = torch.tensor(Delta_CF, dtype=torch.float)

        with torch.no_grad():
            T_t, weights_t = self.trust_aggregator(U_t, S_t, CF_t)

        T = T_t.numpy()
        mean_weights = weights_t.mean(dim=0).tolist()

        from src.evaluation.metrics import compute_trust_calibration_metrics
        metrics = compute_trust_calibration_metrics(T, labels)
        metrics["weights"] = {
            "alpha": mean_weights[0],
            "beta": mean_weights[1],
            "gamma": mean_weights[2],
        }
        return metrics

    def generate_comparison_table(self, results_dict: Dict) -> str:
        """
        Format baseline comparison as a readable table.

        Args:
            results_dict: Output of run_all_baselines()

        Returns:
            Formatted string table
        """
        lines = []
        lines.append("\n" + "=" * 90)
        lines.append("  BASELINE COMPARISON")
        lines.append("=" * 90)
        header = f"{'Method':<25} | {'Acc':>7} | {'AUC-ROC':>8} | {'AUC-PR':>7} | {'ECE':>6} | {'MRR':>7} | {'H@10':>7}"
        lines.append(header)
        lines.append("-" * 90)

        for name, metrics in results_dict.items():
            row = (
                f"{name:<25} | "
                f"{metrics.get('accuracy', 0):.4f}  | "
                f"{metrics.get('auc_roc', 0):.4f}   | "
                f"{metrics.get('auc_pr', 0):.4f}  | "
                f"{metrics.get('ece', 0):.4f} | "
                f"{metrics.get('mrr', 0):.4f}  | "
                f"{metrics.get('hits@10', 0):.4f}"
            )
            lines.append(row)

        lines.append("=" * 90)
        return "\n".join(lines)

    def run_ablation_study(
        self,
        trust_data: Dict,
        ablation_configs: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        Run ablation study removing each trust component one at a time.

        Ablation configs:
            - remove_uncertainty: T = β·S_prov + γ·Δ_CF (renormalized)
            - remove_provenance:  T = α(1-U) + γ·Δ_CF (renormalized)
            - remove_counterfactual: T = α(1-U) + β·S_prov (renormalized)
            - fixed_vs_learned: 0.33/0.33/0.33 vs learned

        Returns:
            Dict mapping ablation_name → metrics
        """
        U = np.array(trust_data["U"], dtype=np.float64)
        S_prov = np.array(trust_data["S_prov"], dtype=np.float64)
        Delta_CF = np.array(trust_data["Delta_CF"], dtype=np.float64)
        labels = np.array(trust_data["labels"], dtype=np.float64)

        ablations = {
            "full_system": self._eval_full_system(U, S_prov, Delta_CF, labels),
            "remove_uncertainty": self._eval_fixed_weights(
                U, S_prov, Delta_CF, labels, alpha=0.0, beta=0.5, gamma=0.5
            ),
            "remove_provenance": self._eval_fixed_weights(
                U, S_prov, Delta_CF, labels, alpha=0.5, beta=0.0, gamma=0.5
            ),
            "remove_counterfactual": self._eval_fixed_weights(
                U, S_prov, Delta_CF, labels, alpha=0.5, beta=0.5, gamma=0.0
            ),
            "fixed_equal": self._eval_fixed_weights(
                U, S_prov, Delta_CF, labels, alpha=1/3, beta=1/3, gamma=1/3
            ),
        }

        return ablations

    def statistical_significance_test(
        self,
        results_a: np.ndarray,
        results_b: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Paired t-test between two result arrays (e.g., trust scores from two systems).

        Args:
            results_a: (N,) scores from system A
            results_b: (N,) scores from system B

        Returns:
            (t_statistic, p_value)
        """
        t_stat, p_value = stats.ttest_rel(results_a, results_b)
        return float(t_stat), float(p_value)


def generate_latex_table(results_by_dataset: Dict, caption: str = "") -> str:
    """
    Generate a LaTeX table from multi-dataset comparison results.

    Args:
        results_by_dataset: Dict mapping dataset_name → baseline_results_dict
        caption: LaTeX table caption

    Returns:
        LaTeX table string (ready to paste into paper)
    """
    methods = ["vanilla_nbfnet", "nbfnet_uncertainty", "nbfnet_provenance",
               "fixed_weight_trust", "full_system"]
    method_display = {
        "vanilla_nbfnet": "Vanilla NBFNet",
        "nbfnet_uncertainty": "NBFNet + Uncertainty",
        "nbfnet_provenance": "NBFNet + Provenance",
        "fixed_weight_trust": "Fixed Weights (1/3, 1/3, 1/3)",
        "full_system": "\\textbf{Full System (Ours)}",
    }

    datasets = list(results_by_dataset.keys())

    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{" + caption + "}",
        "\\label{tab:main_results}",
    ]

    # Column spec
    ncols = 1 + len(datasets) * 3
    col_spec = "l" + "ccc" * len(datasets)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # Header
    dataset_headers = " & ".join(
        f"\\multicolumn{{3}}{{c}}{{{ds.upper()}}}" for ds in datasets
    )
    lines.append(f"Method & {dataset_headers} \\\\")

    sub_headers = " & ".join(
        "MRR & H@10 & AUC" for _ in datasets
    )
    lines.append(f" & {sub_headers} \\\\")
    lines.append("\\midrule")

    # Rows
    for method in methods:
        row_parts = [method_display.get(method, method)]
        for ds in datasets:
            m = results_by_dataset.get(ds, {}).get(method, {})
            mrr = m.get("mrr", 0.0)
            h10 = m.get("hits@10", 0.0)
            auc = m.get("auc_roc", 0.0)
            row_parts.append(f"{mrr:.3f} & {h10:.3f} & {auc:.3f}")
        lines.append(" & ".join(row_parts) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
    ])

    return "\n".join(lines)
