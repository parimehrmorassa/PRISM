"""
Calibration plots and visualization for trust scores, weights, and reliability.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _get_plt():
    """Get matplotlib.pyplot with Agg backend (safe for headless servers)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError("Install matplotlib: pip install matplotlib")


def plot_reliability_diagram(
    scores: np.ndarray,
    labels: np.ndarray,
    save_path: str,
    title: str = "Reliability Diagram",
    num_bins: int = 10,
):
    """
    Plot calibration reliability diagram (confidence vs accuracy).

    A perfectly calibrated model would lie on the diagonal.

    Args:
        scores: (N,) predicted probabilities ∈ [0, 1]
        labels: (N,) binary ground truth labels
        save_path: Output file path
        title: Plot title
        num_bins: Number of calibration bins
    """
    plt = _get_plt()
    scores = np.array(scores, dtype=np.float64)
    labels = np.array(labels, dtype=np.float64)

    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_confidences = []
    bin_accuracies = []
    bin_sizes = []

    for i in range(num_bins):
        if i == num_bins - 1:
            mask = (scores >= bin_edges[i]) & (scores <= bin_edges[i + 1])
        else:
            mask = (scores >= bin_edges[i]) & (scores < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_confidences.append(scores[mask].mean())
        bin_accuracies.append(labels[mask].mean())
        bin_sizes.append(mask.sum())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Reliability diagram
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax.bar(
        bin_confidences,
        bin_accuracies,
        width=1.0 / num_bins,
        alpha=0.6,
        color="steelblue",
        label="Model",
    )
    ax.set_xlabel("Mean Predicted Confidence")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Reliability Diagram")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Score distribution histogram
    ax2 = axes[1]
    ax2.hist(scores[labels == 1], bins=20, alpha=0.6, label="Correct (1)", color="green")
    ax2.hist(scores[labels == 0], bins=20, alpha=0.6, label="Incorrect (0)", color="red")
    ax2.set_xlabel("Trust Score")
    ax2.set_ylabel("Count")
    ax2.set_title("Score Distribution")
    ax2.legend()

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Reliability diagram saved to {save_path}")


def plot_trust_distribution(
    trust_scores: np.ndarray,
    labels: np.ndarray,
    save_path: str,
    title: str = "Trust Score Distribution",
):
    """
    Plot histograms of trust scores split by correct/incorrect predictions.

    Args:
        trust_scores: (N,) trust scores ∈ [0, 1]
        labels: (N,) binary labels
        save_path: Output file path
        title: Plot title
    """
    plt = _get_plt()
    trust_scores = np.array(trust_scores)
    labels = np.array(labels)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(trust_scores[labels == 0], bins=30, alpha=0.6, color="red", label="Incorrect (0)")
    ax.hist(trust_scores[labels == 1], bins=30, alpha=0.6, color="green", label="Correct (1)")
    ax.set_xlabel("Trust Score T")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(0, 1)

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Trust distribution saved to {save_path}")


def plot_weight_distributions(
    weights_by_type: Dict[str, np.ndarray],
    save_path: str,
    title: str = "Learned Weights α, β, γ by Query Type",
):
    """
    Generate the key novelty figure: box/bar plot of α, β, γ by query type.

    This is the central paper figure showing adaptive weight learning.

    Args:
        weights_by_type: Dict mapping query_type → (N, 3) array of [α, β, γ]
        save_path: Output file path
        title: Figure title
    """
    plt = _get_plt()
    query_types = list(weights_by_type.keys())
    n_types = len(query_types)

    if n_types == 0:
        logger.warning("No data to plot weight distributions")
        return

    weight_names = ["α (Uncertainty)", "β (Provenance)", "γ (Counterfactual)"]
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)

    for col_idx, (weight_name, color) in enumerate(zip(weight_names, colors)):
        ax = axes[col_idx]
        data_per_type = []
        for qt in query_types:
            arr = weights_by_type[qt]
            if arr.ndim == 2 and arr.shape[1] >= 3:
                data_per_type.append(arr[:, col_idx])
            elif arr.ndim == 1:
                data_per_type.append(arr)
            else:
                data_per_type.append(np.array([1 / 3]))

        if data_per_type and any(len(d) > 0 for d in data_per_type):
            bp = ax.boxplot(
                [d for d in data_per_type if len(d) > 0],
                patch_artist=True,
                notch=False,
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        valid_labels = [qt for qt, d in zip(query_types, data_per_type) if len(d) > 0]
        ax.set_xticks(range(1, len(valid_labels) + 1))
        ax.set_xticklabels(valid_labels, rotation=15, ha="right")
        ax.set_title(weight_name, fontsize=12)
        ax.set_ylabel("Weight Value")
        ax.set_ylim(0, 1)
        ax.axhline(1.0 / 3, linestyle="--", color="gray", alpha=0.5, label="Equal (1/3)")
        ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Weight distribution plot saved to {save_path}")


def plot_delta_cf_distribution(
    delta_cf_scores: np.ndarray,
    save_path: str,
    title: str = "Counterfactual Sensitivity Distribution",
    critical_impacts: Optional[np.ndarray] = None,
    random_impacts: Optional[np.ndarray] = None,
):
    """
    Plot the distribution of Delta_CF scores with optional critical vs. random comparison.

    Args:
        delta_cf_scores: (N,) Delta_CF values ∈ [0, 1]
        save_path: Output file path
        title: Plot title
        critical_impacts: Optional (K,) array of critical edge impacts
        random_impacts: Optional (K,) array of random edge impacts
    """
    plt = _get_plt()
    delta_cf_scores = np.array(delta_cf_scores)

    n_plots = 2 if (critical_impacts is not None and random_impacts is not None) else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    # Delta_CF distribution
    ax = axes[0]
    ax.hist(delta_cf_scores, bins=30, color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(delta_cf_scores.mean(), color="red", linestyle="--",
               label=f"Mean={delta_cf_scores.mean():.3f}")
    ax.set_xlabel("Δ_CF (Counterfactual Sensitivity)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.legend()

    # Critical vs random comparison
    if n_plots == 2:
        ax2 = axes[1]
        ax2.hist(critical_impacts, bins=20, alpha=0.6, color="orange", label="Critical edges")
        ax2.hist(random_impacts, bins=20, alpha=0.6, color="gray", label="Random edges")
        ax2.set_xlabel("Edge Impact (delta_e)")
        ax2.set_ylabel("Count")
        ax2.set_title("Critical vs. Random Edge Impacts")
        ax2.legend()

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Delta_CF distribution saved to {save_path}")
