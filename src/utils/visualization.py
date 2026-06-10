"""
Visualization utilities for trust scores, weights, and calibration plots.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _require_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting. Install it with: pip install matplotlib")


def plot_trust_score_histogram(
    trust_scores: np.ndarray,
    labels: np.ndarray,
    save_path: str,
    title: str = "Trust Score Distribution",
):
    """Plot histogram of trust scores separated by correct/incorrect predictions."""
    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 5))

    correct = trust_scores[labels == 1]
    incorrect = trust_scores[labels == 0]

    ax.hist(incorrect, bins=30, alpha=0.6, label="Incorrect (label=0)", color="red")
    ax.hist(correct, bins=30, alpha=0.6, label="Correct (label=1)", color="green")
    ax.set_xlabel("Trust Score T")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(0, 1)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Trust distribution plot saved to {save_path}")


def plot_component_scatter(
    U: np.ndarray,
    S_prov: np.ndarray,
    Delta_CF: np.ndarray,
    trust_scores: np.ndarray,
    save_path: str,
):
    """Scatter plot of each trust component vs final trust score."""
    plt = _require_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    components = [(1 - U, "1 - U (Certainty)"), (S_prov, "S_prov"), (Delta_CF, "Δ_CF")]
    for ax, (comp, label) in zip(axes, components):
        ax.scatter(comp, trust_scores, alpha=0.3, s=5)
        ax.set_xlabel(label)
        ax.set_ylabel("Trust Score T")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    fig.suptitle("Trust Components vs Final Score")
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Component scatter saved to {save_path}")


def plot_weight_distributions_by_type(
    weights_by_type: Dict[str, np.ndarray],
    save_path: str,
    title: str = "Learned Weights α, β, γ by Query Type",
):
    """
    Box/bar plot of α, β, γ distributions grouped by query type.

    This is the key novelty figure for the paper.

    Args:
        weights_by_type: Dict mapping query type → (N, 3) array of [α, β, γ] weights
        save_path: Where to save the figure
        title: Figure title
    """
    plt = _require_matplotlib()
    import matplotlib.patches as mpatches

    query_types = list(weights_by_type.keys())
    n_types = len(query_types)
    weight_names = ["α (Uncertainty)", "β (Provenance)", "γ (Counterfactual)"]
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)

    for col_idx, (weight_name, color) in enumerate(zip(weight_names, colors)):
        ax = axes[col_idx]
        data_per_type = []
        for qt in query_types:
            w = weights_by_type[qt]
            data_per_type.append(w[:, col_idx] if w.ndim == 2 else w)

        bp = ax.boxplot(data_per_type, patch_artist=True, notch=False)
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xticks(range(1, n_types + 1))
        ax.set_xticklabels(query_types, rotation=15)
        ax.set_title(weight_name)
        ax.set_ylabel("Weight Value")
        ax.set_ylim(0, 1)
        ax.axhline(1.0 / 3, linestyle="--", color="gray", alpha=0.5, label="Equal (1/3)")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Weight distribution plot saved to {save_path}")


def plot_training_curves(
    history: List[Dict],
    save_path: str,
    title: str = "Training Curves",
):
    """Plot loss and MRR training curves."""
    plt = _require_matplotlib()
    steps = [h["step"] for h in history if h.get("loss") is not None]
    losses = [h["loss"] for h in history if h.get("loss") is not None]
    mrr_steps = [h["step"] for h in history if h.get("valid_mrr") is not None]
    mrrs = [h["valid_mrr"] for h in history if h.get("valid_mrr") is not None]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    if steps and losses:
        ax1.plot(steps, losses, label="Train Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Loss")
        ax1.legend()

    if mrr_steps and mrrs:
        ax2.plot(mrr_steps, mrrs, label="Validation MRR", color="orange")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("MRR")
        ax2.set_title("Validation MRR")
        ax2.legend()

    fig.suptitle(title)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Training curves saved to {save_path}")
