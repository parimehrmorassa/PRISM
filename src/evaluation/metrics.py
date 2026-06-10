"""
Evaluation metrics for trust-aware KG link prediction.

Implements:
  - MRR, Hits@K, MR  (link prediction quality)
  - Trust calibration: accuracy, AUC-ROC, AUC-PR, ECE, F1
  - Uncertainty metrics: coverage, set size, error correlation
  - Counterfactual metrics: mean Delta_CF, critical vs random significance
  - Explanation metrics: mean LExT, completeness, mean length
"""

import logging
from typing import Dict, List, Optional

import numpy as np
from scipy import stats
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
)

logger = logging.getLogger(__name__)


# ─── Link Prediction Metrics ──────────────────────────────────────────────────

def compute_mrr(ranks: np.ndarray) -> float:
    """
    Compute Mean Reciprocal Rank.

    Args:
        ranks: (N,) array of positive integer ranks (1-indexed)

    Returns:
        MRR ∈ [0, 1]
    """
    ranks = np.array(ranks, dtype=np.float64)
    return float(np.mean(1.0 / ranks))


def compute_hits_at_k(ranks: np.ndarray, k: int) -> float:
    """
    Compute Hits@K (fraction of queries where rank ≤ k).

    Args:
        ranks: (N,) array of ranks (1-indexed)
        k: Threshold

    Returns:
        Hits@K ∈ [0, 1]
    """
    ranks = np.array(ranks, dtype=np.float64)
    return float(np.mean(ranks <= k))


def compute_mean_rank(ranks: np.ndarray) -> float:
    """
    Compute Mean Rank (lower is better).

    Args:
        ranks: (N,) array of ranks

    Returns:
        Mean rank (positive float)
    """
    return float(np.mean(np.array(ranks, dtype=np.float64)))


def compute_link_prediction_metrics(ranks: np.ndarray) -> Dict:
    """
    Compute all standard link prediction metrics from rank array.

    Returns:
        dict with mrr, hits@1, hits@3, hits@10, mr
    """
    ranks = np.array(ranks, dtype=np.float64)
    return {
        "mrr": compute_mrr(ranks),
        "hits@1": compute_hits_at_k(ranks, 1),
        "hits@3": compute_hits_at_k(ranks, 3),
        "hits@10": compute_hits_at_k(ranks, 10),
        "mr": compute_mean_rank(ranks),
    }


# ─── Trust Calibration Metrics ────────────────────────────────────────────────

def compute_expected_calibration_error(
    scores: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    Partitions [0, 1] into num_bins equal-width bins.
    ECE = sum_b (|b| / N) * |avg_confidence_b - avg_accuracy_b|

    Args:
        scores: (N,) predicted probabilities ∈ [0, 1]
        labels: (N,) binary ground truth labels
        num_bins: Number of calibration bins

    Returns:
        ECE ∈ [0, 1] (lower is better)
    """
    scores = np.array(scores, dtype=np.float64)
    labels = np.array(labels, dtype=np.float64)
    n = len(scores)
    ece = 0.0

    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    for i in range(num_bins):
        mask = (scores >= bin_edges[i]) & (scores < bin_edges[i + 1])
        if i == num_bins - 1:  # include right edge in last bin
            mask = (scores >= bin_edges[i]) & (scores <= bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_confidence = float(scores[mask].mean())
        bin_accuracy = float(labels[mask].mean())
        bin_weight = float(mask.sum()) / n
        ece += bin_weight * abs(bin_confidence - bin_accuracy)

    return float(ece)


def compute_trust_calibration_metrics(
    trust_scores: np.ndarray,
    labels: np.ndarray,
) -> Dict:
    """
    Compute all trust calibration metrics.

    Args:
        trust_scores: (N,) trust scores ∈ [0, 1]
        labels: (N,) binary labels (1 = correct prediction)

    Returns:
        dict with: accuracy, auc_roc, auc_pr, ece, f1
    """
    trust_scores = np.array(trust_scores, dtype=np.float64)
    labels = np.array(labels, dtype=np.float64)

    preds = (trust_scores >= 0.5).astype(int)
    accuracy = float(accuracy_score(labels, preds))
    f1 = float(f1_score(labels, preds, zero_division=0))

    if len(np.unique(labels)) > 1:
        auc_roc = float(roc_auc_score(labels, trust_scores))
        auc_pr = float(average_precision_score(labels, trust_scores))
    else:
        auc_roc = 0.5
        auc_pr = float(np.mean(labels))

    ece = compute_expected_calibration_error(trust_scores, labels)

    return {
        "accuracy": accuracy,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "ece": ece,
        "f1": f1,
    }


# ─── Uncertainty Metrics ──────────────────────────────────────────────────────

def compute_uncertainty_metrics(
    uncertainty_scores: np.ndarray,
    prediction_sets: List[List[int]],
    true_labels: np.ndarray,
) -> Dict:
    """
    Compute uncertainty quantification metrics.

    Args:
        uncertainty_scores: (N,) — U values ∈ [0, 1]
        prediction_sets:    N-length list of prediction set entity ID lists
        true_labels:        (N,) — true tail entity IDs

    Returns:
        dict with: coverage, avg_set_size, uncertainty_error_correlation
    """
    uncertainty_scores = np.array(uncertainty_scores, dtype=np.float64)
    true_labels = np.array(true_labels)
    N = len(true_labels)

    # Coverage: fraction of queries where true answer is in prediction set
    in_set = []
    for i, pred_set in enumerate(prediction_sets):
        in_set.append(int(true_labels[i]) in pred_set)
    coverage = float(np.mean(in_set))

    # Average prediction set size
    set_sizes = np.array([len(ps) for ps in prediction_sets], dtype=np.float64)
    avg_set_size = float(set_sizes.mean())

    # Correlation between uncertainty and prediction error
    # (higher U should correlate with wrong predictions)
    errors = (1 - np.array(in_set, dtype=np.float64))  # 1 = error
    if len(np.unique(errors)) > 1 and len(np.unique(uncertainty_scores)) > 1:
        corr, p_val = stats.pearsonr(uncertainty_scores, errors)
        uncertainty_error_correlation = float(corr)
    else:
        uncertainty_error_correlation = 0.0

    return {
        "coverage": coverage,
        "avg_set_size": avg_set_size,
        "uncertainty_error_correlation": uncertainty_error_correlation,
    }


# ─── Counterfactual Metrics ───────────────────────────────────────────────────

def compute_counterfactual_metrics(counterfactual_results: List[Dict]) -> Dict:
    """
    Compute counterfactual sensitivity metrics.

    Args:
        counterfactual_results: List of dicts, each with:
            - Delta_CF: float
            - critical_impacts: List[float]
            - random_impacts: List[float]

    Returns:
        dict with: mean_delta_cf, std_delta_cf, critical_vs_random_pvalue
    """
    delta_cfs = np.array(
        [r.get("Delta_CF", 0.0) for r in counterfactual_results], dtype=np.float64
    )
    all_critical = []
    all_random = []
    for r in counterfactual_results:
        all_critical.extend(r.get("critical_impacts", []))
        all_random.extend(r.get("random_impacts", []))

    t_stat, p_value = 1.0, 1.0
    if all_critical and all_random:
        t_stat, p_value = stats.ttest_ind(all_critical, all_random, alternative="greater")

    return {
        "mean_delta_cf": float(delta_cfs.mean()) if len(delta_cfs) > 0 else 0.0,
        "std_delta_cf": float(delta_cfs.std()) if len(delta_cfs) > 0 else 0.0,
        "mean_critical_impact": float(np.mean(all_critical)) if all_critical else 0.0,
        "mean_random_impact": float(np.mean(all_random)) if all_random else 0.0,
        "ttest_statistic": float(t_stat),
        "critical_vs_random_pvalue": float(p_value),
    }


# ─── Explanation Metrics ──────────────────────────────────────────────────────

def compute_explanation_metrics(
    explanations: List[str],
    query_results: List[Dict],
) -> Dict:
    """
    Compute explanation quality metrics.

    Args:
        explanations: List of explanation strings
        query_results: List of query result dicts (with is_faithful field)

    Returns:
        dict with: pct_faithful, mean_word_count
    """
    if not explanations:
        return {"pct_faithful": 0.0, "mean_word_count": 0.0}

    num_faithful = sum(1 for qr in query_results if qr.get("is_faithful", False))
    pct_faithful = 100.0 * num_faithful / len(explanations)

    word_counts = [len(e.split()) for e in explanations]
    mean_word_count = float(np.mean(word_counts))

    return {
        "pct_faithful": pct_faithful,
        "mean_word_count": mean_word_count,
    }
