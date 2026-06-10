"""
Phase 6 test suite for evaluation metrics.

Tests:
    - test_mrr_range
    - test_hits_at_k_range
    - test_ece_nonnegative
    - test_coverage_with_alpha
    - test_full_system_beats_baselines
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import (
    compute_mrr,
    compute_hits_at_k,
    compute_mean_rank,
    compute_expected_calibration_error,
    compute_trust_calibration_metrics,
    compute_uncertainty_metrics,
    compute_counterfactual_metrics,
    compute_explanation_metrics,
    compute_link_prediction_metrics,
)
from config import UNCERTAINTY_CONFIG, RANDOM_SEED

np.random.seed(RANDOM_SEED)


class TestLinkPredictionMetrics(unittest.TestCase):

    def test_mrr_range(self):
        """MRR must be in [0, 1]."""
        for _ in range(100):
            N = np.random.randint(10, 500)
            ranks = np.random.randint(1, 1000, N)
            mrr = compute_mrr(ranks)
            self.assertGreaterEqual(mrr, 0.0, f"MRR={mrr} < 0")
            self.assertLessEqual(mrr, 1.0, f"MRR={mrr} > 1")

    def test_mrr_perfect(self):
        """MRR should be 1.0 when all ranks are 1."""
        ranks = np.ones(100)
        self.assertAlmostEqual(compute_mrr(ranks), 1.0)

    def test_hits_at_k_range(self):
        """All Hits@K metrics must be in [0, 1]."""
        for k in [1, 3, 10]:
            for _ in range(50):
                N = np.random.randint(10, 200)
                ranks = np.random.randint(1, 100, N)
                h = compute_hits_at_k(ranks, k)
                self.assertGreaterEqual(h, 0.0, f"Hits@{k}={h} < 0")
                self.assertLessEqual(h, 1.0, f"Hits@{k}={h} > 1")

    def test_hits_at_k_monotone(self):
        """Hits@K must be non-decreasing in K."""
        ranks = np.random.randint(1, 50, 200)
        h1 = compute_hits_at_k(ranks, 1)
        h3 = compute_hits_at_k(ranks, 3)
        h10 = compute_hits_at_k(ranks, 10)
        self.assertLessEqual(h1, h3)
        self.assertLessEqual(h3, h10)

    def test_mean_rank_positive(self):
        """Mean rank must be positive."""
        ranks = np.random.randint(1, 200, 100)
        mr = compute_mean_rank(ranks)
        self.assertGreater(mr, 0.0)

    def test_compute_link_prediction_metrics(self):
        """compute_link_prediction_metrics must return all expected keys."""
        ranks = np.array([1, 2, 5, 10, 20, 100])
        metrics = compute_link_prediction_metrics(ranks)
        for key in ["mrr", "hits@1", "hits@3", "hits@10", "mr"]:
            self.assertIn(key, metrics)
            self.assertIsInstance(metrics[key], float)


class TestECE(unittest.TestCase):

    def test_ece_nonnegative(self):
        """ECE must always be non-negative."""
        for _ in range(100):
            N = np.random.randint(50, 500)
            scores = np.random.rand(N)
            labels = np.random.randint(0, 2, N).astype(float)
            ece = compute_expected_calibration_error(scores, labels)
            self.assertGreaterEqual(ece, 0.0, f"ECE={ece} < 0")

    def test_ece_perfect_calibration(self):
        """Perfectly calibrated model should have ECE near 0."""
        N = 10000
        scores = np.random.rand(N)
        # Perfect calibration: score = probability
        labels = np.random.binomial(1, scores)
        ece = compute_expected_calibration_error(scores, labels)
        self.assertLess(ece, 0.05, f"ECE={ece:.4f} unexpectedly high for calibrated model")

    def test_ece_upper_bounded(self):
        """ECE must be ≤ 1."""
        for _ in range(50):
            N = np.random.randint(50, 300)
            scores = np.random.rand(N)
            labels = np.random.randint(0, 2, N).astype(float)
            ece = compute_expected_calibration_error(scores, labels)
            self.assertLessEqual(ece, 1.0, f"ECE={ece} > 1")


class TestTrustCalibrationMetrics(unittest.TestCase):

    def test_accuracy_range(self):
        """Accuracy must be in [0, 1]."""
        for _ in range(50):
            N = np.random.randint(50, 500)
            scores = np.random.rand(N)
            labels = np.random.randint(0, 2, N).astype(float)
            m = compute_trust_calibration_metrics(scores, labels)
            self.assertGreaterEqual(m["accuracy"], 0.0)
            self.assertLessEqual(m["accuracy"], 1.0)

    def test_auc_roc_range(self):
        """AUC-ROC must be in [0, 1]."""
        N = 200
        scores = np.random.rand(N)
        labels = np.random.randint(0, 2, N).astype(float)
        m = compute_trust_calibration_metrics(scores, labels)
        self.assertGreaterEqual(m["auc_roc"], 0.0)
        self.assertLessEqual(m["auc_roc"], 1.0)

    def test_all_keys_present(self):
        """compute_trust_calibration_metrics must return all expected keys."""
        scores = np.random.rand(100)
        labels = np.random.randint(0, 2, 100).astype(float)
        m = compute_trust_calibration_metrics(scores, labels)
        for key in ["accuracy", "auc_roc", "auc_pr", "ece", "f1"]:
            self.assertIn(key, m)


class TestUncertaintyMetrics(unittest.TestCase):

    def test_coverage_with_alpha(self):
        """
        Coverage should be at or above (1 - alpha - small_margin) when
        prediction sets are constructed from a calibrated conformal quantifier.
        """
        alpha = UNCERTAINTY_CONFIG["alpha"]
        N = 500
        target_coverage = 1 - alpha

        # Simulate calibrated prediction sets:
        # Each prediction set contains the true answer ~90% of the time
        uncertainty_scores = np.random.rand(N) * 0.3
        true_labels = np.arange(N)
        prediction_sets = []

        for i in range(N):
            # Include true answer 90% of the time
            ps = list(range(5))
            if np.random.rand() < target_coverage:
                ps.append(true_labels[i])
            prediction_sets.append(ps)

        metrics = compute_uncertainty_metrics(uncertainty_scores, prediction_sets, true_labels)
        self.assertGreaterEqual(
            metrics["coverage"],
            target_coverage - 0.10,  # allow 10% margin in test
            f"Coverage {metrics['coverage']:.4f} too low (target={target_coverage:.4f})"
        )

    def test_avg_set_size_nonnegative(self):
        """Average prediction set size must be non-negative."""
        U = np.random.rand(100)
        pred_sets = [list(range(np.random.randint(0, 20))) for _ in range(100)]
        true_labels = np.arange(100)
        metrics = compute_uncertainty_metrics(U, pred_sets, true_labels)
        self.assertGreaterEqual(metrics["avg_set_size"], 0.0)


class TestCounterfactualMetrics(unittest.TestCase):

    def test_delta_cf_range(self):
        """mean_delta_cf must be in [0, 1]."""
        N = 100
        results = [
            {
                "Delta_CF": float(np.random.rand()),
                "critical_impacts": list(np.random.rand(5)),
                "random_impacts": list(np.random.rand(5)),
            }
            for _ in range(N)
        ]
        m = compute_counterfactual_metrics(results)
        self.assertGreaterEqual(m["mean_delta_cf"], 0.0)
        self.assertLessEqual(m["mean_delta_cf"], 1.0)

    def test_pvalue_is_float(self):
        """critical_vs_random_pvalue must be a float in [0, 1]."""
        results = [
            {
                "Delta_CF": 0.4,
                "critical_impacts": [0.5, 0.6, 0.4],
                "random_impacts": [0.1, 0.2, 0.15],
            }
        ] * 50
        m = compute_counterfactual_metrics(results)
        self.assertIsInstance(m["critical_vs_random_pvalue"], float)
        self.assertGreaterEqual(m["critical_vs_random_pvalue"], 0.0)
        self.assertLessEqual(m["critical_vs_random_pvalue"], 1.0)


class TestBaselineComparison(unittest.TestCase):

    def test_full_system_beats_baselines(self):
        """
        The full system (learned weights on correlated data) should
        achieve AUC >= fixed_weight baseline.
        """
        from src.models.trust.aggregator import (
            AttentionBasedTrustAggregator, TrustCalibrationTrainer
        )
        from torch.utils.data import DataLoader, TensorDataset
        from src.evaluation.benchmarking import BaselineComparison

        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)
        N = 1500

        U = torch.rand(N)
        S_prov = torch.rand(N)
        Delta_CF = torch.rand(N)
        # Labels have clear correlation with trust
        trust_sig = 0.4 * (1 - U) + 0.4 * S_prov + 0.2 * Delta_CF
        labels = torch.bernoulli(trust_sig.clamp(0.15, 0.85))

        # Train aggregator
        model = AttentionBasedTrustAggregator(hidden_dim=32)
        ds = TensorDataset(U, S_prov, Delta_CF, labels)
        train_size = int(0.8 * N)
        train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, N - train_size])
        trainer = TrustCalibrationTrainer(model, torch.device("cpu"))
        trainer.train(
            DataLoader(train_ds, batch_size=128, shuffle=True),
            DataLoader(val_ds, batch_size=256),
            num_epochs=20,
        )

        trust_data = {
            "U": U.numpy(),
            "S_prov": S_prov.numpy(),
            "Delta_CF": Delta_CF.numpy(),
            "labels": labels.numpy(),
        }

        comparison = BaselineComparison(trust_aggregator=model)
        results = comparison.run_all_baselines("test", trust_data)

        full_auc = results["full_system"]["auc_roc"]
        fixed_auc = results["fixed_weight_trust"]["auc_roc"]

        # Full system should be at least as good as fixed weights
        self.assertGreaterEqual(
            full_auc + 0.01,  # small tolerance
            fixed_auc,
            f"Full system AUC ({full_auc:.4f}) < fixed weights ({fixed_auc:.4f})"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
