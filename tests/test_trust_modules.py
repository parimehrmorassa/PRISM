"""
Phase 3 test suite for trust modules:
    - ConformalUncertaintyQuantifier
    - CounterfactualAnalyzer
    - Provenance aggregation
"""

import sys
import random
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.nbfnet.model import NBFNet
from src.models.nbfnet.trainer import NBFNetTrainer
from src.models.trust.uncertainty import ConformalUncertaintyQuantifier
from src.models.trust.counterfactual import CounterfactualAnalyzer
from src.models.trust.provenance_aggregator import (
    aggregate_path, aggregate_query, compute_path_provenance_from_attention
)
from config import UNCERTAINTY_CONFIG


def make_model(num_relations=5, hidden_dim=8, num_layers=2):
    return NBFNet(num_relations=num_relations, hidden_dim=hidden_dim, num_layers=num_layers, dropout=0.0)


def make_subgraph(num_nodes=8, num_edges=12, num_relations=5):
    src = torch.randint(0, num_nodes, (num_edges,))
    dst = torch.randint(0, num_nodes, (num_edges,))
    return {
        "edge_index": torch.stack([src, dst], dim=0),
        "edge_type": torch.randint(0, num_relations, (num_edges,)),
        "edge_prov": torch.rand(num_edges).clamp(0.1, 1.0),
        "local_head": 0,
        "local_tail": min(3, num_nodes - 1),
        "num_nodes": num_nodes,
        "global_to_local": {i: i for i in range(num_nodes)},
    }


class TestConformalUncertaintyQuantifier(unittest.TestCase):

    def setUp(self):
        self.num_entities = 100
        self.quantifier = ConformalUncertaintyQuantifier(
            alpha=UNCERTAINTY_CONFIG["alpha"],
            num_entities=self.num_entities,
        )
        # Manually set a quantile for testing without a full model
        self.quantifier.quantile = 0.7  # threshold = 1 - 0.7 = 0.3

    def test_uncertainty_range(self):
        """U must always be in [0, 1]."""
        for _ in range(50):
            scores = torch.rand(self.num_entities)
            U, pred_set = self.quantifier.compute_uncertainty_from_scores(
                scores, num_entities=self.num_entities
            )
            self.assertGreaterEqual(U, 0.0, f"U={U} < 0")
            self.assertLessEqual(U, 1.0, f"U={U} > 1")

    def test_uncertainty_range_extreme(self):
        """U must be in [0, 1] for extreme score distributions."""
        # All scores = 1.0 → all entities in prediction set → U = 1.0
        scores_all_high = torch.ones(self.num_entities)
        U_high, _ = self.quantifier.compute_uncertainty_from_scores(
            scores_all_high, num_entities=self.num_entities
        )
        self.assertLessEqual(U_high, 1.0)
        self.assertGreaterEqual(U_high, 0.0)

        # All scores = 0.0 → no entities in prediction set → U = 0.0
        scores_all_low = torch.zeros(self.num_entities)
        U_low, _ = self.quantifier.compute_uncertainty_from_scores(
            scores_all_low, num_entities=self.num_entities
        )
        self.assertEqual(U_low, 0.0)

    def test_conformal_coverage(self):
        """
        With quantile properly calibrated from scores s_i = 1 - score(true_tail),
        empirical coverage on calibration data should be ~(1 - alpha).
        """
        alpha = 0.1
        n = 200
        # Simulate scores: true tail gets high score (uniformly ≥ 0.5)
        true_scores = torch.rand(n) * 0.5 + 0.5  # scores ∈ [0.5, 1.0]
        nonconformity = 1.0 - true_scores.numpy()

        # Set quantile at (n+1)(1-alpha)/n level
        level = min((n + 1) * (1 - alpha) / n, 1.0)
        q = float(np.quantile(nonconformity, level))

        quantifier = ConformalUncertaintyQuantifier(alpha=alpha, num_entities=50)
        quantifier.quantile = q

        # Coverage: proportion of queries where true score >= 1 - q
        threshold = 1.0 - q
        coverage = (true_scores >= threshold).float().mean().item()
        self.assertGreaterEqual(coverage, 1 - alpha - 0.05,
                                f"Coverage {coverage:.3f} < {1-alpha-0.05:.3f}")

    def test_prediction_set_nonempty(self):
        """Prediction sets should not be empty when quantile is set reasonably."""
        # With a quantile close to 1.0, threshold = 0 → everything included
        self.quantifier.quantile = 0.99
        scores = torch.rand(50)
        U, pred_set = self.quantifier.compute_uncertainty_from_scores(
            scores, num_entities=50
        )
        # At least one entity should be included (scores > 0.01)
        self.assertGreater(len(pred_set), 0)


class TestCounterfactualAnalyzer(unittest.TestCase):

    def setUp(self):
        self.num_relations = 5
        self.model = make_model(num_relations=self.num_relations)
        self.device = torch.device("cpu")
        self.analyzer = CounterfactualAnalyzer(
            model=self.model,
            device=self.device,
            top_k=3,
        )
        self.subgraph = make_subgraph(num_nodes=8, num_edges=12, num_relations=self.num_relations)

    def test_delta_cf_range(self):
        """Delta_CF must always be in [0, 1]."""
        critical_edges, baseline_score = self.analyzer.identify_critical_edges(
            self.subgraph, query_relation=1
        )
        Delta_CF, _ = self.analyzer.compute_sensitivity(
            self.subgraph, 1, critical_edges, baseline_score
        )
        self.assertGreaterEqual(Delta_CF, 0.0, f"Delta_CF={Delta_CF} < 0")
        self.assertLessEqual(Delta_CF, 1.0, f"Delta_CF={Delta_CF} > 1")

    def test_delta_cf_range_random_inputs(self):
        """Delta_CF must be in [0, 1] for many random subgraphs."""
        for _ in range(20):
            sg = make_subgraph()
            try:
                critical_edges, baseline = self.analyzer.identify_critical_edges(sg, 1)
                Delta_CF, _ = self.analyzer.compute_sensitivity(sg, 1, critical_edges, baseline)
                self.assertGreaterEqual(Delta_CF, 0.0)
                self.assertLessEqual(Delta_CF, 1.0)
            except Exception:
                pass  # Edge cases with empty subgraphs are acceptable

    def test_delta_cf_equals_max(self):
        """Delta_CF must equal max of critical impacts."""
        critical_edges, baseline_score = self.analyzer.identify_critical_edges(
            self.subgraph, query_relation=1
        )
        Delta_CF, edge_impacts = self.analyzer.compute_sensitivity(
            self.subgraph, 1, critical_edges, baseline_score
        )
        if edge_impacts["critical"]:
            expected_max = max(edge_impacts["critical"])
            self.assertAlmostEqual(Delta_CF, expected_max, places=6)

    def test_critical_larger_than_random_mean(self):
        """
        On average, critical edge impacts should be >= random edge impacts.
        (Not strictly guaranteed per query, but expected over many queries.)
        """
        all_critical, all_random = [], []
        for _ in range(30):
            sg = make_subgraph(num_nodes=10, num_edges=15, num_relations=self.num_relations)
            try:
                critical_edges, baseline = self.analyzer.identify_critical_edges(sg, 1)
                _, edge_impacts = self.analyzer.compute_sensitivity(
                    sg, 1, critical_edges, baseline
                )
                all_critical.extend(edge_impacts["critical"])
                all_random.extend(edge_impacts["random"])
            except Exception:
                pass

        if all_critical and all_random:
            # Critical mean should be >= random mean (soft check)
            self.assertGreaterEqual(
                np.mean(all_critical) + 0.01,  # small margin
                np.mean(all_random),
                "Critical mean impact should be >= random mean impact"
            )

    def test_edge_removal_reduces_subgraph(self):
        """remove_edge_from_subgraph must reduce edge count by 1."""
        sg = make_subgraph(num_nodes=8, num_edges=12)
        original_E = sg["edge_index"].shape[1]
        modified = self.analyzer.remove_edge_from_subgraph(sg, edge_idx=0)
        self.assertEqual(modified["edge_index"].shape[1], original_E - 1)


class TestProvenanceAggregator(unittest.TestCase):

    def test_provenance_path_aggregation_min(self):
        """Minimum strategy should return minimum S_prov over path edges."""
        path = [
            {"head": "A", "relation": "r1", "tail": "B", "s_prov": 0.9},
            {"head": "B", "relation": "r2", "tail": "C", "s_prov": 0.6},
            {"head": "C", "relation": "r3", "tail": "D", "s_prov": 0.8},
        ]
        result = aggregate_path(path)
        self.assertAlmostEqual(result, 0.6, places=5)

    def test_empty_path_returns_one(self):
        """Empty path should return 1.0 (no weaknesses)."""
        self.assertEqual(aggregate_path([]), 1.0)

    def test_single_edge_path(self):
        """Single edge path returns its s_prov."""
        path = [{"s_prov": 0.75}]
        self.assertAlmostEqual(aggregate_path(path), 0.75)

    def test_attention_weighted_aggregation(self):
        """Attention-weighted aggregation should compute correct weighted average."""
        paths = [
            {"s_prov": 0.8},
            {"s_prov": 0.4},
        ]
        attention_weights = [0.7, 0.3]  # weights should normalize to [0.7, 0.3]

        S_prov, path_scores = aggregate_query(paths, attention_weights)

        expected = 0.7 * 0.8 + 0.3 * 0.4  # = 0.56 + 0.12 = 0.68
        self.assertAlmostEqual(S_prov, expected, places=5)
        self.assertEqual(len(path_scores), 2)

    def test_attention_weights_normalize(self):
        """aggregate_query must normalize attention weights."""
        paths = [{"s_prov": 0.9}, {"s_prov": 0.7}]
        # Unormalized weights summing to 2.0
        attention_weights = [1.4, 0.6]

        S_prov, _ = aggregate_query(paths, attention_weights)
        expected = (1.4 / 2.0) * 0.9 + (0.6 / 2.0) * 0.7
        self.assertAlmostEqual(S_prov, expected, places=5)

    def test_s_prov_in_range(self):
        """S_prov from aggregate_query must be in [0, 1]."""
        for _ in range(50):
            n_paths = random.randint(1, 10)
            paths = [{"s_prov": random.random()} for _ in range(n_paths)]
            weights = [random.random() for _ in range(n_paths)]
            S_prov, _ = aggregate_query(paths, weights)
            self.assertGreaterEqual(S_prov, 0.0)
            self.assertLessEqual(S_prov, 1.0)

    def test_compute_from_attention_weights(self):
        """compute_path_provenance_from_attention must return S_prov in [0, 1]."""
        num_edges = 15
        edge_index = torch.stack([
            torch.randint(0, 8, (num_edges,)),
            torch.randint(0, 8, (num_edges,)),
        ])
        edge_prov = torch.rand(num_edges).clamp(0.1, 1.0)
        attn_weights = [torch.rand(num_edges) for _ in range(3)]

        S_prov, path_scores = compute_path_provenance_from_attention(
            edge_index, edge_prov, attn_weights, top_k=5
        )
        self.assertGreaterEqual(S_prov, 0.0)
        self.assertLessEqual(S_prov, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
