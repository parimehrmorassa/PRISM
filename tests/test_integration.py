"""
Phase 7 integration tests: end-to-end pipeline validation.

Tests the complete pipeline from data loading through inference,
without requiring actual trained models or API keys.
"""

import sys
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.nbfnet.model import NBFNet
from src.models.nbfnet.trainer import NBFNetTrainer
from src.models.trust.uncertainty import ConformalUncertaintyQuantifier
from src.models.trust.counterfactual import CounterfactualAnalyzer
from src.models.trust.provenance_aggregator import aggregate_path, aggregate_query
from src.models.trust.aggregator import AttentionBasedTrustAggregator, TrustCalibrationTrainer
from src.models.explainer.llm_interface import LLMExplainer, _MockLLMClient
from src.evaluation.metrics import (
    compute_mrr, compute_hits_at_k, compute_trust_calibration_metrics,
    compute_expected_calibration_error, compute_counterfactual_metrics,
)
from src.evaluation.benchmarking import BaselineComparison
from config import RANDOM_SEED

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class TestEndToEndPipeline(unittest.TestCase):
    """
    Integration tests for the complete trustworthy KG pipeline.
    Uses small synthetic data — no GPU or API required.
    """

    NUM_ENTITIES = 50
    NUM_RELATIONS = 8
    NUM_EDGES = 100
    HIDDEN_DIM = 16
    NUM_LAYERS = 2

    @classmethod
    def setUpClass(cls):
        """Set up shared model and data structures."""
        torch.manual_seed(RANDOM_SEED)

        cls.model = NBFNet(
            num_relations=cls.NUM_RELATIONS,
            hidden_dim=cls.HIDDEN_DIM,
            num_layers=cls.NUM_LAYERS,
            dropout=0.0,
        )

        # Synthetic graph
        cls.src = torch.randint(0, cls.NUM_ENTITIES, (cls.NUM_EDGES,))
        cls.dst = torch.randint(0, cls.NUM_ENTITIES, (cls.NUM_EDGES,))
        cls.edge_index = torch.stack([cls.src, cls.dst], dim=0)
        cls.edge_type = torch.randint(0, cls.NUM_RELATIONS, (cls.NUM_EDGES,))
        cls.edge_prov = torch.rand(cls.NUM_EDGES).clamp(0.1, 1.0)

        # Synthetic trust dataset
        N = 500
        cls.U = torch.rand(N)
        cls.S_prov = torch.rand(N)
        cls.Delta_CF = torch.rand(N)
        trust_sig = 0.4 * (1 - cls.U) + 0.4 * cls.S_prov + 0.2 * cls.Delta_CF
        cls.labels = torch.bernoulli(trust_sig.clamp(0.1, 0.9))
        cls.N = N

    def _make_subgraph(self, head=0, tail=5, relation=1):
        return {
            "edge_index": self.edge_index,
            "edge_type": self.edge_type,
            "edge_prov": self.edge_prov,
            "local_head": head,
            "local_tail": tail,
            "num_nodes": self.NUM_ENTITIES,
            "global_to_local": {i: i for i in range(self.NUM_ENTITIES)},
        }

    # ─── NBFNet Pipeline ─────────────────────────────────────────────────────────

    def test_nbfnet_forward(self):
        """NBFNet forward pass produces valid scores."""
        self.model.eval()
        with torch.no_grad():
            scores, attn = self.model(
                edge_index=self.edge_index,
                edge_type=self.edge_type,
                edge_prov=self.edge_prov,
                query_head=0,
                query_relation=1,
                num_nodes=self.NUM_ENTITIES,
            )
        self.assertEqual(scores.shape, (self.NUM_ENTITIES,))
        self.assertTrue((scores >= 0).all() and (scores <= 1).all())
        self.assertEqual(len(attn), self.NUM_LAYERS)

    def test_nbfnet_reasoning_paths(self):
        """Reasoning path extraction returns valid dicts."""
        self.model.eval()
        with torch.no_grad():
            _, attn = self.model(
                edge_index=self.edge_index,
                edge_type=self.edge_type,
                edge_prov=self.edge_prov,
                query_head=0,
                query_relation=1,
                num_nodes=self.NUM_ENTITIES,
            )
        paths = self.model.extract_reasoning_paths(
            self.edge_index, self.edge_type, self.edge_prov, attn, 0, 5, top_k=3
        )
        self.assertIsInstance(paths, list)
        if paths:
            self.assertIn("edge_idx", paths[0])
            self.assertIn("attention", paths[0])

    # ─── Uncertainty Pipeline ────────────────────────────────────────────────────

    def test_conformal_uncertainty_pipeline(self):
        """Full uncertainty quantification pipeline produces valid U."""
        quantifier = ConformalUncertaintyQuantifier(alpha=0.1, num_entities=self.NUM_ENTITIES)
        # Manually calibrate with synthetic scores
        n = 200
        cal_scores = np.random.rand(n)
        level = min((n + 1) * 0.9 / n, 1.0)
        quantifier.quantile = float(np.quantile(cal_scores, level))

        scores = torch.rand(self.NUM_ENTITIES)
        U, pred_set = quantifier.compute_uncertainty_from_scores(scores)
        self.assertGreaterEqual(U, 0.0)
        self.assertLessEqual(U, 1.0)

    def test_conformal_save_load(self):
        """ConformalUncertaintyQuantifier save/load roundtrip."""
        quantifier = ConformalUncertaintyQuantifier(alpha=0.1, num_entities=self.NUM_ENTITIES)
        quantifier.quantile = 0.75

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "quantile.pkl")
            quantifier.save(path)
            loaded = ConformalUncertaintyQuantifier.load(path)
            self.assertAlmostEqual(loaded.quantile, 0.75)

    # ─── Counterfactual Pipeline ─────────────────────────────────────────────────

    def test_counterfactual_pipeline(self):
        """Counterfactual pipeline produces Delta_CF in [0, 1]."""
        analyzer = CounterfactualAnalyzer(self.model, torch.device("cpu"), top_k=3)
        subgraph = self._make_subgraph()
        critical_edges, baseline = analyzer.identify_critical_edges(subgraph, 1)
        Delta_CF, impacts = analyzer.compute_sensitivity(subgraph, 1, critical_edges, baseline)
        self.assertGreaterEqual(Delta_CF, 0.0)
        self.assertLessEqual(Delta_CF, 1.0)
        self.assertIn("critical", impacts)
        self.assertIn("random", impacts)

    # ─── Provenance Aggregation ──────────────────────────────────────────────────

    def test_provenance_min_aggregation(self):
        """Minimum provenance aggregation returns correct value."""
        path = [{"s_prov": 0.9}, {"s_prov": 0.6}, {"s_prov": 0.8}]
        result = aggregate_path(path)
        self.assertAlmostEqual(result, 0.6, places=5)

    def test_provenance_attention_weighted(self):
        """Attention-weighted query aggregation computes correctly."""
        paths = [{"s_prov": 0.8}, {"s_prov": 0.4}]
        weights = [0.7, 0.3]
        S_prov, _ = aggregate_query(paths, weights)
        expected = 0.7 * 0.8 + 0.3 * 0.4
        self.assertAlmostEqual(S_prov, expected, places=5)

    # ─── Trust Aggregator Pipeline ───────────────────────────────────────────────

    def test_trust_aggregator_pipeline(self):
        """Trust aggregator produces T ∈ [0, 1] with weights summing to 1."""
        aggregator = AttentionBasedTrustAggregator(hidden_dim=32)
        B = 32
        U = torch.rand(B)
        S = torch.rand(B)
        CF = torch.rand(B)
        T, weights = aggregator(U, S, CF)
        self.assertTrue((T >= 0).all() and (T <= 1).all())
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones(B), atol=1e-5, rtol=1e-5)

    def test_trust_training_pipeline(self):
        """Trust aggregator training converges without errors."""
        aggregator = AttentionBasedTrustAggregator(hidden_dim=32)
        trainer = TrustCalibrationTrainer(aggregator, torch.device("cpu"))

        N = 400
        train_ds = TensorDataset(
            self.U[:int(0.8*N)], self.S_prov[:int(0.8*N)],
            self.Delta_CF[:int(0.8*N)], self.labels[:int(0.8*N)]
        )
        val_ds = TensorDataset(
            self.U[int(0.8*N):N], self.S_prov[int(0.8*N):N],
            self.Delta_CF[int(0.8*N):N], self.labels[int(0.8*N):N]
        )
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=128)

        history = trainer.train(train_loader, val_loader, num_epochs=5)
        self.assertGreater(len(history), 0)
        self.assertIn("loss", history[-1])

    # ─── Explanation Pipeline ────────────────────────────────────────────────────

    def test_explanation_pipeline(self):
        """LLM explainer produces valid explanations."""
        explainer = LLMExplainer()
        explainer._client = _MockLLMClient()

        qr = {
            "query": ("EntityA", "relates_to", "EntityB"),
            "head": "EntityA",
            "relation": "relates_to",
            "prediction": "EntityB",
            "confidence": 0.8,
            "trust_score": 0.72,
            "trust_breakdown": {
                "uncertainty": 0.2, "provenance": 0.85, "counterfactual": 0.4,
                "weights": {"alpha": 0.3, "beta": 0.5, "gamma": 0.2},
            },
            "reasoning_paths": [
                {"head": "EntityA", "relation": "r1", "tail": "EntityB",
                 "s_prov": 0.85, "attention": 0.6, "head_id": 0, "tail_id": 1,
                 "relation_id": 0, "edge_idx": 0, "layer_attns": [0.5, 0.7]},
            ],
            "critical_edge": {"head": "EntityA", "relation": "r1", "tail": "EntityB"},
            "max_delta": 0.3,
        }

        exp = explainer.generate_explanation(qr)
        self.assertIsInstance(exp, str)
        self.assertGreater(len(exp.strip()), 0)

        is_faithful, _ = explainer.verify_faithfulness(exp, qr)
        self.assertTrue(is_faithful)

    # ─── Evaluation Pipeline ─────────────────────────────────────────────────────

    def test_evaluation_metrics_pipeline(self):
        """All evaluation metrics compute without error."""
        N = 100
        ranks = np.random.randint(1, 100, N)
        scores = np.random.rand(N)
        labels = np.random.randint(0, 2, N).astype(float)

        mrr = compute_mrr(ranks)
        h10 = compute_hits_at_k(ranks, 10)
        ece = compute_expected_calibration_error(scores, labels)
        trust_m = compute_trust_calibration_metrics(scores, labels)

        self.assertGreaterEqual(mrr, 0.0)
        self.assertLessEqual(mrr, 1.0)
        self.assertGreaterEqual(h10, 0.0)
        self.assertLessEqual(h10, 1.0)
        self.assertGreaterEqual(ece, 0.0)
        for key in ["accuracy", "auc_roc", "ece"]:
            self.assertIn(key, trust_m)

    def test_baseline_comparison_pipeline(self):
        """BaselineComparison produces valid results for all baselines."""
        N = 200
        trust_data = {
            "U": np.random.rand(N),
            "S_prov": np.random.rand(N),
            "Delta_CF": np.random.rand(N),
            "labels": np.random.randint(0, 2, N).astype(float),
            "confidence": np.random.rand(N),
        }

        aggregator = AttentionBasedTrustAggregator(hidden_dim=32)
        comparison = BaselineComparison(trust_aggregator=aggregator)
        results = comparison.run_all_baselines("test", trust_data)

        expected_baselines = ["vanilla_nbfnet", "nbfnet_uncertainty",
                              "nbfnet_provenance", "fixed_weight_trust", "full_system"]
        for bl in expected_baselines:
            self.assertIn(bl, results)
            self.assertIn("accuracy", results[bl])
            self.assertIn("auc_roc", results[bl])

    # ─── Full Pipeline Sanity Check ──────────────────────────────────────────────

    def test_trust_score_bounded(self):
        """Trust score T must always be in [0, 1] across the full pipeline."""
        aggregator = AttentionBasedTrustAggregator(hidden_dim=32)

        # Run 200 random inputs
        for _ in range(200):
            B = np.random.randint(1, 32)
            U = torch.rand(B)
            S = torch.rand(B)
            CF = torch.rand(B)
            T, weights = aggregator(U, S, CF)
            self.assertTrue((T >= 0).all() and (T <= 1).all())
            self.assertTrue(torch.isfinite(T).all())

    def test_checkpoint_save_load_pipeline(self):
        """NBFNet + Trust aggregator can be saved and reloaded cleanly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save NBFNet
            nbf_path = str(Path(tmpdir) / "nbfnet.pt")
            NBFNet.save(self.model, nbf_path)
            loaded_nbf = NBFNet.load(nbf_path, device=torch.device("cpu"))
            self.assertIsInstance(loaded_nbf, NBFNet)

            # Save trust aggregator
            agg = AttentionBasedTrustAggregator(hidden_dim=32)
            agg_path = str(Path(tmpdir) / "aggregator.pt")
            torch.save({"state_dict": agg.state_dict(), "hidden_dim": 32}, agg_path)
            ckpt = torch.load(agg_path, map_location="cpu", weights_only=False)
            loaded_agg = AttentionBasedTrustAggregator(hidden_dim=32)
            loaded_agg.load_state_dict(ckpt["state_dict"])
            self.assertIsInstance(loaded_agg, AttentionBasedTrustAggregator)


if __name__ == "__main__":
    unittest.main(verbosity=2)
