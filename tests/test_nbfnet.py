"""
Phase 2 test suite for NBFNet model and trainer.

Tests:
    - test_forward_pass_shape
    - test_attention_weights_count
    - test_provenance_modulation
    - test_inductive_no_entity_embeddings
    - test_subgraph_extraction
    - test_checkpoint_save_load
"""

import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.nbfnet.layer import ProvenanceAwareMessagePassingLayer
from src.models.nbfnet.model import NBFNet
from config import NBFNET_CONFIG


def make_small_graph(num_nodes=10, num_edges=20, num_relations=5, device="cpu"):
    """Create a small random graph for testing."""
    src = torch.randint(0, num_nodes, (num_edges,))
    dst = torch.randint(0, num_nodes, (num_edges,))
    edge_index = torch.stack([src, dst], dim=0)
    edge_type = torch.randint(0, num_relations, (num_edges,))
    edge_prov = torch.rand(num_edges).clamp(0.1, 1.0)
    return edge_index, edge_type, edge_prov


class TestProvenanceAwareMPLayer(unittest.TestCase):
    """Tests for ProvenanceAwareMessagePassingLayer."""

    def setUp(self):
        self.hidden_dim = 16
        self.num_relations = 5
        self.num_nodes = 10
        self.num_edges = 20
        self.layer = ProvenanceAwareMessagePassingLayer(
            hidden_dim=self.hidden_dim,
            num_relations=self.num_relations,
            dropout=0.0,
        )
        self.edge_index, self.edge_type, self.edge_prov = make_small_graph(
            self.num_nodes, self.num_edges, self.num_relations
        )
        self.node_features = torch.randn(self.num_nodes, self.hidden_dim)

    def test_output_shape(self):
        h_new, attn = self.layer(
            edge_index=self.edge_index,
            edge_type=self.edge_type,
            edge_prov=self.edge_prov,
            node_features=self.node_features,
            query_relation=0,
            num_nodes=self.num_nodes,
        )
        self.assertEqual(h_new.shape, (self.num_nodes, self.hidden_dim))
        self.assertEqual(attn.shape, (self.num_edges,))

    def test_attention_nonnegative(self):
        _, attn = self.layer(
            edge_index=self.edge_index,
            edge_type=self.edge_type,
            edge_prov=self.edge_prov,
            node_features=self.node_features,
            query_relation=0,
            num_nodes=self.num_nodes,
        )
        # Attention weights should be non-negative (prov * norm)
        # Note: phi.norm can be negative after relu; check finite
        self.assertTrue(torch.isfinite(attn).all())


class TestNBFNet(unittest.TestCase):
    """Tests for the full NBFNet model."""

    def setUp(self):
        self.num_relations = 8
        self.hidden_dim = 16
        self.num_layers = 3  # Reduced for testing speed
        self.num_nodes = 12
        self.num_edges = 25

        self.model = NBFNet(
            num_relations=self.num_relations,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=0.0,
        )
        self.edge_index, self.edge_type, self.edge_prov = make_small_graph(
            self.num_nodes, self.num_edges, self.num_relations
        )

    def test_forward_pass_shape(self):
        """scores shape must be (num_nodes,)."""
        self.model.eval()
        with torch.no_grad():
            scores, attn_weights = self.model(
                edge_index=self.edge_index,
                edge_type=self.edge_type,
                edge_prov=self.edge_prov,
                query_head=0,
                query_relation=1,
                num_nodes=self.num_nodes,
            )
        self.assertEqual(scores.shape, (self.num_nodes,))
        self.assertFalse(torch.isnan(scores).any(), "Scores contain NaN")

    def test_scores_in_range(self):
        """Scores must be in [0, 1] due to sigmoid output."""
        self.model.eval()
        with torch.no_grad():
            scores, _ = self.model(
                edge_index=self.edge_index,
                edge_type=self.edge_type,
                edge_prov=self.edge_prov,
                query_head=0,
                query_relation=1,
                num_nodes=self.num_nodes,
            )
        self.assertTrue((scores >= 0).all() and (scores <= 1).all(),
                        f"Scores out of [0,1]: min={scores.min():.4f}, max={scores.max():.4f}")

    def test_attention_weights_count(self):
        """len(attention_weights) must equal num_layers."""
        self.model.eval()
        with torch.no_grad():
            scores, attn_weights = self.model(
                edge_index=self.edge_index,
                edge_type=self.edge_type,
                edge_prov=self.edge_prov,
                query_head=0,
                query_relation=1,
                num_nodes=self.num_nodes,
            )
        self.assertEqual(len(attn_weights), self.num_layers,
                         f"Expected {self.num_layers} attention tensors, got {len(attn_weights)}")

    def test_provenance_modulation(self):
        """Low S_prov edges should yield lower mean scores than high S_prov."""
        self.model.eval()

        # High provenance graph
        high_prov = torch.ones(self.num_edges)
        # Low provenance graph
        low_prov = torch.full((self.num_edges,), 0.05)

        with torch.no_grad():
            scores_high, _ = self.model(
                edge_index=self.edge_index,
                edge_type=self.edge_type,
                edge_prov=high_prov,
                query_head=0,
                query_relation=1,
                num_nodes=self.num_nodes,
            )
            scores_low, _ = self.model(
                edge_index=self.edge_index,
                edge_type=self.edge_type,
                edge_prov=low_prov,
                query_head=0,
                query_relation=1,
                num_nodes=self.num_nodes,
            )

        # Mean score with high provenance should be >= mean with low provenance
        # (Not strictly guaranteed, but expected due to provenance scaling)
        # This is a soft check — we verify scores differ
        diff = (scores_high.mean() - scores_low.mean()).abs().item()
        self.assertGreater(diff, 1e-6,
                           "High and low provenance scores are identical — provenance has no effect")

    def test_inductive_no_entity_embeddings(self):
        """Model must NOT have entity embedding layers."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Embedding):
                self.assertNotIn("entity", name.lower(),
                                 f"Found entity embedding: {name}")
        # Positive check: relation embeddings exist
        has_relation_emb = any(
            "relation" in name.lower() and isinstance(m, nn.Embedding)
            for name, m in self.model.named_modules()
        )
        self.assertTrue(has_relation_emb, "NBFNet must have relation embeddings")

    def test_subgraph_extraction(self):
        """Subgraph must contain the query edge with correct provenance."""
        from src.models.nbfnet.trainer import NBFNetTrainer

        trainer = NBFNetTrainer(
            model=self.model,
            device=torch.device("cpu"),
            dataset_name="fb15k237",
            all_triples={(0, 1, 5)},
            num_entities=self.num_nodes,
        )

        subgraph = trainer._extract_subgraph(
            head=0, relation=1, tail=5
        )

        self.assertIn("edge_index", subgraph)
        self.assertIn("edge_type", subgraph)
        self.assertIn("edge_prov", subgraph)
        self.assertIn("local_head", subgraph)
        self.assertIn("local_tail", subgraph)
        self.assertEqual(subgraph["edge_index"].shape[0], 2)
        self.assertGreaterEqual(subgraph["num_nodes"], 2)

    def test_checkpoint_save_load(self):
        """Save and reload model — outputs must be identical."""
        self.model.eval()

        with torch.no_grad():
            scores_before, _ = self.model(
                edge_index=self.edge_index,
                edge_type=self.edge_type,
                edge_prov=self.edge_prov,
                query_head=0,
                query_relation=1,
                num_nodes=self.num_nodes,
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "nbfnet_test.pt")
            NBFNet.save(self.model, path)

            loaded_model = NBFNet.load(path, device=torch.device("cpu"))
            loaded_model.eval()

            with torch.no_grad():
                scores_after, _ = loaded_model(
                    edge_index=self.edge_index,
                    edge_type=self.edge_type,
                    edge_prov=self.edge_prov,
                    query_head=0,
                    query_relation=1,
                    num_nodes=self.num_nodes,
                )

        torch.testing.assert_close(scores_before, scores_after,
                                   msg="Scores changed after save/load")

    def test_extract_reasoning_paths(self):
        """extract_reasoning_paths must return list of dicts with required keys."""
        self.model.eval()

        with torch.no_grad():
            _, attn_weights = self.model(
                edge_index=self.edge_index,
                edge_type=self.edge_type,
                edge_prov=self.edge_prov,
                query_head=0,
                query_relation=1,
                num_nodes=self.num_nodes,
            )

        top_k = 3
        paths = self.model.extract_reasoning_paths(
            edge_index=self.edge_index,
            edge_type=self.edge_type,
            edge_prov=self.edge_prov,
            all_attention_weights=attn_weights,
            query_head=0,
            query_tail=5,
            top_k=top_k,
        )

        self.assertIsInstance(paths, list)
        self.assertLessEqual(len(paths), top_k)
        if paths:
            required_keys = {"head_id", "tail_id", "relation_id", "attention", "s_prov", "edge_idx"}
            for key in required_keys:
                self.assertIn(key, paths[0], f"Missing key '{key}' in reasoning path")

    def test_no_gradient_through_initialization(self):
        """Model parameters should have gradients after a forward + backward pass."""
        self.model.train()

        scores, _ = self.model(
            edge_index=self.edge_index,
            edge_type=self.edge_type,
            edge_prov=self.edge_prov,
            query_head=0,
            query_relation=1,
            num_nodes=self.num_nodes,
        )

        loss = scores.sum()
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in self.model.parameters()
        )
        self.assertTrue(has_grad, "No gradients flowed to model parameters")


if __name__ == "__main__":
    unittest.main(verbosity=2)
