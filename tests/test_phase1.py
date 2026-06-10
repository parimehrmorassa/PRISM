"""
Phase 1 test suite for data loading, provenance, and configuration.

Tests:
    - test_config_keys_present
    - test_provenance_metadata_dataclass
    - test_provenance_score_range
    - test_dataset_stats_defined
    - test_gpu_config_valid
    - test_data_directories_creatable
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    NBFNET_CONFIG, GPU_CONFIG, UNCERTAINTY_CONFIG,
    COUNTERFACTUAL_CONFIG, TRUST_AGG_CONFIG, LLM_CONFIG,
    PROVENANCE_CONFIG, PROCESSED_DATA, CHECKPOINT_DIR,
    TRUST_CALIB_DIR, RESULTS_DIR, DATASET_STATS, RANDOM_SEED,
)
from src.data.provenance import ProvenanceMetadata


class TestConfiguration(unittest.TestCase):
    """Test that config.py has all required keys."""

    def test_nbfnet_config_keys(self):
        """NBFNET_CONFIG must have all required hyperparameters."""
        required = [
            "hidden_dim", "num_layers", "dropout", "num_negative_samples",
            "learning_rate", "weight_decay", "batch_size", "num_epochs",
            "save_every_n_epochs", "patience", "k_hop", "top_k_paths",
            "use_amp", "amp_dtype",
        ]
        for key in required:
            self.assertIn(key, NBFNET_CONFIG, f"Missing key: {key}")

    def test_gpu_config_keys(self):
        """GPU_CONFIG must have training and analysis keys."""
        self.assertIn("training", GPU_CONFIG)
        self.assertIn("analysis", GPU_CONFIG)
        self.assertIsInstance(GPU_CONFIG["training"], int)
        self.assertIsInstance(GPU_CONFIG["analysis"], int)

    def test_uncertainty_config(self):
        """UNCERTAINTY_CONFIG must have alpha in (0, 1)."""
        self.assertIn("alpha", UNCERTAINTY_CONFIG)
        alpha = UNCERTAINTY_CONFIG["alpha"]
        self.assertGreater(alpha, 0.0)
        self.assertLess(alpha, 1.0)

    def test_trust_agg_config_keys(self):
        """TRUST_AGG_CONFIG must have required keys."""
        for key in ["hidden_dim", "dropout", "learning_rate", "batch_size", "num_epochs"]:
            self.assertIn(key, TRUST_AGG_CONFIG, f"Missing: {key}")

    def test_provenance_config_source_weights(self):
        """Provenance source weights must be in [0, 1]."""
        for source, weight in PROVENANCE_CONFIG["source_weights"].items():
            self.assertGreaterEqual(weight, 0.0, f"{source} weight < 0")
            self.assertLessEqual(weight, 1.0, f"{source} weight > 1")

    def test_processed_data_keys(self):
        """PROCESSED_DATA must have all three datasets."""
        for ds in ["fb15k237", "wn18rr", "hetionet"]:
            self.assertIn(ds, PROCESSED_DATA, f"Missing dataset: {ds}")

    def test_dataset_stats_defined(self):
        """DATASET_STATS must have num_entities and num_relations for all datasets."""
        for ds in ["fb15k237", "wn18rr"]:
            stats = DATASET_STATS.get(ds, {})
            self.assertIn("num_entities", stats)
            self.assertIn("num_relations", stats)
            self.assertGreater(stats["num_entities"], 0)
            self.assertGreater(stats["num_relations"], 0)

    def test_random_seed_is_int(self):
        """RANDOM_SEED must be an integer."""
        self.assertIsInstance(RANDOM_SEED, int)


class TestProvenanceMetadata(unittest.TestCase):
    """Tests for ProvenanceMetadata dataclass."""

    def test_basic_creation(self):
        """ProvenanceMetadata can be created with valid arguments."""
        pm = ProvenanceMetadata(
            source_name="DRUGBANK",
            source_weight=0.95,
            edge_index=0,
            head_id=10,
            relation_id=2,
            tail_id=42,
        )
        self.assertEqual(pm.source_name, "DRUGBANK")
        self.assertAlmostEqual(pm.source_weight, 0.95)
        self.assertEqual(pm.edge_index, 0)

    def test_s_prov_alias(self):
        """s_prov must alias provenance_score."""
        pm = ProvenanceMetadata(
            source_name="PUBMED",
            source_weight=0.85,
            edge_index=1,
            head_id=5,
            relation_id=3,
            tail_id=7,
        )
        self.assertAlmostEqual(pm.s_prov, pm.provenance_score)

    def test_invalid_weight_raises(self):
        """source_weight outside [0, 1] must raise AssertionError."""
        with self.assertRaises(AssertionError):
            ProvenanceMetadata(
                source_name="BAD",
                source_weight=1.5,
                edge_index=0,
                head_id=0,
                relation_id=0,
                tail_id=0,
            )

    def test_to_dict_roundtrip(self):
        """to_dict/from_dict roundtrip must preserve all fields."""
        pm = ProvenanceMetadata(
            source_name="UNIPROT",
            source_weight=0.9,
            edge_index=5,
            head_id=1,
            relation_id=2,
            tail_id=3,
            supporting_sources=["GO", "REACTOME"],
            metadata={"year": 2021},
        )
        d = pm.to_dict()
        pm2 = ProvenanceMetadata.from_dict(d)
        self.assertEqual(pm2.source_name, pm.source_name)
        self.assertAlmostEqual(pm2.source_weight, pm.source_weight)
        self.assertEqual(pm2.supporting_sources, pm.supporting_sources)
        self.assertEqual(pm2.metadata, pm.metadata)

    def test_default_provenance_score(self):
        """Default provenance_score should equal source_weight."""
        pm = ProvenanceMetadata(
            source_name="MANUAL_CURATION",
            source_weight=1.0,
            edge_index=0,
            head_id=0,
            relation_id=0,
            tail_id=0,
        )
        self.assertAlmostEqual(pm.provenance_score, 1.0)


class TestDirectoryStructure(unittest.TestCase):
    """Test that required directories can be created."""

    def test_data_directories_creatable(self):
        """Required data directories should be creatable."""
        from config import ensure_dirs
        try:
            ensure_dirs()
        except Exception as e:
            self.fail(f"ensure_dirs() raised {e}")

    def test_results_dir_exists_after_ensure(self):
        """RESULTS_DIR should exist after ensure_dirs()."""
        from config import ensure_dirs
        ensure_dirs()
        self.assertTrue(Path(RESULTS_DIR).exists())


class TestDataset(unittest.TestCase):
    """Test KGDataset when processed data is available."""

    def test_kg_dataset_import(self):
        """KGDataset module must be importable."""
        try:
            from src.data.dataset import KGDataset, build_dataloaders, kg_collate_fn
        except ImportError as e:
            self.fail(f"Cannot import dataset module: {e}")

    def test_augmentation_import(self):
        """Augmentation module must be importable."""
        try:
            from src.data.augmentation import (
                add_inverse_relations, compute_entity_degrees,
                compute_relation_frequencies, subsample_triples
            )
        except ImportError as e:
            self.fail(f"Cannot import augmentation module: {e}")

    def test_augmentation_inverse_relations(self):
        """add_inverse_relations must double the number of triples."""
        import torch
        from src.data.augmentation import add_inverse_relations

        triples = torch.tensor([[0, 1, 2], [3, 1, 4], [0, 2, 5]], dtype=torch.long)
        num_relations = 5

        aug, new_num_rel, _ = add_inverse_relations(triples, num_relations)
        self.assertEqual(len(aug), 2 * len(triples))
        self.assertEqual(new_num_rel, 2 * num_relations)

    def test_compute_entity_degrees(self):
        """compute_entity_degrees must count in + out edges."""
        import torch
        from src.data.augmentation import compute_entity_degrees

        triples = torch.tensor([[0, 0, 1], [0, 0, 2], [1, 0, 2]], dtype=torch.long)
        degrees = compute_entity_degrees(triples, num_entities=5)
        self.assertEqual(degrees[0].item(), 2)  # 0 appears as head twice
        self.assertEqual(degrees[2].item(), 2)  # 2 appears as tail twice


if __name__ == "__main__":
    unittest.main(verbosity=2)
