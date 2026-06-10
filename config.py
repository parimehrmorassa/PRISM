"""
Central configuration for the Trustworthy Knowledge Graph Reasoning Framework.
All hyperparameters and paths are defined here.
"""

import os
from pathlib import Path

# ─── Project Root ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_ROOT / "raw"
PROCESSED_DATA_ROOT = DATA_ROOT / "processed"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
TRUST_CALIB_DIR = PROJECT_ROOT / "trust_calibration"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RESULTS_DIR = EXPERIMENTS_DIR / "results"

# ─── GPU Configuration ─────────────────────────────────────────────────────────
GPU_CONFIG = {
    "training": 0,
    "analysis": 0,
    "hetionet": 0,
    "fb15k237": 0,
    "wn18rr": 0,
}

# ─── Dataset Paths ─────────────────────────────────────────────────────────────
DATASETS = {
    "fb15k237": {
        "raw_dir": str(RAW_DATA_DIR / "fb15k237"),
        "url": "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/FB15k-237",
        "files": ["train.txt", "valid.txt", "test.txt"],
    },
    "wn18rr": {
        "raw_dir": str(RAW_DATA_DIR / "wn18rr"),
        "url": "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/WN18RR",
        "files": ["train.txt", "valid.txt", "test.txt"],
    },
    "hetionet": {
        "raw_dir": str(RAW_DATA_DIR / "hetionet"),
        "url": "https://github.com/hetio/hetionet",
        "files": ["hetionet-v1.0-edges.sif", "hetionet-v1.0-nodes.tsv"],
    },
}

PROCESSED_DATA = {
    "fb15k237": str(PROCESSED_DATA_ROOT / "fb15k237"),
    "wn18rr": str(PROCESSED_DATA_ROOT / "wn18rr"),
    "hetionet": str(PROCESSED_DATA_ROOT / "hetionet"),
}

# ─── NBFNet Hyperparameters ────────────────────────────────────────────────────
NBFNET_CONFIG = {
    # Architecture
    "hidden_dim": 64,
    "num_layers": 6,
    "dropout": 0.1,
    "num_negative_samples": 64,    # was 32

    # Training
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "batch_size": 512,             # was 64
    "num_epochs": {
        "fb15k237": 50,
        "wn18rr": 30,
        "hetionet": 20,
    },
    "save_every_n_epochs": 5,
    "patience": 15,

    # Subgraph extraction
    "k_hop": 2,
    "max_nodes_per_hop": 200,

    # Scoring
    "top_k_paths": 5,

    # Mixed precision
    "use_amp": True,
    "amp_dtype": "bfloat16",

    # DataLoader — critical for H200 utilization
    "num_workers": 16,             # ADD THIS
    "pin_memory": True,            # ADD THIS
    "prefetch_factor": 4,          # ADD THIS
    "persistent_workers": True,    # ADD THIS
}

# ─── Uncertainty / Conformal Prediction ───────────────────────────────────────
UNCERTAINTY_CONFIG = {
    "alpha": 0.1,            # target miscoverage rate → ~90% coverage
    "min_set_size": 1,
    "max_set_size": None,    # no upper bound
}

# ─── Counterfactual Analysis ───────────────────────────────────────────────────
COUNTERFACTUAL_CONFIG = {
    "top_k_critical": 10,
    "batch_size": 256,             # was 64 — use H200 memory
    "gpu": 0,
    "random_seed": 42,
}
# ─── Provenance Encoding ───────────────────────────────────────────────────────
PROVENANCE_CONFIG = {
    # Rule-based source weights (higher = more reliable)
    "source_weights": {
        "DRUGBANK": 0.95,
        "PUBMED": 0.85,
        "WIKIPEDIA": 0.70,
        "UNIPROT": 0.90,
        "OMIM": 0.80,
        "REACTOME": 0.85,
        "GO": 0.88,
        "MESH": 0.82,
        "NCI": 0.80,
        "MANUAL_CURATION": 1.00,
        "AUTOMATED": 0.50,
        "fb15k237": 0.75,   # general FB15k-237 synthetic provenance
        "wn18rr": 0.80,     # general WN18RR synthetic provenance
        "default": 0.60,
    },
    "aggregation": "minimum",  # weakest-link principle for path aggregation
}

# ─── Trust Aggregator ─────────────────────────────────────────────────────────
TRUST_AGG_CONFIG = {
    "hidden_dim": 64,
    "dropout": 0.2,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 256,
    "num_epochs": 50,
    "patience": 10,

    # Query type thresholds for weight analysis
    "rare_relation_freq_threshold": 100,
    "hub_entity_degree_threshold": 50,
}

# ─── LLM Explainer ────────────────────────────────────────────────────────────
LLM_CONFIG = {
    "provider": "anthropic",         # "anthropic" or "openai"
    "model": "claude-3-haiku-20240307",
    "temperature": 0.3,
    "max_tokens": 300,
    "api_key_env": "ANTHROPIC_API_KEY",  # name of the env var holding your API key

    # Sentence-transformer for LExT scoring
    "sentence_transformer_model": "all-MiniLM-L6-v2",

    # Faithfulness verification
    "verify_faithfulness": True,
}

# ─── Evaluation ───────────────────────────────────────────────────────────────
EVALUATION_CONFIG = {
    "hits_at_k": [1, 3, 10],
    "ece_num_bins": 10,
    "num_explanation_samples": 200,
    "significance_level": 0.05,
}

# ─── Wandb / Logging ──────────────────────────────────────────────────────────
LOGGING_CONFIG = {
    "use_wandb": False,           # set True to enable W&B logging
    "project_name": "trustworthy-kg",
    "log_every_n_steps": 10,
    "weight_log_every_n_epochs": 10,
}

# ─── Dataset Statistics (filled after preprocessing) ──────────────────────────
# These are updated by download_datasets.py
DATASET_STATS = {
    "fb15k237": {
        "num_entities": 14541,
        "num_relations": 237,
        "num_train": 272115,
        "num_valid": 17535,
        "num_test": 20466,
    },
    "wn18rr": {
        "num_entities": 40943,
        "num_relations": 11,
        "num_train": 86835,
        "num_valid": 3034,
        "num_test": 3134,
    },
    "hetionet": {
        "num_entities": 47031,
        "num_relations": 24,
        "num_train": None,   # set after processing
        "num_valid": None,
        "num_test": None,
    },
}

# ─── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ─── Ensure directories exist ─────────────────────────────────────────────────
def ensure_dirs():
    """Create all required directories."""
    dirs = [
        RAW_DATA_DIR, PROCESSED_DATA_ROOT, CHECKPOINT_DIR,
        TRUST_CALIB_DIR, RESULTS_DIR,
    ]
    for dataset in ["fb15k237", "wn18rr", "hetionet"]:
        dirs += [
            RAW_DATA_DIR / dataset,
            PROCESSED_DATA_ROOT / dataset,
            CHECKPOINT_DIR / dataset,
            TRUST_CALIB_DIR / dataset,
            RESULTS_DIR / dataset,
        ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print("All directories created.")
