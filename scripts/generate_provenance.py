"""
generate_provenance.py

Generates provenance weights and metadata for each edge in a knowledge graph
dataset.

  - FB15k-237 / WN18RR: synthetic provenance weights derived from a
    deterministic hash of the relation type, drawn from
    PROVENANCE_CONFIG["source_weights"].
  - Hetionet: real source-name field (DRUGBANK, PUBMED, etc.) loaded from
    hetionet_edge_sources.pt produced by download_datasets.py.

Outputs (written to PROCESSED_DATA[dataset]/):
  provenance_weights.pt      - float32 tensor, shape (num_train_edges,), in [0,1]
  provenance_metadata.json   - list of ProvenanceMetadata dicts (training edges)

Usage:
    python scripts/generate_provenance.py --dataset fb15k237
    python scripts/generate_provenance.py --dataset wn18rr
    python scripts/generate_provenance.py --dataset hetionet
"""

import sys
import os

# Add project root to sys.path for config.py and src package imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from config import (
    PROCESSED_DATA,
    PROVENANCE_CONFIG,
    RANDOM_SEED,
)
from src.data.provenance import ProvenanceMetadata

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SOURCE_WEIGHTS: Dict[str, float] = PROVENANCE_CONFIG["source_weights"]
DEFAULT_WEIGHT: float = SOURCE_WEIGHTS.get("default", 0.60)

# Ordered list of biomedical sources (used for synthetic assignment)
_ORDERED_SOURCES = [
    k for k in SOURCE_WEIGHTS
    if k not in ("fb15k237", "wn18rr", "default")
]


# ===========================================================================
# Helpers
# ===========================================================================

def _load_triples(processed_dir: Path, split: str) -> torch.Tensor:
    """Load triples_{split}.pt from *processed_dir*. Returns (N, 3) LongTensor."""
    path = processed_dir / f"triples_{split}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Triples file not found: {path}. "
            "Run download_datasets.py first."
        )
    return torch.load(str(path), weights_only=True)


def _load_relation2id(processed_dir: Path) -> Dict[str, int]:
    """Load relation2id.json from *processed_dir*."""
    path = processed_dir / "relation2id.json"
    if not path.exists():
        raise FileNotFoundError(
            f"relation2id.json not found: {path}. "
            "Run download_datasets.py first."
        )
    with open(path, "r") as f:
        return json.load(f)


def _relation_hash_weight(relation_id: int, dataset_key: str) -> Tuple[str, float]:
    """
    Deterministically derive a source name and weight for a synthetic edge
    by hashing its relation ID.

    The hash maps the relation to one of the known source buckets so that
    the same relation always gets the same provenance source — simulating
    the idea that edges of a given type typically come from one database.
    """
    # Use a stable hash of (dataset, relation_id)
    token = f"{dataset_key}:{relation_id}".encode("utf-8")
    digest = int(hashlib.md5(token).hexdigest(), 16)  # noqa: S324

    if _ORDERED_SOURCES:
        source_name = _ORDERED_SOURCES[digest % len(_ORDERED_SOURCES)]
    else:
        source_name = dataset_key

    # Fall back to dataset-level weight if source not in config
    weight = SOURCE_WEIGHTS.get(source_name, SOURCE_WEIGHTS.get(dataset_key, DEFAULT_WEIGHT))
    return source_name, weight


# ===========================================================================
# Per-dataset provenance generators
# ===========================================================================

def _generate_synthetic_provenance(
    triples: torch.Tensor,
    dataset_key: str,
) -> Tuple[torch.Tensor, List[ProvenanceMetadata]]:
    """
    Generate provenance for FB15k-237 or WN18RR via relation-type hashing.

    Args:
        triples:     (N, 3) LongTensor [head_id, relation_id, tail_id]
        dataset_key: "fb15k237" or "wn18rr"

    Returns:
        weights_tensor: (N,) float32 tensor in [0, 1]
        metadata_list:  list of ProvenanceMetadata objects
    """
    weights: List[float] = []
    metadata_list: List[ProvenanceMetadata] = []

    for edge_idx in range(triples.size(0)):
        head_id = int(triples[edge_idx, 0].item())
        rel_id = int(triples[edge_idx, 1].item())
        tail_id = int(triples[edge_idx, 2].item())

        source_name, weight = _relation_hash_weight(rel_id, dataset_key)
        weights.append(weight)

        meta = ProvenanceMetadata(
            source_name=source_name,
            source_weight=weight,
            edge_index=edge_idx,
            head_id=head_id,
            relation_id=rel_id,
            tail_id=tail_id,
            provenance_score=weight,
            supporting_sources=[],
            metadata={"dataset": dataset_key, "synthetic": True},
        )
        metadata_list.append(meta)

        if (edge_idx + 1) % 50_000 == 0:
            log.info("  Processed %d / %d edges ...", edge_idx + 1, triples.size(0))

    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    return weights_tensor, metadata_list


def _generate_hetionet_provenance(
    triples: torch.Tensor,
    processed_dir: Path,
) -> Tuple[torch.Tensor, List[ProvenanceMetadata]]:
    """
    Generate provenance for Hetionet using REAL source_name field.

    Loads hetionet_edge_sources.pt  (source-name indices per training edge)
    and hetionet_source_names.json  (ordered list of source name strings).

    Args:
        triples:       (N, 3) LongTensor [head_id, relation_id, tail_id]
        processed_dir: directory containing Hetionet processed files

    Returns:
        weights_tensor: (N,) float32 tensor in [0, 1]
        metadata_list:  list of ProvenanceMetadata objects
    """
    edge_sources_path = processed_dir / "hetionet_edge_sources.pt"
    source_names_path = processed_dir / "hetionet_source_names.json"

    if not edge_sources_path.exists() or not source_names_path.exists():
        raise FileNotFoundError(
            f"Hetionet source files not found in {processed_dir}. "
            "Run download_datasets.py --dataset hetionet first."
        )

    edge_source_indices: torch.Tensor = torch.load(
        str(edge_sources_path), weights_only=True
    )
    with open(source_names_path, "r") as f:
        source_names_list: List[str] = json.load(f)

    if edge_source_indices.size(0) != triples.size(0):
        raise ValueError(
            f"Mismatch: {edge_source_indices.size(0)} source indices but "
            f"{triples.size(0)} training triples."
        )

    weights: List[float] = []
    metadata_list: List[ProvenanceMetadata] = []

    for edge_idx in range(triples.size(0)):
        head_id = int(triples[edge_idx, 0].item())
        rel_id = int(triples[edge_idx, 1].item())
        tail_id = int(triples[edge_idx, 2].item())

        src_idx = int(edge_source_indices[edge_idx].item())
        source_name = source_names_list[src_idx]
        weight = SOURCE_WEIGHTS.get(source_name, DEFAULT_WEIGHT)

        weights.append(weight)

        meta = ProvenanceMetadata(
            source_name=source_name,
            source_weight=weight,
            edge_index=edge_idx,
            head_id=head_id,
            relation_id=rel_id,
            tail_id=tail_id,
            provenance_score=weight,
            supporting_sources=[],
            metadata={"dataset": "hetionet", "real_source": True},
        )
        metadata_list.append(meta)

        if (edge_idx + 1) % 50_000 == 0:
            log.info("  Processed %d / %d edges ...", edge_idx + 1, triples.size(0))

    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    return weights_tensor, metadata_list


# ===========================================================================
# Main processing function
# ===========================================================================

def generate_provenance(dataset: str) -> None:
    """Full provenance-generation pipeline for a single dataset."""
    log.info("=== Generating provenance for: %s ===", dataset)

    processed_dir = Path(PROCESSED_DATA[dataset])
    if not processed_dir.exists():
        raise RuntimeError(
            f"Processed data directory not found: {processed_dir}. "
            "Run download_datasets.py first."
        )

    # Load training triples (provenance is generated only for training edges)
    log.info("Loading training triples ...")
    train_triples = _load_triples(processed_dir, "train")
    log.info("Loaded %d training triples.", train_triples.size(0))

    # Generate provenance
    if dataset == "hetionet":
        weights_tensor, metadata_list = _generate_hetionet_provenance(
            train_triples, processed_dir
        )
    else:
        weights_tensor, metadata_list = _generate_synthetic_provenance(
            train_triples, dataset
        )

    # Validate
    assert weights_tensor.shape == (train_triples.size(0),), (
        f"Weight tensor shape mismatch: {weights_tensor.shape}"
    )
    assert weights_tensor.min() >= 0.0 and weights_tensor.max() <= 1.0, (
        "Provenance weights out of [0, 1] range."
    )

    # Save provenance_weights.pt
    weights_path = processed_dir / "provenance_weights.pt"
    torch.save(weights_tensor, str(weights_path))
    log.info("Saved provenance_weights.pt  shape=%s", tuple(weights_tensor.shape))

    # Save provenance_metadata.json
    metadata_dicts = [m.to_dict() for m in metadata_list]
    metadata_path = processed_dir / "provenance_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata_dicts, f, indent=2)
    log.info("Saved provenance_metadata.json  (%d entries)", len(metadata_dicts))

    # Summary statistics
    log.info(
        "Provenance summary — min=%.4f  max=%.4f  mean=%.4f",
        float(weights_tensor.min()),
        float(weights_tensor.max()),
        float(weights_tensor.mean()),
    )

    # Count source distribution
    source_counts: Dict[str, int] = {}
    for m in metadata_list:
        source_counts[m.source_name] = source_counts.get(m.source_name, 0) + 1
    log.info("Source distribution: %s", source_counts)
    log.info("Provenance generation complete for %s.\n", dataset)


# ===========================================================================
# CLI entry point
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate provenance weights and metadata for KG edges.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["fb15k237", "wn18rr", "hetionet"],
        help="Dataset to generate provenance for.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    generate_provenance(args.dataset)


if __name__ == "__main__":
    main()
