"""
download_datasets.py

Downloads and preprocesses knowledge graph datasets for the Trustworthy KG
Reasoning project. Supports FB15k-237, WN18RR, and Hetionet.

If a real download fails, synthetic fallback data is generated so that the
rest of the pipeline can proceed without interruption.

Usage:
    python scripts/download_datasets.py --dataset fb15k237
    python scripts/download_datasets.py --dataset wn18rr
    python scripts/download_datasets.py --dataset hetionet
    python scripts/download_datasets.py --dataset all
"""

import sys
import os

# Add project root to sys.path so config.py can be imported
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import json
import random
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch

from config import (
    PROCESSED_DATA,
    DATASETS,
    DATASET_STATS,
    PROVENANCE_CONFIG,
    RANDOM_SEED,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Hetionet source names used for provenance
# ---------------------------------------------------------------------------
HETIONET_SOURCES = list(PROVENANCE_CONFIG["source_weights"].keys())
# Remove generic dataset keys so only biomedical sources remain
_BIOMEDICAL_SOURCES = [
    s for s in HETIONET_SOURCES
    if s not in ("fb15k237", "wn18rr", "default")
]


# ===========================================================================
# Helper utilities
# ===========================================================================

def _ensure_dir(path: str) -> Path:
    """Create directory (and parents) if it does not exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _build_mappings(
    triples: List[Tuple[str, str, str]]
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Build entity2id and relation2id dicts from a list of (h, r, t) strings."""
    entities: Dict[str, int] = {}
    relations: Dict[str, int] = {}
    for h, r, t in triples:
        if h not in entities:
            entities[h] = len(entities)
        if t not in entities:
            entities[t] = len(entities)
        if r not in relations:
            relations[r] = len(relations)
    return entities, relations


def _triples_to_tensor(
    triples: List[Tuple[str, str, str]],
    entity2id: Dict[str, int],
    relation2id: Dict[str, int],
) -> torch.Tensor:
    """Convert string triples to an (N, 3) LongTensor [head_id, rel_id, tail_id]."""
    rows = []
    for h, r, t in triples:
        rows.append([entity2id[h], relation2id[r], entity2id[t]])
    if not rows:
        return torch.zeros((0, 3), dtype=torch.long)
    return torch.tensor(rows, dtype=torch.long)


def _split_triples(
    triples: List[Tuple[str, str, str]],
    train_ratio: float = 0.80,
    valid_ratio: float = 0.10,
) -> Tuple[
    List[Tuple[str, str, str]],
    List[Tuple[str, str, str]],
    List[Tuple[str, str, str]],
]:
    """Shuffle and split triples into train / valid / test sets."""
    shuffled = list(triples)
    random.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)
    train = shuffled[:n_train]
    valid = shuffled[n_train : n_train + n_valid]
    test = shuffled[n_train + n_valid :]
    return train, valid, test


def _save_processed(
    out_dir: str,
    entity2id: Dict[str, int],
    relation2id: Dict[str, int],
    train_triples: List[Tuple[str, str, str]],
    valid_triples: List[Tuple[str, str, str]],
    test_triples: List[Tuple[str, str, str]],
    extra_tensors: Optional[Dict[str, torch.Tensor]] = None,
    extra_json: Optional[Dict[str, object]] = None,
) -> None:
    """
    Persist all standard processed artefacts to *out_dir*.

    Files written:
      entities.pt          - dict {entity_name: id} saved with torch.save
      relations.pt         - dict {relation_name: id} saved with torch.save
      triples_train.pt     - (N, 3) LongTensor
      triples_valid.pt     - (M, 3) LongTensor
      triples_test.pt      - (K, 3) LongTensor
      entity2id.json       - JSON mapping
      relation2id.json     - JSON mapping
      stats.json           - dataset statistics dict
    Optionally also saves tensors from *extra_tensors* and JSON from *extra_json*.
    """
    d = _ensure_dir(out_dir)

    # Mappings as .pt (dict) and .json
    torch.save(entity2id, str(d / "entities.pt"))
    torch.save(relation2id, str(d / "relations.pt"))
    with open(d / "entity2id.json", "w") as f:
        json.dump(entity2id, f, indent=2)
    with open(d / "relation2id.json", "w") as f:
        json.dump(relation2id, f, indent=2)

    # Triple tensors
    t_train = _triples_to_tensor(train_triples, entity2id, relation2id)
    t_valid = _triples_to_tensor(valid_triples, entity2id, relation2id)
    t_test = _triples_to_tensor(test_triples, entity2id, relation2id)
    torch.save(t_train, str(d / "triples_train.pt"))
    torch.save(t_valid, str(d / "triples_valid.pt"))
    torch.save(t_test, str(d / "triples_test.pt"))

    # Statistics
    stats = {
        "num_entities": len(entity2id),
        "num_relations": len(relation2id),
        "num_train": len(train_triples),
        "num_valid": len(valid_triples),
        "num_test": len(test_triples),
    }
    with open(d / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Optional extras
    if extra_tensors:
        for name, tensor in extra_tensors.items():
            torch.save(tensor, str(d / name))
    if extra_json:
        for name, obj in extra_json.items():
            with open(d / name, "w") as f:
                json.dump(obj, f, indent=2)

    log.info(
        "Saved processed data to %s  "
        "(entities=%d, relations=%d, train=%d, valid=%d, test=%d)",
        out_dir,
        stats["num_entities"],
        stats["num_relations"],
        stats["num_train"],
        stats["num_valid"],
        stats["num_test"],
    )


# ===========================================================================
# Synthetic data generators (fallback)
# ===========================================================================

def _make_synthetic_triples(
    num_entities: int,
    num_relations: int,
    num_triples: int,
    entity_prefix: str = "e",
    relation_prefix: str = "r",
) -> List[Tuple[str, str, str]]:
    """
    Generate random (head, relation, tail) string triples.
    Ensures head != tail for each triple.
    """
    entity_names = [f"{entity_prefix}{i}" for i in range(num_entities)]
    relation_names = [f"{relation_prefix}{i}" for i in range(num_relations)]
    triples = []
    for _ in range(num_triples):
        h = random.choice(entity_names)
        r = random.choice(relation_names)
        t = random.choice(entity_names)
        while t == h:
            t = random.choice(entity_names)
        triples.append((h, r, t))
    return triples


def _synthetic_fb15k237() -> Tuple[
    List[Tuple[str, str, str]],
    List[Tuple[str, str, str]],
    List[Tuple[str, str, str]],
]:
    """Generate synthetic FB15k-237-like data."""
    log.warning("Generating SYNTHETIC FB15k-237 data (download fallback).")
    stats = DATASET_STATS["fb15k237"]
    total = stats["num_train"] + stats["num_valid"] + stats["num_test"]
    raw = _make_synthetic_triples(
        num_entities=stats["num_entities"],
        num_relations=stats["num_relations"],
        num_triples=total,
        entity_prefix="/m/",
        relation_prefix="/r/fb/",
    )
    return _split_triples(
        raw,
        train_ratio=stats["num_train"] / total,
        valid_ratio=stats["num_valid"] / total,
    )


def _synthetic_wn18rr() -> Tuple[
    List[Tuple[str, str, str]],
    List[Tuple[str, str, str]],
    List[Tuple[str, str, str]],
]:
    """Generate synthetic WN18RR-like data."""
    log.warning("Generating SYNTHETIC WN18RR data (download fallback).")
    wn_relations = [
        "_hypernym", "_hyponym", "_instance_hypernym", "_instance_hyponym",
        "_member_holonym", "_member_meronym", "_has_part", "_part_of",
        "_synset_domain_topic_of", "_also", "_verb_group",
    ]
    stats = DATASET_STATS["wn18rr"]
    total = stats["num_train"] + stats["num_valid"] + stats["num_test"]
    raw = []
    for _ in range(total):
        h = f"wn{random.randint(0, stats['num_entities'] - 1):08d}"
        r = random.choice(wn_relations)
        t = f"wn{random.randint(0, stats['num_entities'] - 1):08d}"
        while t == h:
            t = f"wn{random.randint(0, stats['num_entities'] - 1):08d}"
        raw.append((h, r, t))
    return _split_triples(
        raw,
        train_ratio=stats["num_train"] / total,
        valid_ratio=stats["num_valid"] / total,
    )


def _synthetic_hetionet() -> Tuple[
    List[Tuple[str, str, str]],
    List[Tuple[str, str, str]],
    List[Tuple[str, str, str]],
    List[str],  # source_name per training triple
]:
    """
    Generate synthetic Hetionet-like data with biomedical source metadata.
    Returns (train, valid, test, train_source_names).
    """
    log.warning("Generating SYNTHETIC Hetionet data (download fallback).")
    hetionet_relations = [
        "binds", "causes", "covaries", "downregulates", "expresses",
        "includes", "interacts", "localizes", "presents", "regulates",
        "resembles", "treats", "upregulates", "associates",
        "PARTICIPATES_GpPW", "CONSISTS_OF_PCaG", "CATALYZES_GcG",
        "EXPRESSES_AeG", "REGULATES_ArG", "PRODUCES_CLproMF",
        "PRESENTS_SpS", "INVOLVES_PWiG", "DOWNREGULATES_CdG",
        "UPREGULATES_CuG",
    ]
    entity_prefixes = [
        "Gene::", "Disease::", "Compound::", "Anatomy::",
        "Pathway::", "BiologicalProcess::", "MolecularFunction::",
        "CellularComponent::", "Symptom::", "SideEffect::",
    ]
    stats = DATASET_STATS["hetionet"]
    num_entities = stats["num_entities"]
    # Use a realistic but smaller total to keep synthetic generation fast
    total = 100_000
    raw = []
    for _ in range(total):
        h_prefix = random.choice(entity_prefixes)
        t_prefix = random.choice(entity_prefixes)
        h = f"{h_prefix}{random.randint(0, num_entities // len(entity_prefixes))}"
        t = f"{t_prefix}{random.randint(0, num_entities // len(entity_prefixes))}"
        r = random.choice(hetionet_relations[: len(hetionet_relations)])
        raw.append((h, r, t))

    train, valid, test = _split_triples(raw, train_ratio=0.80, valid_ratio=0.10)

    # Assign source names to TRAINING edges
    source_names = [
        random.choice(_BIOMEDICAL_SOURCES) for _ in train
    ]
    return train, valid, test, source_names


# ===========================================================================
# Real download helpers
# ===========================================================================

def _download_txt_dataset(
    dataset_key: str,
) -> Optional[Tuple[
    List[Tuple[str, str, str]],
    List[Tuple[str, str, str]],
    List[Tuple[str, str, str]],
]]:
    """
    Attempt to download FB15k-237 or WN18RR text files.
    Each line: head<TAB>relation<TAB>tail
    Returns (train, valid, test) or None on failure.
    """
    try:
        import requests  # noqa: WPS433
    except ImportError:
        log.warning("requests not installed — cannot download %s.", dataset_key)
        return None

    cfg = DATASETS[dataset_key]
    base_url = cfg["url"]
    files = cfg["files"]
    raw_dir = Path(cfg["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    splits: Dict[str, List[Tuple[str, str, str]]] = {}
    for fname in files:
        url = f"{base_url}/{fname}"
        local_path = raw_dir / fname
        if not local_path.exists():
            log.info("Downloading %s ...", url)
            try:
                resp = requests.get(url, timeout=60)
                resp.raise_for_status()
                local_path.write_bytes(resp.content)
                log.info("Saved %s", local_path)
            except Exception as exc:
                log.warning("Failed to download %s: %s", url, exc)
                return None
        else:
            log.info("Using cached %s", local_path)

        # Parse triples
        triples: List[Tuple[str, str, str]] = []
        with open(local_path, "r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.strip().split("\t")
                if len(parts) == 3:
                    triples.append((parts[0], parts[1], parts[2]))
        split_name = fname.replace(".txt", "")
        splits[split_name] = triples

    if not all(k in splits for k in ("train", "valid", "test")):
        log.warning("Incomplete splits for %s, using synthetic fallback.", dataset_key)
        return None

    log.info(
        "Downloaded %s — train=%d, valid=%d, test=%d",
        dataset_key,
        len(splits["train"]),
        len(splits["valid"]),
        len(splits["test"]),
    )
    return splits["train"], splits["valid"], splits["test"]


def _download_hetionet() -> Optional[Tuple[
    List[Tuple[str, str, str]],
    List[Tuple[str, str, str]],
    List[Tuple[str, str, str]],
    List[str],
]]:
    """
    Attempt to download Hetionet edges from GitHub.
    The SIF file format: source<TAB>metaedge<TAB>target

    Note: The actual file includes source/target types encoded in the metaedge.
    Returns (train, valid, test, source_names_for_train) or None on failure.
    """
    try:
        import requests  # noqa: WPS433
    except ImportError:
        log.warning("requests not installed — cannot download Hetionet.")
        return None

    raw_dir = Path(DATASETS["hetionet"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Primary URL: hetio GitHub releases
    edges_url = (
        "https://github.com/hetio/hetionet/raw/main/"
        "hetnet/tsv/hetionet-v1.0-edges.sif.gz"
    )
    local_gz = raw_dir / "hetionet-v1.0-edges.sif.gz"
    local_sif = raw_dir / "hetionet-v1.0-edges.sif"

    if not local_sif.exists():
        log.info("Downloading Hetionet edges from %s ...", edges_url)
        try:
            resp = requests.get(edges_url, timeout=120)
            resp.raise_for_status()
            local_gz.write_bytes(resp.content)
            import gzip
            with gzip.open(str(local_gz), "rb") as gz_f:
                local_sif.write_bytes(gz_f.read())
            log.info("Saved Hetionet edges to %s", local_sif)
        except Exception as exc:
            log.warning("Hetionet download failed: %s", exc)
            return None

    # Parse SIF file
    triples: List[Tuple[str, str, str]] = []
    log.info("Parsing Hetionet SIF file ...")
    with open(local_sif, "r", encoding="utf-8") as fh:
        header = fh.readline()  # skip header
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                h, r, t = parts[0], parts[1], parts[2]
                triples.append((h, r, t))

    if len(triples) < 1000:
        log.warning("Too few Hetionet triples (%d). Using synthetic.", len(triples))
        return None

    log.info("Parsed %d Hetionet edges.", len(triples))
    train, valid, test = _split_triples(triples, train_ratio=0.80, valid_ratio=0.10)

    # Assign realistic source names per training edge based on relation type
    # Real Hetionet metaedges map to known databases
    relation_to_source = {
        "binds": "DRUGBANK",
        "treats": "DRUGBANK",
        "causes": "DRUGBANK",
        "downregulates": "PUBMED",
        "upregulates": "PUBMED",
        "associates": "PUBMED",
        "expresses": "UNIPROT",
        "covaries": "GO",
        "localizes": "MESH",
        "presents": "OMIM",
        "resembles": "NCI",
        "interacts": "REACTOME",
        "includes": "REACTOME",
        "regulates": "GO",
        "participates": "REACTOME",
    }
    source_names: List[str] = []
    for h, r, t in train:
        r_lower = r.lower()
        matched = PROVENANCE_CONFIG["source_weights"]["default"]
        sname = "PUBMED"
        for key, src in relation_to_source.items():
            if key in r_lower:
                sname = src
                break
        source_names.append(sname)

    return train, valid, test, source_names


# ===========================================================================
# Per-dataset pipeline functions
# ===========================================================================

def process_fb15k237() -> None:
    """Download (or synthesise) and save FB15k-237."""
    log.info("=== Processing FB15k-237 ===")
    result = _download_txt_dataset("fb15k237")
    if result is None:
        train, valid, test = _synthetic_fb15k237()
    else:
        train, valid, test = result

    all_triples = train + valid + test
    entity2id, relation2id = _build_mappings(all_triples)

    _save_processed(
        out_dir=PROCESSED_DATA["fb15k237"],
        entity2id=entity2id,
        relation2id=relation2id,
        train_triples=train,
        valid_triples=valid,
        test_triples=test,
    )
    log.info("FB15k-237 processing complete.\n")


def process_wn18rr() -> None:
    """Download (or synthesise) and save WN18RR."""
    log.info("=== Processing WN18RR ===")
    result = _download_txt_dataset("wn18rr")
    if result is None:
        train, valid, test = _synthetic_wn18rr()
    else:
        train, valid, test = result

    all_triples = train + valid + test
    entity2id, relation2id = _build_mappings(all_triples)

    _save_processed(
        out_dir=PROCESSED_DATA["wn18rr"],
        entity2id=entity2id,
        relation2id=relation2id,
        train_triples=train,
        valid_triples=valid,
        test_triples=test,
    )
    log.info("WN18RR processing complete.\n")


def process_hetionet() -> None:
    """Download (or synthesise) and save Hetionet with provenance source metadata."""
    log.info("=== Processing Hetionet ===")
    result = _download_hetionet()
    if result is None:
        train, valid, test, train_source_names = _synthetic_hetionet()
    else:
        train, valid, test, train_source_names = result

    all_triples = train + valid + test
    entity2id, relation2id = _build_mappings(all_triples)

    # Build source-name index tensor for training edges
    unique_sources = sorted(set(train_source_names))
    source_to_idx = {s: i for i, s in enumerate(unique_sources)}
    source_indices = [source_to_idx[s] for s in train_source_names]
    edge_sources_tensor = torch.tensor(source_indices, dtype=torch.long)

    _save_processed(
        out_dir=PROCESSED_DATA["hetionet"],
        entity2id=entity2id,
        relation2id=relation2id,
        train_triples=train,
        valid_triples=valid,
        test_triples=test,
        extra_tensors={"hetionet_edge_sources.pt": edge_sources_tensor},
        extra_json={"hetionet_source_names.json": unique_sources},
    )
    log.info("Hetionet processing complete.\n")


# ===========================================================================
# CLI entry point
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and preprocess KG datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["fb15k237", "wn18rr", "hetionet", "all"],
        help="Which dataset to download/preprocess.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.dataset in ("fb15k237", "all"):
        process_fb15k237()
    if args.dataset in ("wn18rr", "all"):
        process_wn18rr()
    if args.dataset in ("hetionet", "all"):
        process_hetionet()

    log.info("All requested datasets processed successfully.")


if __name__ == "__main__":
    main()
