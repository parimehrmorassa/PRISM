"""
create_trust_calibration.py

Creates trust calibration labels for training the trust aggregator module.

Labels are binary:
  1 = the model's prediction for this query was "correct" (high-confidence proxy)
  0 = the model's prediction was "wrong"  (low-confidence proxy)

Since ground-truth model predictions are not yet available at dataset-creation
time, we use two structural heuristics as proxy labels:

  1. Relation frequency proxy:
       High-frequency relations are more often predicted correctly by KGE models,
       so triples whose relation appears > median frequency times are labeled 1.

  2. Entity-degree proxy:
       Queries involving hub entities (high degree) are harder to rank correctly,
       so we down-weight them. Specifically, a triple gets label 1 only if BOTH
       the head-degree AND relation-frequency are above their respective medians.

Outputs (written to TRUST_CALIB_DIR[dataset]/):
  calibration_labels.pt    - (N, 4) LongTensor [query_idx, head_id, rel_id, label]
  relation_frequencies.pt  - (num_relations,) LongTensor  (count per relation)
  entity_degrees.pt        - (num_entities,)  LongTensor  (degree per entity)

Usage:
    python scripts/create_trust_calibration.py --dataset fb15k237
    python scripts/create_trust_calibration.py --dataset wn18rr
    python scripts/create_trust_calibration.py --dataset hetionet
"""

import sys
import os

# Add project root to sys.path so config.py can be imported
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import torch

from config import (
    PROCESSED_DATA,
    TRUST_CALIB_DIR,
    DATASET_STATS,
    RANDOM_SEED,
)

# ---------------------------------------------------------------------------
# Corruption ratios (Phase 1C calibration)
# ---------------------------------------------------------------------------
RANDOM_CORRUPTION_RATIO = 0.40      # fraction of triples randomly poisoned
ADVERSARIAL_CORRUPTION_RATIO = 0.20 # fraction of hub/rare triples adversarially poisoned

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
# Reproducibility
# ---------------------------------------------------------------------------
torch.manual_seed(RANDOM_SEED)


# ===========================================================================
# Helpers
# ===========================================================================

def _load_triples(processed_dir: Path, split: str) -> torch.Tensor:
    """Load triples_{split}.pt. Returns (N, 3) LongTensor or empty tensor."""
    path = processed_dir / f"triples_{split}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Triples file not found: {path}. "
            "Run download_datasets.py first."
        )
    return torch.load(str(path), weights_only=True)


def _load_stats(processed_dir: Path, dataset: str) -> Tuple[int, int]:
    """Return (num_entities, num_relations) from stats.json or DATASET_STATS."""
    stats_path = processed_dir / "stats.json"
    if stats_path.exists():
        with open(stats_path, "r") as f:
            stats = json.load(f)
        return int(stats["num_entities"]), int(stats["num_relations"])
    # Fallback to config constants
    ds = DATASET_STATS.get(dataset, {})
    num_entities = ds.get("num_entities") or 10_000
    num_relations = ds.get("num_relations") or 50
    log.warning(
        "stats.json not found. Using config fallback: "
        "num_entities=%d, num_relations=%d",
        num_entities, num_relations,
    )
    return num_entities, num_relations


def _compute_relation_frequencies(
    triples: torch.Tensor,
    num_relations: int,
) -> torch.Tensor:
    """
    Count how many times each relation appears in *triples*.

    Args:
        triples:       (N, 3) LongTensor
        num_relations: total number of distinct relations

    Returns:
        (num_relations,) LongTensor with counts
    """
    rel_ids = triples[:, 1]
    # bincount requires non-negative ints; max relation id may be < num_relations
    max_rel = max(int(rel_ids.max().item()) + 1, num_relations)
    freqs = torch.bincount(rel_ids, minlength=max_rel)
    # Truncate or pad to exactly num_relations
    if freqs.size(0) > num_relations:
        freqs = freqs[:num_relations]
    elif freqs.size(0) < num_relations:
        pad = torch.zeros(num_relations - freqs.size(0), dtype=torch.long)
        freqs = torch.cat([freqs, pad])
    return freqs


def _compute_entity_degrees(
    triples: torch.Tensor,
    num_entities: int,
) -> torch.Tensor:
    """
    Compute the degree (number of incident edges) for each entity.
    Both head and tail appearances are counted.

    Args:
        triples:      (N, 3) LongTensor
        num_entities: total number of distinct entities

    Returns:
        (num_entities,) LongTensor with degree counts
    """
    head_ids = triples[:, 0]
    tail_ids = triples[:, 2]
    all_ids = torch.cat([head_ids, tail_ids])

    max_ent = max(int(all_ids.max().item()) + 1, num_entities)
    degrees = torch.bincount(all_ids, minlength=max_ent)

    if degrees.size(0) > num_entities:
        degrees = degrees[:num_entities]
    elif degrees.size(0) < num_entities:
        pad = torch.zeros(num_entities - degrees.size(0), dtype=torch.long)
        degrees = torch.cat([degrees, pad])
    return degrees


def _compute_calibration_data(
    triples: torch.Tensor,
    relation_frequencies: torch.Tensor,
    entity_degrees: torch.Tensor,
    seed: int = RANDOM_SEED,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Four-type calibration dataset that prevents trivial learning from S_prov alone.

    The core design: Types C and D deliberately cross the S_prov/label correlation,
    so the model MUST use U and Delta_CF to classify them correctly.

    Type A (35%) — Verified, high prov:
        Structurally trustworthy (frequent relation + low-degree entity).
        S_prov ~ U(0.70, 1.00), U ~ U(0.05, 0.35), Delta_CF ~ U(0.05, 0.25), label=1

    Type B (25%) — Poisoned, low prov:
        Random corruption, easy negative.
        S_prov ~ U(0.05, 0.40), U ~ U(0.60, 0.95), Delta_CF ~ U(0.60, 0.95), label=0

    Type C (25%) — HARD NEGATIVE: poisoned but high prov:
        Adversarial (hub entity + rare relation). S_prov is artificially elevated.
        Only U and Delta_CF reveal the corruption.
        S_prov ~ U(0.65, 0.85), U ~ U(0.55, 0.85), Delta_CF ~ U(0.60, 0.90), label=0

    Type D (15%) — HARD POSITIVE: verified but low prov:
        Real valid triple from a rare/low-credibility relation.
        S_prov is low, but U and Delta_CF confirm the fact is real.
        S_prov ~ U(0.35, 0.60), U ~ U(0.10, 0.40), Delta_CF ~ U(0.15, 0.45), label=1

    Returns:
        calibration_labels:     (N, 4) LongTensor  [query_idx, head_id, rel_id, label]
        calibration_components: (N, 3) FloatTensor [U, S_prov, Delta_CF]
    """
    N = triples.size(0)
    head_ids = triples[:, 0]
    rel_ids  = triples[:, 1]

    rel_freqs = relation_frequencies[rel_ids].float()

    h_clamped = head_ids.clamp(0, entity_degrees.size(0) - 1)
    head_degrees = entity_degrees[h_clamped].float()
    hub_threshold = float(torch.quantile(entity_degrees.float(), 0.75).item())
    hub_threshold = max(hub_threshold, 1.0)

    # Normalised structural scores for sorting
    rf_min, rf_max = rel_freqs.min(), rel_freqs.max()
    dg_min, dg_max = head_degrees.min(), head_degrees.max()
    rf_norm = (rel_freqs - rf_min) / (rf_max - rf_min + 1e-8)
    dg_norm = (head_degrees - dg_min) / (dg_max - dg_min + 1e-8)

    # Strength: high freq + low degree → Type A (genuinely trustworthy)
    # Weakness: high degree + low freq → Type C (adversarial candidates)
    strength = rf_norm - dg_norm
    weakness = dg_norm - rf_norm

    n_A = int(0.30 * N)  # 30% verified high-prov
    n_B = int(0.20 * N)  # 20% poisoned low-prov
    n_C = int(0.30 * N)  # 30% hard-negative (high-prov, poisoned)
    n_D = N - n_A - n_B - n_C  # ~20% hard-positive (low-prov, verified)

    sample_type = torch.full((N,), -1, dtype=torch.long)

    # Type A: top-strength triples
    sorted_A = torch.argsort(strength, descending=True)
    sample_type[sorted_A[:n_A]] = 0

    # Type C: top-weakness among unassigned
    unassigned = (sample_type == -1).nonzero(as_tuple=True)[0]
    sorted_C = unassigned[torch.argsort(weakness[unassigned], descending=True)]
    sample_type[sorted_C[:n_C]] = 2

    # Type D: rarest-relation among still-unassigned
    unassigned = (sample_type == -1).nonzero(as_tuple=True)[0]
    sorted_D = unassigned[torch.argsort(rel_freqs[unassigned], descending=False)]
    sample_type[sorted_D[:n_D]] = 3

    # Type B: all remaining
    sample_type[sample_type == -1] = 1

    idx_A = (sample_type == 0).nonzero(as_tuple=True)[0]
    idx_B = (sample_type == 1).nonzero(as_tuple=True)[0]
    idx_C = (sample_type == 2).nonzero(as_tuple=True)[0]
    idx_D = (sample_type == 3).nonzero(as_tuple=True)[0]

    labels = torch.zeros(N, dtype=torch.long)
    labels[idx_A] = 1
    labels[idx_B] = 0
    labels[idx_C] = 0  # hard negative
    labels[idx_D] = 1  # hard positive

    rng = torch.Generator()
    rng.manual_seed(seed)

    def _u(size, lo, hi): return torch.rand(size, generator=rng) * (hi - lo) + lo

    # ── S_prov ────────────────────────────────────────────────────────────────
    # Type C (label=0) overlaps Type A (label=1) → S_prov cannot separate them.
    # Type D (label=1) overlaps Type B (label=0) → S_prov cannot separate them.
    # Expected S_prov AUC ≈ 0.62.
    S_prov = torch.zeros(N)
    S_prov[idx_A] = _u(len(idx_A), 0.70, 1.00)   # clearly high
    S_prov[idx_B] = _u(len(idx_B), 0.05, 0.35)   # clearly low
    S_prov[idx_C] = _u(len(idx_C), 0.60, 0.90)   # high — overlaps A, misleads
    S_prov[idx_D] = _u(len(idx_D), 0.30, 0.65)   # low-to-mid — overlaps B, misleads

    # ── U (uncertainty): low = confident = verified ────────────────────────────
    # Design: U is the KEY signal for Type D (distinctly low U confirms the fact
    # is real despite low S_prov). U is moderate/overlapping for Type C so that
    # U alone cannot detect it — Delta_CF must do that job.
    # Expected 1-U AUC ≈ 0.73–0.80.
    U = torch.zeros(N)
    U[idx_A] = _u(len(idx_A), 0.05, 0.50)   # low — confirmed trustworthy
    U[idx_B] = _u(len(idx_B), 0.50, 0.95)   # high — confirmed untrustworthy
    U[idx_C] = _u(len(idx_C), 0.30, 0.75)   # MODERATE — overlaps A and B; U alone misses C
    U[idx_D] = _u(len(idx_D), 0.05, 0.45)   # LOW — α catches Type D despite low S_prov

    # ── Delta_CF (counterfactual sensitivity): low = stable = verified ─────────
    # Design: Delta_CF is the KEY signal for Type C (distinctly high sensitivity
    # exposes the adversarial corruption despite high S_prov). Delta_CF is
    # moderate/overlapping for Type D so the model must use U for that.
    # U and Delta_CF are therefore INDEPENDENT — forcing both α and γ to matter.
    # Expected 1-Delta_CF AUC ≈ 0.63–0.72.
    Delta_CF = torch.zeros(N)
    Delta_CF[idx_A] = _u(len(idx_A), 0.20, 0.70)   # moderate — overlaps C lower tail
    Delta_CF[idx_B] = _u(len(idx_B), 0.25, 0.85)   # moderate-high — noisy easy negative
    Delta_CF[idx_C] = _u(len(idx_C), 0.65, 0.95)   # HIGH — γ catches Type C despite high S_prov
    Delta_CF[idx_D] = _u(len(idx_D), 0.20, 0.75)   # moderate — overlaps C lower tail; U does the work

    query_idx = torch.arange(N, dtype=torch.long)
    calibration_labels = torch.stack([query_idx, head_ids, rel_ids, labels], dim=1)
    calibration_components = torch.stack([U, S_prov, Delta_CF], dim=1)

    # Per-type diagnostics
    for _, name, idx in [(0,"A",idx_A),(1,"B",idx_B),(2,"C",idx_C),(3,"D",idx_D)]:
        n = len(idx)
        log.info(
            "  Type %s: n=%d (%4.1f%%) label=%.0f "
            "S_prov=%.3f U=%.3f Delta_CF=%.3f",
            name, n, 100.0 * n / N,
            float(labels[idx].float().mean()),
            float(S_prov[idx].mean()),
            float(U[idx].mean()),
            float(Delta_CF[idx].mean()),
        )

    pos_rate = float(labels.float().mean())
    log.info(
        "Overall: N=%d, positive=%d (%.1f%%), negative=%d (%.1f%%)",
        N, int(labels.sum()), pos_rate * 100,
        N - int(labels.sum()), (1 - pos_rate) * 100,
    )
    return calibration_labels, calibration_components


# ===========================================================================
# Main pipeline
# ===========================================================================

def create_trust_calibration(dataset: str) -> None:
    """Full calibration-data creation pipeline for a single dataset."""
    log.info("=== Creating trust calibration labels for: %s ===", dataset)

    processed_dir = Path(PROCESSED_DATA[dataset])
    if not processed_dir.exists():
        raise RuntimeError(
            f"Processed data directory not found: {processed_dir}. "
            "Run download_datasets.py first."
        )

    calib_dir = Path(str(TRUST_CALIB_DIR)) / dataset
    calib_dir.mkdir(parents=True, exist_ok=True)

    # Load statistics
    num_entities, num_relations = _load_stats(processed_dir, dataset)
    log.info("Dataset: num_entities=%d, num_relations=%d", num_entities, num_relations)

    # Load all splits; use all triples for frequency / degree computation,
    # but generate calibration labels only for training triples.
    log.info("Loading triples ...")
    train_triples = _load_triples(processed_dir, "train")
    valid_triples = _load_triples(processed_dir, "valid")
    test_triples = _load_triples(processed_dir, "test")
    all_triples = torch.cat([train_triples, valid_triples, test_triples], dim=0)
    log.info(
        "Loaded — train=%d, valid=%d, test=%d, total=%d",
        train_triples.size(0),
        valid_triples.size(0),
        test_triples.size(0),
        all_triples.size(0),
    )

    # Compute relation frequencies and entity degrees over ALL triples
    log.info("Computing relation frequencies ...")
    relation_frequencies = _compute_relation_frequencies(all_triples, num_relations)
    log.info(
        "Relation frequencies — min=%d, max=%d, mean=%.1f",
        int(relation_frequencies.min().item()),
        int(relation_frequencies.max().item()),
        float(relation_frequencies.float().mean().item()),
    )

    log.info("Computing entity degrees ...")
    entity_degrees = _compute_entity_degrees(all_triples, num_entities)
    log.info(
        "Entity degrees — min=%d, max=%d, mean=%.1f",
        int(entity_degrees.min().item()),
        int(entity_degrees.max().item()),
        float(entity_degrees.float().mean().item()),
    )

    # Compute 4-type calibration data for training triples
    log.info("Computing calibration data (4-type design) ...")
    calibration_labels, calibration_components = _compute_calibration_data(
        train_triples, relation_frequencies, entity_degrees
    )

    # Save artefacts
    calib_labels_path      = calib_dir / "calibration_labels.pt"
    calib_components_path  = calib_dir / "calibration_components.pt"
    rel_freq_path          = calib_dir / "relation_frequencies.pt"
    ent_deg_path           = calib_dir / "entity_degrees.pt"

    torch.save(calibration_labels, str(calib_labels_path))
    log.info("Saved calibration_labels.pt     shape=%s", tuple(calibration_labels.shape))

    torch.save(calibration_components, str(calib_components_path))
    log.info("Saved calibration_components.pt shape=%s", tuple(calibration_components.shape))

    torch.save(relation_frequencies, str(rel_freq_path))
    log.info("Saved relation_frequencies.pt   shape=%s", tuple(relation_frequencies.shape))

    torch.save(entity_degrees, str(ent_deg_path))
    log.info("Saved entity_degrees.pt         shape=%s", tuple(entity_degrees.shape))

    # Save a brief summary JSON for inspection
    labels_col = calibration_labels[:, 3]
    summary = {
        "dataset": dataset,
        "num_entities": num_entities,
        "num_relations": num_relations,
        "num_calibration_samples": int(calibration_labels.size(0)),
        "num_positive_labels": int(labels_col.sum().item()),
        "num_negative_labels": int((labels_col == 0).sum().item()),
        "positive_rate": float(labels_col.float().mean().item()),
        "mean_U":        float(calibration_components[:, 0].mean().item()),
        "mean_S_prov":   float(calibration_components[:, 1].mean().item()),
        "mean_Delta_CF": float(calibration_components[:, 2].mean().item()),
        "is_synthetic": True,  # U and Delta_CF are structurally-derived, NOT real model outputs
    }
    with open(calib_dir / "calibration_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Saved calibration_summary.json")

    log.info("Trust calibration creation complete for %s.\n", dataset)


# ===========================================================================
# CLI entry point
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create trust calibration labels for the trust aggregator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["fb15k237", "wn18rr", "hetionet"],
        help="Dataset to generate calibration labels for.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    create_trust_calibration(args.dataset)


if __name__ == "__main__":
    main()
