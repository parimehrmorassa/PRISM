"""
Data augmentation utilities for knowledge graph training.
"""

import sys
import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import RANDOM_SEED

logger = logging.getLogger(__name__)


def add_inverse_relations(
    triples: torch.Tensor,
    num_relations: int,
    provenance_weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, int, Optional[torch.Tensor]]:
    """
    Add inverse triples (t, r+num_relations, h) for each (h, r, t).

    This is a standard KG augmentation that helps with reciprocal reasoning.

    Args:
        triples: (N, 3) tensor [head, relation, tail]
        num_relations: Original number of relations
        provenance_weights: Optional (N,) tensor; inverted triples get same provenance

    Returns:
        (augmented_triples, new_num_relations, augmented_provenance)
    """
    h, r, t = triples[:, 0], triples[:, 1], triples[:, 2]
    inv_r = r + num_relations
    inv_triples = torch.stack([t, inv_r, h], dim=1)
    augmented = torch.cat([triples, inv_triples], dim=0)
    new_num_relations = num_relations * 2

    aug_prov = None
    if provenance_weights is not None:
        aug_prov = torch.cat([provenance_weights, provenance_weights], dim=0)

    logger.info(
        f"Added {len(inv_triples)} inverse triples. "
        f"Num relations: {num_relations} → {new_num_relations}"
    )
    return augmented, new_num_relations, aug_prov


def filter_triples_by_degree(
    triples: torch.Tensor,
    num_entities: int,
    min_degree: int = 1,
    max_degree: Optional[int] = None,
) -> torch.Tensor:
    """
    Filter triples to only include entities with degree in [min_degree, max_degree].

    Useful for removing very rare entities or hub entities.
    """
    degrees = torch.zeros(num_entities, dtype=torch.long)
    for col in [0, 2]:
        for e in triples[:, col]:
            degrees[e] += 1

    if max_degree is None:
        max_degree = degrees.max().item()

    valid_mask = (degrees >= min_degree) & (degrees <= max_degree)
    triple_mask = valid_mask[triples[:, 0]] & valid_mask[triples[:, 2]]
    filtered = triples[triple_mask]
    logger.info(
        f"Filtered triples: {len(triples)} → {len(filtered)} "
        f"(degree range [{min_degree}, {max_degree}])"
    )
    return filtered


def subsample_triples(
    triples: torch.Tensor,
    fraction: float = 1.0,
    seed: int = RANDOM_SEED,
) -> torch.Tensor:
    """
    Randomly subsample a fraction of triples (for faster debugging).

    Args:
        triples: (N, 3) tensor
        fraction: Fraction in (0, 1] to keep
        seed: Random seed

    Returns:
        Subsampled triples tensor
    """
    if fraction >= 1.0:
        return triples
    torch.manual_seed(seed)
    n = int(len(triples) * fraction)
    idx = torch.randperm(len(triples))[:n]
    return triples[idx]


def compute_entity_degrees(triples: torch.Tensor, num_entities: int) -> torch.Tensor:
    """
    Compute degree of each entity (in + out edges).

    Args:
        triples: (N, 3) tensor [head, relation, tail]
        num_entities: Total number of entities

    Returns:
        (num_entities,) tensor of degrees
    """
    degrees = torch.zeros(num_entities, dtype=torch.long)
    for e in triples[:, 0]:
        degrees[e] += 1
    for e in triples[:, 2]:
        degrees[e] += 1
    return degrees


def compute_relation_frequencies(
    triples: torch.Tensor, num_relations: int
) -> torch.Tensor:
    """
    Compute frequency (count) of each relation in triples.

    Args:
        triples: (N, 3) tensor [head, relation, tail]
        num_relations: Total number of relations

    Returns:
        (num_relations,) tensor of frequencies
    """
    frequencies = torch.zeros(num_relations, dtype=torch.long)
    for r in triples[:, 1]:
        frequencies[r] += 1
    return frequencies


def build_adjacency_from_triples(
    triples: torch.Tensor,
    num_entities: int,
    num_relations: int,
    provenance_weights: Optional[torch.Tensor] = None,
) -> dict:
    """
    Build a simple adjacency structure from triples.

    Returns a dict with:
        edge_index: (2, E) tensor [src, dst]
        edge_type:  (E,) tensor of relation IDs
        edge_prov:  (E,) tensor of provenance weights (if provided)
    """
    src = triples[:, 0]
    rel = triples[:, 1]
    dst = triples[:, 2]

    adj = {
        "edge_index": torch.stack([src, dst], dim=0),
        "edge_type": rel,
        "num_entities": num_entities,
        "num_relations": num_relations,
    }
    if provenance_weights is not None:
        adj["edge_prov"] = provenance_weights
    else:
        adj["edge_prov"] = torch.ones(len(triples))

    return adj
