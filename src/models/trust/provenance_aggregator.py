"""
Provenance quality score aggregation for reasoning paths.

Strategy: Minimum provenance (weakest-link principle)
  - A reasoning chain is only as reliable as its weakest evidence link
  - Path S_prov = min(s_prov for each edge in path)
  - Query S_prov = attention-weighted average of path scores

This is auditable and interpretable — unlike learned aggregation.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from config import PROVENANCE_CONFIG

logger = logging.getLogger(__name__)


def aggregate_path(reasoning_path: List[Dict]) -> float:
    """
    Compute provenance score for a single reasoning path.

    Strategy: minimum provenance (weakest-link principle).
    A reasoning chain is only as reliable as its weakest evidence link.

    Args:
        reasoning_path: List of edge dicts, each must have 's_prov' key.
            Example edge: {'head': ..., 'relation': ..., 'tail': ..., 's_prov': 0.85}

    Returns:
        Path provenance score ∈ [0, 1].
        Returns 1.0 for empty paths (no evidence to contradict).
    """
    if not reasoning_path:
        return 1.0

    strategy = PROVENANCE_CONFIG.get("aggregation", "minimum")

    if strategy == "minimum":
        # Weakest-link: minimum over all edges in path
        scores = [edge.get("s_prov", 1.0) for edge in reasoning_path]
        result = float(min(scores))
    elif strategy == "mean":
        scores = [edge.get("s_prov", 1.0) for edge in reasoning_path]
        result = float(np.mean(scores))
    elif strategy == "product":
        scores = [edge.get("s_prov", 1.0) for edge in reasoning_path]
        result = float(np.prod(scores))
    else:
        raise ValueError(f"Unknown aggregation strategy: {strategy}")

    return float(np.clip(result, 0.0, 1.0))


def aggregate_query(
    reasoning_paths: List[Dict],
    attention_weights: Optional[List[float]] = None,
) -> Tuple[float, List[float]]:
    """
    Compute query-level provenance score as attention-weighted average of path scores.

    Args:
        reasoning_paths: List of reasoning path dicts.
            Each path dict may have:
              - 'edges': List of edge dicts with 's_prov' key
              - OR 's_prov': Pre-computed path provenance score
              - 'attention': Attention weight for this path (used if attention_weights=None)
        attention_weights: Optional explicit attention weights (same length as paths).
            If None, uses 'attention' key from path dicts.

    Returns:
        (S_prov, path_scores):
            S_prov: Attention-weighted average provenance score ∈ [0, 1]
            path_scores: Per-path provenance scores
    """
    if not reasoning_paths:
        return 1.0, []

    # Compute individual path scores
    path_scores = []
    for path in reasoning_paths:
        if "s_prov" in path and not isinstance(path.get("edges"), list):
            # Pre-computed path score
            path_scores.append(float(path["s_prov"]))
        elif "edges" in path:
            # Aggregate from constituent edges
            path_scores.append(aggregate_path(path["edges"]))
        else:
            # Treat path dict as a single edge
            path_scores.append(float(path.get("s_prov", 1.0)))

    # Get attention weights
    if attention_weights is not None:
        weights = list(attention_weights)
    else:
        weights = [float(p.get("attention", 1.0)) for p in reasoning_paths]

    # Normalize weights to sum to 1
    weights_arr = np.array(weights, dtype=np.float64)
    w_sum = weights_arr.sum()
    if w_sum > 1e-8:
        weights_arr = weights_arr / w_sum
    else:
        # Uniform weights
        weights_arr = np.ones(len(weights)) / len(weights)

    # Weighted average
    S_prov = float(np.dot(weights_arr, np.array(path_scores, dtype=np.float64)))
    S_prov = float(np.clip(S_prov, 0.0, 1.0))

    return S_prov, path_scores


def aggregate_edge_provenance(
    edge_prov_tensor,
    edge_index,
    head: int,
    tail: int,
    attention_weights=None,
) -> float:
    """
    Aggregate provenance from raw edge tensors for a query (head → tail).

    Identifies all edges connecting head to tail (direct or multi-hop)
    and aggregates their provenance scores.

    Args:
        edge_prov_tensor: (E,) tensor of per-edge provenance scores
        edge_index: (2, E) tensor of [src, dst] node indices
        head: Head entity node index
        tail: Tail entity node index
        attention_weights: Optional (E,) tensor of attention weights

    Returns:
        Aggregated provenance score ∈ [0, 1]
    """
    import torch
    src = edge_index[0]
    dst = edge_index[1]

    # Find direct edges head → tail
    direct_mask = (src == head) & (dst == tail)

    if direct_mask.any():
        direct_prov = edge_prov_tensor[direct_mask]
        if attention_weights is not None:
            direct_attn = attention_weights[direct_mask]
            w_sum = direct_attn.sum()
            if w_sum > 1e-8:
                S_prov = (direct_attn * direct_prov).sum() / w_sum
            else:
                S_prov = direct_prov.mean()
        else:
            S_prov = direct_prov.min()  # weakest-link
        return float(S_prov.clamp(0.0, 1.0).item())

    # No direct edges — use global minimum over all edges
    if len(edge_prov_tensor) > 0:
        return float(edge_prov_tensor.min().item())
    return 1.0


def compute_path_provenance_from_attention(
    edge_index,
    edge_prov,
    all_attention_weights: List,
    top_k: int = 5,
) -> Tuple[float, List[float]]:
    """
    Compute query-level S_prov from model attention weights.

    1. Aggregate attention across layers (mean)
    2. Get top-k edges by attention
    3. Apply weakest-link over those edges

    Args:
        edge_index: (2, E) edge index
        edge_prov: (E,) provenance weights
        all_attention_weights: List of (E,) attention tensors per layer
        top_k: Number of top edges to consider

    Returns:
        (S_prov, top_edge_prov_scores)
    """
    import torch

    if not all_attention_weights:
        return 1.0, []

    stacked = torch.stack(all_attention_weights, dim=0)  # (L, E)
    agg_attn = stacked.mean(dim=0)  # (E,)

    k = min(top_k, agg_attn.shape[0])
    top_vals, top_idx = agg_attn.topk(k)

    top_prov = edge_prov[top_idx]     # (k,)
    top_attn = top_vals              # (k,)

    # Attention-weighted average
    w = top_attn / (top_attn.sum() + 1e-8)
    S_prov = float((w * top_prov).sum().clamp(0.0, 1.0).item())
    path_prov_scores = top_prov.tolist()

    return S_prov, path_prov_scores
