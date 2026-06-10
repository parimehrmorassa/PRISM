"""
Counterfactual Sensitivity Analyzer for Knowledge Graph predictions.

Method: Structural Edge Ablation
  - Identify top-k critical edges by aggregated attention weight
  - For each edge: remove it → re-run NBFNet → measure score drop
  - Delta_CF = max(delta_e for e in critical_edges)
  - Also compute random baseline for statistical significance testing

Formula:
    delta_e = clip(1 - (intervened_score / baseline_score), 0, 1)
    Delta_CF = max(delta_e for e in top-k critical edges)
"""

import sys
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from config import COUNTERFACTUAL_CONFIG, NBFNET_CONFIG
from src.utils.gpu_utils import autocast_ctx, get_amp_dtype

logger = logging.getLogger(__name__)


class CounterfactualAnalyzer:
    """
    Counterfactual sensitivity analysis via structural edge ablation.

    For each query (h, r, ?):
      1. Run NBFNet to get baseline score and attention weights
      2. Identify top-k critical edges by aggregated attention
      3. For each critical edge: remove it and re-run → delta_e
      4. Delta_CF = max(delta_e)  ∈ [0, 1]
      5. Also compute random baseline (same k, random edges)

    Args:
        model: Trained NBFNet model
        device: Torch device for computation
        top_k: Number of critical edges to identify (from COUNTERFACTUAL_CONFIG)
    """

    def __init__(self, model, device: torch.device, top_k: int = None):
        self.model = model
        self.device = device
        self.top_k = top_k or COUNTERFACTUAL_CONFIG["top_k_critical"]
        self.use_amp = NBFNET_CONFIG["use_amp"] and device.type == "cuda"
        self.amp_dtype = get_amp_dtype(NBFNET_CONFIG["amp_dtype"])
        random.seed(COUNTERFACTUAL_CONFIG["random_seed"])

    def identify_critical_edges(
        self,
        subgraph: Dict,
        query_relation: int,
        top_k: int = None,
    ) -> Tuple[List[Dict], float]:
        """
        Run NBFNet on a subgraph and identify top-k critical edges by attention.

        Args:
            subgraph: Dict from trainer._extract_subgraph()
            query_relation: Query relation ID
            top_k: Number of critical edges to return

        Returns:
            (critical_edges, baseline_score):
                critical_edges: List of dicts {head, relation, tail, attention, edge_idx}
                baseline_score: Score for the true tail entity
        """
        top_k = top_k or self.top_k
        edge_index = subgraph["edge_index"].to(self.device)
        edge_type = subgraph["edge_type"].to(self.device)
        edge_prov = subgraph["edge_prov"].to(self.device)
        local_head = subgraph["local_head"]
        local_tail = subgraph["local_tail"]
        num_nodes = subgraph["num_nodes"]

        self.model.eval()
        with torch.no_grad():
            with autocast_ctx(self.device, self.use_amp, self.amp_dtype):
                scores, attn_weights = self.model(
                    edge_index=edge_index,
                    edge_type=edge_type,
                    edge_prov=edge_prov,
                    query_head=local_head,
                    query_relation=query_relation,
                    num_nodes=num_nodes,
                )

        baseline_score = scores[local_tail].item()

        # Aggregate attention across all layers
        stacked = torch.stack(attn_weights, dim=0)   # (num_layers, E)
        agg_attn = stacked.mean(dim=0)                # (E,)

        E = edge_index.shape[1]
        k = min(top_k, E)
        top_vals, top_idx = agg_attn.topk(k)

        src = edge_index[0]
        dst = edge_index[1]

        critical_edges = []
        for rank_i, edge_i in enumerate(top_idx.tolist()):
            critical_edges.append({
                "head": src[edge_i].item(),
                "relation": edge_type[edge_i].item(),
                "tail": dst[edge_i].item(),
                "attention": top_vals[rank_i].item(),
                "edge_idx": edge_i,
            })

        return critical_edges, baseline_score

    def remove_edge_from_subgraph(self, subgraph: Dict, edge_idx: int) -> Dict:
        """
        Return a new subgraph with the given edge removed.

        Args:
            subgraph: Original subgraph dict
            edge_idx: Index of the edge to remove

        Returns:
            Modified subgraph dict (does not modify original)
        """
        E = subgraph["edge_index"].shape[1]
        keep_mask = torch.ones(E, dtype=torch.bool)
        keep_mask[edge_idx] = False

        modified = {
            "edge_index": subgraph["edge_index"][:, keep_mask],
            "edge_type":  subgraph["edge_type"][keep_mask],
            "edge_prov":  subgraph["edge_prov"][keep_mask],
            "local_head": subgraph["local_head"],
            "local_tail": subgraph["local_tail"],
            "num_nodes":  subgraph["num_nodes"],
            "global_to_local": subgraph.get("global_to_local", {}),
        }
        return modified

    def compute_sensitivity(
        self,
        subgraph: Dict,
        query_relation: int,
        critical_edges: List[Dict],
        baseline_score: float,
    ) -> Tuple[float, Dict]:
        """
        Compute Delta_CF by ablating each critical edge.

        Also computes random baseline (same number of random edges).

        Args:
            subgraph: The query subgraph
            query_relation: Query relation ID
            critical_edges: Top-k critical edges from identify_critical_edges()
            baseline_score: Baseline score for the true tail

        Returns:
            (Delta_CF, edge_impacts):
                Delta_CF: max delta_e over critical edges ∈ [0, 1]
                edge_impacts: dict with 'critical' and 'random' impact lists
        """
        self.model.eval()
        E = subgraph["edge_index"].shape[1]
        local_tail = subgraph["local_tail"]

        critical_impacts = []
        for edge_dict in critical_edges:
            edge_idx = edge_dict["edge_idx"]
            if edge_idx >= E:
                continue

            modified = self.remove_edge_from_subgraph(subgraph, edge_idx)
            intervened_score = self._score_tail(modified, query_relation, local_tail)

            if baseline_score > 1e-8:
                delta_e = 1.0 - (intervened_score / baseline_score)
            else:
                delta_e = 0.0
            delta_e = float(np.clip(delta_e, 0.0, 1.0))
            critical_impacts.append(delta_e)

        # Random baseline: pick same number of random edges
        k = len(critical_edges)
        critical_indices = {e["edge_idx"] for e in critical_edges}
        candidate_random = [i for i in range(E) if i not in critical_indices]
        random_k = min(k, len(candidate_random))
        random_indices = random.sample(candidate_random, random_k) if candidate_random else []

        random_impacts = []
        for edge_idx in random_indices:
            modified = self.remove_edge_from_subgraph(subgraph, edge_idx)
            intervened_score = self._score_tail(modified, query_relation, local_tail)
            if baseline_score > 1e-8:
                delta_e = 1.0 - (intervened_score / baseline_score)
            else:
                delta_e = 0.0
            delta_e = float(np.clip(delta_e, 0.0, 1.0))
            random_impacts.append(delta_e)

        Delta_CF = max(critical_impacts) if critical_impacts else 0.0
        Delta_CF = float(np.clip(Delta_CF, 0.0, 1.0))

        assert 0.0 <= Delta_CF <= 1.0, f"Delta_CF out of range: {Delta_CF}"

        edge_impacts = {
            "critical": critical_impacts,
            "random": random_impacts,
        }
        return Delta_CF, edge_impacts

    def _score_tail(
        self,
        subgraph: Dict,
        query_relation: int,
        local_tail: int,
    ) -> float:
        """Run model on modified subgraph and return score for the tail node."""
        edge_index = subgraph["edge_index"].to(self.device)
        edge_type = subgraph["edge_type"].to(self.device)
        edge_prov = subgraph["edge_prov"].to(self.device)
        num_nodes = subgraph["num_nodes"]

        # If all edges removed, model gets boundary condition only
        if edge_index.shape[1] == 0:
            # Minimal graph: just head and tail nodes, no edges
            edge_index = torch.zeros(2, 0, dtype=torch.long, device=self.device)
            edge_type = torch.zeros(0, dtype=torch.long, device=self.device)
            edge_prov = torch.ones(0, dtype=torch.float, device=self.device)

        with torch.no_grad():
            with autocast_ctx(self.device, self.use_amp, self.amp_dtype):
                scores, _ = self.model(
                    edge_index=edge_index,
                    edge_type=edge_type,
                    edge_prov=edge_prov,
                    query_head=subgraph["local_head"],
                    query_relation=query_relation,
                    num_nodes=num_nodes,
                )

        tail_idx = min(local_tail, scores.shape[0] - 1)
        return scores[tail_idx].item()

    def compute_sensitivity_batch(
        self,
        queries: List[Dict],
        trainer,
        top_k: int = None,
    ) -> List[Dict]:
        """
        Compute Delta_CF for a batch of queries with progress bar.

        Args:
            queries: List of dicts with keys: head, relation, tail, query_idx
            trainer: NBFNetTrainer (for subgraph extraction)
            top_k: Number of critical edges (default from config)

        Returns:
            List of result dicts per query:
                {query_idx, Delta_CF, critical_impacts, random_impacts,
                 baseline_score, critical_edges}
        """
        top_k = top_k or self.top_k
        results = []

        for query in tqdm(queries, desc="Computing counterfactuals"):
            h = query["head"]
            r = query["relation"]
            t = query["tail"]
            q_idx = query.get("query_idx", 0)

            subgraph = trainer._extract_subgraph(h, r, t)
            if subgraph is None:
                results.append({
                    "query_idx": q_idx,
                    "Delta_CF": 0.0,
                    "critical_impacts": [],
                    "random_impacts": [],
                    "baseline_score": 0.0,
                    "critical_edges": [],
                })
                continue

            critical_edges, baseline_score = self.identify_critical_edges(
                subgraph, r, top_k=top_k
            )
            Delta_CF, edge_impacts = self.compute_sensitivity(
                subgraph, r, critical_edges, baseline_score
            )

            results.append({
                "query_idx": q_idx,
                "Delta_CF": Delta_CF,
                "critical_impacts": edge_impacts["critical"],
                "random_impacts": edge_impacts["random"],
                "baseline_score": baseline_score,
                "critical_edges": critical_edges,
            })

        return results
