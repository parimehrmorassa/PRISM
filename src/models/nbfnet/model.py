"""
NBFNet model: Neural Bellman-Ford Network with provenance-aware message passing.

Key design decisions (from config.py / design specs):
1. ONLY relation embeddings — no entity embeddings (maintains inductive property)
2. 6 ProvenanceAwareMessagePassingLayer layers
3. Scoring MLP: Linear → ReLU → Linear → scalar
4. extract_reasoning_paths returns top-k paths by attention
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from config import NBFNET_CONFIG
from src.models.nbfnet.layer import ProvenanceAwareMessagePassingLayer

logger = logging.getLogger(__name__)


class NBFNet(nn.Module):
    """
    Provenance-aware Neural Bellman-Ford Network for KG link prediction.

    Architecture:
        - Relation embeddings only (inductive — no entity embeddings)
        - 6 ProvenanceAwareMessagePassingLayer layers
        - Scoring MLP: Linear(D→D) → ReLU → Linear(D→1)

    The forward pass:
        1. Initialize head node with query relation embedding (boundary condition)
        2. Run 6 message passing layers
        3. Score each candidate entity using the head/tail node features

    Args:
        num_relations: Total number of relation types in the graph.
        hidden_dim: Dimension of node/relation representations.
        num_layers: Number of message passing layers.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        num_relations: int,
        hidden_dim: int = None,
        num_layers: int = None,
        dropout: float = None,
    ):
        super().__init__()
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim or NBFNET_CONFIG["hidden_dim"]
        self.num_layers = num_layers or NBFNET_CONFIG["num_layers"]
        self.dropout = dropout if dropout is not None else NBFNET_CONFIG["dropout"]

        # Relation embeddings (NO entity embeddings — inductive!)
        self.relation_embeddings = nn.Embedding(
            num_relations, self.hidden_dim
        )

        # Message passing layers
        self.layers = nn.ModuleList([
            ProvenanceAwareMessagePassingLayer(
                hidden_dim=self.hidden_dim,
                num_relations=num_relations,
                dropout=self.dropout,
            )
            for _ in range(self.num_layers)
        ])

        # Scoring MLP: D → D → 1
        self.score_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )

        self._reset_parameters()
        logger.info(
            f"NBFNet initialized: {num_relations} relations, "
            f"hidden_dim={self.hidden_dim}, num_layers={self.num_layers}"
        )

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
        for layer_module in self.score_mlp:
            if isinstance(layer_module, nn.Linear):
                nn.init.xavier_uniform_(layer_module.weight)
                nn.init.zeros_(layer_module.bias)

    def _initialize_node_features(
        self,
        query_head: int,
        query_relation: int,
        num_nodes: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Initialize node features using the boundary condition:
          - Head node gets the query relation embedding
          - All other nodes get zero vector

        Args:
            query_head: Index of the head entity
            query_relation: Index of the query relation
            num_nodes: Total number of nodes in the subgraph
            device: Target device

        Returns:
            (num_nodes, hidden_dim) tensor
        """
        h = torch.zeros(num_nodes, self.hidden_dim, device=device)
        # Boundary condition: head node initialized with relation embedding
        with torch.no_grad():
            query_emb = self.relation_embeddings(
                torch.tensor(query_relation, device=device)
            )
        h[query_head] = query_emb
        return h

    def forward(
        self,
        edge_index: torch.Tensor,    # (2, E) graph edges
        edge_type: torch.Tensor,     # (E,) relation IDs
        edge_prov: torch.Tensor,     # (E,) provenance weights
        query_head: int,             # head entity index in subgraph
        query_relation: int,         # query relation ID
        num_nodes: int,              # total nodes in subgraph
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Run NBFNet forward pass on a subgraph.

        Args:
            edge_index:     (2, E) edge connectivity
            edge_type:      (E,) relation IDs
            edge_prov:      (E,) provenance weights in [0, 1]
            query_head:     Index of the head entity node
            query_relation: Query relation ID
            num_nodes:      Number of nodes in the subgraph

        Returns:
            (scores, all_attention_weights):
                scores:               (num_nodes,) — link prediction score per node
                all_attention_weights: List of (E,) tensors, one per layer
        """
        device = edge_index.device

        # Initialize node features
        h = self._initialize_node_features(query_head, query_relation, num_nodes, device)

        # Ensure edge_prov and edge_type are on same device
        edge_prov = edge_prov.to(device)
        edge_type = edge_type.to(device)

        all_attention_weights = []

        # Run message passing layers
        for layer in self.layers:
            h, attn = layer(
                edge_index=edge_index,
                edge_type=edge_type,
                edge_prov=edge_prov,
                node_features=h,
                query_relation=query_relation,
                num_nodes=num_nodes,
            )
            all_attention_weights.append(attn)

        # Score each node (potential tail entity)
        scores = self.score_mlp(h).squeeze(-1)  # (num_nodes,)
        scores = torch.sigmoid(scores)           # normalize to [0, 1]

        return scores, all_attention_weights

    def forward_batched(
        self,
        edge_index: torch.Tensor,    # (2, total_E) — block-diagonal combined graph
        edge_type: torch.Tensor,     # (total_E,)
        edge_prov: torch.Tensor,     # (total_E,)
        query_heads: List[int],      # K offsets into the combined node space
        query_relations: List[int],  # K relation IDs (one per query)
        total_nodes: int,
    ) -> torch.Tensor:               # (total_nodes,) sigmoid scores
        """
        Run NBFNet on K independent subgraphs packed into one block-diagonal graph.

        Because the sub-graphs are disconnected, message passing within block i
        is identical to running a separate forward pass for query i — but we pay
        the Python / CUDA-launch overhead only once per call instead of K times.

        Args:
            edge_index:      (2, total_E) — edge indices already offset per block
            edge_type:       (total_E,)
            edge_prov:       (total_E,)
            query_heads:     List of K head-node indices (already offset)
            query_relations: List of K query relation IDs
            total_nodes:     Sum of num_nodes across all K subgraphs

        Returns:
            scores: (total_nodes,) — link-prediction score per node
        """
        device = edge_index.device

        # Boundary condition: each query head gets its relation embedding; rest = 0
        h = torch.zeros(total_nodes, self.hidden_dim, device=device)
        if query_heads:
            with torch.no_grad():
                rel_t = torch.tensor(query_relations, device=device)
                rel_embs = self.relation_embeddings(rel_t)   # (K, D)
            head_t = torch.tensor(query_heads, device=device)
            h[head_t] = rel_embs

        edge_prov = edge_prov.to(device)
        edge_type = edge_type.to(device)

        for layer in self.layers:
            # query_relation arg exists in the signature but is not used by the
            # layer — it only uses edge_type for relation transforms.
            h, _ = layer(
                edge_index=edge_index,
                edge_type=edge_type,
                edge_prov=edge_prov,
                node_features=h,
                query_relation=0,   # unused placeholder
                num_nodes=total_nodes,
            )

        scores = self.score_mlp(h).squeeze(-1)   # (total_nodes,)
        return torch.sigmoid(scores)

    def extract_reasoning_paths(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_prov: torch.Tensor,
        all_attention_weights: List[torch.Tensor],
        query_head: int,
        query_tail: int,
        id2entity: Optional[Dict] = None,
        id2relation: Optional[Dict] = None,
        top_k: int = None,
    ) -> List[Dict]:
        """
        Extract top-k reasoning paths from attention weights.

        Aggregates attention across all layers (mean) and returns
        the top-k edges by aggregated attention weight.

        Args:
            edge_index:           (2, E) edge connectivity
            edge_type:            (E,) relation IDs
            edge_prov:            (E,) provenance weights
            all_attention_weights: List of (E,) attention tensors per layer
            query_head:           Head entity index
            query_tail:           Tail entity index (predicted)
            id2entity:            Optional mapping from ID to entity name
            id2relation:          Optional mapping from ID to relation name
            top_k:                Number of paths to return

        Returns:
            List of dicts with keys:
                head, relation, tail, attention, s_prov, edge_idx, layer_attns
        """
        top_k = top_k or NBFNET_CONFIG["top_k_paths"]

        # Aggregate attention across layers
        stacked = torch.stack(all_attention_weights, dim=0)  # (num_layers, E)
        agg_attention = stacked.mean(dim=0)                  # (E,)

        # Get top-k edges
        k = min(top_k, agg_attention.shape[0])
        top_vals, top_idx = agg_attention.topk(k)

        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]

        paths = []
        for i, edge_i in enumerate(top_idx.tolist()):
            h_node = src_nodes[edge_i].item()
            t_node = dst_nodes[edge_i].item()
            rel_id = edge_type[edge_i].item()
            prov_w = edge_prov[edge_i].item()
            attn_val = top_vals[i].item()

            path = {
                "head": id2entity.get(h_node, str(h_node)) if id2entity else h_node,
                "relation": id2relation.get(rel_id, str(rel_id)) if id2relation else rel_id,
                "tail": id2entity.get(t_node, str(t_node)) if id2entity else t_node,
                "head_id": h_node,
                "tail_id": t_node,
                "relation_id": rel_id,
                "attention": attn_val,
                "s_prov": prov_w,
                "edge_idx": edge_i,
                "layer_attns": [aw[edge_i].item() for aw in all_attention_weights],
            }
            paths.append(path)

        return paths

    @classmethod
    def save(cls, model: "NBFNet", path: str):
        """Save model state dict and constructor args."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "num_relations": model.num_relations,
                "hidden_dim": model.hidden_dim,
                "num_layers": model.num_layers,
                "dropout": model.dropout,
            },
            path,
        )
        logger.info(f"NBFNet saved to {path}")

    @classmethod
    def load(cls, path: str, device: torch.device = None) -> "NBFNet":
        """Load model from checkpoint."""
        if device is None:
            device = torch.device("cpu")
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = cls(
            num_relations=ckpt["num_relations"],
            hidden_dim=ckpt["hidden_dim"],
            num_layers=ckpt["num_layers"],
            dropout=ckpt["dropout"],
        )
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        logger.info(f"NBFNet loaded from {path}")
        return model
