"""
Provenance-Aware Message Passing Layer for NBFNet.

Uses PyTorch Geometric (PyG) for graph operations.
Message: m_{u→v} = S_prov(u,r',v) · φ(r',r_query) · h_u
Aggregation: scatter_add over destination nodes
"""

import sys
import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from config import NBFNET_CONFIG

logger = logging.getLogger(__name__)


class ProvenanceAwareMessagePassingLayer(nn.Module):
    """
    Single message-passing layer for NBFNet with provenance weighting.

    Message from node u to node v via relation r':
        m_{u→v} = S_prov(u,r',v) · φ(r',r_query) · h_u

    where:
        S_prov(u,r',v)  - provenance weight on edge (u,r',v)
        φ(r',r_query)   - relation compatibility vector (bilinear)
        h_u             - hidden state of source node

    Aggregation: h_v^new = scatter_add(m_{u→v}) + self_loop(h_v)

    Args:
        hidden_dim: Dimension of node hidden states and relation embeddings.
        num_relations: Total number of relations.
        dropout: Dropout rate applied to messages.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_relations: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self.dropout = dropout

        # Relation compatibility: φ(r', r_query) = W_r' applied to boundary condition
        # We learn a (num_relations, hidden_dim, hidden_dim) bilinear map
        # For efficiency: use two separate linear projections
        self.message_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.relation_transform = nn.Embedding(num_relations, hidden_dim * hidden_dim)

        # Self-loop transformation
        self.self_loop_linear = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.drop = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.message_linear.weight)
        nn.init.xavier_uniform_(self.self_loop_linear.weight)
        nn.init.zeros_(self.self_loop_linear.bias)
        # Initialize relation transforms near identity
        with torch.no_grad():
            eye = torch.eye(self.hidden_dim).flatten()
            self.relation_transform.weight.data.copy_(
                eye.unsqueeze(0).expand(self.num_relations, -1) * 0.1
            )

    def forward(
        self,
        edge_index: torch.Tensor,    # (2, E) — [src, dst]
        edge_type: torch.Tensor,     # (E,) — relation IDs
        edge_prov: torch.Tensor,     # (E,) — provenance weights in [0,1]
        node_features: torch.Tensor, # (N, D) — hidden states
        query_relation: int,         # scalar relation ID
        num_nodes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of one message-passing layer.

        Args:
            edge_index: (2, E) — source and destination node indices
            edge_type:  (E,)   — relation ID for each edge
            edge_prov:  (E,)   — provenance weight per edge, in [0, 1]
            node_features: (N, D) — current node hidden states
            query_relation: Integer ID of the query relation
            num_nodes: Total number of nodes

        Returns:
            (h_new, attention_weights):
                h_new:            (N, D) updated node features
                attention_weights: (E,) per-edge attention values
        """
        src, dst = edge_index[0], edge_index[1]
        E = src.shape[0]
        D = self.hidden_dim
        device = node_features.device

        # ── Compute relation compatibility φ(r', r_query) ──────────────────────
        # Get relation transform matrices: (E, D, D)
        rel_matrices = self.relation_transform(edge_type)  # (E, D*D)
        rel_matrices = rel_matrices.view(E, D, D)

        # Apply relation transform to source features
        h_src = node_features[src]  # (E, D)
        # Transform: (E, D, D) x (E, D, 1) → (E, D)
        phi = torch.bmm(rel_matrices, h_src.unsqueeze(-1)).squeeze(-1)  # (E, D)
        phi = self.message_linear(phi)  # (E, D)

        # ── Provenance weighting ────────────────────────────────────────────────
        # S_prov: (E,) → broadcast to (E, D)
        prov_weighted = edge_prov.unsqueeze(-1) * phi  # (E, D)

        # ── Attention weights (for reasoning path extraction) ───────────────────
        # attention = S_prov * ||phi||
        attention_weights = edge_prov * phi.norm(dim=-1)  # (E,)

        # ── Apply dropout ───────────────────────────────────────────────────────
        messages = self.drop(prov_weighted)  # (E, D)

        # ── Scatter-add aggregation ─────────────────────────────────────────────
        h_new = torch.zeros(num_nodes, D, device=device, dtype=node_features.dtype)
        h_new.scatter_add_(0, dst.unsqueeze(-1).expand(-1, D), messages)

        # ── Self-loop ───────────────────────────────────────────────────────────
        h_new = h_new + self.self_loop_linear(node_features)

        # ── Layer norm + activation ─────────────────────────────────────────────
        h_new = self.layer_norm(h_new)
        h_new = F.relu(h_new)

        return h_new, attention_weights
