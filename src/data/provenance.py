"""
ProvenanceMetadata dataclass for tracking data source quality.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict


@dataclass
class ProvenanceMetadata:
    """
    Metadata about the provenance (source quality) of a knowledge graph triple.

    Attributes:
        source_name:   Name of the data source (e.g., DRUGBANK, PUBMED).
        source_weight: Quality score in [0, 1] from PROVENANCE_CONFIG source_weights.
        edge_index:    Index of the edge in the dataset.
        head_id:       Integer ID of the head entity.
        relation_id:   Integer ID of the relation.
        tail_id:       Integer ID of the tail entity.
        provenance_score: Final provenance score in [0, 1] after aggregation.
        supporting_sources: Additional sources that confirm this edge.
        metadata:      Extra metadata (year, DOI, confidence, etc.).
    """
    source_name: str
    source_weight: float
    edge_index: int
    head_id: int
    relation_id: int
    tail_id: int
    provenance_score: float = 0.0
    supporting_sources: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        assert 0.0 <= self.source_weight <= 1.0, (
            f"source_weight must be in [0,1], got {self.source_weight}"
        )
        if self.provenance_score == 0.0:
            self.provenance_score = self.source_weight

    @property
    def s_prov(self) -> float:
        """Alias for provenance_score (used in trust formula)."""
        return self.provenance_score

    def to_dict(self) -> Dict:
        return {
            "source_name": self.source_name,
            "source_weight": self.source_weight,
            "edge_index": self.edge_index,
            "head_id": self.head_id,
            "relation_id": self.relation_id,
            "tail_id": self.tail_id,
            "provenance_score": self.provenance_score,
            "supporting_sources": self.supporting_sources,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "ProvenanceMetadata":
        return cls(
            source_name=d["source_name"],
            source_weight=d["source_weight"],
            edge_index=d["edge_index"],
            head_id=d["head_id"],
            relation_id=d["relation_id"],
            tail_id=d["tail_id"],
            provenance_score=d.get("provenance_score", d["source_weight"]),
            supporting_sources=d.get("supporting_sources", []),
            metadata=d.get("metadata", {}),
        )
