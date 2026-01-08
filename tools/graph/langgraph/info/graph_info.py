"""
GraphInfo dataclass for graph statistics.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class GraphInfo:
    """Graph statistics information."""
    total_triples: int
    total_entities: int
    graph_nodes: int
    graph_edges: int
    triples_changed: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "total_triples": self.total_triples,
            "total_entities": self.total_entities,
            "graph_nodes": self.graph_nodes,
            "graph_edges": self.graph_edges,
            "triples_changed": self.triples_changed,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "GraphInfo":
        """Create from dictionary."""
        return cls(
            total_triples=data.get("total_triples", 0),
            total_entities=data.get("total_entities", 0),
            graph_nodes=data.get("graph_nodes", 0),
            graph_edges=data.get("graph_edges", 0),
            triples_changed=data.get("triples_changed", 0),
        )

