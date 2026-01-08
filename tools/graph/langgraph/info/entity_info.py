"""
EntityInfo dataclass for entity information.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class ConnectedEntity:
    """Information about a connected entity."""
    name: str
    relation: str
    direction: str  # "incoming" or "outgoing"


@dataclass
class EntityInfo:
    """Information about an entity."""
    name: str
    id: str
    connections: int = 0
    properties: Dict[str, Any] = field(default_factory=dict)
    connected_entities: List[ConnectedEntity] = field(default_factory=list)
    triples: List["TripleInfo"] = field(default_factory=list)
    label: Optional[str] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = {
            "name": self.name,
            "id": self.id,
            "connections": self.connections,
            "properties": self.properties,
            "connected_entities": [ce.__dict__ for ce in self.connected_entities],
            "triples": [t.to_dict() if hasattr(t, 'to_dict') else t for t in self.triples],
        }
        if self.label:
            result["label"] = self.label
        if self.error:
            result["error"] = self.error
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> "EntityInfo":
        """Create from dictionary."""
        from tools.graph.langgraph.info.triple_info import TripleInfo
        
        connected_entities = [
            ConnectedEntity(**ce) if isinstance(ce, dict) else ce
            for ce in data.get("connected_entities", [])
        ]
        
        triples = [
            TripleInfo.from_dict(t) if isinstance(t, dict) else t
            for t in data.get("triples", [])
        ]
        
        return cls(
            name=data.get("name", ""),
            id=data.get("id", ""),
            connections=data.get("connections", 0),
            properties=data.get("properties", {}),
            connected_entities=connected_entities,
            triples=triples,
            label=data.get("label"),
            error=data.get("error"),
        )

