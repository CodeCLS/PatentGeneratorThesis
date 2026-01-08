"""
TripleInfo dataclass for triple information.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any


@dataclass
class TripleInfo:
    """Information about a triple."""
    index: int
    head: str
    relation: str
    tail: str
    head_id: Optional[str] = None
    tail_id: Optional[str] = None
    head_properties: Dict[str, Any] = field(default_factory=dict)
    tail_properties: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = {
            "index": self.index,
            "head": self.head,
            "relation": self.relation,
            "tail": self.tail,
        }
        if self.head_id:
            result["head_id"] = self.head_id
        if self.tail_id:
            result["tail_id"] = self.tail_id
        if self.head_properties:
            result["head_properties"] = self.head_properties
        if self.tail_properties:
            result["tail_properties"] = self.tail_properties
        if self.error:
            result["error"] = self.error
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> "TripleInfo":
        """Create from dictionary."""
        return cls(
            index=data.get("index", -1),
            head=data.get("head", ""),
            relation=data.get("relation", ""),
            tail=data.get("tail", ""),
            head_id=data.get("head_id"),
            tail_id=data.get("tail_id"),
            head_properties=data.get("head_properties", {}),
            tail_properties=data.get("tail_properties", {}),
            error=data.get("error"),
        )

