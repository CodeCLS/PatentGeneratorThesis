"""
Question dataclass for graph validation questions.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


@dataclass
class Question:
    """A validation question about the graph."""
    id: str
    text: str
    category: str = "unclear"
    priority: int = 5
    answered: bool = False
    num_responses: int = 0,
    entities_contained: List[str] = field(default_factory=list),
    context: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Question":
        """Create a Question from a dictionary."""
        return cls(
            id=data.get("id", ""),
            text=data.get("text", ""),
            category=data.get("category", "unclear"),
            priority=data.get("priority", 5),
            answered=data.get("answered", False),
            num_responses=data.get("num_responses", 0),
            entities_contained=data.get("entities_contained", []),
            context=data.get("context", {}),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Question to dictionary."""
        return {
            "id": self.id,
            "text": self.text,
            "category": self.category,
            "priority": self.priority,
            "answered": self.answered,
            "num_responses": self.num_responses,
            "entities_contained": self.entities_contained,
            "context": self.context,
        }
 
