from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

@dataclass
class ChatContextInformation:
    intent: Optional[str] = None
    entities_in_focus: List[str] = field(default_factory=list)
    relevant_triples: List[int] = field(default_factory=list)
    additional_context: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatContextInformation':
        return cls(
            intent=data.get("intent"),
            entities_in_focus=data.get("entities_in_focus", []),
            relevant_triples=data.get("relevant_triples", []),
            additional_context=data.get("additional_context", {})
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "entities_in_focus": self.entities_in_focus,
            "relevant_triples": self.relevant_triples,
            "additional_context": self.additional_context
        }
