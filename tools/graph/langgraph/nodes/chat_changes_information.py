from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

@dataclass
class ChatChangesInformation:
    added_triples: List[Dict[str, Any]] = field(default_factory=list)
    deleted_triples: List[int] = field(default_factory=list)
    merged_entities: List[Dict[str, Any]] = field(default_factory=list)
    renamed_entities: List[Dict[str, Any]] = field(default_factory=list)
    modified_triples: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatChangesInformation':
        return cls(
            added_triples=data.get("added_triples", []),
            deleted_triples=data.get("deleted_triples", []),
            merged_entities=data.get("merged_entities", []),
            renamed_entities=data.get("renamed_entities", []),
            modified_triples=data.get("modified_triples", [])
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "added_triples": self.added_triples,
            "deleted_triples": self.deleted_triples,
            "merged_entities": self.merged_entities,
            "renamed_entities": self.renamed_entities,
            "modified_triples": self.modified_triples
        }
