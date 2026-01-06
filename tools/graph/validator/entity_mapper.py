"""
Entity Mapper: Maps between entity names and IDs.
"""
from typing import Optional, Dict, List
from tools.graph.Triple import Triple


class EntityMapper:
    """Maps between entity names and IDs."""
    
    def __init__(
        self,
        id_to_name: Dict[str, str],
        triples: List[Triple],
    ):
        self.id_to_name = id_to_name
        self.triples = triples
    
    def name_to_id(self, entity_name: str) -> Optional[str]:
        """
        Convert entity name to ID using id_to_name mapping (reverse lookup).
        
        Args:
            entity_name: Human-readable entity name
            
        Returns:
            Entity ID if found, or None
        """
        # Reverse lookup: find ID by name
        for entity_id, name in self.id_to_name.items():
            if name == entity_name:
                return entity_id
        
        # Also check in triples
        for triple in self.triples:
            head_name = getattr(triple.head, "name", None) or getattr(triple.head, "text", None) or str(triple.head)
            tail_name = getattr(triple.tail, "name", None) or getattr(triple.tail, "text", None) or str(triple.tail)
            
            if head_name == entity_name:
                return getattr(triple.head, "ref", None) or getattr(triple.head, "id", None) or getattr(triple.head, "ref_short", None)
            if tail_name == entity_name:
                return getattr(triple.tail, "ref", None) or getattr(triple.tail, "id", None) or getattr(triple.tail, "ref_short", None)
        
        return None
    
    def id_to_name(self, entity_id: str) -> Optional[str]:
        """Convert entity ID to name."""
        return self.id_to_name.get(entity_id)
    
    def update_mapping(self, entity_id: str, entity_name: str) -> None:
        """Update the ID to name mapping."""
        self.id_to_name[entity_id] = entity_name
    
    def remove_mapping(self, entity_id: str) -> None:
        """Remove an entity from the mapping."""
        self.id_to_name.pop(entity_id, None)

