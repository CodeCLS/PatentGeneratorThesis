from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, Protocol, List, Set, Callable, TYPE_CHECKING
import uuid

from typing import Dict

if TYPE_CHECKING:
    from tools.graph.data.Triple import Triple


class InMemoryEntityRepository:
    def __init__(self, entities: list[Entity] | None = None):
        self._entities: Dict[str, Entity] = {
            (e.ref or e.id or e.ref_short): e for e in (entities or []) if (e.ref or e.id or e.ref_short)
        }
    def getAll(self):
        return self._entities
    def get_by_id(self, entity_id: str) -> Entity:
        """Get entity by ref (or id/ref_short). Note: parameter name kept as entity_id for backward compatibility."""
        try:
            return self._entities[entity_id]
        except KeyError:
            raise KeyError(f"Entity with id {entity_id} not found")
    def save(self, entity: "Entity") -> None:
        entity_key = entity.ref or entity.id or entity.ref_short
        if entity_key:
            self._entities[entity_key] = entity


from dataclasses import dataclass, field, asdict
from typing import Optional
import uuid

@dataclass
class Entity:
    """
    Represents an entity mention in a specific sentence.
    Offsets are relative to the sentence text (0..len(sentence)).
    """
    name: str
    label: str
    ref_short: str

    start: int = 0
    end: int = 0

    entity_type: Optional[str] = None
    id: Optional[str] = None  # Deprecated: use ref instead
    ref: str = field(default_factory=lambda: str(uuid.uuid4()))  # Primary identifier (shorter than old id)
    sentence_id: Optional[str] = None

    def __repr__(self) -> str:
        return f"Entity({self.name}@{self.start}:{self.end})"

    def to_dict(self):
        return asdict(self)


class EnhancedEntityTripleRepository(InMemoryEntityRepository):
    """
    Enhanced repository that extends InMemoryEntityRepository with:
    - Advanced entity search and editing capabilities
    - Triple management (search, edit, delete)
    - Bidirectional indexing for efficient lookups
    """
    
    def __init__(self, entities: List[Entity] | None = None, triples: List["Triple"] | None = None):
        """
        Initialize the repository with entities and triples.
        
        Args:
            entities: List of Entity objects
            triples: List of Triple objects (from tools.graph.Triple)
        """
        super().__init__(entities)
        
        # Import Triple here to avoid circular imports
        
        # Store triples by ID
        self._triples: Dict[str, Triple] = {
            t.id: t for t in (triples or [])
        }
        
        # Index triples by head/tail entity IDs for fast lookups
        self._triples_by_head: Dict[str, Set[str]] = {}  # entity_id -> set of triple_ids
        self._triples_by_tail: Dict[str, Set[str]] = {}  # entity_id -> set of triple_ids
        self._triples_by_relation: Dict[str, Set[str]] = {}  # relation -> set of triple_ids
        
        # Build indices
        for triple in self._triples.values():
            self._index_triple(triple)
    
    def _index_triple(self, triple: "Triple") -> None:
        """Index a triple for fast lookups."""
        head_id = triple.head.ref or triple.head.id or triple.head.ref_short
        tail_id = triple.tail.ref or triple.tail.id or triple.tail.ref_short
        relation = triple.relation
        
        # Index by head
        if head_id not in self._triples_by_head:
            self._triples_by_head[head_id] = set()
        self._triples_by_head[head_id].add(triple.id)
        
        # Index by tail
        if tail_id not in self._triples_by_tail:
            self._triples_by_tail[tail_id] = set()
        self._triples_by_tail[tail_id].add(triple.id)
        
        # Index by relation
        if relation not in self._triples_by_relation:
            self._triples_by_relation[relation] = set()
        self._triples_by_relation[relation].add(triple.id)
    
    def _unindex_triple(self, triple_id: str) -> None:
        """Remove a triple from indices."""
        if triple_id not in self._triples:
            return
        
        triple = self._triples[triple_id]
        head_id = triple.head.ref or triple.head.id or triple.head.ref_short
        tail_id = triple.tail.ref or triple.tail.id or triple.tail.ref_short
        relation = triple.relation
        
        # Remove from head index
        if head_id in self._triples_by_head:
            self._triples_by_head[head_id].discard(triple_id)
            if not self._triples_by_head[head_id]:
                del self._triples_by_head[head_id]
        
        # Remove from tail index
        if tail_id in self._triples_by_tail:
            self._triples_by_tail[tail_id].discard(triple_id)
            if not self._triples_by_tail[tail_id]:
                del self._triples_by_tail[tail_id]
        
        # Remove from relation index
        if relation in self._triples_by_relation:
            self._triples_by_relation[relation].discard(triple_id)
            if not self._triples_by_relation[relation]:
                del self._triples_by_relation[relation]
    
    # ==================== Entity Methods ====================
    
    def search_entities(
        self,
        name: Optional[str] = None,
        label: Optional[str] = None,
        entity_type: Optional[str] = None,
        ref_short: Optional[str] = None,
        sentence_id: Optional[str] = None,
        name_contains: Optional[str] = None,
        label_contains: Optional[str] = None,
    ) -> List[Entity]:
        """
        Search entities by various criteria.
        
        Args:
            name: Exact match on entity name
            label: Exact match on entity label
            entity_type: Exact match on entity_type
            ref_short: Exact match on ref_short
            sentence_id: Exact match on sentence_id
            name_contains: Substring match on entity name (case-insensitive)
            label_contains: Substring match on entity label (case-insensitive)
        
        Returns:
            List of matching Entity objects
        """
        results = list(self._entities.values())
        
        if name is not None:
            results = [e for e in results if e.name == name]
        
        if label is not None:
            results = [e for e in results if e.label == label]
        
        if entity_type is not None:
            results = [e for e in results if e.entity_type == entity_type]
        
        if ref_short is not None:
            results = [e for e in results if e.ref_short == ref_short]
        
        if sentence_id is not None:
            results = [e for e in results if e.sentence_id == sentence_id]
        
        if name_contains is not None:
            name_lower = name_contains.lower()
            results = [e for e in results if name_lower in e.name.lower()]
        
        if label_contains is not None:
            label_lower = label_contains.lower()
            results = [e for e in results if label_lower in e.label.lower()]
        
        return results
    
    def update_entity(
        self,
        entity_id: str,
        name: Optional[str] = None,
        label: Optional[str] = None,
        ref_short: Optional[str] = None,
        ref: Optional[str] = None,
        entity_type: Optional[str] = None,
        sentence_id: Optional[str] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> Entity:
        """
        Update entity properties.
        
        Args:
            entity_id: ID of entity to update
            name: New name (if provided)
            label: New label (if provided)
            ref_short: New ref_short (if provided)
            ref: New ref (if provided)
            entity_type: New entity_type (if provided)
            sentence_id: New sentence_id (if provided)
            start: New start position (if provided)
            end: New end position (if provided)
        
        Returns:
            Updated Entity object
        
        Raises:
            KeyError: If entity not found
        """
        entity = self.get_by_id(entity_id)
        
        if name is not None:
            entity.name = name
        if label is not None:
            entity.label = label
        if ref_short is not None:
            entity.ref_short = ref_short
        if ref is not None:
            entity.ref = ref
        if entity_type is not None:
            entity.entity_type = entity_type
        if sentence_id is not None:
            entity.sentence_id = sentence_id
        if start is not None:
            entity.start = start
        if end is not None:
            entity.end = end
        
        self.save(entity)
        return entity
    
    def delete_entity(self, entity_id: str) -> bool:
        """
        Delete an entity and all triples that reference it.
        
        Args:
            entity_id: ID of entity to delete
        
        Returns:
            True if deleted, False if not found
        """
        if entity_id not in self._entities:
            return False
        
        # Find all triples that reference this entity
        triple_ids_to_delete = set()
        if entity_id in self._triples_by_head:
            triple_ids_to_delete.update(self._triples_by_head[entity_id])
        if entity_id in self._triples_by_tail:
            triple_ids_to_delete.update(self._triples_by_tail[entity_id])
        
        # Delete all referencing triples
        for triple_id in triple_ids_to_delete:
            self.delete_triple(triple_id)
        
        # Delete the entity
        del self._entities[entity_id]
        return True
    
    # ==================== Triple Methods ====================
    
    def add_triple(self, triple: "Triple") -> "Triple":
        """
        Add a new triple to the repository.
        
        Args:
            triple: Triple object to add
        
        Returns:
            The added Triple object
        """
        # Ensure entities exist in repository
        head_key = triple.head.ref or triple.head.id or triple.head.ref_short
        tail_key = triple.tail.ref or triple.tail.id or triple.tail.ref_short
        if head_key and head_key not in self._entities:
            self.save(triple.head)
        if tail_key and tail_key not in self._entities:
            self.save(triple.tail)
        
        self._triples[triple.id] = triple
        self._index_triple(triple)
        return triple
    
    def get_triple(self, triple_id: str) -> "Triple":
        """
        Get a triple by ID.
        
        Args:
            triple_id: ID of triple to retrieve
        
        Returns:
            Triple object
        
        Raises:
            KeyError: If triple not found
        """
        if triple_id not in self._triples:
            raise KeyError(f"Triple with id {triple_id} not found")
        return self._triples[triple_id]
    
    def get_all_triples(self) -> Dict[str, "Triple"]:
        """
        Get all triples.
        
        Returns:
            Dictionary mapping triple IDs to Triple objects
        """
        return self._triples.copy()
    
    def search_triples(
        self,
        head_id: Optional[str] = None,
        tail_id: Optional[str] = None,
        relation: Optional[str] = None,
        relation_contains: Optional[str] = None,
        head_name: Optional[str] = None,
        tail_name: Optional[str] = None,
        head_label: Optional[str] = None,
        tail_label: Optional[str] = None,
        head_name_contains: Optional[str] = None,
        tail_name_contains: Optional[str] = None,
    ) -> List["Triple"]:
        """
        Search triples by various criteria.
        
        Args:
            head_id: Exact match on head entity ID
            tail_id: Exact match on tail entity ID
            relation: Exact match on relation
            relation_contains: Substring match on relation (case-insensitive)
            head_name: Exact match on head entity name
            tail_name: Exact match on tail entity name
            head_label: Exact match on head entity label
            tail_label: Exact match on tail entity label
            head_name_contains: Substring match on head entity name (case-insensitive)
            tail_name_contains: Substring match on tail entity name (case-insensitive)
        
        Returns:
            List of matching Triple objects
        """
        # Start with all triples or use indices for faster lookup
        if head_id is not None:
            triple_ids = self._triples_by_head.get(head_id, set())
            results = [self._triples[tid] for tid in triple_ids if tid in self._triples]
        elif tail_id is not None:
            triple_ids = self._triples_by_tail.get(tail_id, set())
            results = [self._triples[tid] for tid in triple_ids if tid in self._triples]
        elif relation is not None:
            triple_ids = self._triples_by_relation.get(relation, set())
            results = [self._triples[tid] for tid in triple_ids if tid in self._triples]
        else:
            results = list(self._triples.values())
        
        # Apply additional filters
        if head_id is not None and tail_id is not None:
            results = [t for t in results if (t.head.ref or t.head.id or t.head.ref_short) == head_id and (t.tail.ref or t.tail.id or t.tail.ref_short) == tail_id]
        elif head_id is not None:
            results = [t for t in results if (t.head.ref or t.head.id or t.head.ref_short) == head_id]
        elif tail_id is not None:
            results = [t for t in results if (t.tail.ref or t.tail.id or t.tail.ref_short) == tail_id]
        
        if relation is not None:
            results = [t for t in results if t.relation == relation]
        
        if relation_contains is not None:
            relation_lower = relation_contains.lower()
            results = [t for t in results if relation_lower in t.relation.lower()]
        
        if head_name is not None:
            results = [t for t in results if t.head.name == head_name]
        
        if tail_name is not None:
            results = [t for t in results if t.tail.name == tail_name]
        
        if head_label is not None:
            results = [t for t in results if t.head.label == head_label]
        
        if tail_label is not None:
            results = [t for t in results if t.tail.label == tail_label]
        
        if head_name_contains is not None:
            head_name_lower = head_name_contains.lower()
            results = [t for t in results if head_name_lower in t.head.name.lower()]
        
        if tail_name_contains is not None:
            tail_name_lower = tail_name_contains.lower()
            results = [t for t in results if tail_name_lower in t.tail.name.lower()]
        
        return results
    
    def get_triples_by_entity(self, entity_id: str) -> List["Triple"]:
        """
        Get all triples where the entity appears as head or tail.
        
        Args:
            entity_id: ID of entity
        
        Returns:
            List of Triple objects
        """
        triple_ids = set()
        if entity_id in self._triples_by_head:
            triple_ids.update(self._triples_by_head[entity_id])
        if entity_id in self._triples_by_tail:
            triple_ids.update(self._triples_by_tail[entity_id])
        
        return [self._triples[tid] for tid in triple_ids if tid in self._triples]
    
    def get_triples_by_head(self, entity_id: str) -> List["Triple"]:
        """
        Get all triples where the entity is the head.
        
        Args:
            entity_id: ID of head entity
        
        Returns:
            List of Triple objects
        """
        triple_ids = self._triples_by_head.get(entity_id, set())
        return [self._triples[tid] for tid in triple_ids if tid in self._triples]
    
    def get_triples_by_tail(self, entity_id: str) -> List["Triple"]:
        """
        Get all triples where the entity is the tail.
        
        Args:
            entity_id: ID of tail entity
        
        Returns:
            List of Triple objects
        """
        triple_ids = self._triples_by_tail.get(entity_id, set())
        return [self._triples[tid] for tid in triple_ids if tid in self._triples]
    
    def update_triple(
        self,
        triple_id: str,
        head: Optional[Entity] = None,
        tail: Optional[Entity] = None,
        relation: Optional[str] = None,
        importance: Optional[float] = None,
        info_quality: Optional[float] = None,
        novelty: Optional[float] = None,
        tags: Optional[List[str]] = None,
    ) -> "Triple":
        """
        Update triple properties.
        
        Args:
            triple_id: ID of triple to update
            head: New head Entity (if provided)
            tail: New tail Entity (if provided)
            relation: New relation string (if provided)
            importance: New importance score (if provided)
            info_quality: New info_quality score (if provided)
            novelty: New novelty score (if provided)
            tags: New tags list (if provided, replaces existing)
        
        Returns:
            Updated Triple object
        
        Raises:
            KeyError: If triple not found
        """
        triple = self.get_triple(triple_id)
        
        # Unindex old triple if head/tail/relation changes
        needs_reindex = False
        if head is not None or tail is not None or relation is not None:
            self._unindex_triple(triple_id)
            needs_reindex = True
        
        if head is not None:
            # Ensure new head entity exists in repository
            if head.id not in self._entities:
                self.save(head)
            triple.head = head
        
        if tail is not None:
            # Ensure new tail entity exists in repository
            if tail.id not in self._entities:
                self.save(tail)
            triple.tail = tail
        
        if relation is not None:
            triple.relation = relation.strip()
        
        if importance is not None:
            triple.importance = max(0.0, min(1.0, float(importance)))
        
        if info_quality is not None:
            triple.info_quality = max(0.0, min(1.0, float(info_quality)))
        
        if novelty is not None:
            triple.novelty = max(0.0, min(1.0, float(novelty))) if novelty is not None else None
        
        if tags is not None:
            triple.tags = tags
        
        # Reindex if needed
        if needs_reindex:
            self._index_triple(triple)
        
        return triple
    
    def delete_triple(self, triple_id: str) -> bool:
        """
        Delete a triple from the repository.
        
        Args:
            triple_id: ID of triple to delete
        
        Returns:
            True if deleted, False if not found
        """
        if triple_id not in self._triples:
            return False
        
        self._unindex_triple(triple_id)
        del self._triples[triple_id]
        return True
    
    def delete_triples_by_entity(self, entity_id: str) -> int:
        """
        Delete all triples that reference a specific entity (as head or tail).
        
        Args:
            entity_id: ID of entity
        
        Returns:
            Number of triples deleted
        """
        triple_ids = set()
        if entity_id in self._triples_by_head:
            triple_ids.update(self._triples_by_head[entity_id])
        if entity_id in self._triples_by_tail:
            triple_ids.update(self._triples_by_tail[entity_id])
        
        count = 0
        for triple_id in triple_ids:
            if self.delete_triple(triple_id):
                count += 1
        
        return count
    
    def get_triple_count(self) -> int:
        """Get total number of triples."""
        return len(self._triples)
    
    def get_entity_count(self) -> int:
        """Get total number of entities."""
        return len(self._entities)
