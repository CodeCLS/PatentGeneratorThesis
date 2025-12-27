from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, Protocol
import uuid

from typing import Dict


class InMemoryEntityRepository:
    def __init__(self, entities: list[Entity] | None = None):
        self._entities: Dict[str, Entity] = {
            e.id: e for e in (entities or [])
        }
    def getAll(self):
        return self._entities
    def get_by_id(self, entity_id: str) -> Entity:
        try:
            return self._entities[entity_id]
        except KeyError:
            raise KeyError(f"Entity with id {entity_id} not found")
    def save(self, entity: "Entity") -> None:
        self._entities[entity.id] = entity


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
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    ref: Optional[str] = None
    sentence_id: Optional[str] = None

    def __repr__(self) -> str:
        return f"Entity({self.name}@{self.start}:{self.end})"

    def to_dict(self):
        return asdict(self)
