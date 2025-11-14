from dataclasses import dataclass, field, asdict
from typing import Optional
import uuid


@dataclass
class Entity:
    REFERENCE = "Reference"
    """
    Represents an entity (subject/object/concept) in a specific sentence.
    """
    name: str                                 # e.g. "pump motor"
    label: str
    ref_short: str              # optional short reference label like "E1"

    entity_type: Optional[str] = None         # e.g. "component"
    uid: str = field(default_factory=lambda: str(uuid.uuid4()))
    ref: Optional[str] = None                 # optional short reference label like "E1"

    sentence_id: Optional[str] = None         # ID of the sentence where it occurs

    def __repr__(self) -> str:
        return f"Entity({self.name})"

    def to_dict(self):
        return asdict(self)