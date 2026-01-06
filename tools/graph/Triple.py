# tools/sentence/triple.py
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import uuid

from tools.sentence.entity import Entity


@dataclass(slots=True)
class Triple:
    head: Entity
    relation: str
    tail: Entity

    # IDs / graph hooks
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Language / meta
    lang: str = "en"

    # Bounded metrics
    importance: float = 0.5          # clamped to [0, 1]
    info_quality: float = 0.5        # clamped to [0, 1]
    novelty: Optional[float] = None  # clamped to [0, 1] if not None

    # Optional extras
    embedding: List[float] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)  # Relation properties/qualifiers

    def __post_init__(self) -> None:
        # clamp bounded scores
        self.importance = max(0.0, min(1.0, float(self.importance)))
        self.info_quality = max(0.0, min(1.0, float(self.info_quality)))
        if self.novelty is not None:
            self.novelty = max(0.0, min(1.0, float(self.novelty)))

        # normalize relation
        self.relation = str(self.relation).strip()

        # validate embedding
        if not isinstance(self.embedding, list):
            raise TypeError("embedding must be a List[float]")
        for x in self.embedding:
            if not isinstance(x, (int, float)):
                raise TypeError("embedding must be a List[float]")
        self.embedding = [float(x) for x in self.embedding]

        # validate tags
        if not isinstance(self.tags, list) or any(not isinstance(t, str) for t in self.tags):
            raise TypeError("tags must be a List[str]")

    def score(self, w_imp: float = 0.5, w_info: float = 0.5, w_novel: float = 0.0) -> float:
        """Weighted utility score combining bounded metrics."""
        imp = self.importance
        info = self.info_quality
        nov = self.novelty if self.novelty is not None else 0.0
        total_w = max(1e-9, w_imp + w_info + w_novel)
        return (w_imp * imp + w_info * info + w_novel * nov) / total_w

    def set_embedding(self, vec: List[float]) -> None:
        """Attach/replace embedding."""
        if not isinstance(vec, list) or any(not isinstance(x, (int, float)) for x in vec):
            raise TypeError("embedding must be List[float]")
        self.embedding = [float(x) for x in vec]

    def add_tag(self, tag: str) -> None:
        tag = (tag or "").strip()
        if tag and tag not in self.tags:
            self.tags.append(tag)
    
    def set_property(self, key: str, value: Any) -> None:
        """Set a property/qualifier on the relation."""
        if key:
            self.properties[key] = value
    
    def get_property(self, key: str, default: Any = None) -> Any:
        """Get a property/qualifier from the relation."""
        return self.properties.get(key, default)
    
    def has_properties(self) -> bool:
        """Check if the relation has any properties/qualifiers."""
        return len(self.properties) > 0

    def to_dict(self) -> dict:
        return asdict(self)
