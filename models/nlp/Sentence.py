from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Sentence:
    """
    Represents a processed or raw sentence unit.
    Useful for graph construction, classification, or LLM preprocessing.
    """

    id: int
    text: str
    tokens: Optional[List[str]] = field(default_factory=list)
    lemma: Optional[List[str]] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    pos_tags: Optional[List[str]] = field(default_factory=list)
    dependencies: Optional[List[str]] = field(default_factory=list)
    entities: Optional[List[str]] = field(default_factory=list)
    importance: Optional[float] = None
    parent_ids: Optional[List[int]] = field(default_factory=list)
    child_ids: Optional[List[int]] = field(default_factory=list)

    def __post_init__(self):
        # Basic cleanup for safety
        self.text = self.text.strip()

    def summary(self) -> str:
        """Return a compact summary string of the sentence."""
        return f"Sentence[{self.id}]: '{self.text[:60]}' (tokens={len(self.tokens)})"

    def add_child(self, child_id: int):
        """Attach a new child sentence by ID."""
        if child_id not in self.child_ids:
            self.child_ids.append(child_id)

    def add_parent(self, parent_id: int):
        """Attach a parent sentence by ID."""
        if parent_id not in self.parent_ids:
            self.parent_ids.append(parent_id)
