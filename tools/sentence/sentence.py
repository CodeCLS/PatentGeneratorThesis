# tools/sentence/sentence.py
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Dict
import uuid
from tools.sentence.entity import Entity

@dataclass(slots=True)
class Sentence:
    # Core
    text: str
    index: int = 0                       # order in sequence
    source: str = ""                     # doc/page/paragraph id
    span: Optional[Tuple[int, int]] = None  # (char_start, char_end) in source

    # IDs / graph hooks
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    kg_node_id: Optional[str] = None     # node id once inserted into KG

    # NLP / features
    embedding: Optional[List[float]] = None
    tokens: List[str] = field(default_factory=list)
    entities: List[Entity] = field(default_factory=list)

    tags: List[str] = field(default_factory=list)      # arbitrary labels ("claim", "boilerplate", ...)

    # Scores (0..1)
    importance: float = 0.5              # task-specific weight (centrality, etc.)
    info_quality: float = 0.5            # informativeness / specificity
    novelty: Optional[float] = None      # optional 0..1 (higher = more novel)

    # Language / meta
    lang: str = "en"

    def __post_init__(self):
        # clamp bounded scores
        self.importance = max(0.0, min(1.0, self.importance))
        self.info_quality = max(0.0, min(1.0, self.info_quality))
        if self.novelty is not None:
            self.novelty = max(0.0, min(1.0, self.novelty))

    def score(self, w_imp: float = 0.5, w_info: float = 0.5, w_novel: float = 0.0) -> float:
        """Weighted utility score combining bounded metrics."""
        imp = self.importance
        info = self.info_quality
        nov = self.novelty if self.novelty is not None else 0.0
        total_w = max(1e-9, w_imp + w_info + w_novel)
        return (w_imp * imp + w_info * info + w_novel * nov) / total_w

    def set_embedding(self, vec: List[float]) -> None:
        """Attach/replace embedding."""
        if not isinstance(vec, list) or (vec and not isinstance(vec[0], (int, float))):
            raise TypeError("embedding must be List[float]")
        self.embedding = [float(x) for x in vec]

    def add_tag(self, tag: str) -> None:
        if tag and tag not in self.tags:
            self.tags.append(tag)

    def to_dict(self) -> dict:
        return asdict(self)
