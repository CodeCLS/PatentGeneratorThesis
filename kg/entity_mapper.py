"""
Entity mapping utilities for mapping spaCy entities to Sentence objects.
"""
from dataclasses import dataclass
from typing import List

from tools.sentence.entity import Entity


@dataclass(frozen=True)
class JoinedText:
    """Represents joined text with sentence start positions."""
    text: str
    starts: List[int]


def join_sentences(sentences, sep=" "):
    """Join sentences into a single text, tracking start positions."""
    parts, starts, cur = [], [], 0
    for i, s in enumerate(sentences):
        starts.append(cur)
        parts.append(s.text)
        cur += len(s.text)
        if i < len(sentences) - 1:
            parts.append(sep)
            cur += len(sep)
    return JoinedText("".join(parts), starts)


class EntityMapper:
    """Maps spaCy document entities to Sentence objects."""
    
    def __init__(self, sentence_cls, dedupe: bool = True):
        self.Sentence = sentence_cls
        self.dedupe = dedupe

    @staticmethod
    def _sent_index(starts: list[int], pos: int) -> int:
        """Find sentence index for a given character position."""
        return max(i for i, s in enumerate(starts) if s <= pos)

    @staticmethod
    def _ref_short(ref: str | None) -> str:
        """Get short reference ID (last 4 characters)."""
        if isinstance(ref, str) and len(ref) >= 4:
            return ref[-4:]
        if isinstance(ref, str) and len(ref) > 0:
            return ref
        return ""

    def _maybe_add(self, sent, ent: Entity, seen: set) -> None:
        """Add entity to sentence if not duplicate."""
        if not self.dedupe:
            sent.entities.append(ent)
            return

        key = (ent.start, ent.end, ent.ref, ent.label, ent.sentence_id)
        if key in seen:
            return
        seen.add(key)
        sent.entities.append(ent)

    def map_to_sentences(self, doc, sentences, joined: JoinedText):
        """Map document entities and coref clusters to sentences."""
        seen: set = set()

        # 1) Map NER entities
        for sp in doc.ents:
            ref = getattr(sp._, "kb_id", None) or sp.text
            idx = self._sent_index(joined.starts, sp.start_char)
            sent = sentences[idx]

            start = sp.start_char - joined.starts[idx]
            end = sp.end_char - joined.starts[idx]

            ent = Entity(
                name=sp.text,
                label=getattr(sp, "label_", None) or "REFERENCE",
                ref_short=self._ref_short(ref),
                start=start,
                end=end,
                ref=ref,
                sentence_id=f"s{idx}",
                entity_type=getattr(sp, "label_", None),
            )
            self._maybe_add(sent, ent, seen)

        # 2) Map coreference clusters
        for cluster in (doc._.coref_clusters or []):
            cluster_ref = getattr(cluster[0]._, "kb_id", None) or cluster[0].text
            for sp in cluster:
                idx = self._sent_index(joined.starts, sp.start_char)
                sent = sentences[idx]

                start = sp.start_char - joined.starts[idx]
                end = sp.end_char - joined.starts[idx]

                ent = Entity(
                    name=sp.text,
                    label="REFERENCE",
                    ref_short=self._ref_short(cluster_ref),
                    start=start,
                    end=end,
                    ref=cluster_ref,
                    sentence_id=f"s{idx}",
                    entity_type="COREF",
                )
                self._maybe_add(sent, ent, seen)

        return doc._.coref_clusters

