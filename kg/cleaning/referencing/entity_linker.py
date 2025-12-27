"""
Local entity linking component for spaCy pipeline.
Creates unique IDs for entities and coreference clusters.
"""
import hashlib
from spacy.language import Language
from spacy.tokens import Doc, Span


class LocalEntityLinker:
    """Links entities to unique IDs based on normalized labels and context."""
    
    def __init__(self, hash_len=6):
        self.hash_len = hash_len

    def _canon(self, sp: Span) -> str:
        """Get canonical form of span."""
        return (sp._.norm_label or sp.text).lower()

    def _id(self, canon: str, sp: Span) -> str:
        """Generate unique ID for entity."""
        ctx = f"{canon}_{sp.sent.start_char}"
        h = hashlib.md5(ctx.encode()).hexdigest()[: self.hash_len]
        return f"ent::{canon.replace(' ', '_')}::{h}"

    def __call__(self, doc: Doc) -> Doc:
        """Link entities and coreference clusters to unique IDs."""
        canon2id = {}

        # Link named entities
        for ent in doc.ents:
            c = self._canon(ent)
            canon2id.setdefault(c, self._id(c, ent))
            ent._.kb_id = canon2id[c]

        # Link coreference clusters
        for cluster in doc._.coref_clusters:
            rep = max(cluster, key=lambda s: len(s))
            c = self._canon(rep)
            canon2id.setdefault(c, self._id(c, rep))
            for sp in cluster:
                sp._.kb_id = canon2id[c]

        return doc


@Language.factory("local_entity_linker", default_config={"hash_len": 6})
def make_local_entity_linker(nlp, name, hash_len):
    """Factory function for local entity linker component."""
    return LocalEntityLinker(hash_len)

