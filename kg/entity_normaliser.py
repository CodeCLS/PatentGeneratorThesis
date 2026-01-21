"""
Entity normalization component for spaCy pipeline.
"""
from spacy.language import Language
from spacy.tokens import Doc


class EntityNormaliser:
    """Normalizes entity labels using lemmatization or raw text."""
    
    def __init__(self, mode="lemma"):
        self.mode = mode

    def __call__(self, doc: Doc) -> Doc:
        """Normalize all entities in the document."""
        for ent in doc.ents:
            base = ent.root.lemma_ if self.mode == "lemma" else ent.text
            ent._.norm_label = " ".join(base.lower().split())
        return doc


@Language.factory("entity_normaliser", default_config={"mode": "lemma"})
def make_entity_normaliser(nlp, name, mode):
    """Factory function for entity normaliser component."""
    return EntityNormaliser(mode)

