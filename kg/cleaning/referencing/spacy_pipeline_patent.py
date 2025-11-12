# pip install spacy fastcoref
# python -m spacy download en_core_web_trf
import hashlib
from typing import List, Optional
import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span
from fastcoref import FCoref

# ---- Safe extensions (avoid shared mutable defaults)
if not Span.has_extension("norm_label"):
    Span.set_extension("norm_label", default=None)
if not Span.has_extension("kb_id"):
    Span.set_extension("kb_id", default=None)
if not Doc.has_extension("coref_clusters"):
    Doc.set_extension("coref_clusters", default=None)
if not Doc.has_extension("triples"):
    Doc.set_extension("triples", default=None)

# =========================
# Components (as classes)
# =========================

class EntityNormaliser:
    def __init__(self, mode: str = "lemma"):
        self.mode = mode

    def __call__(self, doc: Doc) -> Doc:
        for ent in doc.ents:
            head = ent.root
            if self.mode == "lemma":
                canon = (head.lemma_ or ent.text).lower()
            else:
                canon = ent.text.lower()
            ent._.norm_label = canon
        return doc


class CorefResolver:
    def __init__(self, device: str = "auto"):
        if device == "auto":
            device = "cuda" if spacy.prefer_gpu() else "cpu"
        self.coref = FCoref(device=device)

    def __call__(self, doc: Doc) -> Doc:
        out = self.coref.predict(texts=[doc.text], is_split_into_words=False)[0]
        clusters: List[List[Span]] = []
        for cl in out.get_clusters():  # list of (char_start, char_end)
            spans = []
            for (start, end) in cl:
                sp = doc.char_span(start, end, alignment_mode="expand")
                if sp is not None:
                    spans.append(sp)
            if spans:
                clusters.append(spans)
        doc._.coref_clusters = clusters
        return doc


class LocalEntityLinker:
    def __init__(self, hash_len: int = 6):
        self.hash_len = hash_len

    def _pick_kb_id(self, span: Span) -> str:
        base = (span._.norm_label or span.root.lemma_ or span.text).lower().strip().replace(" ", "_")
        ctx = f"{base}_{span.sent.start}_{span.sent.end}"
        suffix = hashlib.md5(ctx.encode()).hexdigest()[: self.hash_len]
        return f"ent::{base}::{suffix}"

    def __call__(self, doc: Doc) -> Doc:
        # 1) IDs for NER spans
        for ent in doc.ents:
            ent._.kb_id = self._pick_kb_id(ent)

        # 2) Propagate across coref clusters
        clusters = doc._.coref_clusters or []
        for cluster in clusters:
            # Prefer a span covered by a NER entity
            source: Optional[Span] = None
            for sp in cluster:
                covering = [e for e in doc.ents if e.start <= sp.start and e.end >= sp.end]
                if covering:
                    source = covering[0]
                    break
            if source is None:
                source = max(cluster, key=lambda s: len(s))

            if source._.norm_label is None:
                source._.norm_label = (source.root.lemma_ or source.text).lower()
            if source._.kb_id is None:
                source._.kb_id = self._pick_kb_id(source)

            for sp in cluster:
                sp._.norm_label = source._.norm_label
                sp._.kb_id = source._.kb_id

        return doc

# =========================
# Register factories
# =========================

@Language.factory(
    "entity_normaliser",
    default_config={"mode": "lemma"},
)
def make_entity_normaliser(nlp: Language, name: str, mode: str):
    return EntityNormaliser(mode=mode)

@Language.factory(
    "fastcoref_resolver",
    default_config={"device": "auto"},
)
def make_coref_resolver(nlp: Language, name: str, device: str):
    return CorefResolver(device=device)

@Language.factory(
    "local_entity_linker",
    default_config={"hash_len": 6},
)
def make_local_entity_linker(nlp: Language, name: str, hash_len: int):
    return LocalEntityLinker(hash_len=hash_len)
