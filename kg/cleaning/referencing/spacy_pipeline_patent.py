"""
Safer, cleaner coref+NER pipeline.

Hard guarantee vs your ForwardRef error:
- NO `from __future__ import annotations`
- NO type hints on ANY spaCy factory parameters (nlp/name/others)
- NO @Language.factory on classes (only on functions)

Also:
- fastcoref cluster output normalized across versions
- Sentence joining with explicit separator + offsets
"""

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span

from transformers import pipeline as hf_pipeline

# registers spaCy component "fastcoref"
from fastcoref import spacy_component  # noqa: F401

# Your custom Entity class
from tools.sentence.entity import Entity


# ---------------------------------------------------------------------
# 1) Custom extensions
# ---------------------------------------------------------------------

if not Span.has_extension("norm_label"):
    Span.set_extension("norm_label", default=None)

if not Span.has_extension("kb_id"):
    Span.set_extension("kb_id", default=None)

# we store normalized List[List[Span]] here
if not Doc.has_extension("coref_clusters"):
    Doc.set_extension("coref_clusters", default=None)

if not Doc.has_extension("triples"):
    Doc.set_extension("triples", default=None)


# ---------------------------------------------------------------------
# 2) Pipeline components (NO factory decorators on classes)
# ---------------------------------------------------------------------

class EntityNormaliser:
    def __init__(self, mode: str = "lemma"):
        self.mode = mode

    def __call__(self, doc: Doc) -> Doc:
        for ent in doc.ents:
            if self.mode == "lemma":
                canon = (ent.root.lemma_ or ent.text).lower()
            else:
                canon = ent.text.lower()
            ent._.norm_label = " ".join(canon.split())
        return doc


@Language.factory("entity_normaliser", default_config={"mode": "lemma"})
def make_entity_normaliser(nlp, name, mode):
    # IMPORTANT: no type hints in factory signature
    return EntityNormaliser(mode=mode)


class HFNER:
    """
    HuggingFace NER as a spaCy component.
    Keeps longest non-overlapping spans.
    """
    def __init__(self, model_path: str, aggregation_strategy: str = "simple", device: int = -1):
        self.ner = hf_pipeline(
            "ner",
            model=model_path,
            aggregation_strategy=aggregation_strategy,
            device=device,  # -1 CPU, 0 GPU
        )

    def __call__(self, doc: Doc) -> Doc:
        outputs = self.ner(doc.text)

        candidates: List[Span] = []
        for o in outputs:
            sp = doc.char_span(
                o["start"],
                o["end"],
                label=o.get("entity_group") or o.get("entity"),
                alignment_mode="contract",
            )
            if sp is not None and len(sp) > 0:
                candidates.append(sp)

        # longest first
        candidates.sort(key=lambda s: (s.end_char - s.start_char), reverse=True)

        selected: List[Span] = []
        occupied = set()
        for sp in candidates:
            sp_tokens = set(range(sp.start, sp.end))
            if not (sp_tokens & occupied):
                selected.append(sp)
                occupied.update(sp_tokens)

        doc.ents = tuple(selected)
        return doc


@Language.factory(
    "hf_ner",
    default_config={
        "model_path": "training/ner/done/hf/ner_model",
        "aggregation_strategy": "simple",
        "device": -1,
    },
)
def make_hf_ner(nlp, name, model_path, aggregation_strategy, device):
    return HFNER(model_path=model_path, aggregation_strategy=aggregation_strategy, device=device)


class CorefNormalizer:
    """
    Normalizes fastcoref outputs into doc._.coref_clusters = List[List[Span]].
    This ensures downstream components (like LocalEntityLinker) see a consistent format.
    """
    def __call__(self, doc: Doc) -> Doc:
        doc._.coref_clusters = _normalize_clusters_to_spans(doc)
        return doc


@Language.component("coref_normalizer")
def coref_normalizer(doc):
    # Stateless wrapper for CorefNormalizer logic
    doc._.coref_clusters = _normalize_clusters_to_spans(doc)
    return doc


class LocalEntityLinker:
    """
    Assign stable local kb_id to:
    - all doc.ents
    - all coref mentions by cluster propagation
    """
    def __init__(self, hash_len: int = 6):
        self.hash_len = hash_len

    def _canon(self, span: Span) -> str:
        base = (span._.norm_label or span.text or "").lower().strip()
        return " ".join(base.split())

    def _make_id(self, canon: str, span: Span) -> str:
        ctx = f"{canon}_{span.sent.start_char}_{span.sent.end_char}"
        suffix = hashlib.md5(ctx.encode("utf-8")).hexdigest()[: self.hash_len]
        safe = canon.replace(" ", "_")
        return f"ent::{safe}::{suffix}"

    def _choose_representative(self, doc: Doc, cluster_spans: List[Span]) -> Span:
        # Prefer a mention that overlaps a NER entity
        for ent in doc.ents:
            for sp in cluster_spans:
                if ent.start < sp.end and sp.start < ent.end:
                    return ent

        # Otherwise prefer longest and penalize 1-token pronouns
        def score(sp: Span) -> Tuple[int, int]:
            is_pron = int(len(sp) == 1 and sp[0].pos_ == "PRON")
            return (len(sp), -is_pron)

        return max(cluster_spans, key=score)

    def __call__(self, doc: Doc) -> Doc:
        canon2id: Dict[str, str] = {}

        # 1) NER entities
        for ent in doc.ents:
            canon = self._canon(ent)
            kb_id = canon2id.get(canon)
            if kb_id is None:
                kb_id = self._make_id(canon, ent)
                canon2id[canon] = kb_id
            ent._.norm_label = canon
            ent._.kb_id = kb_id

        # 2) Coref clusters (already normalized by coref_normalizer)
        clusters: List[List[Span]] = doc._.coref_clusters or []
        for cluster in clusters:
            if not cluster:
                continue

            rep = self._choose_representative(doc, cluster)
            canon = self._canon(rep)

            kb_id = canon2id.get(canon)
            if kb_id is None:
                kb_id = self._make_id(canon, rep)
                canon2id[canon] = kb_id

            for sp in cluster:
                sp._.norm_label = canon
                sp._.kb_id = kb_id

        return doc


@Language.factory("local_entity_linker", default_config={"hash_len": 6})
def make_local_entity_linker(nlp, name, hash_len):
    return LocalEntityLinker(hash_len=hash_len)


# ---------------------------------------------------------------------
# 3) fastcoref cluster normalization across versions
# ---------------------------------------------------------------------

def _get_fastcoref_clusters(doc: Doc):
    # fastcoref versions differ in attribute names / shapes
    if hasattr(doc._, "coref_clusters") and doc._.coref_clusters:
        return doc._.coref_clusters
    if hasattr(doc._, "coref_clusters_") and getattr(doc._, "coref_clusters_"):
        return getattr(doc._, "coref_clusters_")
    return None


def _normalize_clusters_to_spans(doc: Doc) -> List[List[Span]]:
    """
    Normalize fastcoref output to List[List[Span]].

    Handles clusters shaped as:
    - objects with .mentions (Cluster objects)
    - list of Span
    - list of (start,end) or dict {start,end}
    """
    raw = _get_fastcoref_clusters(doc)
    if not raw:
        return []

    norm: List[List[Span]] = []

    for cl in raw:
        if hasattr(cl, "mentions"):
            mentions = cl.mentions
        elif isinstance(cl, (list, tuple)):
            mentions = cl
        else:
            continue

        spans: List[Span] = []
        for m in mentions:
            if isinstance(m, Span):
                spans.append(m)
                continue

            if isinstance(m, (list, tuple)) and len(m) >= 2:
                try:
                    s = int(m[0]); e = int(m[1])
                except Exception:
                    continue
                sp = doc.char_span(s, e, alignment_mode="contract")
                if sp is not None and len(sp) > 0:
                    spans.append(sp)
                continue

            if isinstance(m, dict) and "start" in m and "end" in m:
                try:
                    s = int(m["start"]); e = int(m["end"])
                except Exception:
                    continue
                sp = doc.char_span(s, e, alignment_mode="contract")
                if sp is not None and len(sp) > 0:
                    spans.append(sp)

        if spans:
            norm.append(spans)

    # keep only clusters with >=2 mentions (real coref)
    norm = [c for c in norm if len(c) >= 2]
    return norm


# ---------------------------------------------------------------------
# 4) Pipeline builder
# ---------------------------------------------------------------------

class PipelineBuilder:
    """
    Builds a spaCy pipeline:
      tokenizer/tagger (from spaCy model)
      hf_ner
      entity_normaliser
      fastcoref (LingMessCoref)
      coref_normalizer (to doc._.coref_clusters)
      local_entity_linker
    """
    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        device: str = "auto",
        coref_arch: str = "LingMessCoref",
        coref_model_path: str = "biu-nlp/lingmess-coref",
        hf_ner_model_path: str = "training/ner/done/hf/ner_model",
        hf_ner_device: int = -1,
    ):
        self.spacy_model = spacy_model
        self.device = device
        self.coref_arch = coref_arch
        self.coref_model_path = coref_model_path
        self.hf_ner_model_path = hf_ner_model_path
        self.hf_ner_device = hf_ner_device
        self._nlp: Optional[Language] = None

    def build(self) -> Language:
        nlp = spacy.load(self.spacy_model)

        # Replace spaCy NER with HF component
        if "ner" in nlp.pipe_names:
            nlp.remove_pipe("ner")

        nlp.add_pipe(
            "hf_ner",
            name="ner",
            config={
                "model_path": self.hf_ner_model_path,
                "aggregation_strategy": "simple",
                "device": self.hf_ner_device,
            },
        )

        nlp.add_pipe("entity_normaliser", after="ner")

        nlp.add_pipe(
            "fastcoref",
            config={
                "model_architecture": self.coref_arch,
                "model_path": self.coref_model_path,
                "device": self.device,
            },
        )

        # Normalize fastcoref output to doc._.coref_clusters
        nlp.add_pipe("coref_normalizer", after="fastcoref")

        # After coref so we can propagate kb_ids into clusters
        nlp.add_pipe("local_entity_linker", after="coref_normalizer")

        self._nlp = nlp
        return nlp

    @property
    def nlp(self) -> Language:
        if self._nlp is None:
            self._nlp = self.build()
        return self._nlp


# ---------------------------------------------------------------------
# 5) Sentence joining + mapping back to your Sentence objects
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class JoinedText:
    text: str
    starts: List[int]
    sep: str


def join_sentences(sentences: List[Any], sep: str = " ") -> JoinedText:
    starts: List[int] = []
    parts: List[str] = []
    cursor = 0

    for i, s in enumerate(sentences):
        starts.append(cursor)
        parts.append(s.text)
        cursor += len(s.text)
        if i != len(sentences) - 1:
            parts.append(sep)
            cursor += len(sep)

    return JoinedText(text="".join(parts), starts=starts, sep=sep)


class EntityMapper:
    """
    Maps doc.ents + doc._.coref_clusters back onto your custom Sentence objects.
    Assumes doc.text == join_sentences(sentences, sep=...).text
    """
    def __init__(self, sentence_cls: Any):
        self.Sentence = sentence_cls

    def _find_sentence_index(self, char_pos: int, sentence_starts: List[int]) -> int:
        for i, start in enumerate(sentence_starts):
            if i == len(sentence_starts) - 1 or char_pos < sentence_starts[i + 1]:
                return i
        return len(sentence_starts) - 1

    def _cluster_spans_set(self, clusters: List[List[Span]]) -> set:
        return {(sp.start_char, sp.end_char) for cl in clusters for sp in cl}

    def map_to_sentences(self, doc: Doc, sentences: List[Any], joined: JoinedText) -> List[List[Span]]:
        sentence_starts = joined.starts

        clusters: List[List[Span]] = doc._.coref_clusters or []

        # 1) Coref clusters -> reference entities
        for cluster in clusters:
            if not cluster:
                continue
            kb_id = cluster[0]._.kb_id

            for sp in cluster:
                sent_idx = self._find_sentence_index(sp.start_char, sentence_starts)
                sent = sentences[sent_idx]
                sent_start = sentence_starts[sent_idx]

                local_start = sp.start_char - sent_start
                local_end = sp.end_char - sent_start

                sent.entities[(local_start, local_end)] = Entity(
                    name=sp.text,
                    ref=kb_id,
                    ref_short=kb_id[-4:].upper() if kb_id else "",
                    label=Entity.REFERENCE,
                    sentence_id=f"s{sent_idx}",
                )

        # 2) NER ents not already in coref mentions
        covered = self._cluster_spans_set(clusters)
        for ent in doc.ents:
            if (ent.start_char, ent.end_char) in covered:
                continue

            kb_id = ent._.kb_id
            sent_idx = self._find_sentence_index(ent.start_char, sentence_starts)
            sent = sentences[sent_idx]
            sent_start = sentence_starts[sent_idx]

            local_start = ent.start_char - sent_start
            local_end = ent.end_char - sent_start

            if (local_start, local_end) not in sent.entities:
                sent.entities[(local_start, local_end)] = Entity(
                    name=ent.text,
                    ref=kb_id,
                    ref_short=ent.text,
                    label=ent.label_,
                    sentence_id=f"s{sent_idx}",
                )

        self._normalise_other_refs_to_first(sentences)
        return clusters

    def _normalise_other_refs_to_first(self, sentences: List[Any]) -> None:
        canonical: Dict[str, str] = {}
        for s in sentences:
            for _, ent in sorted(s.entities.items(), key=lambda kv: kv[0][0]):
                if not ent.ref or ent.ref in canonical:
                    continue
                canonical[ent.ref] = ent.name

        for s in sentences:
            for ent in s.entities.values():
                if ent.ref in canonical:
                    ent.ref_short = canonical[ent.ref]


# ---------------------------------------------------------------------
# 6) Example usage
# ---------------------------------------------------------------------

def run_pipeline_on_sentences(sentences: List[Any], sep: str = " ") -> List[List[Span]]:
    builder = PipelineBuilder(
        spacy_model="en_core_web_sm",
        device="auto",
        coref_arch="LingMessCoref",
        coref_model_path="biu-nlp/lingmess-coref",
        hf_ner_model_path="training/ner/done/hf/ner_model",
        hf_ner_device=-1,  # set 0 to run HF NER on GPU
    )

    joined = join_sentences(sentences, sep=sep)
    doc = builder.nlp(joined.text)

    mapper = EntityMapper(sentence_cls=type(sentences[0]) if sentences else Any)
    clusters = mapper.map_to_sentences(doc, sentences, joined)
    return clusters
