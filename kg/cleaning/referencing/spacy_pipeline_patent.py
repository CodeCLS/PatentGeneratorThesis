import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span

from transformers import pipeline as hf_pipeline
from fastcoref import spacy_component  # registers "fastcoref"

from tools.sentence.entity import Entity


# ---------------------------------------------------------------------
# 1) Extensions
# ---------------------------------------------------------------------

if not Span.has_extension("norm_label"):
    Span.set_extension("norm_label", default=None)

if not Span.has_extension("kb_id"):
    Span.set_extension("kb_id", default=None)



if not Doc.has_extension("coref_clusters"):
    Doc.set_extension("coref_clusters", default=[])


# ---------------------------------------------------------------------
# 2) Components
# ---------------------------------------------------------------------

class EntityNormaliser:
    def __init__(self, mode="lemma"):
        self.mode = mode

    def __call__(self, doc):
        for ent in doc.ents:
            base = ent.root.lemma_ if self.mode == "lemma" else ent.text
            ent._.norm_label = " ".join(base.lower().split())
        return doc


@Language.factory("entity_normaliser", default_config={"mode": "lemma"})
def make_entity_normaliser(nlp, name, mode):
    return EntityNormaliser(mode)


# ---------------------------------------------------------------------

class HFNER:
    """
    Sentence-wise HF NER (fixes 512-token crash)
    """
    def __init__(self, model_path, aggregation_strategy="simple", device=-1):
        self.ner = hf_pipeline(
            "ner",
            model=model_path,
            aggregation_strategy=aggregation_strategy,
            device=device,
        )

    def __call__(self, doc):
        spans = []

        # guarantee sentence boundaries
        sents = list(doc.sents) if doc.has_annotation("SENT_START") else [doc]

        for sent in sents:
            base = sent.start_char
            for o in self.ner(sent.text):
                s = base + int(o["start"])
                e = base + int(o["end"])
                sp = doc.char_span(
                    s, e,
                    label=o.get("entity_group") or o.get("entity"),
                    alignment_mode="contract",
                )
                if sp is not None:
                    spans.append(sp)

        # keep longest non-overlapping
        spans.sort(key=lambda s: (s.end_char - s.start_char), reverse=True)
        chosen, occupied = [], set()
        for sp in spans:
            toks = set(range(sp.start, sp.end))
            if not toks & occupied:
                chosen.append(sp)
                occupied |= toks

        doc.ents = tuple(chosen)
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
    return HFNER(model_path, aggregation_strategy, device)


# ---------------------------------------------------------------------
# 3) Windowed coref (THIS FIXES YOUR CRASH)
# ---------------------------------------------------------------------

def _char_windows(text, size, overlap):
    i = 0
    while i < len(text):
        j = min(len(text), i + size)
        yield i, text[i:j]
        if j == len(text):
            break
        i = max(0, j - overlap)


def _dedupe(spans):
    seen, out = set(), []
    for sp in spans:
        k = (sp.start_char, sp.end_char)
        if k not in seen:
            seen.add(k)
            out.append(sp)
    return out


def _merge_clusters(clusters):
    merged = []
    keys = []

    for cl in clusters:
        s = {(sp.start_char, sp.end_char) for sp in cl}
        hit = None
        for i, k in enumerate(keys):
            if s & k:
                hit = i
                break
        if hit is None:
            merged.append(cl)
            keys.append(set(s))
        else:
            merged[hit] = _dedupe(merged[hit] + cl)
            keys[hit] |= s

    return [c for c in merged if len(c) >= 2]




@Language.factory(
    "windowed_fastcoref",
    default_config={
        "chunk_chars": 12000,
        "overlap": 1200,
        "model_architecture": "LingMessCoref",
        "model_path": "biu-nlp/lingmess-coref",
        "device": "cpu",
    },
)
def make_windowed_fastcoref(
    nlp, name, chunk_chars, overlap, model_architecture, model_path, device
):
    # mini pipeline used ONLY for coref on chunks
    coref_nlp = spacy.blank(nlp.lang)
    coref_nlp.add_pipe("sentencizer")
    coref_nlp.add_pipe(
        "fastcoref",
        config={
            "model_architecture": model_architecture,
            "model_path": model_path,
            "device": device,
        },
    )

    def component(doc):
        all_clusters = []

        for base, chunk in _char_windows(doc.text, chunk_chars, overlap):
            try:
                cdoc = coref_nlp(chunk)
            except Exception:
                continue
            print("fastcoref ext keys:", [k for k in cdoc._.extensions])
            print("coref_clusters type:", type(getattr(cdoc._, "coref_clusters", None)))


            raw = getattr(cdoc._, "coref_clusters", None) or []
            for cl in raw:
                mentions = cl.mentions if hasattr(cl, "mentions") else cl
                spans = []
                for m in mentions:
                    s = base + m.start_char
                    e = base + m.end_char
                    sp = doc.char_span(s, e, alignment_mode="contract")
                    if sp is not None:
                        spans.append(sp)
                if len(spans) >= 2:
                    all_clusters.append(_dedupe(spans))

        doc._.coref_clusters = _merge_clusters(all_clusters)
        return doc

    return component


# ---------------------------------------------------------------------
# 4) Entity linker
# ---------------------------------------------------------------------

class LocalEntityLinker:
    def __init__(self, hash_len=6):
        self.hash_len = hash_len

    def _canon(self, sp):
        return (sp._.norm_label or sp.text).lower()

    def _id(self, canon, sp):
        ctx = f"{canon}_{sp.sent.start_char}"
        h = hashlib.md5(ctx.encode()).hexdigest()[: self.hash_len]
        return f"ent::{canon.replace(' ', '_')}::{h}"

    def __call__(self, doc):
        canon2id = {}

        for ent in doc.ents:
            c = self._canon(ent)
            canon2id.setdefault(c, self._id(c, ent))
            ent._.kb_id = canon2id[c]

        for cluster in doc._.coref_clusters:
            rep = max(cluster, key=lambda s: len(s))
            c = self._canon(rep)
            canon2id.setdefault(c, self._id(c, rep))
            for sp in cluster:
                sp._.kb_id = canon2id[c]

        return doc


@Language.factory("local_entity_linker", default_config={"hash_len": 6})
def make_local_entity_linker(nlp, name, hash_len):
    return LocalEntityLinker(hash_len)


# ---------------------------------------------------------------------
# 5) PipelineBuilder
# ---------------------------------------------------------------------

class PipelineBuilder:
    def __init__(self):
        self._nlp = None

    def build(self):
        nlp = spacy.load("en_core_web_sm")

        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer", first=True)

        if "ner" in nlp.pipe_names:
            nlp.remove_pipe("ner")

        nlp.add_pipe("hf_ner", name="ner")
        nlp.add_pipe("entity_normaliser", after="ner")
        nlp.add_pipe(
            "windowed_fastcoref",
            after="entity_normaliser",
            config={
                "chunk_chars": 12000,
                "overlap": 1200,
                "model_architecture": "LingMessCoref",
                "model_path": "biu-nlp/lingmess-coref",
                "device": "cpu",
            },
        )
        nlp.add_pipe("local_entity_linker", after="windowed_fastcoref")



       

        self._nlp = nlp
        return nlp

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = self.build()
        return self._nlp


# ---------------------------------------------------------------------
# 6) Sentence join + mapping
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class JoinedText:
    text: str
    starts: List[int]


def join_sentences(sentences, sep=" "):
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
    def __init__(self, sentence_cls):
        self.Sentence = sentence_cls

    def map_to_sentences(self, doc, sentences, joined):
        # 1) Map NER entities first (uses kb_id if your linker set it)
        for ent in doc.ents:
            ref = getattr(ent._, "kb_id", None) or ent.text
            idx = max(i for i, s in enumerate(joined.starts) if s <= ent.start_char)
            sent = sentences[idx]
            start = ent.start_char - joined.starts[idx]
            end = ent.end_char - joined.starts[idx]

            sent.entities[(start, end)] = Entity(
                name=ent.text,
                ref=ref,
                ref_short=ref[-4:] if isinstance(ref, str) and len(ref) >= 4 else None,
                label=getattr(ent, "label_", None) or Entity.REFERENCE,
                sentence_id=f"s{idx}",
            )

        # 2) If coref clusters exist, unify IDs across mentions (optional)
        for cluster in (doc._.coref_clusters or []):
            ref = cluster[0]._.kb_id
            for sp in cluster:
                idx = max(i for i, s in enumerate(joined.starts) if s <= sp.start_char)
                sent = sentences[idx]
                start = sp.start_char - joined.starts[idx]
                end = sp.end_char - joined.starts[idx]

                sent.entities[(start, end)] = Entity(
                    name=sp.text,
                    ref=ref,
                    ref_short=ref[-4:],
                    label=Entity.REFERENCE,
                    sentence_id=f"s{idx}",
                )

        return doc._.coref_clusters
