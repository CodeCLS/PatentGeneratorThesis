"""
Windowed coreference resolution component for spaCy pipeline.
Handles long documents by processing in overlapping windows.
"""
import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span
from fastcoref import spacy_component  # registers "fastcoref"


def _char_windows(text: str, size: int, overlap: int):
    """Generate overlapping character windows from text."""
    i = 0
    while i < len(text):
        j = min(len(text), i + size)
        yield i, text[i:j]
        if j == len(text):
            break
        i = max(0, j - overlap)


def _dedupe(spans: list[Span]) -> list[Span]:
    """Remove duplicate spans based on character positions."""
    seen, out = set(), []
    for sp in spans:
        k = (sp.start_char, sp.end_char)
        if k not in seen:
            seen.add(k)
            out.append(sp)
    return out


def _merge_clusters(clusters: list[list[Span]]) -> list[list[Span]]:
    """Merge overlapping coreference clusters."""
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
    """
    Factory function for windowed fastcoref component.
    Creates a component that processes documents in chunks to avoid memory issues.
    """
    # Mini pipeline used ONLY for coref on chunks
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

    def component(doc: Doc) -> Doc:
        """Process document in windows and merge coreference clusters."""
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