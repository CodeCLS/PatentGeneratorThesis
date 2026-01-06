"""
HuggingFace NER component for spaCy pipeline.
Handles sentence-wise NER to avoid 512-token limit issues.
"""
from spacy.language import Language
from spacy.tokens import Doc
from transformers import pipeline as hf_pipeline


class HFNER:
    """
    Sentence-wise HF NER (fixes 512-token crash).
    Processes each sentence separately to avoid token limit issues.
    """
    
    def __init__(self, model_path, aggregation_strategy="simple", device=-1):
        self.ner = hf_pipeline(
            "ner",
            model=model_path,
            aggregation_strategy=aggregation_strategy,
            device=device,
        )
        device_name = "GPU" if device >= 0 else "CPU"
        print(f"HF NER initialized on {device_name} (device={device})")

    def __call__(self, doc: Doc) -> Doc:
        """Extract named entities from document using sentence-wise processing."""
        spans = []

        # Guarantee sentence boundaries
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

        # Keep longest non-overlapping spans
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
    """Factory function for HF NER component."""
    return HFNER(model_path, aggregation_strategy, device)

