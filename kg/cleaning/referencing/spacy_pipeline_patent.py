# pip install spacy fastcoref
# python -m spacy download en_core_web_trf
import hashlib
from typing import List, Optional
import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span
from fastcoref import FCoref
from tools.sentence.entity import Entity
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
        raw_clusters = out.get_clusters(as_strings=False)

        clusters: List[List[Span]] = []
        for cl in raw_clusters:  # list of (char_start, char_end)
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

    def _canon(self, span: Span) -> str:
        base = (span._.norm_label or span.root.lemma_ or span.text).lower().strip()
        # normalize whitespace
        return " ".join(base.split())

    def _pick_kb_id(self, canon: str, span: Span) -> str:
        ctx = f"{canon}_{span.sent.start}_{span.sent.end}"
        suffix = hashlib.md5(ctx.encode()).hexdigest()[: self.hash_len]
        safe = canon.replace(" ", "_")
        return f"ent::{safe}::{suffix}"

    def __call__(self, doc: Doc) -> Doc:
        # canon string -> kb_id
        canon2id = {}

        # 1) Assign IDs to NER entities first
        for ent in doc.ents:
            canon = self._canon(ent)
            kb_id = canon2id.get(canon)
            if kb_id is None:
                kb_id = self._pick_kb_id(canon, ent)
                canon2id[canon] = kb_id
            ent._.norm_label = canon
            ent._.kb_id = kb_id

        # 2) Propagate into coref clusters (pronouns etc.)
        clusters = doc._.coref_clusters or []
        for cluster in clusters:
            # Prefer a span that is (fully) inside a NER entity
            source = None
            for sp in cluster:
                covering = [e for e in doc.ents if e.start <= sp.start < e.end]
                if covering:
                    source = covering[0]
                    break
            if source is None:
                # fall back to longest span in cluster
                source = max(cluster, key=lambda s: len(s))

            canon = self._canon(source)
            kb_id = canon2id.get(canon)
            if kb_id is None:
                kb_id = self._pick_kb_id(canon, source)
                canon2id[canon] = kb_id

            source._.norm_label = canon
            source._.kb_id = kb_id

            for sp in cluster:
                sp._.norm_label = canon
                sp._.kb_id = kb_id

        return doc
class SpacyPipeline:
    def __init__(self):
        pass
    def build_nlp(self):
        nlp = spacy.load("en_core_web_trf")
        nlp.add_pipe("entity_normaliser", after="ner")
        nlp.add_pipe("fastcoref_resolver")
        nlp.add_pipe("local_entity_linker")
        return nlp
    def indexes_sentences(self, sentences):
        sentence_starts = []
        offset = 0
        for s in sentences:
            sentence_starts.append(offset)
            offset += len(s.text)
        return sentence_starts
    def nlp(self,text):
<<<<<<< HEAD
        nlp =  self.build_nlp()
        return nlp(text)
=======
        return self.build_nlp(text)
>>>>>>> 4eec9490a922b26d8df35968d1d7e431807dd75a
    
    def find_sentence_index(self,char_pos, sentence_starts):
        """
        Given a global character position, return the index of the sentence
        it belongs to.
        """
        # sentence_starts is sorted: [0, len(s1), len(s1)+len(s2), ...]
        idx = 0
        for i, start in enumerate(sentence_starts):
            # last sentence or before the next sentence start
            if i == len(sentence_starts) - 1 or char_pos < sentence_starts[i+1]:
                return i
        return len(sentence_starts) - 1
<<<<<<< HEAD
    def fill_coref_entities(self,doc,sentence_starts,sentences):
=======
    def fill_coref_entities(self,sentence_starts,sentences):
>>>>>>> 4eec9490a922b26d8df35968d1d7e431807dd75a
        clusters = doc._.coref_clusters or []
        for cl in clusters:
            kb_id = cl[0]._.kb_id  # one ID per cluster (set by LocalEntityLinker)

            for sp in cl:
                # 1) find which Sentence this span belongs to
                sent_idx = self.find_sentence_index(sp.start_char, sentence_starts)
                sent = sentences[sent_idx]

                # 2) compute local char offsets inside that sentence
                sent_start_char = sentence_starts[sent_idx]
                local_start = sp.start_char - sent_start_char
                local_end = sp.end_char - sent_start_char

                # 3) store Entity in that sentence's entities dict
                sent.entities[(local_start, local_end)] = Entity(
                    name=sp.text,
                    ref=kb_id,
                    ref_short=kb_id[-4:].upper(),

                    label = Entity.REFERENCE,
                    sentence_id=f"s{sent_idx}",
                )
        return clusters
    def normalise_other_refs_to_first(self,sentences):
        canonical = {}
        for s in sentences:
            # process entities in left-to-right order within the sentence
            for (start, end), ent in sorted(s.entities.items(), key=lambda kv: kv[0][0]):
                if ent.ref is None:
                    continue
                if ent.ref not in canonical:
                    # first time we see this ref -> freeze its name as canonical
                    canonical[ent.ref] = ent.name
        for s in sentences:
            for ent in s.entities.values():
                if ent.ref in canonical:
                    ent.ref_short = canonical[ent.ref]   # e.g. "Caleb"
    def get_cluster_spans(self,clusters):
            
            cluster_spans = {
                (sp.start_char, sp.end_char)
                for cl in clusters
                for sp in cl
        }
            return cluster_spans
<<<<<<< HEAD
    def add_entities_from_ner(self, doc,sentence_starts,sentences,cluster_spans = []):
=======
    def add_entities_from_ner(self, cluster_spans,sentence_starts,sentences):
>>>>>>> 4eec9490a922b26d8df35968d1d7e431807dd75a
                
        for ent in doc.ents:
            # skip entities that were already added via coref
            if (ent.start_char, ent.end_char) in cluster_spans:
                continue

            kb_id = ent._.kb_id  # set by LocalEntityLinker for every NER ent

            sent_idx = self.find_sentence_index(ent.start_char, sentence_starts)
            sent = sentences[sent_idx]

            sent_start_char = sentence_starts[sent_idx]
            local_start = ent.start_char - sent_start_char
            local_end = ent.end_char - sent_start_char

            # don't overwrite anything that might already exist at the same span
            if (local_start, local_end) not in sent.entities:
                sent.entities[(local_start, local_end)] = Entity(
                    name=ent.text,
                    ref=kb_id,
                    ref_short=ent.text,
                    label = ent.label_ ,
                    sentence_id=f"s{sent_idx}",
                )
<<<<<<< HEAD
    def generate_normalised_text(self,sentences):
        for i, s in enumerate(sentences):
            original = s.text
            spans = sorted(s.entities.items(), key=lambda kv: kv[0][0])
            new_parts = []
            cursor = 0
            for (start, end), ent in spans:
                if start >= end or start < 0 or end > len(original):
                    continue
                new_parts.append(original[cursor:start])
                new_parts.append(ent.ref_short)  
                cursor = end                     
            new_parts.append(original[cursor:])
            s.text = "".join(new_parts)
            print(f"\nSentence {i}: {s.text}")
            for (start, end), ent in s.entities.items():
                print(f"  [{start}:{end}] {ent.name} {ent.label}  ref={ent.ref_short}")
                print(s.text)

=======
>>>>>>> 4eec9490a922b26d8df35968d1d7e431807dd75a




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

