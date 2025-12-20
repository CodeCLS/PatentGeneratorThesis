import hashlib
from typing import List, Optional, Dict, Tuple, Any
import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span, Token
from fastcoref import FCoref
# Assuming 'tools.sentence.entity.Entity' is available as described
from tools.sentence.entity import Entity 
from spacy.language import Language
from spacy.tokens import Doc, Span
from transformers import pipeline
from spacy.language import Language
from spacy.tokens import Span

from fastcoref import FCoref, LingMessCoref
from spacy.language import Language
from spacy.tokens import Doc
import spacy

# --- 1. Custom Extension Setup ---

# Ensure all custom attributes (extensions) are set on Span and Doc objects.
# These are essential for storing and passing information between pipeline components.

if not Span.has_extension("norm_label"):
    Span.set_extension("norm_label", default=None)  # Canonicalized string of the entity
if not Span.has_extension("kb_id"):
    Span.set_extension("kb_id", default=None)     # Locally linked Knowledge Base ID
if not Doc.has_extension("coref_clusters"):
    Doc.set_extension("coref_clusters", default=None) # List of lists of coreferent Spans
if not Doc.has_extension("triples"):
    Doc.set_extension("triples", default=None)       # Placeholder for future relation triples

# --- 2. spaCy Pipeline Components ---
# These classes are registered as spaCy pipeline factories.

# --- extensions as before ---


class EntityNormaliser:
    def __init__(self, nlp: Language, name: str, mode: str = "lemma"):
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


@Language.factory(
    "entity_normaliser",
    default_config={"mode": "lemma"},
)
def make_entity_normaliser(nlp: Language, name: str, mode: str):
    return EntityNormaliser(nlp, name, mode=mode)

from spacy.language import Language
from transformers import pipeline

@Language.factory("hf_ner")
class HFNER:
    def __init__(self, nlp, name):
        self.ner = pipeline(
            "ner",
            model="training/ner/done/hf/ner_model",
            aggregation_strategy="simple",
        )

    def __call__(self, doc):
        candidates = []
        outputs = self.ner(doc.text)

        # 1. Create candidate spans
        for o in outputs:
            span = doc.char_span(
                o["start"],
                o["end"],
                label=o["entity_group"],
                alignment_mode="contract",
            )
            if span is not None:
                candidates.append(span)

        # 2. Sort spans by length (longest first)
        candidates.sort(key=lambda s: (s.end_char - s.start_char), reverse=True)

        # 3. Keep largest non-overlapping spans
        selected = []
        occupied = set()

        for span in candidates:
            span_range = set(range(span.start, span.end))
            if not span_range & occupied:
                selected.append(span)
                occupied.update(span_range)

        # 4. Assign entities
        doc.ents = tuple(selected)
        return doc


@Language.factory(
    "fastcoref_resolver",
    default_config={
        "device": "auto",
        "model_architecture": "FCoref",
        "model_path": None,
    },
)
class CorefResolver:
    def __init__(self, nlp, name, device="auto", model_architecture="FCoref", model_path=None):
        if device == "auto":
            device = "cuda" if spacy.prefer_gpu() else "cpu"

        if model_architecture == "LingMessCoref":
            self.coref = LingMessCoref(device=device, model_name_or_path=model_path or "biu-nlp/lingmess-coref")
        else:
            self.coref = FCoref(device=device, model_name_or_path=model_path or "biu-nlp/f-coref")

    def __call__(self, doc: Doc) -> Doc:
        predictions = self.coref.predict(texts=[doc.text])
        pred = predictions[0]

        clusters: list[list[Span]] = []
        for cluster in pred.get("clusters", []):
            spans = [doc[start:end+1] for start, end in cluster]
            if spans:
                clusters.append(spans)

        doc._.coref_clusters = clusters
        return doc




@Language.factory(
    "local_entity_linker",
    default_config={"hash_len": 6},
)
class LocalEntityLinker:
    """
    Assigns a unique, stable Knowledge Base ID (kb_id) to all entities 
    (NER and coreferent spans).
    """
    def __init__(self, nlp: Language, name: str, hash_len: int = 6):
        self.hash_len = hash_len

    def _get_canonical_string(self, span: Span) -> str:
        """Helper to get a cleaned, canonical string for ID generation."""
        # Prioritize pre-calculated norm_label, fallback to lemma/text
        base: str = (span._.norm_label or span.root.lemma_ or span.text).lower().strip()
        # Normalize whitespace (e.g., replace multiple spaces with one)
        return " ".join(base.split())

    def _generate_kb_id(self, canon: str, span: Span) -> str:
        """Generates a stable, unique ID based on canonical string and sentence context."""
        # Context includes sentence bounds to differentiate entities with same name in different sentences
        ctx: str = f"{canon}_{span.sent.start}_{span.sent.end}"
        # Use MD5 hash for a stable, short suffix
        suffix: str = hashlib.md5(ctx.encode()).hexdigest()[: self.hash_len]
        # Clean canonical string for a safe ID part
        safe: str = canon.replace(" ", "_")
        return f"ent::{safe}::{suffix}"

    def __call__(self, doc: Doc) -> Doc:
        # Maps canonical string (e.g., "john smith") to its assigned KB ID
        canon2id: Dict[str, str] = {}

        # 1) Assign IDs to NER entities
        for ent in doc.ents:
            canon: str = self._get_canonical_string(ent)
            kb_id: Optional[str] = canon2id.get(canon)
            
            if kb_id is None:
                # First time seeing this canonical form; generate and store new ID
                kb_id = self._generate_kb_id(canon, ent)
                canon2id[canon] = kb_id
            
            # Store canon and ID back on the Span extension
            ent._.norm_label = canon
            ent._.kb_id = kb_id

        # 2) Propagate IDs and canonical labels into coref clusters (including pronouns)
        clusters: List[List[Span]] = doc._.coref_clusters or []
        for cluster in clusters:
            # Determine the 'source' span to represent the entire cluster
            source: Span
            
            # a) Prefer a span that is fully or partially covered by a NER entity
            covering_ents = [e for e in doc.ents if any(e.start <= sp.start < e.end for sp in cluster)]
            if covering_ents:
                 # Use the first NER entity that covers any cluster span
                source = covering_ents[0]
            else:
                # b) Fall back to the longest span in the cluster (e.g., for non-NER names/phrases)
                source = max(cluster, key=lambda s: len(s))

            # Ensure the source span has an ID and canonical label
            canon = self._get_canonical_string(source)
            kb_id = canon2id.get(canon)
            if kb_id is None:
                kb_id = self._generate_kb_id(canon, source)
                canon2id[canon] = kb_id

            # Apply the ID and canonical label to ALL spans in the cluster
            for sp in cluster:
                sp._.norm_label = canon
                sp._.kb_id = kb_id

        return doc

# --- 3. Pipeline Orchestration ---

class PipelineBuilder:
    def __init__(self):
        self._nlp: Optional[Language] = None

    def _build_nlp(self) -> Language:
        nlp = spacy.load("en_core_web_trf")

        if "ner" in nlp.pipe_names:
            nlp.remove_pipe("ner")

        nlp.add_pipe("hf_ner", name="ner")              # <-- FIXED
        nlp.add_pipe("entity_normaliser", after="ner")
        nlp.add_pipe("fastcoref_resolver", config={"model_architecture": "LingMessCoref"})
        nlp.add_pipe("local_entity_linker")

        self._nlp = nlp
        return nlp

    def nlp(self, text: str) -> Doc:
        """Processes the input text using the built pipeline."""
        if self._nlp is None:
            self._nlp = self._build_nlp()
            
        return self._nlp(text)

# --- 4. Post-Processing and Entity Mapping ---

class EntityMapper:
    """
    Handles the post-processing of a spaCy Doc, mapping its entities 
    and coreference clusters back to a custom list of Sentence objects.
    """
    def __init__(self, sentence_cls: Any):
        """
        Initializes the mapper.
        :param sentence_cls: The custom Sentence class (e.g., from tools.sentence.entity)
        """
        self.Sentence = sentence_cls

    def _index_sentences(self, sentences: List[Any]) -> List[int]:
        """Calculates the global character start index for each sentence."""
        sentence_starts: List[int] = []
        offset: int = 0
        for s in sentences:
            sentence_starts.append(offset)
            # Add length of sentence text + 1 for a single space/separator, 
            # though here it assumes a continuous string concatenation length
            offset += len(s.text) 
        return sentence_starts

    def _find_sentence_index(self, char_pos: int, sentence_starts: List[int]) -> int:
        """
        Given a global character position, returns the index of the sentence
        it belongs to (binary search would be faster, but linear search is fine for typical documents).
        """
        for i, start in enumerate(sentence_starts):
            # Check if this is the last sentence or if the position is before the next sentence's start
            if i == len(sentence_starts) - 1 or char_pos < sentence_starts[i+1]:
                return i
        return len(sentence_starts) - 1 # Should handle the very end of the document

    def map_to_sentences(self, doc: Doc, sentences: List[Any]) -> List[List[Span]]:
        """
        Main function to map spaCy Doc information (NER and coref) to custom Sentence objects.
        Returns the list of coref clusters for use in other components.
        """
        sentence_starts = self._index_sentences(sentences)
        
        # 1. Map Coreference Spans
        clusters = self._fill_coref_entities(doc, sentence_starts, sentences)
        cluster_spans = self._get_cluster_spans(clusters)
        
        # 2. Map Remaining NER Entities
        self._add_entities_from_ner(doc, sentence_starts, sentences, cluster_spans)
        
        # 3. Normalize Entity References
        self._normalise_other_refs_to_first(sentences)
        
        # 4. Generate Text (Optional: only for debugging/display)
        self._generate_normalised_text(sentences)

        return clusters

    def _fill_coref_entities(self, doc: Doc, sentence_starts: List[int], sentences: List[Any]) -> List[List[Span]]:
        """Populates entities from coreference clusters."""
        clusters: List[List[Span]] = doc._.coref_clusters or []
        
        for cl in clusters:
            # KB ID is guaranteed to be set by LocalEntityLinker on all spans in a cluster
            kb_id: Optional[str] = cl[0]._.kb_id 

            for sp in cl:
                # 1) Find which Sentence this span belongs to
                sent_idx: int = self._find_sentence_index(sp.start_char, sentence_starts)
                sent: Any = sentences[sent_idx]

                # 2) Compute local char offsets inside that sentence
                sent_start_char: int = sentence_starts[sent_idx]
                local_start: int = sp.start_char - sent_start_char
                local_end: int = sp.end_char - sent_start_char

                # 3) Store Entity in that sentence's entities dict
                # Coreferent spans are typically labeled as a generic REFERENCE type
                sent.entities[(local_start, local_end)] = Entity(
                    name=sp.text,
                    ref=kb_id,
                    ref_short=kb_id[-4:].upper(),  # Initial short reference from hash suffix
                    label=Entity.REFERENCE,        # Assuming Entity.REFERENCE is defined
                    sentence_id=f"s{sent_idx}",
                )
        return clusters

    def _get_cluster_spans(self, clusters: List[List[Span]]) -> set[Tuple[int, int]]:
        """Returns a set of (start_char, end_char) for all coreferent spans."""
        return {
            (sp.start_char, sp.end_char)
            for cl in clusters
            for sp in cl
        }

    def _add_entities_from_ner(self, doc: Doc, sentence_starts: List[int], sentences: List[Any], cluster_spans: set[Tuple[int, int]]):
        """Populates entities from NER that were *not* part of a coreference cluster."""
        for ent in doc.ents:
            # Skip entities already mapped via coreference
            if (ent.start_char, ent.end_char) in cluster_spans:
                continue

            kb_id: Optional[str] = ent._.kb_id 

            sent_idx: int = self._find_sentence_index(ent.start_char, sentence_starts)
            sent: Any = sentences[sent_idx]

            sent_start_char: int = sentence_starts[sent_idx]
            local_start: int = ent.start_char - sent_start_char
            local_end: int = ent.end_char - sent_start_char

            # Only add if the span is not already occupied (ensures no accidental overwrite)
            if (local_start, local_end) not in sent.entities:
                sent.entities[(local_start, local_end)] = Entity(
                    name=ent.text,
                    ref=kb_id,
                    ref_short=ent.text,  #
                    label=ent.label_, 
                    sentence_id=f"s{sent_idx}",
                )

    def _normalise_other_refs_to_first(self, sentences: List[Any]):
        """
        Replaces the 'ref_short' for all entities in a cluster with the name 
        of the first entity encountered for that cluster (KB ID).
        """
        canonical: Dict[str, str] = {}
        # First pass: find the canonical name for each KB ID
        for s in sentences:
            # Process entities in left-to-right order to find the first instance
            for _, ent in sorted(s.entities.items(), key=lambda kv: kv[0][0]):
                if ent.ref is None or ent.ref in canonical:
                    continue
                # First time we see this ref -> freeze its name as canonical
                canonical[ent.ref] = ent.name
        
        # Second pass: apply the canonical name
        for s in sentences:
            for ent in s.entities.values():
                if ent.ref in canonical:
                    # Overwrite the hash suffix with the canonical name
                    ent.ref_short = canonical[ent.ref] 

    def _generate_normalised_text(self, sentences: List[Any]):
        """Reconstructs and prints the sentence text, replacing entity spans with ref_short."""
        for i, s in enumerate(sentences):
            original: str = s.text
            # Sort spans by start position
            spans = sorted(s.entities.items(), key=lambda kv: kv[0][0])
            new_parts: List[str] = []
            cursor: int = 0
            
            for (start, end), ent in spans:
                # Basic bounds check
                if start >= end or start < 0 or end > len(original):
                    continue
                
                # Add text between cursor and current span start
                new_parts.append(original[cursor:start])
                # Add the normalized entity reference
                new_parts.append(ent.ref_short)  
                # Advance cursor past the entity
                cursor = end 
            
            # Add remaining text after the last entity
            new_parts.append(original[cursor:])
            s.text = "".join(new_parts)
            
            # Debug/Display output (as in original code)
            print(f"\nSentence {i}: {s.text}")
            for (start, end), ent in s.entities.items():
                print(f"  [{start}:{end}] {ent.name} {ent.label}  ref={ent.ref_short}")