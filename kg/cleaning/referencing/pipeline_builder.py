"""
Pipeline builder for patent processing spaCy pipeline.
"""
import spacy
from spacy.language import Language


class PipelineBuilder:
    """Builds and manages the spaCy pipeline for patent processing."""
    
    def __init__(self):
        self._nlp = None

    def build(self) -> Language:
        """Build the complete spaCy pipeline with all components."""
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
    def nlp(self) -> Language:
        """Lazy-load the pipeline if not already built."""
        if self._nlp is None:
            self._nlp = self.build()
        return self._nlp

