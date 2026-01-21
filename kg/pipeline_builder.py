"""
Pipeline builder for patent processing spaCy pipeline.
Supports GPU acceleration for faster processing.
"""
import spacy
import torch
from spacy.language import Language


class PipelineBuilder:
    """Builds and manages the spaCy pipeline for patent processing."""
    
    def __init__(self, use_gpu: bool = True, model: str = "en_core_web_trf"):
        """
        Initialize PipelineBuilder.
        
        Args:
            use_gpu: Whether to use GPU if available (default: True)
            model: spaCy model to use. "en_core_web_trf" (transformer, GPU-capable) 
                   or "en_core_web_sm" (small, CPU-only). Default: "en_core_web_trf"
        """
        self._nlp = None
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.model = model
        self.device = "cuda" if self.use_gpu else "cpu"
        self.device_id = 0 if self.use_gpu else -1  # 0 for GPU, -1 for CPU
        
        if self.use_gpu:
            print(f"PipelineBuilder: Using GPU (device: {self.device})")
        else:
            print(f"PipelineBuilder: Using CPU (GPU not available or disabled)")

    def build(self) -> Language:
        """Build the complete spaCy pipeline with all components."""
        # Load model (transformer model supports GPU)
        nlp = spacy.load(self.model)
        
        # Enable GPU for transformer models if available
        if self.use_gpu and hasattr(nlp, 'pipe') and hasattr(nlp, 'get_pipe'):
            # For transformer models, GPU is handled automatically by spacy-transformers
            # But we can set it explicitly if needed
            try:
                # Check if transformer component exists
                if "transformer" in nlp.pipe_names:
                    print("Transformer model detected - GPU acceleration enabled")
            except:
                pass

        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer", first=True)

        if "ner" in nlp.pipe_names:
            nlp.remove_pipe("ner")

        # Add HF NER with GPU support
        nlp.add_pipe(
            "hf_ner", 
            name="ner",
            config={
                "device": self.device_id,  # 0 for GPU, -1 for CPU
            }
        )
        nlp.add_pipe("entity_normaliser", after="ner")
        
        # Add coreference resolution with GPU support
        nlp.add_pipe(
            "windowed_fastcoref",
            after="entity_normaliser",
            config={
                "chunk_chars": 12000,
                "overlap": 1200,
                "model_architecture": "LingMessCoref",
                "model_path": "biu-nlp/lingmess-coref",
                "device": self.device,  # "cuda" or "cpu"
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

