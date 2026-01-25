"""
SpaCy pipeline components for patent processing.

This package provides:
- Entity normalization
- HuggingFace NER integration
- Windowed coreference resolution
- Local entity linking
- Pipeline builder
- Entity mapping utilities
"""

# Import extensions first to register them
from kg import extensions  # noqa: F401

# Import components
from kg.entity_normaliser import EntityNormaliser  # noqa: F401
from kg.hf_ner import HFNER  # noqa: F401
from kg.entity_linker import LocalEntityLinker  # noqa: F401
from kg.windowed_coref import make_windowed_fastcoref  # noqa: F401
from kg.pipeline_builder import PipelineBuilder  # noqa: F401
from kg.entity_mapper import EntityMapper, JoinedText, join_sentences  # noqa: F401

__all__ = [
    "EntityNormaliser",
    "HFNER",
    "LocalEntityLinker",
    "make_windowed_fastcoref",
    "PipelineBuilder",
    "EntityMapper",
    "JoinedText",
    "join_sentences",
]

