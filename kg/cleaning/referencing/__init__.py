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
from kg.cleaning.referencing import extensions  # noqa: F401

# Import components
from kg.cleaning.referencing.entity_normaliser import EntityNormaliser, make_entity_normaliser  # noqa: F401
from kg.cleaning.referencing.hf_ner import HFNER, make_hf_ner  # noqa: F401
from kg.cleaning.referencing.windowed_coref import make_windowed_fastcoref  # noqa: F401
from kg.cleaning.referencing.entity_linker import LocalEntityLinker, make_local_entity_linker  # noqa: F401
from kg.cleaning.referencing.pipeline_builder import PipelineBuilder  # noqa: F401
from kg.cleaning.referencing.entity_mapper import EntityMapper, JoinedText, join_sentences  # noqa: F401

__all__ = [
    "EntityNormaliser",
    "HFNER",
    "LocalEntityLinker",
    "PipelineBuilder",
    "EntityMapper",
    "JoinedText",
    "join_sentences",
]

