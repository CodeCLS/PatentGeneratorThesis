"""
Helper functions for LangGraph validator.
"""

from tools.sentence.entity import Entity
from tools.graph.Triple import Triple


def extract_text_from_response(response) -> str:
    """Extract text from LLM response (handles various response formats)."""
    if isinstance(response, str):
        return response
    if hasattr(response, "content"):
        return str(response.content)
    if hasattr(response, "text"):
        return str(response.text)
    return str(response)


def get_entity_id(entity: Entity) -> str:
    """Get entity ID (ref or id)."""
    return entity.ref or entity.id or entity.ref_short or ""


def get_entity_name(entity: Entity) -> str:
    """Get entity name."""
    return entity.name


def get_triple_head_id(triple: Triple) -> str:
    """Get head entity ID from triple."""
    return get_entity_id(triple.head)


def get_triple_tail_id(triple: Triple) -> str:
    """Get tail entity ID from triple."""
    return get_entity_id(triple.tail)


def get_triple_head_name(triple: Triple) -> str:
    """Get head entity name from triple."""
    return get_entity_name(triple.head)


def get_triple_tail_name(triple: Triple) -> str:
    """Get tail entity name from triple."""
    return get_entity_name(triple.tail)
