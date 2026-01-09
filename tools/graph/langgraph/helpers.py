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


def extract_retrieved_info(retrieved_info_text: str) -> dict | None:
    """
    Extract JSON data from a retrieved information message.
    
    Format: "[Retrieved Information]\n{json}\n\nReason: ..."
    
    Args:
        retrieved_info_text: The full retrieved information message text
        
    Returns:
        Parsed JSON data as dict, or None if parsing fails
    """
    from tools.helper.json_helper import JsonHelper
    from tools.graph.constants_graph import KEY_RETRIEVED_INFO_MARKER, KEY_REASON
    
    if not retrieved_info_text or KEY_RETRIEVED_INFO_MARKER not in retrieved_info_text:
        return None
    
    try:
        # Extract JSON from message (format: "[Retrieved Information]\n{json}\n\nReason: ...")
        lines = retrieved_info_text.split('\n')
        json_lines = []
        in_json = False
        
        for line in lines:
            if line.strip().startswith('{'):
                in_json = True
            if in_json:
                if line.strip().startswith(KEY_REASON):
                    break
                json_lines.append(line)
        
        if json_lines:
            json_text = '\n'.join(json_lines)
            return JsonHelper.parse_json(json_text)
    except Exception:
        pass
    
    return None


def process_retrieved_info_for_widget(retrieved_info_text: str) -> tuple[list | None, dict | None]:
    """
    Process retrieved information and determine if it should be shown as a widget.
    
    Args:
        retrieved_info_text: The full retrieved information message text
        
    Returns:
        Tuple of (triples_list, full_info_dict):
        - triples_list: List of triples if related_triples found, None otherwise
        - full_info_dict: Full parsed info dict, None if parsing fails
    """
    from tools.graph.constants_graph import KEY_RELATED_TRIPLES
    
    info_data = extract_retrieved_info(retrieved_info_text)
    
    if not info_data:
        return None, None
    
    # Check if we have related_triples
    if KEY_RELATED_TRIPLES in info_data:
        triples = info_data[KEY_RELATED_TRIPLES]
        if isinstance(triples, list):
            return triples, info_data
    
    return None, info_data