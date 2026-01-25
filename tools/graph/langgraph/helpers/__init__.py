"""
Helper modules for LangGraph validator.
"""

from typing import List, Union, Dict, Any, Optional, Tuple
from tools.sentence.entity import Entity
from tools.graph.data.Triple import Triple
from tools.graph.langgraph.helpers.modifier_actions import ModifierActions

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
    """
    from tools.helper.json_helper import JsonHelper
    from tools.graph.constants_graph import KEY_RETRIEVED_INFO_MARKER, KEY_REASON
    
    if not retrieved_info_text or KEY_RETRIEVED_INFO_MARKER not in retrieved_info_text:
        return None
    
    try:
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
    """
    from tools.graph.constants_graph import KEY_RELATED_TRIPLES
    
    info_data = extract_retrieved_info(retrieved_info_text)
    
    if not info_data:
        return None, None
    
    if KEY_RELATED_TRIPLES in info_data:
        triples = info_data[KEY_RELATED_TRIPLES]
        if isinstance(triples, list):
            return triples, info_data
    
    return None, info_data

def format_conversation_history(messages: List[Union[Dict[str, Any], Any]], limit: int = 10, include_system: bool = False) -> str:
    """
    Format conversation messages into a string for LLM prompts.
    Provides consistent, non-truncated history to all agents.
    """
    from tools.graph.langgraph.message import MessageRole
    
    history_lines = []
    
    target_messages = messages
    if not include_system:
        filtered = []
        for m in messages:
            role = m.get("role") if isinstance(m, dict) else getattr(m, "role", "user")
            if hasattr(role, "value"):
                role = role.value
            if str(role).lower() != "system":
                filtered.append(m)
        target_messages = filtered
    
    if limit > 0:
        target_messages = target_messages[-limit:]
    
    for msg in target_messages:
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
        else:
            role = getattr(msg, "role", "user")
            content = getattr(msg, "content", "")
            
        if hasattr(role, "value"):
            role_str = str(role.value)
        else:
            role_str = str(role)
            
        history_lines.append(f"{role_str.upper()}: {content}")
        
    return "\n".join(history_lines)

__all__ = [
    'ModifierActions',
    'extract_text_from_response',
    'get_entity_id',
    'get_entity_name',
    'get_triple_head_id',
    'get_triple_tail_id',
    'get_triple_head_name',
    'get_triple_tail_name',
    'extract_retrieved_info',
    'process_retrieved_info_for_widget',
    'format_conversation_history',
]
