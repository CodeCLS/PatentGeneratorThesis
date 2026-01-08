"""
Retriever node - fetches detailed information about entities and triples.
"""

import json
from typing import TYPE_CHECKING, Union
from tools.helper.json_helper import JsonHelper
from tools.graph.langgraph.state import GraphValidatorState
from tools.graph.langgraph.message import Message, MessageRole
from tools.graph.langgraph.prompts import get_registry
from tools.graph.constants_graph import (
    AGENT_COMMUNICATOR,
    AGENT_RETRIEVER,
    STATE_MESSAGES,
    STATE_NEXT_AGENT,
)

if TYPE_CHECKING:
    from tools.graph.langgraph.validator import GraphValidatorLangGraph


def retriever_node(validator: "GraphValidatorLangGraph", state: "GraphValidatorState") -> "GraphValidatorState":
    """Retrieval agent - fetches detailed information about entities and triples."""
    messages = state.get(STATE_MESSAGES, [])
    
    # Get the last user message (most important for understanding the request)
    user_message = None
    last_bot_message = None
    for msg in reversed(messages):
        # Handle both Message objects and dicts
        if isinstance(msg, Message):
            if (msg.role == MessageRole.USER or (isinstance(msg.role, str) and msg.role == "user")) and user_message is None:
                user_message = msg.content
            elif (msg.role == MessageRole.BOT or (isinstance(msg.role, str) and msg.role == "bot")) and last_bot_message is None:
                last_bot_message = msg.content
        elif isinstance(msg, dict):
            if msg.get("role") == "user" and user_message is None:
                user_message = msg.get("content", "")
            elif msg.get("role") == "bot" and last_bot_message is None:
                last_bot_message = msg.get("content", "")
        if user_message and last_bot_message:
            break
    
    registry = get_registry()
    prompt = registry.build_prompt(
        AGENT_RETRIEVER,
        user_message=user_message or "",
        last_bot_message=last_bot_message or ""
    )
    
    response = validator.api_repo.chat(prompt)
    action_data = JsonHelper.parse_json(str(response))
    if not action_data:
        action_data = {"action": "get_entity_info", "parameters": {}, "reason": "Default"}
    
    action = action_data.get("action", "get_entity_info")
    params = action_data.get("parameters", {})
    
    if action == "get_entity_info":
        entity_info = validator.tools.get_entity_info(params.get("entity_name", ""))
        info = entity_info.to_dict() if hasattr(entity_info, 'to_dict') else entity_info
    elif action == "get_triple_info":
        triple_info = validator.tools.get_triple_info(params.get("triple_index", -1))
        info = triple_info.to_dict() if hasattr(triple_info, 'to_dict') else triple_info
    elif action == "get_related_triples":
        related_triples = validator.tools.get_related_triples(params.get("entity_name", ""))
        info = {"related_triples": [t.to_dict() if hasattr(t, 'to_dict') else t for t in related_triples]}
    elif action == "search_entities":
        info = {"search_results": validator.tools.search_entities(params.get("query", ""))}
    else:
        info = {"error": f"Unknown action: {action}"}
    
    info_text = json.dumps(info, indent=2)
    retrieval_message = f"[Retrieved Information]\n{info_text}\n\nReason: {action_data.get('reason', 'N/A')}"
    
    return {
        **state,
        STATE_MESSAGES: messages + [Message(role=MessageRole.SYSTEM, content=retrieval_message)],
        STATE_NEXT_AGENT: AGENT_COMMUNICATOR,
    }
