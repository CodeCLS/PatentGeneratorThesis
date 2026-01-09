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
    MESSAGE_ROLE_USER,
    MESSAGE_ROLE_BOT,
    KEY_ROLE,
    KEY_CONTENT,
    KEY_ACTION,
    KEY_PARAMETERS,
    KEY_RELATED_TRIPLES,
    KEY_SEARCH_RESULTS,
    KEY_ERROR,
    KEY_NAME,
    ACTION_GET_ENTITY_INFO,
    ACTION_GET_TRIPLE_INFO,
    ACTION_GET_RELATED_TRIPLES,
    ACTION_SEARCH_ENTITIES,
    KEY_RETRIEVED_INFO_MARKER,
    DEFAULT_REASON,
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
            if (msg.role == MessageRole.USER or (isinstance(msg.role, str) and msg.role == MESSAGE_ROLE_USER)) and user_message is None:
                user_message = msg.content
            elif (msg.role == MessageRole.BOT or (isinstance(msg.role, str) and msg.role == MESSAGE_ROLE_BOT)) and last_bot_message is None:
                last_bot_message = msg.content
        elif isinstance(msg, dict):
            if msg.get(KEY_ROLE) == MESSAGE_ROLE_USER and user_message is None:
                user_message = msg.get(KEY_CONTENT, "")
            elif msg.get(KEY_ROLE) == MESSAGE_ROLE_BOT and last_bot_message is None:
                last_bot_message = msg.get(KEY_CONTENT, "")
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
        action_data = {KEY_ACTION: ACTION_GET_ENTITY_INFO, KEY_PARAMETERS: {}, "reason": DEFAULT_REASON}
    
    action = action_data.get(KEY_ACTION, ACTION_GET_ENTITY_INFO)
    params = action_data.get(KEY_PARAMETERS, {})
    
    if action == ACTION_GET_ENTITY_INFO:
        entity_info = validator.tools.get_entity_info(params.get(KEY_NAME, ""))
        info = entity_info.to_dict() if hasattr(entity_info, 'to_dict') else entity_info
    elif action == ACTION_GET_TRIPLE_INFO:
        triple_info = validator.tools.get_triple_info(params.get(KEY_INDEX, -1))
        info = triple_info.to_dict() if hasattr(triple_info, 'to_dict') else triple_info
    elif action == ACTION_GET_RELATED_TRIPLES:
        related_triples = validator.tools.get_related_triples(params.get(KEY_NAME, ""))
        info = {KEY_RELATED_TRIPLES: [t.to_dict() if hasattr(t, 'to_dict') else t for t in related_triples]}
    elif action == ACTION_SEARCH_ENTITIES:
        info = {KEY_SEARCH_RESULTS: validator.tools.search_entities(params.get("query", ""))}
    else:
        info = {KEY_ERROR: f"Unknown action: {action}"}
    
    info_text = json.dumps(info, indent=2)
    retrieval_message = f"{KEY_RETRIEVED_INFO_MARKER}\n{info_text}\n\nReason: {action_data.get('reason', DEFAULT_N_A)}"
    
    return {
        **state,
        STATE_MESSAGES: messages + [Message(role=MessageRole.SYSTEM, content=retrieval_message)],
        STATE_NEXT_AGENT: AGENT_COMMUNICATOR,
    }
