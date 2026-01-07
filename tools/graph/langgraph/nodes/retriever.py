"""
Retriever node - fetches detailed information about entities and triples.
"""

import json
from typing import TYPE_CHECKING
from tools.helper.json_helper import JsonHelper
from tools.graph.langgraph.state import GraphValidatorState
from tools.graph.constants_graph import (
    AGENT_COMMUNICATOR,
    STATE_MESSAGES,
    STATE_NEXT_AGENT,
)

if TYPE_CHECKING:
    from tools.graph.langgraph.validator import GraphValidatorLangGraph


def retriever_node(validator: "GraphValidatorLangGraph", state: "GraphValidatorState") -> "GraphValidatorState":
    """Retrieval agent - fetches detailed information about entities and triples."""
    messages = state.get(STATE_MESSAGES, [])
    last_bot_message = None
    for msg in reversed(messages):
        if msg.get("role") == "bot":
            last_bot_message = msg.get("content", "")
            break
    
    prompt = (
        "You are a retrieval agent. Identify what information needs to be retrieved.\n\n"
        f"Last bot message: {last_bot_message}\n\n"
        "Return JSON:\n"
        '{"action": "get_entity_info|get_triple_info|get_related_triples|search_entities", '
        '"parameters": {"entity_name": "...", "triple_index": 0}, '
        '"reason": "Why this information is needed"}\n'
    )
    
    response = validator.api_repo.chat(prompt)
    action_data = JsonHelper.parse_json(str(response))
    if not action_data:
        action_data = {"action": "get_entity_info", "parameters": {}, "reason": "Default"}
    
    action = action_data.get("action", "get_entity_info")
    params = action_data.get("parameters", {})
    
    if action == "get_entity_info":
        info = validator.tools.get_entity_info(params.get("entity_name", ""))
    elif action == "get_triple_info":
        info = validator.tools.get_triple_info(params.get("triple_index", -1))
    elif action == "get_related_triples":
        info = {"related_triples": validator.tools.get_related_triples(params.get("entity_name", ""))}
    elif action == "search_entities":
        info = {"search_results": validator.tools.search_entities(params.get("query", ""))}
    else:
        info = {"error": f"Unknown action: {action}"}
    
    info_text = json.dumps(info, indent=2)
    retrieval_message = f"[Retrieved Information]\n{info_text}\n\nReason: {action_data.get('reason', 'N/A')}"
    
    return {
        **state,
        STATE_MESSAGES: messages + [{"role": "system", "content": retrieval_message}],
        STATE_NEXT_AGENT: AGENT_COMMUNICATOR,
    }
