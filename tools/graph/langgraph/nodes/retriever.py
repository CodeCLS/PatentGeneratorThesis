"""
Retriever node - fetches detailed information about entities and triples.
"""

import json
from typing import TYPE_CHECKING

from tools.helper.json_helper import JsonHelper

# Import GraphValidatorState at runtime (not just TYPE_CHECKING)
# This is needed because LangGraph might inspect type hints at runtime
from tools.graph.langgraph.state import GraphValidatorState

if TYPE_CHECKING:
    from tools.graph.langgraph.validator import GraphValidatorLangGraph


def retriever_node(validator: "GraphValidatorLangGraph", state: "GraphValidatorState") -> "GraphValidatorState":
    """
    Retrieval agent - fetches detailed information about entities and triples.
    """
    messages = state.get("messages", [])
    last_bot_message = None
    for msg in reversed(messages):
        if msg.get("role") == "bot":
            last_bot_message = msg.get("content", "")
            break
    
    # Parse what to retrieve from the last message
    prompt = (
        "You are a retrieval agent. Your job is to identify what information needs to be retrieved.\n\n"
        f"Last bot message: {last_bot_message}\n\n"
        "Based on the conversation, determine what needs to be retrieved:\n"
        "- Entity information (use get_entity_info)\n"
        "- Triple information (use get_triple_info)\n"
        "- Related triples (use get_related_triples)\n"
        "- Entity search (use search_entities)\n\n"
        "CRITICAL: Return ONLY valid JSON. No reasoning, no explanation, just the JSON object.\n\n"
        "Return JSON:\n"
        '{"action": "get_entity_info|get_triple_info|get_related_triples|search_entities", '
        '"parameters": {"entity_name": "...", "triple_index": 0, etc.}, '
        '"reason": "Why this information is needed"}\n'
    )
    
    response = validator.api_repo.chat(prompt)
    # Use JsonHelper for robust JSON parsing (handles fences and extraction automatically)
    action_data = JsonHelper.parse_json(str(response))
    if action_data is None:
        action_data = {"action": "get_entity_info", "parameters": {}, "reason": "Default"}
    
    # Execute retrieval
    action = action_data.get("action", "get_entity_info")
    params = action_data.get("parameters", {})
    
    if action == "get_entity_info":
        entity_name = params.get("entity_name", "")
        if entity_name:
            info = validator.tools.get_entity_info(entity_name)
        else:
            info = {"error": "No entity name provided"}
    elif action == "get_triple_info":
        triple_index = params.get("triple_index")
        # Handle None case - use -1 as default if key doesn't exist, but None if explicitly set to null
        if triple_index is None and "triple_index" not in params:
            triple_index = -1  # Key doesn't exist, use default
        # triple_index can still be None if JSON had "triple_index": null
        info = validator.tools.get_triple_info(triple_index)
    elif action == "get_related_triples":
        entity_name = params.get("entity_name", "")
        info = {"related_triples": validator.tools.get_related_triples(entity_name)}
    elif action == "search_entities":
        query = params.get("query", "")
        info = {"search_results": validator.tools.search_entities(query)}
    else:
        info = {"error": f"Unknown action: {action}"}
    
    # Format retrieved information for the communicator
    info_text = json.dumps(info, indent=2)
    retrieval_message = f"[Retrieved Information]\n{info_text}\n\nReason: {action_data.get('reason', 'N/A')}"
    
    return {
        **state,
        "messages": messages + [{"role": "system", "content": retrieval_message}],
        "next_agent": "communicator",  # Return to communicator with retrieved info
    }

