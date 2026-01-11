"""
Modifier node - applies graph modifications based on user requests.
"""

from typing import TYPE_CHECKING, List, Dict, Any
from tools.graph.langgraph.state import GraphValidatorState, get_last_message, consume_agent
from tools.graph.langgraph.message import Message, MessageRole
from tools.graph.langgraph.prompts import get_registry
from tools.helper.json_helper import JsonHelper
from tools.graph.langgraph.nodes.chat_changes_information import ChatChangesInformation
from tools.graph.constants_graph import (
    AGENT_MODIFIER,
    STATE_MESSAGES,
    STATE_CURRENT_QUESTION,
    STATE_CHAT_CHANGES_INFORMATION,
    STATE_CHANGES_SUMMARY,
    STATE_AGENT_QUEUE,
    STATE_PLAN
)

if TYPE_CHECKING:
    from tools.graph.langgraph.validator import GraphValidatorLangGraph


def modifier_node(validator: "GraphValidatorLangGraph", state: "GraphValidatorState") -> "GraphValidatorState":
    """Modification agent - identifies and applies graph modifications."""
    messages = state.get(STATE_MESSAGES, [])
    question = state.get(STATE_CURRENT_QUESTION)
    current_plan = state.get(STATE_PLAN)
    
    # Get full conversation history (last 5 messages for context)
    conversation_history = ""
    for msg in messages[-5:]:
        role = msg.role if isinstance(msg, Message) else msg.get("role")
        content = msg.content if isinstance(msg, Message) else msg.get("content", "")
        if isinstance(role, MessageRole):
            role = role.value
        # Show full content - don't truncate for better context
        conversation_history += f"{role.upper()}: {content}\n"
    
    user_msg_obj = get_last_message(messages, MessageRole.USER)
    last_bot_msg_obj = get_last_message(messages, MessageRole.BOT)
    
    user_message = user_msg_obj.content if user_msg_obj else ""
    last_bot_message = last_bot_msg_obj.content if last_bot_msg_obj else ""

    # Format id_to_name mapping for the prompt (show first 30 entries)
    id_to_name_mapping = ""
    if validator.id_to_name:
        mapping_items = []
        for i, (eid, name) in enumerate(validator.id_to_name.items()):
            if i < 30:
                mapping_items.append(f"  '{name}' -> {eid}")
            else:
                break
        id_to_name_mapping = "\n".join(mapping_items)
        if len(validator.id_to_name) > 30:
            id_to_name_mapping += f"\n  ... (and {len(validator.id_to_name) - 30} more mappings)"

    registry = get_registry()
    prompt = registry.build_prompt(
        AGENT_MODIFIER,
        plan=current_plan,
        user_message=user_message,
        last_bot_message=last_bot_message,
        current_question=question.text if question else "N/A",
        conversation_history=conversation_history,
        id_to_name=validator.id_to_name,
        id_to_name_mapping=id_to_name_mapping
    )
    
    response = validator.api_repo.chat(prompt)
    action_data = JsonHelper.parse_json(str(response))
    
    # Actually apply changes using ModifierActions
    from tools.graph.langgraph.helpers.modifier_actions import ModifierActions
    
    modifier_actions = ModifierActions()
    changes_summary = []
    
    if isinstance(action_data, list):
        for item in action_data:
            action_type = item.get("type", "")
            params = item.get("parameters", {})
            if action_type:
                modifier_actions.apply(
                    action_type,
                    params,
                    validator.triples,
                    validator.id_to_name,
                    changes_summary,
                    graph=validator.graph
                )
    elif isinstance(action_data, dict):
        action_type = action_data.get("type", "")
        params = action_data.get("parameters", {})
        if action_type:
            modifier_actions.apply(
                action_type,
                params,
                validator.triples,
                validator.id_to_name,
                changes_summary,
                graph=validator.graph
            )
    
    # Consume current agent from queue
    updated_state = consume_agent(state, AGENT_MODIFIER)
    
    return {
        **updated_state,
        STATE_CHANGES_SUMMARY: state.get(STATE_CHANGES_SUMMARY, []) + changes_summary,
    }
