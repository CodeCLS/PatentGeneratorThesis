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
    user_msg_obj = get_last_message(messages, MessageRole.USER)
    last_bot_msg_obj = get_last_message(messages, MessageRole.BOT)
    
    user_message = user_msg_obj.content if user_msg_obj else ""
    last_bot_message = last_bot_msg_obj.content if last_bot_msg_obj else ""

    registry = get_registry()
    prompt = registry.build_prompt(
        AGENT_MODIFIER,
        plan=current_plan,
        user_message=user_message,
        last_bot_message=last_bot_message,
        current_question=question.text if question else "N/A"
    )
    
    response = validator.api_repo.chat(prompt)
    action_data = JsonHelper.parse_json(str(response))
    
    new_changes = []
    if isinstance(action_data, list):
        for item in action_data:
            new_changes.append(ChatChangesInformation.from_dict(item))
    elif isinstance(action_data, dict):
        new_changes.append(ChatChangesInformation.from_dict(action_data))

    changes_summary = []
    # Note: In a real implementation, you'd call validator.tools methods here
    # to actually apply the changes described in new_changes.
    # For now, we'll just record that changes were "processed".
    
    if new_changes:
        for change in new_changes:
            # Placeholder for actual tool application logic
            if change.added_triples:
                changes_summary.append(f"Added {len(change.added_triples)} triples")
            if change.deleted_triples:
                changes_summary.append(f"Deleted {len(change.deleted_triples)} triples")
            if change.merged_entities:
                changes_summary.append(f"Merged {len(change.merged_entities)} entities")
            if change.renamed_entities:
                changes_summary.append(f"Renamed {len(change.renamed_entities)} entities")
            if change.modified_triples:
                changes_summary.append(f"Modified {len(change.modified_triples)} triples")

    # Update validator instance data (if tools applied changes)
    # validator.triples = updated_triples
    # ...
    
    # Consume current agent from queue
    updated_state = consume_agent(state, AGENT_MODIFIER)
    
    return {
        **updated_state,
        STATE_CHAT_CHANGES_INFORMATION: state.get(STATE_CHAT_CHANGES_INFORMATION, []) + new_changes,
        STATE_CHANGES_SUMMARY: state.get(STATE_CHANGES_SUMMARY, []) + changes_summary,
    }
