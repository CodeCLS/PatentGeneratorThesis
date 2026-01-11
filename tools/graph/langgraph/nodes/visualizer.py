"""
Visualizer node - decides which widgets to show and with what data.
"""

from typing import TYPE_CHECKING, Union
from tools.helper.json_helper import JsonHelper
from tools.graph.langgraph.state import GraphValidatorState, consume_agent
from tools.graph.langgraph.prompts import get_registry
from tools.graph.langgraph.nodes.chat_visual_info import ChatVisualInfo
from tools.graph.constants_graph import (
    AGENT_COMMUNICATOR,
    AGENT_VISUALIZER,
    STATE_MESSAGES,
    STATE_CURRENT_QUESTION,
    STATE_DISPLAY_ACTIONS,
    STATE_AGENT_QUEUE,
    STATE_PLAN
)

if TYPE_CHECKING:
    from tools.graph.langgraph.validator import GraphValidatorLangGraph


def visualizer_node(validator: "GraphValidatorLangGraph", state: "GraphValidatorState") -> "GraphValidatorState":
    """Visualization agent - decides which widgets to show."""
    messages = state.get(STATE_MESSAGES, [])
    current_question = state.get(STATE_CURRENT_QUESTION)
    current_plan = state.get(STATE_PLAN)
    
    registry = get_registry()
    prompt = registry.build_prompt(
        AGENT_VISUALIZER,
        plan=current_plan,
        current_question=current_question.text if current_question else "N/A",
        recent_conversation=str(messages[-3:]) if messages else "No conversation yet"
    )
    
    response = validator.api_repo.chat(prompt)
    widget_data = JsonHelper.parse_json(str(response))
    
    visual_info = ChatVisualInfo.from_dict(widget_data or {})
    
    # Update display actions list
    new_display_actions = state.get(STATE_DISPLAY_ACTIONS, [])
    if visual_info.show_widget:
        new_display_actions = new_display_actions + [visual_info]
    
    # Use consume_agent helper which returns updated state
    updated_state = consume_agent(state, AGENT_VISUALIZER)
    
    return {
        **updated_state,
        STATE_DISPLAY_ACTIONS: new_display_actions,
    }
