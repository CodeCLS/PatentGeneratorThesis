"""
Visualizer node - decides which widgets to show and with what data.
"""

from typing import TYPE_CHECKING, Union
from tools.helper.json_helper import JsonHelper
from tools.graph.langgraph.state import GraphValidatorState
from tools.graph.langgraph.message.widgets import Widget, EdgesWidget
from tools.graph.langgraph.prompts import get_registry
from tools.graph.constants_graph import (
    AGENT_COMMUNICATOR,
    AGENT_VISUALIZER,
    STATE_MESSAGES,
    STATE_CURRENT_QUESTION_TEXT,
    STATE_SHOW_WIDGET,
    STATE_WIDGET_TYPE,
    STATE_WIDGET_DATA,
    STATE_NEXT_AGENT,
    STATE_AGENT_QUEUE,
    STATE_INTERNAL_RETRIEVED_TRIPLES,
    WIDGET_TYPE_EDGES,
    KEY_TRIPLES,
)

if TYPE_CHECKING:
    from tools.graph.langgraph.validator import GraphValidatorLangGraph


def visualizer_node(validator: "GraphValidatorLangGraph", state: "GraphValidatorState") -> "GraphValidatorState":
    """Visualization agent - decides which widgets to show."""
    messages = state.get(STATE_MESSAGES, [])
    
    # Check if we have retrieved triples to show
    retrieved_triples = state.get(STATE_INTERNAL_RETRIEVED_TRIPLES)
    if retrieved_triples:
        # Automatically show edges_widget with the retrieved triples
        # Clear _retrieved_triples to prevent reprocessing
        new_state = {**state}
        new_state.pop(STATE_INTERNAL_RETRIEVED_TRIPLES, None)
        return {
            **new_state,
            STATE_SHOW_WIDGET: True,
            STATE_WIDGET_TYPE: WIDGET_TYPE_EDGES,
            STATE_WIDGET_DATA: {KEY_TRIPLES: retrieved_triples},
            STATE_NEXT_AGENT: AGENT_COMMUNICATOR,
        }
    
    # Otherwise, let LLM decide
    registry = get_registry()
    prompt = registry.build_prompt(
        AGENT_VISUALIZER,
        current_question=state.get(STATE_CURRENT_QUESTION_TEXT, 'N/A'),
        recent_conversation=str(messages[-3:]) if messages else "No conversation yet"
    )
    
    response = validator.api_repo.chat(prompt)
    widget_data = JsonHelper.parse_json(str(response))
    if not widget_data:
        widget_data = {"show_widget": False, "widget_type": None, "widget_data": {}}
    
    # Convert widget_data dict to Widget object if show_widget is True
    widget_obj = None
    if widget_data.get("show_widget", False):
        widget_type = widget_data.get("widget_type")
        widget_data_dict = widget_data.get("widget_data", {})
        if widget_type:
            widget_dict = {"widget_type": widget_type, "data": widget_data_dict}
            widget_obj = Widget.from_dict(widget_dict)
    
    # Consume current agent from queue if it's at the front
    agent_queue = state.get(STATE_AGENT_QUEUE, [])
    if agent_queue and agent_queue[0] == AGENT_VISUALIZER:
        agent_queue = agent_queue[1:]
    
    return {
        **state,
        STATE_SHOW_WIDGET: widget_data.get("show_widget", False),
        STATE_WIDGET_TYPE: widget_data.get("widget_type"),
        STATE_WIDGET_DATA: widget_obj.to_dict().get("data", {}) if widget_obj else widget_data.get("widget_data", {}),
        STATE_NEXT_AGENT: AGENT_COMMUNICATOR,
        STATE_AGENT_QUEUE: agent_queue,
    }
