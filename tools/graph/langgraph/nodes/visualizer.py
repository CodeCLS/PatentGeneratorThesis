"""
Visualizer node - decides which widgets to show and with what data.
"""

from typing import TYPE_CHECKING
from tools.helper.json_helper import JsonHelper
from tools.graph.langgraph.state import GraphValidatorState
from tools.graph.constants_graph import (
    AGENT_COMMUNICATOR,
    STATE_MESSAGES,
    STATE_CURRENT_QUESTION_TEXT,
    STATE_SHOW_WIDGET,
    STATE_WIDGET_TYPE,
    STATE_WIDGET_DATA,
    STATE_NEXT_AGENT,
)

if TYPE_CHECKING:
    from tools.graph.langgraph.validator import GraphValidatorLangGraph


def visualizer_node(validator: "GraphValidatorLangGraph", state: "GraphValidatorState") -> "GraphValidatorState":
    """Visualization agent - decides which widgets to show."""
    messages = state.get(STATE_MESSAGES, [])
    
    prompt = (
        "You are a visualization agent. Decide what widgets to show.\n\n"
        f"Current question: {state.get(STATE_CURRENT_QUESTION_TEXT, 'N/A')}\n"
        f"Recent conversation: {messages[-3:]}\n\n"
        "Widget types: triple_editor, entity_selector, importance_selector, graph_viewer, confirmation_dialog\n\n"
        "Return JSON:\n"
        '{"show_widget": true/false, '
        '"widget_type": "triple_editor|entity_selector|...", '
        '"widget_data": {"triple_index": 0, "entities": [...]}}\n'
    )
    
    response = validator.api_repo.chat(prompt)
    widget_data = JsonHelper.parse_json(str(response))
    if not widget_data:
        widget_data = {"show_widget": False, "widget_type": None, "widget_data": {}}
    
    return {
        **state,
        STATE_SHOW_WIDGET: widget_data.get("show_widget", False),
        STATE_WIDGET_TYPE: widget_data.get("widget_type"),
        STATE_WIDGET_DATA: widget_data.get("widget_data", {}),
        STATE_NEXT_AGENT: AGENT_COMMUNICATOR,
    }
