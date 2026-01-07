"""
Visualizer node - decides which widgets to show and with what data.
"""

from typing import TYPE_CHECKING
from tools.helper.json_helper import JsonHelper
from tools.graph.langgraph.state import GraphValidatorState

if TYPE_CHECKING:
    from tools.graph.langgraph.validator import GraphValidatorLangGraph


def visualizer_node(validator: "GraphValidatorLangGraph", state: "GraphValidatorState") -> "GraphValidatorState":
    """Visualization agent - decides which widgets to show."""
    messages = state.get("messages", [])
    
    prompt = (
        "You are a visualization agent. Decide what widgets to show.\n\n"
        f"Current question: {state.get('current_question_text', 'N/A')}\n"
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
        "show_widget": widget_data.get("show_widget", False),
        "widget_type": widget_data.get("widget_type"),
        "widget_data": widget_data.get("widget_data", {}),
        "next_agent": "communicator",
    }
