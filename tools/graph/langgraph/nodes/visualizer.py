"""
Visualizer node - decides which widgets to show and with what data.
"""

from typing import TYPE_CHECKING

from tools.helper.json_helper import JsonHelper

# Import GraphValidatorState at runtime (not just TYPE_CHECKING)
# This is needed because LangGraph might inspect type hints at runtime
from tools.graph.langgraph.state import GraphValidatorState

if TYPE_CHECKING:
    from tools.graph.langgraph.validator import GraphValidatorLangGraph


def visualizer_node(validator: "GraphValidatorLangGraph", state: "GraphValidatorState") -> "GraphValidatorState":
    """
    Visualization agent - decides which widgets to show and with what data.
    """
    messages = state.get("messages", [])
    current_question_id = state.get("current_question_id")
    # Get triples from instance, not state (to avoid serialization)
    triples = validator.triples
    
    # Analyze what should be visualized
    prompt = (
        "You are a visualization agent. Your job is to decide what widgets to show.\n\n"
        f"Current question: {state.get('current_question_text', 'N/A')}\n"
        f"Recent conversation: {messages[-3:] if len(messages) >= 3 else messages}\n\n"
        "Widget types available:\n"
        "- triple_editor: Edit a specific triple\n"
        "- entity_selector: Select entities from a list\n"
        "- importance_selector: Rate importance of a triple\n"
        "- graph_viewer: Show a subgraph\n"
        "- confirmation_dialog: Confirm an action\n\n"
        "CRITICAL: Return ONLY valid JSON. No reasoning, no explanation, just the JSON object.\n\n"
        "Return JSON:\n"
        '{"show_widget": true/false, '
        '"widget_type": "triple_editor|entity_selector|...", '
        '"widget_data": {"triple_index": 0, "entities": [...], etc.}, '
        '"reason": "Why this widget is needed"}\n'
    )
    
    response = validator.api_repo.chat(prompt)
    # Use JsonHelper for robust JSON parsing (handles fences and extraction automatically)
    widget_data = JsonHelper.parse_json(str(response))
    if widget_data is None:
        widget_data = {"show_widget": False, "widget_type": None, "widget_data": {}}
    
    return {
        **state,
        "show_widget": widget_data.get("show_widget", False),
        "widget_type": widget_data.get("widget_type"),
        "widget_data": widget_data.get("widget_data", {}),
        "next_agent": "communicator",
    }

