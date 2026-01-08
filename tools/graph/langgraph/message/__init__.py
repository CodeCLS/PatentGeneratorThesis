"""
Message and Widget dataclasses for LangGraph validator.
"""

from tools.graph.langgraph.message.message import Message, MessageRole
from tools.graph.langgraph.message.widgets import (
    Widget,
    EdgesWidget,
    GraphWidget,
    GraphSubsectionWidget,
    QuestionWidgetGeneral,
    QuestionWidgetTriple,
    QuestionWidgetEntity,
    QuestionWidgetClusterTriple,
    ValidationSummaryWidget,
    PatentAnalysisWidget,
    ConnectionCheckWidget,
    SuggestionWidget,
)

__all__ = [
    "Message",
    "MessageRole",
    "Widget",
    "EdgesWidget",
    "GraphWidget",
    "GraphSubsectionWidget",
    "QuestionWidgetGeneral",
    "QuestionWidgetTriple",
    "QuestionWidgetEntity",
    "QuestionWidgetClusterTriple",
    "ValidationSummaryWidget",
    "PatentAnalysisWidget",
    "ConnectionCheckWidget",
    "SuggestionWidget",
]

