"""
Routing functions for LangGraph validator nodes.
"""

from typing import TYPE_CHECKING

try:
    from langgraph.graph import END
except ImportError:
    END = "__end__"

from tools.graph.langgraph.state import GraphValidatorState

# Ensure GraphValidatorState is in module globals for type hint evaluation
globals()['GraphValidatorState'] = GraphValidatorState

if TYPE_CHECKING:
    from tools.graph.langgraph.validator import GraphValidatorLangGraph


def route_from_communicator(state: GraphValidatorState):
    """Route from communicator to next agent."""
    next_agent = state.get("next_agent")
    
    if next_agent is END or (isinstance(next_agent, str) and next_agent.lower() == "null"):
        return END
    
    if next_agent in ("retriever", "visualizer", "analyzer", "modifier"):
        return next_agent
    
    if state.get("validation_complete", False):
        return END
    
    # If no questions, route to analyzer
    if not state.get("questions", []):
        return "analyzer"
    
    return END


def route_from_retriever(state: GraphValidatorState) -> str:
    return "communicator"


def route_from_visualizer(state: GraphValidatorState) -> str:
    return "communicator"


def route_from_analyzer(state: GraphValidatorState) -> str:
    return "communicator"


def route_from_modifier(state: GraphValidatorState) -> str:
    return "communicator"
