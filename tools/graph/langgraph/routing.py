"""
Routing functions for LangGraph validator nodes.
"""

from typing import TYPE_CHECKING

try:
    from langgraph.graph import END
except ImportError:
    END = "__end__"

from tools.graph.langgraph.state import GraphValidatorState
from tools.graph.constants_graph import (
    AGENT_COMMUNICATOR,
    AGENT_ANALYZER,
    AGENT_MODIFIER,
    AGENT_RETRIEVER,
    AGENT_VISUALIZER,
    STATE_NEXT_AGENT,
    STATE_VALIDATION_COMPLETE,
    STATE_QUESTIONS,
)

# Ensure GraphValidatorState is in module globals for type hint evaluation
globals()['GraphValidatorState'] = GraphValidatorState

if TYPE_CHECKING:
    from tools.graph.langgraph.validator import GraphValidatorLangGraph


def route_from_communicator(state: GraphValidatorState):
    """Route from communicator to next agent."""
    next_agent = state.get(STATE_NEXT_AGENT)
    
    if next_agent is END or (isinstance(next_agent, str) and next_agent.lower() == "null"):
        return END
    
    if next_agent in (AGENT_RETRIEVER, AGENT_VISUALIZER, AGENT_ANALYZER, AGENT_MODIFIER):
        return next_agent
    
    if state.get(STATE_VALIDATION_COMPLETE, False):
        return END
    
    # If no questions, route to analyzer
    if not state.get(STATE_QUESTIONS, []):
        return AGENT_ANALYZER
    
    return END


def route_from_retriever(state: GraphValidatorState) -> str:
    return AGENT_COMMUNICATOR


def route_from_visualizer(state: GraphValidatorState) -> str:
    return AGENT_COMMUNICATOR


def route_from_analyzer(state: GraphValidatorState) -> str:
    return AGENT_COMMUNICATOR


def route_from_modifier(state: GraphValidatorState) -> str:
    return AGENT_COMMUNICATOR
