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
    AGENT_ORCHESTRATOR,
    AGENT_COMMUNICATOR,
    AGENT_ANALYZER,
    AGENT_MODIFIER,
    AGENT_RETRIEVER,
    AGENT_VISUALIZER,
    STATE_AGENT_QUEUE,
    STATE_VALIDATION_COMPLETE
)

# Ensure GraphValidatorState is in module globals for type hint evaluation
globals()['GraphValidatorState'] = GraphValidatorState

def route_from_any_node(state: GraphValidatorState) -> str:
    """Generic router that follows the agent_queue."""
    agent_queue = state.get(STATE_AGENT_QUEUE, [])
    if agent_queue:
        return agent_queue[0]

    return END

def route_from_orchestrator(state: GraphValidatorState) -> str:
    return route_from_any_node(state)

def route_from_communicator(state: GraphValidatorState) -> str:
    return route_from_any_node(state)

def route_from_retriever(state: GraphValidatorState) -> str:
    return route_from_any_node(state)

def route_from_visualizer(state: GraphValidatorState) -> str:
    return route_from_any_node(state)

def route_from_analyzer(state: GraphValidatorState) -> str:
    return route_from_any_node(state)

def route_from_modifier(state: GraphValidatorState) -> str:
    return route_from_any_node(state)
