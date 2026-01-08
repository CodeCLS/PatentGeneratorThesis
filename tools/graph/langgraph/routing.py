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
    AGENT_FORK,
    STATE_NEXT_AGENT,
    STATE_AGENT_QUEUE,
    STATE_VALIDATION_COMPLETE,
    STATE_QUESTIONS,
    STATE_HIDDEN_ACTIONS,
    STATE_SHOW_WIDGET,
    AGENT_MERGE,
)

# Ensure GraphValidatorState is in module globals for type hint evaluation
globals()['GraphValidatorState'] = GraphValidatorState

if TYPE_CHECKING:
    from tools.graph.langgraph.validator import GraphValidatorLangGraph


def route_from_orchestrator(state: GraphValidatorState):
    """Route from orchestrator - pop next agent from queue."""
    agent_queue = state.get(STATE_AGENT_QUEUE, [])
    if agent_queue:
        return agent_queue[0]  # Return first agent in queue
    return AGENT_COMMUNICATOR  # Default fallback


def route_from_communicator(state: GraphValidatorState):
    """Route from communicator - check agent queue or use legacy routing."""
    # Check if there's an agent queue (orchestrator-driven)
    agent_queue = state.get(STATE_AGENT_QUEUE, [])
    if agent_queue:
        # Pop the current agent and get next one
        next_agent = agent_queue[0] if agent_queue else None
        if next_agent:
            return next_agent
    return END


def route_from_retriever(state: GraphValidatorState) -> str:
    """Route from retriever - pop from queue or go to communicator."""
    agent_queue = state.get(STATE_AGENT_QUEUE, [])
    if len(agent_queue) > 1:
        return agent_queue[1]  # Next agent in queue
    return AGENT_COMMUNICATOR


def route_from_visualizer(state: GraphValidatorState) -> str:
    """Route from visualizer - check queue, merge, or communicator."""
    # If we came from fork (parallel execution), route to merge
    if state.get("_from_fork", False):
        return AGENT_MERGE
    
    # Check agent queue
    agent_queue = state.get(STATE_AGENT_QUEUE, [])
    current_idx = 0
    if AGENT_VISUALIZER in agent_queue:
        current_idx = agent_queue.index(AGENT_VISUALIZER)
    if len(agent_queue) > current_idx + 1:
        return agent_queue[current_idx + 1]
    
    return AGENT_COMMUNICATOR


def route_from_analyzer(state: GraphValidatorState) -> str:
    """Route from analyzer - check queue or go to communicator."""
    agent_queue = state.get(STATE_AGENT_QUEUE, [])
    if AGENT_ANALYZER in agent_queue:
        current_idx = agent_queue.index(AGENT_ANALYZER)
        if len(agent_queue) > current_idx + 1:
            return agent_queue[current_idx + 1]
    return AGENT_COMMUNICATOR


def route_from_modifier(state: GraphValidatorState) -> str:
    """Route from modifier - check queue, merge, or communicator."""
    # If we came from fork (parallel execution), route to merge
    if state.get("_from_fork", False):
        return AGENT_MERGE
    
    # Check agent queue
    agent_queue = state.get(STATE_AGENT_QUEUE, [])
    if AGENT_MODIFIER in agent_queue:
        current_idx = agent_queue.index(AGENT_MODIFIER)
        if len(agent_queue) > current_idx + 1:
            return agent_queue[current_idx + 1]
    
    return AGENT_COMMUNICATOR
