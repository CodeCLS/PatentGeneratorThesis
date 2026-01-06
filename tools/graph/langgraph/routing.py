"""
Routing functions for LangGraph validator nodes.
"""

from typing import TYPE_CHECKING

try:
    from langgraph.graph import END
except ImportError:
    END = "__end__"  # Fallback if END is not available

# Import GraphValidatorState at runtime (not just TYPE_CHECKING)
# This is needed because LangGraph's get_type_hints() evaluates forward references
# and needs GraphValidatorState to be available in the module's global namespace
from tools.graph.langgraph.state import GraphValidatorState

# Ensure GraphValidatorState is in module globals for type hint evaluation
# This is critical for get_type_hints() to resolve forward references
import sys
_current_module = sys.modules[__name__]
if not hasattr(_current_module, 'GraphValidatorState'):
    _current_module.GraphValidatorState = GraphValidatorState
globals()['GraphValidatorState'] = GraphValidatorState

if TYPE_CHECKING:
    # Also import for type checkers
    pass  # Already imported above


def route_from_communicator(state: GraphValidatorState):
    """
    Route from communicator to next agent.
    
    This function reads the 'next_agent' value that was set by communicator_node()
    and routes to the appropriate next node.
    
    The routing map defines valid destinations:
    - "retriever", "visualizer", "analyzer", "modifier" -> routes to those nodes
    - END or None -> ends the conversation
    """
    next_agent = state.get("next_agent")
    # Handle 'null' string (from JSON), None, or END constant
    if next_agent:
        # Check if it's the END constant (compare by identity, not value)
        if next_agent is END:
            return END
        # Handle string 'null'
        if isinstance(next_agent, str) and next_agent.lower() == "null":
            next_agent = None
        # Normalize agent name
        elif next_agent in ("retriever", "visualizer", "analyzer", "modifier"):
            return next_agent
    
    if state.get("validation_complete", False):
        return END
    
    # Check if we just came from analyzer (to prevent infinite loop)
    # Look at the last few messages to see if analyzer was recently called
    messages = state.get("messages", [])
    recent_bot_messages = [msg for msg in messages[-3:] if msg.get("role") == "bot"]
    
    # If no questions exist AND we haven't just tried to generate them, route to analyzer
    questions = state.get("questions", [])
    if not questions:
        # Only route to analyzer if we haven't recently tried (prevent recursion)
        # Check if the last bot message suggests we just tried to generate questions
        if not recent_bot_messages or "analyzing" not in recent_bot_messages[-1].get("content", "").lower():
            return "analyzer"
        else:
            # We just tried, but questions weren't generated - end to prevent infinite loop
            return END
    
    # Otherwise, end (wait for user response) - return END constant
    # LangGraph will handle the conversion to '__end__' internally
    return END


def route_from_retriever(state: GraphValidatorState) -> str:
    """Route from retriever."""
    return "communicator"


def route_from_visualizer(state: GraphValidatorState) -> str:
    """Route from visualizer."""
    return "communicator"


def route_from_analyzer(state: GraphValidatorState) -> str:
    """Route from analyzer."""
    return "communicator"


def route_from_modifier(state: GraphValidatorState) -> str:
    """Route from modifier."""
    return "communicator"

