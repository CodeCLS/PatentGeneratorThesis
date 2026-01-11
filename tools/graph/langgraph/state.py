"""
State definition for LangGraph-based Graph Validator.
"""

from typing import Dict, List, Optional, Any, TypedDict, Annotated, Union
from tools.graph.langgraph.question import Question
from tools.graph.langgraph.message import Message
from tools.graph.langgraph.message.message import MessageRole
from tools.graph.constants_graph import *


class GraphValidatorState(TypedDict):
    """State passed between agent nodes in the graph."""
    # Current conversation
    messages: Annotated[List[Union[Message, Dict[str, Any]]], "append"]  # Chat messages
    
    # Current question being handled
    current_question: Optional[Question]  
    questions: List[Question]  # All available questions
    
    # Graph data (metadata only)
    graph_nodes_count: int
    graph_edges_count: int
    triples_count: int
    entities_count: int
    chat_context_information: List[Dict[str, Any]]

    # Agent decisions
    agent_queue: List[str]
    validation_complete: bool
    
    # Orchestrator planning
    mode: Optional[str]
    plan: Optional[Any]  # Flexible plan structure
    needs_retrieval: bool
    write: bool
    
    # Actions and information
    chat_changes_information: List[Dict[str, Any]]
    display_actions: List[Dict[str, Any]]
    changes_summary: List[str]
    
    # Context and metadata
    conversation_turn: int


def create_state(**kwargs) -> GraphValidatorState:
    """Create a GraphValidatorState dict with defaults."""
    # Import here to avoid circular import - only needed at runtime
    from tools.graph.langgraph.nodes.chat_changes_information import ChatChangesInformation
    from tools.graph.langgraph.nodes.chat_context_information import ChatContextInformation
    from tools.graph.langgraph.nodes.chat_visual_info import ChatVisualInfo
    
    defaults: GraphValidatorState = {
        STATE_MESSAGES: [],
        STATE_CURRENT_QUESTION: None,
        STATE_QUESTIONS: [],
        STATE_GRAPH_NODES_COUNT: 0,
        STATE_GRAPH_EDGES_COUNT: 0,
        STATE_TRIPLES_COUNT: 0,
        STATE_ENTITIES_COUNT: 0,
        STATE_CHAT_CONTEXT_INFORMATION: [],
        STATE_AGENT_QUEUE: [],
        STATE_VALIDATION_COMPLETE: False,
        STATE_MODE: None,
        STATE_PLAN: None,
        STATE_NEEDS_RETRIEVAL: False,
        STATE_WRITE: False,
        STATE_CHAT_CHANGES_INFORMATION: [],
        STATE_DISPLAY_ACTIONS: [],
        STATE_CONVERSATION_TURN: 0,
        STATE_CHANGES_SUMMARY: [],
    }
    defaults.update(kwargs)
    return defaults

def get_last_message(messages: List[Union[Message, Dict[str, Any]]], role: Union[MessageRole, str]) -> Optional[Union[Message, Dict[str, Any]]]:
    """Helper to get the last message of a specific role."""
    if not messages:
        return None
    
    target_role_value = role.value if isinstance(role, MessageRole) else role
    
    for msg in reversed(messages):
        current_role = msg.role if isinstance(msg, Message) else msg.get("role")
        if isinstance(current_role, MessageRole):
            current_role = current_role.value
        
        if current_role == target_role_value:
            return msg
    return None

def get_message_content(msg: Optional[Union[Message, Dict[str, Any]]]) -> str:
    """Extract content from a Message object or dict."""
    if msg is None:
        return ""
    if isinstance(msg, Message):
        return msg.content
    elif isinstance(msg, dict):
        return msg.get("content", "")
    return ""

def consume_agent(state: GraphValidatorState, agent: str) -> GraphValidatorState:
    """Consume the agent from the queue and return the updated state."""
    agent_queue = state.get(STATE_AGENT_QUEUE, [])
    if agent_queue and agent_queue[0] == agent:
        # Return a NEW dict to maintain immutability
        new_state = dict(state)
        new_state[STATE_AGENT_QUEUE] = agent_queue[1:]
        return new_state
    return state
