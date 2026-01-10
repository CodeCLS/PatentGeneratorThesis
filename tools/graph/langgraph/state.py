"""
State definition for LangGraph-based Graph Validator.
"""

from typing import Dict, List, Optional, Any, TypedDict, Annotated, Union
from tools.graph.langgraph.question import Question
from tools.graph.langgraph.message import Message
from tools.graph.langgraph.message.widgets import Widget
from tools.graph.langgraph.message.message import MessageRole, Message
from tools.graph.langgraph.nodes.chat_changes_information import ChatChangesInformation
from tools.graph.langgraph.nodes.chat_context_information import ChatContextInformation
from tools.graph.langgraph.nodes.chat_visual_info import ChatVisualInfo
from tools.graph.constants_graph import (
    STATE_MESSAGES,
    STATE_CURRENT_QUESTION_ID,
    STATE_CURRENT_QUESTION_TEXT,
    STATE_QUESTIONS,
    STATE_GRAPH_NODES_COUNT,
    STATE_GRAPH_EDGES_COUNT,
    STATE_TRIPLES_COUNT,
    STATE_ENTITIES_COUNT,
    STATE_NEXT_AGENT,
    STATE_AGENT_QUEUE,
    STATE_VALIDATION_COMPLETE,
    STATE_HIDDEN_ACTIONS,
    STATE_DISPLAY_ACTIONS,
    STATE_SHOW_WIDGET,
    STATE_WIDGET_TYPE,
    STATE_WIDGET_DATA,
    STATE_CONVERSATION_TURN,
    STATE_CHANGES_SUMMARY,
    STATE_STATS,
    STATE_MODE,
    STATE_PLAN,
    STATE_NEEDS_RETRIEVAL,
    STATE_WRITE,
    STATE_RESPONSE_STYLE,
    STATE_CURRENT_QUESTION,
    STATE_CHAT_CHANGES_INFORMATION
)


class GraphValidatorState(TypedDict):
    """State passed between agent nodes in the graph."""
    # Current conversation
    messages: Annotated[List[Union[Message, Dict[str, Any]]], "append"]  # Chat messages: Message objects or dicts for compatibility
    
    # Current question being handled
    current_question: Optional[Question]  # ID of the current question
    questions: List[Question]  # All available questions
    
    # Graph data (metadata only - actual graph stored separately to avoid serialization issues)
    graph_nodes_count: int  # Number of nodes in graph
    graph_edges_count: int  # Number of edges in graph
    triples_count: int  # Number of triples
    entities_count: int  # Number of entities
    chat_context_information: List[ChatContextInformation]  # Graph modification actions

    
    # Agent decisions
    agent_queue: List[str]  # Queue of agents to run in sequence
    validation_complete: bool  # Whether validation is done
    
    # Orchestrator planning
    mode: Optional[str]  # Mode: "WRITE", "Q&A", "EXPLORATION", "DEBUG"
    plan: Optional[Dict[int,str]]  # Plan description
    needs_retrieval: bool  # Whether retrieval is needed
    write: bool  # Whether graph modifications are needed
    
    # Actions to perform
    chat_changes_information: List[ChatChangesInformation]  # Graph modification actions
    display_actions: List[ChatVisualInfo]  # UI display actions
    
    # Widget information
    
    # Context and metadata
    conversation_turn: int  # Current turn number


def create_state(**kwargs) -> GraphValidatorState:
    """Create a GraphValidatorState dict with defaults."""
    defaults: GraphValidatorState = {
        STATE_MESSAGES: [],
        STATE_CURRENT_QUESTION: None,
        STATE_QUESTIONS: [],
        STATE_GRAPH_NODES_COUNT: 0,
        STATE_GRAPH_EDGES_COUNT: 0,
        STATE_TRIPLES_COUNT: 0,
        STATE_ENTITIES_COUNT: 0,
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
def get_last_message(messages: List[Message], role: MessageRole) -> Optional[Message]:
    for msg in reversed(messages):
        if isinstance(msg, Message):
            if msg.role == MessageRole.USER or (isinstance(msg.role, str) and msg.role == role):
                return msg
    return None
def get_message_content(msg: Union[Message]) -> str:
    """Extract content from a Message object or dict."""
    if isinstance(msg, Message):
        return msg.content
    return ""
def consume_agent(state: GraphValidatorState, agent: str) -> GraphValidatorState:
    agent_queue = state.get(STATE_AGENT_QUEUE, [])
    if agent_queue and agent_queue[0] == agent:
        agent_queue = agent_queue[1:]


