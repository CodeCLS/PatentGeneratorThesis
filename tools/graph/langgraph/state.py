"""
State definition for LangGraph-based Graph Validator.
"""

from typing import Dict, List, Optional, Any, TypedDict, Annotated, Union
from tools.graph.langgraph.question import Question
from tools.graph.langgraph.message import Message
from tools.graph.langgraph.message.widgets import Widget
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
)


class GraphValidatorState(TypedDict):
    """State passed between agent nodes in the graph."""
    # Current conversation
    messages: Annotated[List[Union[Message, Dict[str, Any]]], "append"]  # Chat messages: Message objects or dicts for compatibility
    
    # Current question being handled
    current_question_id: Optional[str]  # ID of the current question
    current_question_text: Optional[str]  # Text of the current question
    questions: List[Question]  # All available questions
    
    # Graph data (metadata only - actual graph stored separately to avoid serialization issues)
    graph_nodes_count: int  # Number of nodes in graph
    graph_edges_count: int  # Number of edges in graph
    triples_count: int  # Number of triples
    entities_count: int  # Number of entities
    
    # Agent decisions
    agent_queue: List[str]  # Queue of agents to run in sequence
    validation_complete: bool  # Whether validation is done
    
    # Orchestrator planning
    mode: Optional[str]  # Mode: "WRITE", "Q&A", "EXPLORATION", "DEBUG"
    plan: Optional[str]  # Plan description
    needs_retrieval: bool  # Whether retrieval is needed
    write: bool  # Whether graph modifications are needed
    response_style: Optional[str]  # Response style preference
    
    # Actions to perform
    hidden_actions: List[Dict[str, Any]]  # Graph modification actions
    display_actions: List[Dict[str, Any]]  # UI display actions
    
    # Widget information
    show_widget: bool
    widget_type: Optional[str]
    widget_data: Union[Widget, Dict[str, Any]]  # Widget object or dict for compatibility
    
    # Context and metadata
    conversation_turn: int  # Current turn number
    changes_summary: List[str]  # Summary of changes made
    stats: Dict[str, Any]  # Graph statistics


def create_state(**kwargs) -> GraphValidatorState:
    """Create a GraphValidatorState dict with defaults."""
    defaults: GraphValidatorState = {
        STATE_MESSAGES: [],
        STATE_CURRENT_QUESTION_ID: None,
        STATE_CURRENT_QUESTION_TEXT: None,
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
        STATE_RESPONSE_STYLE: None,
        STATE_HIDDEN_ACTIONS: [],
        STATE_DISPLAY_ACTIONS: [],
        STATE_SHOW_WIDGET: False,
        STATE_WIDGET_TYPE: None,
        STATE_WIDGET_DATA: {},
        STATE_CONVERSATION_TURN: 0,
        STATE_CHANGES_SUMMARY: [],
        STATE_STATS: {},
    }
    defaults.update(kwargs)
    return defaults

