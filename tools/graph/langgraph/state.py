"""
State definition for LangGraph-based Graph Validator.
"""

from typing import Dict, List, Optional, Any, TypedDict, Annotated


class GraphValidatorState(TypedDict):
    """State passed between agent nodes in the graph."""
    # Current conversation
    messages: Annotated[List[Dict[str, str]], "append"]  # Chat messages: [{"role": "user/bot", "content": "..."}]
    
    # Current question being handled
    current_question_id: Optional[str]  # ID of the current question
    current_question_text: Optional[str]  # Text of the current question
    questions: List[Dict[str, Any]]  # All available questions
    
    # Graph data (metadata only - actual graph stored separately to avoid serialization issues)
    graph_nodes_count: int  # Number of nodes in graph
    graph_edges_count: int  # Number of edges in graph
    triples_count: int  # Number of triples
    entities_count: int  # Number of entities
    
    # Agent decisions
    next_agent: Optional[str]  # Which agent to route to next
    validation_complete: bool  # Whether validation is done
    
    # Actions to perform
    hidden_actions: List[Dict[str, Any]]  # Graph modification actions
    display_actions: List[Dict[str, Any]]  # UI display actions
    
    # Widget information
    show_widget: bool
    widget_type: Optional[str]
    widget_data: Dict[str, Any]
    
    # Context and metadata
    conversation_turn: int  # Current turn number
    changes_summary: List[str]  # Summary of changes made
    stats: Dict[str, Any]  # Graph statistics

