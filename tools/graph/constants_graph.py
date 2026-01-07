"""
Constants for graph-related operations.
"""

# Agent names for LangGraph validator
AGENT_COMMUNICATOR = "communicator"
AGENT_ANALYZER = "analyzer"
AGENT_MODIFIER = "modifier"
AGENT_RETRIEVER = "retriever"
AGENT_VISUALIZER = "visualizer"

# Action types for graph modifications
ACTION_ADD_TRIPLES = "add_triples"
ACTION_DELETE_TRIPLES = "delete_triples"
ACTION_MERGE_ENTITIES = "merge_entities"
ACTION_RENAME_ENTITY = "rename_entity"
ACTION_UPDATE_ENTITY_LABEL = "update_entity_label"
ACTION_MODIFY_TRIPLE = "modify_triple"

# State dictionary keys for GraphValidatorState
STATE_MESSAGES = "messages"
STATE_CURRENT_QUESTION_ID = "current_question_id"
STATE_CURRENT_QUESTION_TEXT = "current_question_text"
STATE_QUESTIONS = "questions"
STATE_GRAPH_NODES_COUNT = "graph_nodes_count"
STATE_GRAPH_EDGES_COUNT = "graph_edges_count"
STATE_TRIPLES_COUNT = "triples_count"
STATE_ENTITIES_COUNT = "entities_count"
STATE_NEXT_AGENT = "next_agent"
STATE_VALIDATION_COMPLETE = "validation_complete"
STATE_HIDDEN_ACTIONS = "hidden_actions"
STATE_DISPLAY_ACTIONS = "display_actions"
STATE_SHOW_WIDGET = "show_widget"
STATE_WIDGET_TYPE = "widget_type"
STATE_WIDGET_DATA = "widget_data"
STATE_CONVERSATION_TURN = "conversation_turn"
STATE_CHANGES_SUMMARY = "changes_summary"
STATE_STATS = "stats"

