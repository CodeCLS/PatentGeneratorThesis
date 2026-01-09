"""
Constants for graph-related operations.
"""

AGENT_ORCHESTRATOR = "orchestrator"
AGENT_COMMUNICATOR = "communicator"
AGENT_ANALYZER = "analyzer"
AGENT_MODIFIER = "modifier"
AGENT_RETRIEVER = "retriever"
AGENT_VISUALIZER = "visualizer"
AGENT_FORK = "fork"  # Fork node for parallel execution
AGENT_MERGE = "merge"  # Merge node for combining parallel results

ACTION_ADD_TRIPLES = "add_triples"
ACTION_DELETE_TRIPLES = "delete_triples"
ACTION_MERGE_ENTITIES = "merge_entities"
ACTION_RENAME_ENTITY = "rename_entity"
ACTION_UPDATE_ENTITY_LABEL = "update_entity_label"
ACTION_MODIFY_TRIPLE = "modify_triple"

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
STATE_AGENT_QUEUE = "agent_queue"
STATE_MODE = "mode"
STATE_PLAN = "plan"
STATE_NEEDS_RETRIEVAL = "needs_retrieval"
STATE_WRITE = "write"
STATE_RESPONSE_STYLE = "response_style"
STATE_TEXT = "text"
STATE_NEXT_QUESTION = "next_question"

# Message roles (for compatibility with dict-based messages)
MESSAGE_ROLE_USER = "user"
MESSAGE_ROLE_BOT = "bot"
MESSAGE_ROLE_SYSTEM = "system"

# Dictionary keys for messages and actions
KEY_ROLE = "role"
KEY_CONTENT = "content"
KEY_TYPE = "type"
KEY_ACTION = "action"
KEY_PARAMETERS = "parameters"
KEY_NAME = "name"
KEY_ID = "id"
KEY_INDEX = "index"
KEY_HEAD = "head"
KEY_TAIL = "tail"
KEY_RELATION = "relation"
KEY_TRIPLES = "triples"
KEY_ERROR = "error"

# Retrieved information keys
KEY_RELATED_TRIPLES = "related_triples"
KEY_SEARCH_RESULTS = "search_results"
KEY_RETRIEVED_INFO_MARKER = "[Retrieved Information]"
KEY_REASON = "Reason:"

# Widget types
WIDGET_TYPE_EDGES = "edges_widget"
WIDGET_TYPE_VISUALIZATION = "visualization"

# Action types for retriever
ACTION_GET_ENTITY_INFO = "get_entity_info"
ACTION_GET_TRIPLE_INFO = "get_triple_info"
ACTION_GET_RELATED_TRIPLES = "get_related_triples"
ACTION_SEARCH_ENTITIES = "search_entities"

# Mode values
MODE_INITIAL = "INITIAL"
MODE_QA = "Q&A"
MODE_WRITE = "WRITE"
MODE_EXPLORATION = "EXPLORATION"
MODE_DEBUG = "DEBUG"

# Default values
DEFAULT_UNKNOWN = "UNKNOWN"
DEFAULT_ENTITY = "Entity"
DEFAULT_N_A = "N/A"
DEFAULT_REASON = "Default"

# Internal state keys
STATE_INTERNAL_RETRIEVED_TRIPLES = "_retrieved_triples"
STATE_INTERNAL_RETRIEVED_INFO_PROCESSED = "_retrieved_info_processed"
STATE_INTERNAL_FROM_FORK = "_from_fork"
STATE_INTERNAL_NEEDS_WIDGET = "needs_widget"
STATE_INTERNAL_NODE_TYPE = "node_type"

