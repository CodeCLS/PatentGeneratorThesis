"""
Graph Validator Chat Interface - Simple chat UI for validating and modifying graphs.
"""

# Lazy imports to avoid Flask/Jinja2 import issues until needed

def start_validator_chat(*args, **kwargs):
    """Lazy import to avoid Flask/Jinja2 issues until needed."""
    from web_editor.graph_validator_chat.server import start_validator_chat as _start_validator_chat
    return _start_validator_chat(*args, **kwargs)

def get_validator_state(*args, **kwargs):
    """Lazy import to avoid Flask/Jinja2 issues until needed."""
    from web_editor.graph_validator_chat.server import get_validator_state as _get_validator_state
    return _get_validator_state(*args, **kwargs)

def get_updated_graph(*args, **kwargs):
    """Lazy import to avoid Flask/Jinja2 issues until needed."""
    from web_editor.graph_validator_chat.helper import get_updated_graph as _get_updated_graph
    return _get_updated_graph(*args, **kwargs)

def get_updated_triples(*args, **kwargs):
    """Lazy import to avoid Flask/Jinja2 issues until needed."""
    from web_editor.graph_validator_chat.helper import get_updated_triples as _get_updated_triples
    return _get_updated_triples(*args, **kwargs)

def get_updated_entities(*args, **kwargs):
    """Lazy import to avoid Flask/Jinja2 issues until needed."""
    from web_editor.graph_validator_chat.helper import get_updated_entities as _get_updated_entities
    return _get_updated_entities(*args, **kwargs)

def get_changes_summary(*args, **kwargs):
    """Lazy import to avoid Flask/Jinja2 issues until needed."""
    from web_editor.graph_validator_chat.helper import get_changes_summary as _get_changes_summary
    return _get_changes_summary(*args, **kwargs)

def get_all_updates(*args, **kwargs):
    """Lazy import to avoid Flask/Jinja2 issues until needed."""
    from web_editor.graph_validator_chat.helper import get_all_updates as _get_all_updates
    return _get_all_updates(*args, **kwargs)

__all__ = [
    'start_validator_chat',
    'get_validator_state',
    'get_updated_graph',
    'get_updated_triples',
    'get_updated_entities',
    'get_changes_summary',
    'get_all_updates',
]

