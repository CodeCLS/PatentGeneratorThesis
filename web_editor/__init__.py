"""
Web editor for triples and entities.

This module provides a simple web interface for editing triples and entities
in real-time. Changes are immediately reflected in the repository.
"""

# Lazy imports to avoid Flask/Jinja2 import issues
# Only import Flask when actually starting the server

def start_triple_editor(*args, **kwargs):
    """Lazy import to avoid Flask/Jinja2 issues until needed."""
    from web_editor.server import start_triple_editor as _start_triple_editor
    return _start_triple_editor(*args, **kwargs)

def get_updated_triples(*args, **kwargs):
    """Lazy import to avoid Flask/Jinja2 issues until needed."""
    from web_editor.server import get_updated_triples as _get_updated_triples
    return _get_updated_triples(*args, **kwargs)

def get_repository(*args, **kwargs):
    """Lazy import to avoid Flask/Jinja2 issues until needed."""
    from web_editor.server import get_repository as _get_repository
    return _get_repository(*args, **kwargs)

__all__ = [
    'start_triple_editor',
    'get_updated_triples',
    'get_repository',
]

