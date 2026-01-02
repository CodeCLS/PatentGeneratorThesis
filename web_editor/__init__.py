"""
Web editor for triples and entities.

This module provides a simple web interface for editing triples and entities
in real-time. Changes are immediately reflected in the repository.
"""

from web_editor.server import start_triple_editor, get_updated_triples, get_repository

__all__ = [
    'start_triple_editor',
    'get_updated_triples',
    'get_repository',
]

