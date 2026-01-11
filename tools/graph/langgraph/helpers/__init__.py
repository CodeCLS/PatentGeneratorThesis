"""
Helper modules for LangGraph validator.
"""

# Import ModifierActions from this package
from tools.graph.langgraph.helpers.modifier_actions import ModifierActions

# Import helper functions from the parent module's helpers.py file
# We need to import from the parent module since helpers.py is a sibling file
import importlib.util
import os

# Load the sibling helpers.py module
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_helpers_py_path = os.path.join(_parent_dir, 'helpers.py')

if os.path.exists(_helpers_py_path):
    spec = importlib.util.spec_from_file_location("tools.graph.langgraph._helpers_module", _helpers_py_path)
    _helpers_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_helpers_module)
    
    # Re-export all functions
    extract_text_from_response = _helpers_module.extract_text_from_response
    get_entity_id = _helpers_module.get_entity_id
    get_entity_name = _helpers_module.get_entity_name
    get_triple_head_id = _helpers_module.get_triple_head_id
    get_triple_tail_id = _helpers_module.get_triple_tail_id
    get_triple_head_name = _helpers_module.get_triple_head_name
    get_triple_tail_name = _helpers_module.get_triple_tail_name
    extract_retrieved_info = _helpers_module.extract_retrieved_info
    process_retrieved_info_for_widget = _helpers_module.process_retrieved_info_for_widget
else:
    raise ImportError(f"helpers.py not found at {_helpers_py_path}")

__all__ = [
    'ModifierActions',
    'extract_text_from_response',
    'get_entity_id',
    'get_entity_name',
    'get_triple_head_id',
    'get_triple_tail_id',
    'get_triple_head_name',
    'get_triple_tail_name',
    'extract_retrieved_info',
    'process_retrieved_info_for_widget',
]

