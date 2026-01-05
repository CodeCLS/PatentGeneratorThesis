"""
Server for Graph Validator Chat Interface.
Allows interactive chat-based graph validation and modification.
"""
from typing import List, Optional, Dict, Any
import networkx as nx
import pickle
import base64

from tools.graph.Triple import Triple

# Use simple HTTP server instead of Flask - NO Jinja2 dependency!
from web_editor.graph_validator_chat.simple_server import (
    start_validator_chat as _start_validator_chat,
    get_validator,
)


def start_validator_chat(
    graph: Optional[nx.MultiDiGraph] = None,
    triples: Optional[List[Triple]] = None,
    id_to_name: Optional[Dict[str, str]] = None,
    port: int = 5001,
    open_browser: bool = True,
    debug: bool = False,
    use_langgraph: bool = True,  # Use LangGraph by default if available
) -> None:
    """
    Start the graph validator chat interface.
    
    Args:
        graph: Optional NetworkX graph to validate
        triples: Optional list of Triple objects to validate
        id_to_name: Optional mapping from entity ID to display name
        port: Port number to run the server on (default: 5001)
        open_browser: Whether to automatically open the browser (default: True)
        debug: Enable Flask debug mode (default: False)
        use_langgraph: Whether to use LangGraph-based validator (default: True if available)
    
    Example:
        >>> from tools.graph.visualizer import GraphVisualizer
        >>> from web_editor.graph_validator_chat import start_validator_chat
        >>> 
        >>> visualizer = GraphVisualizer()
        >>> G = visualizer.build_graph(triples)
        >>> 
        >>> # Use LangGraph validator (default)
        >>> start_validator_chat(graph=G, triples=triples, id_to_name=id_to_name)
        >>> 
        >>> # Or use original validator
        >>> start_validator_chat(graph=G, triples=triples, id_to_name=id_to_name, use_langgraph=False)
        >>> 
        >>> # Chat interface opens in browser
        >>> # Answer questions and modify the graph
        >>> # When done, get updated data from get_validator_state()
    """
    # Use simple HTTP server (NO Flask, NO Jinja2!)
    _start_validator_chat(
        graph=graph, 
        triples=triples, 
        id_to_name=id_to_name, 
        port=port, 
        open_browser=open_browser,
        use_langgraph=use_langgraph,
    )


def get_validator_state() -> Dict[str, Any]:
    """
    Get the current state of the validator including updated graph, triples, and entities.
    This function fetches data from the running server via API.
    
    Returns:
        Dictionary with:
        - 'graph': Updated NetworkX graph (pickled as base64)
        - 'triples': Updated list of Triple objects
        - 'entities': List of all entities from triples
        - 'id_to_name': Updated entity ID to name mapping
        - 'changes': Summary of changes made
    
    Example:
        >>> state = get_validator_state()
        >>> 
        >>> # Get updated graph
        >>> import pickle
        >>> import base64
        >>> if state['graph']:
        >>>     graph_data = base64.b64decode(state['graph'])
        >>>     updated_graph = pickle.loads(graph_data)
        >>> 
        >>> # Get updated triples
        >>> updated_triples = state['triples']
        >>> 
        >>> # Get changes summary
        >>> changes = state['changes']
        >>> print(f"Added {changes['triples_added']} triples")
    """
    try:
        import requests
    except ImportError:
        requests = None
    
    try:
        # Try to fetch from running server
        if requests:
            response = requests.get('http://localhost:5001/api/export', timeout=2)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    
    # Fallback: get directly from validator if available
    validator = get_validator()
    if not validator:
        return {
            "graph": None,
            "triples": [],
            "entities": [],
            "id_to_name": {},
            "changes": {},
        }
    
    # Get updated graph and triples
    updated_graph = validator.getUpdatedGraph()
    updated_triples = validator.getUpdatedTriples()
    changes = validator.getChanges()
    
    # Serialize graph as base64 pickle
    graph_data = None
    if updated_graph:
        graph_bytes = pickle.dumps(updated_graph)
        graph_data = base64.b64encode(graph_bytes).decode('utf-8')
    
    # Extract all entities from triples
    entities = []
    entity_ids = set()
    for triple in updated_triples:
        head_id = getattr(triple.head, "id", None) or getattr(triple.head, "ref_short", None)
        tail_id = getattr(triple.tail, "id", None) or getattr(triple.tail, "ref_short", None)
        
        if head_id and head_id not in entity_ids:
            entities.append({
                "id": head_id,
                "name": getattr(triple.head, "name", ""),
                "label": getattr(triple.head, "label", ""),
                "ref_short": getattr(triple.head, "ref_short", ""),
            })
            entity_ids.add(head_id)
        
        if tail_id and tail_id not in entity_ids:
            entities.append({
                "id": tail_id,
                "name": getattr(triple.tail, "name", ""),
                "label": getattr(triple.tail, "label", ""),
                "ref_short": getattr(triple.tail, "ref_short", ""),
            })
            entity_ids.add(tail_id)
    
    return {
        "graph": graph_data,
        "triples": updated_triples,
        "entities": entities,
        "id_to_name": validator.id_to_name,
        "changes": changes,
    }

