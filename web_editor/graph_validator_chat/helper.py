"""
Helper functions to get updated graph, triples, and entities from the validator chat.
Use these in your Jupyter notebook after using the chat interface.
"""
import pickle
import base64
from typing import Dict, Any, Optional, List
import networkx as nx

try:
    import requests
except ImportError:
    requests = None

from tools.graph.Triple import Triple
from tools.sentence.entity import Entity
from web_editor.graph_validator_chat.simple_server import get_validator


def get_updated_graph() -> Optional[nx.MultiDiGraph]:
    """
    Get the updated graph from the validator.
    
    Returns:
        Updated NetworkX graph, or None if not available
    
    Example:
        >>> from web_editor.graph_validator_chat.helper import get_updated_graph
        >>> updated_graph = get_updated_graph()
        >>> if updated_graph:
        >>>     print(f"Graph has {updated_graph.number_of_nodes()} nodes")
    """
    # Try to get from running server first
    try:
        response = requests.get('http://localhost:5001/api/export', timeout=2)
        if response.status_code == 200:
            data = response.json()
            if data.get("graph"):
                graph_bytes = base64.b64decode(data["graph"])
                return pickle.loads(graph_bytes)
    except:
        pass
    
    # Fallback: get directly from validator
    validator = get_validator()
    if not validator:
        return None
    
    return validator.getUpdatedGraph()


def get_updated_triples() -> List[Triple]:
    """
    Get the updated triples from the validator.
    Reconstructs Triple objects from exported data.
    
    Returns:
        Updated list of Triple objects
    
    Example:
        >>> from web_editor.graph_validator_chat.helper import get_updated_triples
        >>> updated_triples = get_updated_triples()
        >>> print(f"Have {len(updated_triples)} triples")
    """
    # Try to get from running server first
    try:
        if requests is None:
            raise ImportError("requests not available")
        response = requests.get('http://localhost:5001/api/export', timeout=2)
        if response.status_code == 200:
            data = response.json()
            triples_data = data.get("triples", [])
            
            # Reconstruct Triple objects
            triples = []
            for t_data in triples_data:
                head_data = t_data.get("head", {})
                tail_data = t_data.get("tail", {})
                
                head_ent = Entity(
                    id=head_data.get("id", ""),
                    name=head_data.get("name", ""),
                    label=head_data.get("label", ""),
                    ref_short=head_data.get("ref_short", ""),
                )
                
                tail_ent = Entity(
                    id=tail_data.get("id", ""),
                    name=tail_data.get("name", ""),
                    label=tail_data.get("label", ""),
                    ref_short=tail_data.get("ref_short", ""),
                )
                
                triple = Triple(
                    head=head_ent,
                    relation=t_data.get("relation", ""),
                    tail=tail_ent,
                )
                triples.append(triple)
            
            return triples
    except:
        pass
    
    # Fallback: get directly from validator
    validator = get_validator()
    if not validator:
        return []
    
    return validator.getUpdatedTriples()


def get_updated_entities() -> List[Dict[str, Any]]:
    """
    Get all entities from the updated triples.
    
    Returns:
        List of entity dictionaries with id, name, label
    
    Example:
        >>> from web_editor.graph_validator_chat.helper import get_updated_entities
        >>> entities = get_updated_entities()
        >>> print(f"Have {len(entities)} unique entities")
    """
    triples = get_updated_triples()
    entities = []
    entity_ids = set()
    
    for triple in triples:
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
    
    return entities


def get_changes_summary() -> Dict[str, Any]:
    """
    Get a summary of changes made to the graph.
    
    Returns:
        Dictionary with change statistics
    
    Example:
        >>> from web_editor.graph_validator_chat.helper import get_changes_summary
        >>> changes = get_changes_summary()
        >>> print(f"Added {changes['triples_added']} triples")
    """
    validator = get_validator()
    if not validator:
        return {}
    
    return validator.getChanges()


def get_all_updates() -> Dict[str, Any]:
    """
    Get all updated data: graph, triples, entities, and changes.
    
    Returns:
        Dictionary with:
        - 'graph': Updated NetworkX graph
        - 'triples': Updated list of Triple objects
        - 'entities': List of all entities
        - 'changes': Summary of changes
        - 'id_to_name': Entity ID to name mapping
    
    Example:
        >>> from web_editor.graph_validator_chat.helper import get_all_updates
        >>> updates = get_all_updates()
        >>> 
        >>> G = updates['graph']
        >>> triples = updates['triples']
        >>> entities = updates['entities']
        >>> changes = updates['changes']
    """
    # Try to get from running server first
    try:
        if requests is not None:
            response = requests.get('http://localhost:5001/api/export', timeout=2)
            if response.status_code == 200:
                data = response.json()
                
                # Reconstruct graph
                graph = None
                if data.get("graph"):
                    graph_bytes = base64.b64decode(data["graph"])
                    graph = pickle.loads(graph_bytes)
                
                # Reconstruct triples
                triples = []
                for t_data in data.get("triples", []):
                    head_data = t_data.get("head", {})
                    tail_data = t_data.get("tail", {})
                    
                    head_ent = Entity(
                        id=head_data.get("id", ""),
                        name=head_data.get("name", ""),
                        label=head_data.get("label", ""),
                        ref_short=head_data.get("ref_short", ""),
                    )
                    
                    tail_ent = Entity(
                        id=tail_data.get("id", ""),
                        name=tail_data.get("name", ""),
                        label=tail_data.get("label", ""),
                        ref_short=tail_data.get("ref_short", ""),
                    )
                    
                    triple = Triple(
                        head=head_ent,
                        relation=t_data.get("relation", ""),
                        tail=tail_ent,
                    )
                    triples.append(triple)
                
                return {
                    "graph": graph,
                    "triples": triples,
                    "entities": data.get("entities", []),
                    "changes": data.get("changes", {}),
                    "id_to_name": data.get("id_to_name", {}),
                }
    except Exception as e:
        # Fall through to direct validator access
        pass
    
    # Fallback: get directly from validator
    validator = get_validator()
    if not validator:
        return {
            "graph": None,
            "triples": [],
            "entities": [],
            "changes": {},
            "id_to_name": {},
        }
    
    return {
        "graph": validator.getUpdatedGraph(),
        "triples": validator.getUpdatedTriples(),
        "entities": get_updated_entities(),
        "changes": validator.getChanges(),
        "id_to_name": validator.id_to_name,
    }

