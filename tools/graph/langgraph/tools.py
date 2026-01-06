"""
Tools for LangGraph-based Graph Validator agents.

This module provides tools that agents can use to interact with the graph.
"""

from typing import Dict, List, Any
import networkx as nx

from tools.graph.Triple import Triple


class GraphValidatorTools:
    """Tools that agents can use to interact with the graph."""
    
    def __init__(
        self,
        graph: nx.MultiDiGraph,
        triples: List[Triple],
        id_to_name: Dict[str, str],
    ):
        self.graph = graph
        self.triples = triples
        self.id_to_name = id_to_name
        self._original_graph = graph.copy() if graph else None
        self._original_triples = triples.copy()
    
    def get_entity_info(self, entity_name: str) -> Dict[str, Any]:
        """Retrieve detailed information about an entity by name."""
        # Find entity ID from name
        entity_id = None
        for eid, name in self.id_to_name.items():
            if name.lower() == entity_name.lower():
                entity_id = eid
                break
        
        if not entity_id:
            return {"error": f"Entity '{entity_name}' not found"}
        
        info = {
            "name": entity_name,
            "id": entity_id,
            "connections": 0,
            "properties": {},
            "connected_entities": [],
            "triples": [],
        }
        
        if self.graph and self.graph.has_node(entity_id):
            node_data = self.graph.nodes[entity_id]
            info["properties"] = {k: v for k, v in node_data.items() 
                                 if k not in ("node_type", "name") and not k.startswith("_")}
            info["label"] = node_data.get("node_type", "UNKNOWN")
            info["connections"] = self.graph.degree(entity_id)
            
            # Get connected entities
            for neighbor in self.graph.neighbors(entity_id):
                neighbor_name = self.id_to_name.get(neighbor, neighbor)
                edge_data = self.graph.get_edge_data(entity_id, neighbor)
                if edge_data:
                    for key, data in edge_data.items():
                        relation = data.get("label", "")
                        info["connected_entities"].append({
                            "name": neighbor_name,
                            "relation": relation,
                            "direction": "outgoing"
                        })
            
            # Get incoming connections
            for predecessor in self.graph.predecessors(entity_id):
                pred_name = self.id_to_name.get(predecessor, predecessor)
                edge_data = self.graph.get_edge_data(predecessor, entity_id)
                if edge_data:
                    for key, data in edge_data.items():
                        relation = data.get("label", "")
                        info["connected_entities"].append({
                            "name": pred_name,
                            "relation": relation,
                            "direction": "incoming"
                        })
        
        # Find triples involving this entity
        for i, triple in enumerate(self.triples):
            head_id = getattr(triple.head, "ref", None) or getattr(triple.head, "id", None) or getattr(triple.head, "ref_short", None) or str(triple.head)
            tail_id = getattr(triple.tail, "ref", None) or getattr(triple.tail, "id", None) or getattr(triple.tail, "ref_short", None) or str(triple.tail)
            if head_id == entity_id or tail_id == entity_id:
                head_name = self.id_to_name.get(head_id, str(triple.head))
                tail_name = self.id_to_name.get(tail_id, str(triple.tail))
                triple_info = {
                    "index": i,
                    "head": head_name,
                    "relation": triple.relation,
                    "tail": tail_name,
                }
                # Include relation properties if present
                if hasattr(triple, "properties") and triple.properties:
                    triple_info["relation_properties"] = dict(triple.properties)
                info["triples"].append(triple_info)
        
        return info
    
    def get_triple_info(self, triple_index: int) -> Dict[str, Any]:
        """Retrieve detailed information about a triple by index."""
        # Safety check - ensure triples list exists and index is valid
        if not self.triples:
            return {"error": "No triples available"}
        
        # Handle None or invalid types
        if triple_index is None:
            return {"error": "Triple index is required (cannot be None)"}
        
        try:
            triple_index = int(triple_index)
        except (TypeError, ValueError):
            return {"error": f"Invalid triple index: {triple_index} (must be an integer)"}
        
        if triple_index < 0 or triple_index >= len(self.triples):
            return {"error": f"Triple index {triple_index} out of range (valid range: 0-{len(self.triples)-1})"}
        
        try:
            triple = self.triples[triple_index]
        except (IndexError, TypeError) as e:
            return {"error": f"Error accessing triple at index {triple_index}: {str(e)}"}
        head_id = getattr(triple.head, "ref", None) or getattr(triple.head, "id", None) or getattr(triple.head, "ref_short", None) or str(triple.head)
        tail_id = getattr(triple.tail, "ref", None) or getattr(triple.tail, "id", None) or getattr(triple.tail, "ref_short", None) or str(triple.tail)
        head_name = self.id_to_name.get(head_id, str(triple.head))
        tail_name = self.id_to_name.get(tail_id, str(triple.tail))
        
        info = {
            "index": triple_index,
            "head": head_name,
            "head_id": head_id,
            "relation": triple.relation,
            "tail": tail_name,
            "tail_id": tail_id,
        }
        
        # Include relation properties/qualifiers (RHF-compatible)
        if hasattr(triple, "properties") and triple.properties:
            info["relation_properties"] = dict(triple.properties)
            info["has_properties"] = True
        else:
            info["relation_properties"] = {}
            info["has_properties"] = False
        
        # Include other triple metadata
        if hasattr(triple, "tags") and triple.tags:
            info["tags"] = list(triple.tags)
        if hasattr(triple, "importance"):
            info["importance"] = triple.importance
        if hasattr(triple, "info_quality"):
            info["info_quality"] = triple.info_quality
        
        # Get additional context from graph
        if self.graph:
            if self.graph.has_node(head_id):
                info["head_properties"] = dict(self.graph.nodes[head_id])
            if self.graph.has_node(tail_id):
                info["tail_properties"] = dict(self.graph.nodes[tail_id])
            
            # Check if edge exists in graph
            if self.graph.has_edge(head_id, tail_id):
                edge_data = self.graph.get_edge_data(head_id, tail_id)
                if edge_data:
                    info["edge_data"] = dict(list(edge_data.values())[0])
        
        return info
    
    def search_entities(self, query: str, limit: int = 10) -> List[Dict[str, str]]:
        """Search for entities by name (fuzzy match)."""
        query_lower = query.lower()
        results = []
        
        for eid, name in self.id_to_name.items():
            if query_lower in name.lower():
                results.append({"id": eid, "name": name})
                if len(results) >= limit:
                    break
        
        return results
    
    def get_related_triples(self, entity_name: str, max_depth: int = 1) -> List[Dict[str, Any]]:
        """Get triples related to an entity (including neighbors)."""
        entity_id = None
        for eid, name in self.id_to_name.items():
            if name.lower() == entity_name.lower():
                entity_id = eid
                break
        
        if not entity_id or not self.graph:
            return []
        
        # Get direct and neighbor entities
        related_entities = {entity_id}
        if self.graph.has_node(entity_id):
            for neighbor in list(self.graph.neighbors(entity_id))[:5]:  # Limit neighbors
                related_entities.add(neighbor)
            for predecessor in list(self.graph.predecessors(entity_id))[:5]:
                related_entities.add(predecessor)
        
        # Find triples involving these entities
        related_triples = []
        for i, triple in enumerate(self.triples):
            head_id = getattr(triple.head, "ref", None) or getattr(triple.head, "id", None) or getattr(triple.head, "ref_short", None) or str(triple.head)
            tail_id = getattr(triple.tail, "ref", None) or getattr(triple.tail, "id", None) or getattr(triple.tail, "ref_short", None) or str(triple.tail)
            if head_id in related_entities or tail_id in related_entities:
                head_name = self.id_to_name.get(head_id, str(triple.head))
                tail_name = self.id_to_name.get(tail_id, str(triple.tail))
                triple_info = {
                    "index": i,
                    "head": head_name,
                    "relation": triple.relation,
                    "tail": tail_name,
                }
                # Include relation properties if present
                if hasattr(triple, "properties") and triple.properties:
                    triple_info["relation_properties"] = dict(triple.properties)
                related_triples.append(triple_info)
        
        return related_triples
    
    def calculate_stats(self) -> Dict[str, Any]:
        """Calculate current graph statistics."""
        return {
            "total_triples": len(self.triples),
            "total_entities": len(self.id_to_name),
            "graph_nodes": self.graph.number_of_nodes() if self.graph else 0,
            "graph_edges": self.graph.number_of_edges() if self.graph else 0,
            "triples_changed": len(self.triples) - len(self._original_triples),
            "entities_changed": len(self.id_to_name) - len(self._original_triples),  # Simplified
        }

