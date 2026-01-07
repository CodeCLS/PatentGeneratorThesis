"""
Tools for LangGraph-based Graph Validator agents.
"""

from typing import Dict, List, Any
import networkx as nx
from tools.graph.Triple import Triple
from tools.graph.langgraph.helpers import get_triple_head_id, get_triple_tail_id, get_triple_head_name, get_triple_tail_name


class GraphValidatorTools:
    """Tools that agents can use to interact with the graph."""
    
    def __init__(self, graph: nx.MultiDiGraph, triples: List[Triple], id_to_name: Dict[str, str]):
        self.graph = graph
        self.triples = triples
        self.id_to_name = id_to_name
        self._original_triples = triples.copy()
    
    def get_entity_info(self, entity_name: str) -> Dict[str, Any]:
        """Retrieve detailed information about an entity by name."""
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
            
            for neighbor in self.graph.neighbors(entity_id):
                neighbor_name = self.id_to_name.get(neighbor, neighbor)
                edge_data = self.graph.get_edge_data(entity_id, neighbor)
                if edge_data:
                    relation = list(edge_data.values())[0].get("label", "")
                    info["connected_entities"].append({
                        "name": neighbor_name,
                        "relation": relation,
                        "direction": "outgoing"
                    })
            
            for predecessor in self.graph.predecessors(entity_id):
                pred_name = self.id_to_name.get(predecessor, predecessor)
                edge_data = self.graph.get_edge_data(predecessor, entity_id)
                if edge_data:
                    relation = list(edge_data.values())[0].get("label", "")
                    info["connected_entities"].append({
                        "name": pred_name,
                        "relation": relation,
                        "direction": "incoming"
                    })
        
        for i, triple in enumerate(self.triples):
            head_id = get_triple_head_id(triple)
            tail_id = get_triple_tail_id(triple)
            if head_id == entity_id or tail_id == entity_id:
                head_name = self.id_to_name.get(head_id, get_triple_head_name(triple))
                tail_name = self.id_to_name.get(tail_id, get_triple_tail_name(triple))
                info["triples"].append({
                    "index": i,
                    "head": head_name,
                    "relation": triple.relation,
                    "tail": tail_name,
                })
        
        return info
    
    def get_triple_info(self, triple_index: int) -> Dict[str, Any]:
        """Retrieve detailed information about a triple by index."""
        if not self.triples or triple_index < 0 or triple_index >= len(self.triples):
            return {"error": f"Triple index {triple_index} out of range"}
        
        triple = self.triples[triple_index]
        head_id = get_triple_head_id(triple)
        tail_id = get_triple_tail_id(triple)
        head_name = self.id_to_name.get(head_id, get_triple_head_name(triple))
        tail_name = self.id_to_name.get(tail_id, get_triple_tail_name(triple))
        
        info = {
            "index": triple_index,
            "head": head_name,
            "head_id": head_id,
            "relation": triple.relation,
            "tail": tail_name,
            "tail_id": tail_id,
        }
        
        if self.graph:
            if self.graph.has_node(head_id):
                info["head_properties"] = dict(self.graph.nodes[head_id])
            if self.graph.has_node(tail_id):
                info["tail_properties"] = dict(self.graph.nodes[tail_id])
        
        return info
    
    def search_entities(self, query: str, limit: int = 10) -> List[Dict[str, str]]:
        """Search for entities by name."""
        query_lower = query.lower()
        results = []
        for eid, name in self.id_to_name.items():
            if query_lower in name.lower():
                results.append({"id": eid, "name": name})
                if len(results) >= limit:
                    break
        return results
    
    def get_related_triples(self, entity_name: str, max_depth: int = 1) -> List[Dict[str, Any]]:
        """Get triples related to an entity."""
        entity_id = None
        for eid, name in self.id_to_name.items():
            if name.lower() == entity_name.lower():
                entity_id = eid
                break
        
        if not entity_id or not self.graph:
            return []
        
        related_entities = {entity_id}
        if self.graph.has_node(entity_id):
            for neighbor in list(self.graph.neighbors(entity_id))[:5]:
                related_entities.add(neighbor)
            for predecessor in list(self.graph.predecessors(entity_id))[:5]:
                related_entities.add(predecessor)
        
        related_triples = []
        for i, triple in enumerate(self.triples):
            head_id = get_triple_head_id(triple)
            tail_id = get_triple_tail_id(triple)
            if head_id in related_entities or tail_id in related_entities:
                head_name = self.id_to_name.get(head_id, get_triple_head_name(triple))
                tail_name = self.id_to_name.get(tail_id, get_triple_tail_name(triple))
                related_triples.append({
                    "index": i,
                    "head": head_name,
                    "relation": triple.relation,
                    "tail": tail_name,
                })
        
        return related_triples
    
    def calculate_stats(self) -> Dict[str, Any]:
        """Calculate current graph statistics."""
        return {
            "total_triples": len(self.triples),
            "total_entities": len(self.id_to_name),
            "graph_nodes": self.graph.number_of_nodes() if self.graph else 0,
            "graph_edges": self.graph.number_of_edges() if self.graph else 0,
            "triples_changed": len(self.triples) - len(self._original_triples),
        }
