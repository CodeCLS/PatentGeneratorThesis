"""
Tools for LangGraph-based Graph Validator agents.
"""

from typing import Dict, List, Any, Optional
import networkx as nx
from tools.graph.Triple import Triple
from tools.graph.langgraph.helpers import get_triple_head_id, get_triple_tail_id, get_triple_head_name, get_triple_tail_name
from tools.graph.langgraph.info import EntityInfo, TripleInfo, GraphInfo, ConnectedEntity
from tools.graph.constants_graph import (
    STATE_INTERNAL_NODE_TYPE,
    KEY_NAME,
    DEFAULT_UNKNOWN,
    KEY_ID,
)
from tools.sentence.entity import Entity


class GraphValidatorTools:
    """Tools that agents can use to interact with the graph."""
    
    def __init__(self, graph: nx.MultiDiGraph, triples: List[Triple], id_to_name: Dict[str, str]):
        self.graph = graph
        self.triples = triples
        self.id_to_name = id_to_name
        self._original_triples = triples.copy()
    
    def get_entity_info(self, entity_name: str, id: str = "") -> EntityInfo:
        """Retrieve detailed information about an entity by name."""
        entity_id = None
        if id:
            entity_id = id
        else:
            for eid, name in self.id_to_name.items():
                if name.lower() == entity_name.lower():
                    entity_id = eid
                    break
            print("Entity ID: ", entity_id)
        if not entity_id:
            return EntityInfo(
                name=entity_name,
                id="",
                error=f"Entity '{entity_name}' not found"
            )
        
        info = EntityInfo(
            name=entity_name,
            id=entity_id,
            connections=0,
            properties={},
            connected_entities=[],
            triples=[],
        )
        
        if self.graph and self.graph.has_node(entity_id):
            node_data = self.graph.nodes[entity_id]
            info.properties = {k: v for k, v in node_data.items() 
                              if k not in (STATE_INTERNAL_NODE_TYPE, KEY_NAME) and not k.startswith("_")}
            info.label = node_data.get(STATE_INTERNAL_NODE_TYPE, DEFAULT_UNKNOWN)
            info.connections = self.graph.degree(entity_id)
            
            for neighbor in self.graph.neighbors(entity_id):
                neighbor_name = self.id_to_name.get(neighbor, neighbor)
                edge_data = self.graph.get_edge_data(entity_id, neighbor)
                if edge_data:
                    relation = list(edge_data.values())[0].get("label", "")
                    info.connected_entities.append(ConnectedEntity(
                        name=neighbor_name,
                        relation=relation,
                        direction="outgoing"
                    ))
            
            for predecessor in self.graph.predecessors(entity_id):
                pred_name = self.id_to_name.get(predecessor, predecessor)
                edge_data = self.graph.get_edge_data(predecessor, entity_id)
                if edge_data:
                    relation = list(edge_data.values())[0].get("label", "")
                    info.connected_entities.append(ConnectedEntity(
                        name=pred_name,
                        relation=relation,
                        direction="incoming"
                    ))
        
        for i, triple in enumerate(self.triples):
            head_id = get_triple_head_id(triple)
            tail_id = get_triple_tail_id(triple)
            if head_id == entity_id or tail_id == entity_id:
                head_name = self.id_to_name.get(head_id, get_triple_head_name(triple))
                tail_name = self.id_to_name.get(tail_id, get_triple_tail_name(triple))
                info.triples.append(TripleInfo(
                    index=i,
                    head=head_name,
                    relation=triple.relation,
                    tail=tail_name,
                ))
        
        return info
    
    def get_triple_info(self, triple_index: int) -> TripleInfo:
        """Retrieve detailed information about a triple by index."""
        
        if not self.triples or triple_index < 0 or triple_index >= len(self.triples):
            return TripleInfo(
                index=triple_index,
                head="",
                relation="",
                tail="",
                error=f"Triple index {triple_index} out of range"
            )
        
        triple = self.triples[triple_index]
        head_id = get_triple_head_id(triple)
        tail_id = get_triple_tail_id(triple)
        head_name = self.id_to_name.get(head_id, get_triple_head_name(triple))
        tail_name = self.id_to_name.get(tail_id, get_triple_tail_name(triple))
        
        info = TripleInfo(
            index=triple_index,
            head=head_name,
            head_id=head_id,
            relation=triple.relation,
            tail=tail_name,
            tail_id=tail_id,
        )
        
        if self.graph:
            if self.graph.has_node(head_id):
                info.head_properties = dict(self.graph.nodes[head_id])
            if self.graph.has_node(tail_id):
                info.tail_properties = dict(self.graph.nodes[tail_id])
        
        return info
    
    def search_entities(self, query: str, limit: int = 10) -> List[Entity]:
        """Search for entities by name."""
        query_lower = query.lower()
        results = []
        for eid, name in self.id_to_name.items():
            if query_lower in name.lower():
                results.append()
                if len(results) >= limit:
                    break
        return results
    
    def get_related_triples(self, entity_name: str, max_depth: int = 1, id: str = "") -> List[TripleInfo]:
        """Get triples directly related to an entity (where entity is head or tail)."""
        entity_id = None
        entity_name_lower = entity_name.lower().strip()
        if id:
            entity_id = id
        else:
        # Try exact match first
            for eid, name in self.id_to_name.items():
                if name.lower() == entity_name_lower:
                    entity_id = eid
                    break
        
        # Try partial match if exact match fails
        if not entity_id:
            for eid, name in self.id_to_name.items():
                name_lower = name.lower()
                # Check if entity_name is contained in name or vice versa
                if entity_name_lower in name_lower or name_lower in entity_name_lower:
                    entity_id = eid
                    break
        
        if not entity_id:
            return []
        
        # Only get triples where the entity is directly involved (head or tail)
        related_triples = []
        for i, triple in enumerate(self.triples):
            head_id = get_triple_head_id(triple)
            tail_id = get_triple_tail_id(triple)
            
            # Only include if the entity is directly the head or tail
            if head_id == entity_id or tail_id == entity_id:
                head_name = self.id_to_name.get(head_id, get_triple_head_name(triple))
                tail_name = self.id_to_name.get(tail_id, get_triple_tail_name(triple))
                related_triples.append(TripleInfo(
                    index=i,
                    head=head_name,
                    relation=triple.relation,
                    tail=tail_name,
                ))
        
        return related_triples
    
    def calculate_stats(self) -> GraphInfo:
        """Calculate current graph statistics."""
        return GraphInfo(
            total_triples=len(self.triples),
            total_entities=len(self.id_to_name),
            graph_nodes=self.graph.number_of_nodes() if self.graph else 0,
            graph_edges=self.graph.number_of_edges() if self.graph else 0,
            triples_changed=len(self.triples) - len(self._original_triples),
        )
