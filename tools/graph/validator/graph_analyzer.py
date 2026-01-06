"""
Graph Analyzer: Analyzes graph structure and builds context for LLM.
"""
from typing import Dict, Any, List, Optional
import networkx as nx
import logging

logger = logging.getLogger(__name__)

from tools.graph.Triple import Triple


class GraphAnalyzer:
    """Analyzes graph structure and builds context for LLM."""
    
    def __init__(
        self,
        graph: Optional[nx.MultiDiGraph] = None,
        triples: Optional[List[Triple]] = None,
        id_to_name: Optional[Dict[str, str]] = None,
    ):
        self.graph = graph
        self.triples = triples or []
        self.id_to_name = id_to_name or {}
    
    def build_context(self) -> Dict[str, Any]:
        """Build context information for the LLM."""
        logger.debug("GraphAnalyzer: Building context")
        context = {
            "num_nodes": 0,
            "num_edges": 0,
            "num_triples": len(self.triples),
            "entities": [],
            "triples_summary": [],
            "potential_issues": [],
        }
        
        if self.graph:
            context["num_nodes"] = self.graph.number_of_nodes()
            context["num_edges"] = self.graph.number_of_edges()
            logger.debug(f"Graph stats: {context['num_nodes']} nodes, {context['num_edges']} edges")
            
            # Extract entity information - use human-readable names with properties
            entities = []
            for node_id, node_data in self.graph.nodes(data=True):
                node_type = node_data.get("node_type", "UNKNOWN")
                name = self.id_to_name.get(node_id, node_data.get("name", node_id))
                
                if node_type != "ASSERTION" and node_type != "CLAIM_CONCEPT":
                    entity_info = {
                        "name": name,  # Only show name, not ID
                        "type": node_type,
                    }
                    
                    # Include entity properties
                    properties = {k: v for k, v in node_data.items() 
                                if k not in ("node_type", "name") and not k.startswith("_")}
                    if properties:
                        entity_info["properties"] = properties
                    
                    # Include connection count
                    in_degree = self.graph.in_degree(node_id)
                    out_degree = self.graph.out_degree(node_id)
                    entity_info["connections"] = in_degree + out_degree
                    
                    entities.append(entity_info)
            
            context["entities"] = entities
        
        # Extract triple information - use human-readable names only
        triples_summary = []
        for i, triple in enumerate(self.triples):
            # Get human-readable names, prefer id_to_name mapping
            head_id = getattr(triple.head, "ref", None) or getattr(triple.head, "id", None) or getattr(triple.head, "ref_short", None)
            tail_id = getattr(triple.tail, "ref", None) or getattr(triple.tail, "id", None) or getattr(triple.tail, "ref_short", None)
            
            head_name = self.id_to_name.get(head_id) or getattr(triple.head, "name", None) or getattr(triple.head, "text", None) or str(triple.head)
            tail_name = self.id_to_name.get(tail_id) or getattr(triple.tail, "name", None) or getattr(triple.tail, "text", None) or str(triple.tail)
            
            # Store IDs internally but don't show them to LLM in prompt
            triples_summary.append({
                "index": i,
                "head": head_name,
                "relation": triple.relation,
                "tail": tail_name,
                "head_id": head_id,  # Internal use only, not shown in prompt
                "tail_id": tail_id,  # Internal use only, not shown in prompt
            })
        
        context["triples_summary"] = triples_summary
        logger.debug(f"Context built: {len(context['entities'])} entities, {len(context['triples_summary'])} triples")
        
        return context

