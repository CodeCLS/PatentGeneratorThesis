"""
GraphRAG: Retrieval-Augmented Generation system for knowledge graphs.
Retrieves relevant subgraphs, entities, and relationships to enhance claim drafting.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
import networkx as nx
from collections import defaultdict, deque
import json

from tools.graph.Triple import Triple
from tools.api.llm_api_repo import LLmApi_Repo
from tools.graph.visualizer import GraphVisualizer


@dataclass
class RetrievedContext:
    """Context retrieved from the graph for RAG."""
    relevant_entities: List[Dict[str, Any]]
    relevant_triples: List[Dict[str, Any]]
    subgraph_summary: str
    key_relationships: List[Dict[str, Any]]
    entity_paths: List[List[str]]  # Paths between important entities


class GraphRAG:
    """
    Graph-based Retrieval-Augmented Generation system.
    
    Retrieves relevant information from a knowledge graph to enhance LLM prompts.
    Uses graph traversal, entity importance, and relationship analysis.
    """
    
    def __init__(
        self,
        G: Optional[nx.MultiDiGraph] = None,
        triples: Optional[List[Triple]] = None,
        id_to_name: Optional[Dict[str, str]] = None,
        api_repo: Optional[LLmApi_Repo] = None,
    ):
        """
        Initialize GraphRAG system.
        
        Args:
            G: NetworkX graph (optional, can be built from triples)
            triples: List of Triple objects (optional, used to build graph if G not provided)
            id_to_name: Mapping from entity ID to display name
            api_repo: Optional LLM API repository for semantic queries
        """
        self.api_repo = api_repo or LLmApi_Repo()
        self.id_to_name = id_to_name or {}
        
        # Build graph if not provided
        if G is None and triples:
            visualizer = GraphVisualizer()
            G = visualizer.build_graph(triples, deduplicate=True)
            # Build id_to_name if not provided
            if not self.id_to_name:
                self.id_to_name = GraphVisualizer.build_id_to_name_map_from_triples(triples)
        
        self.G = G or nx.MultiDiGraph()
        
        # Cache for entity importance scores
        self._importance_cache: Dict[str, float] = {}
        
        # Cache for entity types
        self._entity_types: Dict[str, str] = {}
        self._build_entity_cache()
    
    def _build_entity_cache(self):
        """Build cache of entity types and names from graph."""
        for node_id in self.G.nodes():
            node_data = self.G.nodes[node_id]
            node_type = node_data.get("node_type", "UNKNOWN")
            self._entity_types[node_id] = node_type
            
            # Store name if available
            if "name" in node_data and node_id not in self.id_to_name:
                self.id_to_name[node_id] = node_data["name"]
    
    def retrieve_for_claim_bundle(
        self,
        assertion_ids: List[str],
        max_entities: int = 20,
        max_triples: int = 30,
        max_depth: int = 2,
    ) -> RetrievedContext:
        """
        Retrieve relevant context for a claim bundle based on its assertions.
        
        Args:
            assertion_ids: List of assertion node IDs in the bundle
            max_entities: Maximum number of entities to retrieve
            max_triples: Maximum number of triples/relationships to retrieve
            max_depth: Maximum graph traversal depth from assertion nodes
            
        Returns:
            RetrievedContext with relevant information
        """
        # Find all entities connected to these assertions
        relevant_entity_ids: Set[str] = set()
        relevant_triples_data: List[Dict[str, Any]] = []
        
        # Traverse from assertion nodes
        for assertion_id in assertion_ids:
            if not self.G.has_node(assertion_id):
                continue
            
            # Get entities connected to this assertion (SUBJECT and OBJECT)
            for target in self.G.successors(assertion_id):
                edge_data = self.G.get_edge_data(assertion_id, target)
                if edge_data:
                    for key, data in edge_data.items():
                        if data.get("label") in ("SUBJECT", "OBJECT"):
                            relevant_entity_ids.add(target)
            
            # Also get entities that connect to this assertion's entities
            for entity_id in list(relevant_entity_ids):
                self._traverse_entity_neighborhood(
                    entity_id,
                    relevant_entity_ids,
                    max_depth=max_depth,
                )
        
        # Extract entity information
        relevant_entities = self._extract_entity_info(
            list(relevant_entity_ids)[:max_entities]
        )
        
        # Extract triples/relationships
        relevant_triples_data = self._extract_triples_from_entities(
            list(relevant_entity_ids),
            max_triples=max_triples,
        )
        
        # Find key relationships (important connections)
        key_relationships = self._find_key_relationships(
            list(relevant_entity_ids),
            max_relationships=10,
        )
        
        # Find paths between important entities
        entity_paths = self._find_entity_paths(
            list(relevant_entity_ids),
            max_paths=5,
        )
        
        # Generate subgraph summary
        subgraph_summary = self._generate_subgraph_summary(
            relevant_entities,
            relevant_triples_data,
            key_relationships,
        )
        
        return RetrievedContext(
            relevant_entities=relevant_entities,
            relevant_triples=relevant_triples_data,
            subgraph_summary=subgraph_summary,
            key_relationships=key_relationships,
            entity_paths=entity_paths,
        )
    
    def retrieve_by_query(
        self,
        query: str,
        max_entities: int = 15,
        max_triples: int = 20,
        use_semantic_search: bool = True,
    ) -> RetrievedContext:
        """
        Retrieve relevant context based on a text query.
        
        Args:
            query: Text query describing what to retrieve
            max_entities: Maximum number of entities to retrieve
            max_triples: Maximum number of triples to retrieve
            use_semantic_search: Whether to use LLM for semantic matching
            
        Returns:
            RetrievedContext with relevant information
        """
        # Find relevant entities by matching query terms
        relevant_entity_ids: Set[str] = set()
        
        if use_semantic_search:
            # Use LLM to identify relevant entities
            relevant_entity_ids = self._semantic_entity_search(query, top_k=max_entities)
        else:
            # Simple keyword matching
            query_lower = query.lower()
            for node_id in self.G.nodes():
                node_data = self.G.nodes[node_id]
                node_type = node_data.get("node_type", "")
                
                # Skip assertion and claim nodes
                if node_type in ("ASSERTION", "CLAIM_CONCEPT", "LEGAL_CLAIM_TEXT"):
                    continue
                
                # Check entity name
                entity_name = self.id_to_name.get(node_id, "")
                if entity_name and any(term in entity_name.lower() for term in query_lower.split()):
                    relevant_entity_ids.add(node_id)
        
        # Expand to connected entities
        for entity_id in list(relevant_entity_ids):
            self._traverse_entity_neighborhood(
                entity_id,
                relevant_entity_ids,
                max_depth=1,
            )
        
        # Extract information
        relevant_entities = self._extract_entity_info(
            list(relevant_entity_ids)[:max_entities]
        )
        
        relevant_triples_data = self._extract_triples_from_entities(
            list(relevant_entity_ids),
            max_triples=max_triples,
        )
        
        key_relationships = self._find_key_relationships(
            list(relevant_entity_ids),
            max_relationships=10,
        )
        
        entity_paths = self._find_entity_paths(
            list(relevant_entity_ids),
            max_paths=5,
        )
        
        subgraph_summary = self._generate_subgraph_summary(
            relevant_entities,
            relevant_triples_data,
            key_relationships,
        )
        
        return RetrievedContext(
            relevant_entities=relevant_entities,
            relevant_triples=relevant_triples_data,
            subgraph_summary=subgraph_summary,
            key_relationships=key_relationships,
            entity_paths=entity_paths,
        )
    
    def _traverse_entity_neighborhood(
        self,
        entity_id: str,
        visited: Set[str],
        max_depth: int = 2,
        current_depth: int = 0,
    ):
        """Traverse entity neighborhood to find connected entities."""
        if current_depth >= max_depth or entity_id in visited:
            return
        
        visited.add(entity_id)
        
        if current_depth < max_depth - 1:
            # Get neighbors (both incoming and outgoing)
            for neighbor in list(self.G.successors(entity_id)) + list(self.G.predecessors(entity_id)):
                node_data = self.G.nodes.get(neighbor, {})
                node_type = node_data.get("node_type", "")
                
                # Only traverse entity nodes, skip assertion/claim nodes
                if node_type not in ("ASSERTION", "CLAIM_CONCEPT", "LEGAL_CLAIM_TEXT"):
                    self._traverse_entity_neighborhood(
                        neighbor,
                        visited,
                        max_depth,
                        current_depth + 1,
                    )
    
    def _extract_entity_info(self, entity_ids: List[str]) -> List[Dict[str, Any]]:
        """Extract information about entities."""
        entities_info = []
        
        for entity_id in entity_ids:
            if not self.G.has_node(entity_id):
                continue
            
            node_data = self.G.nodes[entity_id]
            entity_name = self.id_to_name.get(entity_id, entity_id)
            entity_type = node_data.get("node_type", "UNKNOWN")
            
            # Count connections
            in_degree = self.G.in_degree(entity_id)
            out_degree = self.G.out_degree(entity_id)
            
            entities_info.append({
                "id": entity_id,
                "name": entity_name,
                "type": entity_type,
                "in_degree": in_degree,
                "out_degree": out_degree,
                "importance": self._get_entity_importance(entity_id),
            })
        
        # Sort by importance
        entities_info.sort(key=lambda x: x["importance"], reverse=True)
        return entities_info
    
    def _extract_triples_from_entities(
        self,
        entity_ids: List[str],
        max_triples: int = 30,
    ) -> List[Dict[str, Any]]:
        """Extract triples/relationships involving the given entities."""
        triples_data = []
        seen_triples = set()
        
        for entity_id in entity_ids:
            # Outgoing edges (entity -> other)
            for target in self.G.successors(entity_id):
                edge_data = self.G.get_edge_data(entity_id, target)
                if edge_data:
                    for key, data in edge_data.items():
                        # Skip assertion links
                        if data.get("edge_type") == "ASSERTION_LINK":
                            continue
                        
                        relation = data.get("label", "")
                        target_name = self.id_to_name.get(target, target)
                        source_name = self.id_to_name.get(entity_id, entity_id)
                        
                        triple_key = (entity_id, target, relation)
                        if triple_key not in seen_triples:
                            seen_triples.add(triple_key)
                            triples_data.append({
                                "head": source_name,
                                "relation": relation,
                                "tail": target_name,
                                "head_id": entity_id,
                                "tail_id": target,
                            })
            
            # Incoming edges (other -> entity)
            for source in self.G.predecessors(entity_id):
                edge_data = self.G.get_edge_data(source, entity_id)
                if edge_data:
                    for key, data in edge_data.items():
                        # Skip assertion links
                        if data.get("edge_type") == "ASSERTION_LINK":
                            continue
                        
                        relation = data.get("label", "")
                        source_name = self.id_to_name.get(source, source)
                        target_name = self.id_to_name.get(entity_id, entity_id)
                        
                        triple_key = (source, entity_id, relation)
                        if triple_key not in seen_triples:
                            seen_triples.add(triple_key)
                            triples_data.append({
                                "head": source_name,
                                "relation": relation,
                                "tail": target_name,
                                "head_id": source,
                                "tail_id": entity_id,
                            })
        
        # Limit and return
        return triples_data[:max_triples]
    
    def _find_key_relationships(
        self,
        entity_ids: List[str],
        max_relationships: int = 10,
    ) -> List[Dict[str, Any]]:
        """Find key relationships between entities (high-degree connections)."""
        relationships = []
        
        for i, entity1_id in enumerate(entity_ids):
            for entity2_id in entity_ids[i+1:]:
                # Check if there's a direct connection
                if self.G.has_edge(entity1_id, entity2_id):
                    edge_data = self.G.get_edge_data(entity1_id, entity2_id)
                    if edge_data:
                        for key, data in edge_data.items():
                            if data.get("edge_type") == "ASSERTION_LINK":
                                continue
                            
                            relationships.append({
                                "from": self.id_to_name.get(entity1_id, entity1_id),
                                "to": self.id_to_name.get(entity2_id, entity2_id),
                                "relation": data.get("label", ""),
                            })
                
                if self.G.has_edge(entity2_id, entity1_id):
                    edge_data = self.G.get_edge_data(entity2_id, entity1_id)
                    if edge_data:
                        for key, data in edge_data.items():
                            if data.get("edge_type") == "ASSERTION_LINK":
                                continue
                            
                            relationships.append({
                                "from": self.id_to_name.get(entity2_id, entity2_id),
                                "to": self.id_to_name.get(entity1_id, entity1_id),
                                "relation": data.get("label", ""),
                            })
        
        return relationships[:max_relationships]
    
    def _find_entity_paths(
        self,
        entity_ids: List[str],
        max_paths: int = 5,
    ) -> List[List[str]]:
        """Find paths between important entities."""
        paths = []
        
        # Find paths between high-importance entities
        important_entities = [
            eid for eid in entity_ids
            if self._get_entity_importance(eid) > 0.5
        ][:10]  # Limit to top 10 for performance
        
        for i, entity1_id in enumerate(important_entities):
            for entity2_id in important_entities[i+1:]:
                try:
                    # Find shortest path
                    if nx.has_path(self.G, entity1_id, entity2_id):
                        path = nx.shortest_path(self.G, entity1_id, entity2_id)
                        # Convert to names
                        path_names = [
                            self.id_to_name.get(node_id, node_id)
                            for node_id in path
                        ]
                        paths.append(path_names)
                        
                        if len(paths) >= max_paths:
                            break
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
            
            if len(paths) >= max_paths:
                break
        
        return paths
    
    def _get_entity_importance(self, entity_id: str) -> float:
        """Calculate entity importance score (cached)."""
        if entity_id in self._importance_cache:
            return self._importance_cache[entity_id]
        
        if not self.G.has_node(entity_id):
            return 0.0
        
        # Simple importance: based on degree centrality
        in_degree = self.G.in_degree(entity_id)
        out_degree = self.G.out_degree(entity_id)
        total_degree = in_degree + out_degree
        
        # Normalize (assuming max degree is around 20-30)
        importance = min(1.0, total_degree / 20.0)
        
        self._importance_cache[entity_id] = importance
        return importance
    
    def _semantic_entity_search(self, query: str, top_k: int = 15) -> Set[str]:
        """Use LLM to find entities semantically related to the query."""
        # Get all entity names
        entity_list = []
        for node_id in self.G.nodes():
            node_data = self.G.nodes[node_id]
            node_type = node_data.get("node_type", "")
            
            # Skip non-entity nodes
            if node_type in ("ASSERTION", "CLAIM_CONCEPT", "LEGAL_CLAIM_TEXT"):
                continue
            
            entity_name = self.id_to_name.get(node_id, "")
            if entity_name:
                entity_list.append({
                    "id": node_id,
                    "name": entity_name,
                    "type": node_type,
                })
        
        if not entity_list:
            return set()
        
        # Use LLM to find relevant entities
        prompt = (
            f"Given the following query about a patent invention, identify which entities are most relevant.\n\n"
            f"Query: {query}\n\n"
            f"Available entities (up to 50 shown):\n"
        )
        
        for i, entity in enumerate(entity_list[:50], 1):
            prompt += f"{i}. {entity['name']} (type: {entity['type']})\n"
        
        prompt += (
            "\nReturn a JSON array of entity names that are most relevant to the query.\n"
            "Only include entities that are directly relevant to understanding or implementing the query.\n"
            "Return ONLY the JSON array, no other text.\n"
            "Example: [\"water tank\", \"bubble generator\", \"convection\"]\n"
        )
        
        try:
            response = self.api_repo.chat(prompt)
            
            # Parse response
            if isinstance(response, dict):
                response_text = response.get("content", response.get("text", ""))
            else:
                response_text = str(response)
            
            # Extract JSON array
            response_text = response_text.strip()
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            entity_names = json.loads(response_text)
            
            # Map names back to IDs
            name_to_id = {v: k for k, v in self.id_to_name.items()}
            relevant_ids = set()
            
            for name in entity_names:
                if name in name_to_id:
                    relevant_ids.add(name_to_id[name])
                else:
                    # Try fuzzy matching
                    for entity_id, entity_name in self.id_to_name.items():
                        if name.lower() in entity_name.lower() or entity_name.lower() in name.lower():
                            relevant_ids.add(entity_id)
                            break
            
            return relevant_ids
            
        except Exception as e:
            print(f"⚠️  Error in semantic entity search: {e}")
            return set()
    
    def _generate_subgraph_summary(
        self,
        entities: List[Dict[str, Any]],
        triples: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
    ) -> str:
        """Generate a natural language summary of the retrieved subgraph."""
        if not entities:
            return "No relevant entities found."
        
        summary_parts = []
        
        # Entity summary
        summary_parts.append(f"Relevant entities ({len(entities)}):")
        for entity in entities[:10]:  # Top 10
            summary_parts.append(f"  - {entity['name']} ({entity['type']})")
        
        # Key relationships
        if relationships:
            summary_parts.append(f"\nKey relationships ({len(relationships)}):")
            for rel in relationships[:5]:  # Top 5
                summary_parts.append(f"  - {rel['from']} --[{rel['relation']}]--> {rel['to']}")
        
        # Triple summary
        if triples:
            summary_parts.append(f"\nRelevant triples ({len(triples)}):")
            for triple in triples[:10]:  # Top 10
                summary_parts.append(
                    f"  - {triple['head']} --[{triple['relation']}]--> {triple['tail']}"
                )
        
        return "\n".join(summary_parts)
    
    def format_context_for_prompt(self, context: RetrievedContext) -> str:
        """Format retrieved context as text for LLM prompt."""
        parts = []
        
        parts.append("=== RETRIEVED GRAPH CONTEXT ===\n")
        
        # Subgraph summary
        parts.append(context.subgraph_summary)
        
        # Key relationships
        if context.key_relationships:
            parts.append("\n=== KEY RELATIONSHIPS ===")
            for rel in context.key_relationships:
                parts.append(f"{rel['from']} --[{rel['relation']}]--> {rel['to']}")
        
        # Entity paths
        if context.entity_paths:
            parts.append("\n=== ENTITY CONNECTION PATHS ===")
            for path in context.entity_paths:
                parts.append(" → ".join(path))
        
        parts.append("\n=== END RETRIEVED CONTEXT ===\n")
        
        return "\n".join(parts)

