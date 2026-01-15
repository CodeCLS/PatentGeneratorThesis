"""
GraphRAG: Retrieval-Augmented Generation system for knowledge graphs.
Retrieves relevant subgraphs, entities, and relationships to enhance claim drafting.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from dataclasses import dataclass
import networkx as nx
from collections import defaultdict, deque
import json
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️  Warning: faiss not available. Install with: pip install faiss-cpu")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️  Warning: sentence-transformers not available. Install with: pip install sentence-transformers")

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
        self.triples: List[Triple] = triples or []
        
        # Cache for entity importance scores
        self._importance_cache: Dict[str, float] = {}
        
        # Cache for entity types
        self._entity_types: Dict[str, str] = {}
        self._build_entity_cache()
        
        # Faiss index for semantic triple search
        self._faiss_index: Optional[Any] = None
        self._triple_index_map: Dict[int, int] = {}  # Faiss index -> triple index
        self._embedding_model: Optional[Any] = None
        self._embedding_dim: int = 384  # Default for all-MiniLM-L6-v2
        self._faiss_built: bool = False
    
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
    
    def _initialize_embedding_model(self) -> None:
        """Initialize the sentence transformer model for embeddings."""
        if self._embedding_model is not None:
            return
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                # Use a lightweight, fast model suitable for semantic search
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                self._embedding_dim = self._embedding_model.get_sentence_embedding_dimension()
                print(f"✓ Initialized embedding model (dim={self._embedding_dim})")
            except Exception as e:
                print(f"⚠️  Failed to load sentence transformer: {e}")
                self._embedding_model = None
        else:
            print("⚠️  sentence-transformers not available, using fallback embedding")
            self._embedding_model = None
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a text string."""
        if self._embedding_model is None:
            self._initialize_embedding_model()
        
        if self._embedding_model is not None:
            # Use sentence transformer
            embedding = self._embedding_model.encode(text, normalize_embeddings=True)
            return np.asarray(embedding, dtype=np.float32)
        else:
            # Fallback: simple hash-based embedding (not ideal but works)
            return self._hash_embedding_fallback(text)
    
    def _hash_embedding_fallback(self, text: str) -> np.ndarray:
        """Fallback hash-based embedding when sentence-transformers is not available."""
        import hashlib
        # Create a simple hash-based embedding
        hash_obj = hashlib.sha256(text.encode('utf-8'))
        hash_bytes = hash_obj.digest()
        # Convert to float32 array (repeat hash if needed to fill dimension)
        embedding = np.zeros(self._embedding_dim, dtype=np.float32)
        for i in range(self._embedding_dim):
            byte_idx = i % len(hash_bytes)
            embedding[i] = float(hash_bytes[byte_idx]) / 255.0
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.astype(np.float32)
    
    def _triple_to_text(self, triple: Triple) -> str:
        """Convert a triple to a text representation for embedding."""
        head_name = triple.head.name if hasattr(triple.head, 'name') else str(triple.head)
        tail_name = triple.tail.name if hasattr(triple.tail, 'name') else str(triple.tail)
        relation = triple.relation
        
        # Create a descriptive text representation
        text = f"{head_name} {relation} {tail_name}"
        
        # Add labels if available
        if hasattr(triple.head, 'label') and triple.head.label:
            text += f" ({triple.head.label})"
        if hasattr(triple.tail, 'label') and triple.tail.label:
            text += f" ({triple.tail.label})"
        
        return text
    
    def _build_faiss_index(self, triples: Optional[List[Triple]] = None, force_rebuild: bool = False) -> None:
        """Build or update the Faiss index for triple embeddings."""
        if not FAISS_AVAILABLE:
            raise RuntimeError("faiss is not available. Install with: pip install faiss-cpu")
        
        if self._faiss_built and not force_rebuild:
            return
        
        triples_to_index = triples if triples is not None else self.triples
        
        if not triples_to_index:
            print("⚠️  No triples available to index")
            self._faiss_index = None
            self._faiss_built = False
            return
        
        self._initialize_embedding_model()
        
        # Collect embeddings
        embeddings_list = []
        self._triple_index_map = {}
        
        for idx, triple in enumerate(triples_to_index):
            # Check if triple already has an embedding
            if triple.embedding and len(triple.embedding) > 0:
                embedding = np.asarray(triple.embedding, dtype=np.float32)
                # Ensure correct dimension
                if len(embedding) != self._embedding_dim:
                    # Regenerate if dimension mismatch
                    triple_text = self._triple_to_text(triple)
                    embedding = self._get_embedding(triple_text)
                    triple.set_embedding(embedding.tolist())
            else:
                # Generate embedding
                triple_text = self._triple_to_text(triple)
                embedding = self._get_embedding(triple_text)
                triple.set_embedding(embedding.tolist())
            
            embeddings_list.append(embedding)
            self._triple_index_map[len(embeddings_list) - 1] = idx
        
        if not embeddings_list:
            return
        
        # Stack embeddings into a matrix
        embeddings_matrix = np.vstack(embeddings_list).astype(np.float32)
        
        # Normalize for cosine similarity (inner product)
        faiss.normalize_L2(embeddings_matrix)
        
        # Create Faiss index (using inner product for cosine similarity with normalized vectors)
        self._faiss_index = faiss.IndexFlatIP(self._embedding_dim)
        self._faiss_index.add(embeddings_matrix)
        
        self._faiss_built = True
        print(f"✓ Built Faiss index with {len(embeddings_list)} triples")
    
    def find_similar_triples(
        self,
        query_text: str,
        top_k: int = 10,
        similarity_threshold: float = 0.0,
        rebuild_index: bool = False,
    ) -> List[Tuple[Triple, float]]:
        """
        Find triples that are semantically similar to the given sentence(s) using Faiss.
        
        Args:
            query_text: A sentence or multiple sentences (string) to search for
            top_k: Number of top similar triples to return
            similarity_threshold: Minimum similarity score (0.0-1.0) to include a triple
            rebuild_index: Whether to rebuild the Faiss index (useful if triples changed)
            
        Returns:
            List of tuples (triple, similarity_score) sorted by similarity (highest first)
        """
        if not FAISS_AVAILABLE:
            raise RuntimeError("faiss is not available. Install with: pip install faiss-cpu")
        
        # Build index if not built or if forced
        if not self._faiss_built or rebuild_index:
            self._build_faiss_index(force_rebuild=rebuild_index)
        
        if self._faiss_index is None or self._faiss_index.ntotal == 0:
            print("⚠️  Faiss index is empty or not built")
            return []
        
        # Get embedding for query text
        self._initialize_embedding_model()
        query_embedding = self._get_embedding(query_text)
        
        # Normalize query embedding
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_embedding)
        
        # Search in Faiss index
        k = min(top_k, self._faiss_index.ntotal)
        similarities, indices = self._faiss_index.search(query_embedding, k)
        
        # Map results back to triples
        results = []
        triples_to_search = self.triples
        
        for sim_score, faiss_idx in zip(similarities[0], indices[0]):
            if faiss_idx < 0 or faiss_idx >= len(self._triple_index_map):  # Invalid index
                continue
            
            # Map Faiss index to triple index
            triple_idx = self._triple_index_map.get(faiss_idx, faiss_idx)
            
            if 0 <= triple_idx < len(triples_to_search):
                triple = triples_to_search[triple_idx]
                # Clamp similarity to [0, 1] (inner product of normalized vectors = cosine similarity)
                similarity = float(np.clip(sim_score, 0.0, 1.0))
                
                if similarity >= similarity_threshold:
                    results.append((triple, similarity))
        
        # Sort by similarity (already sorted by Faiss, but ensure descending order)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def find_similar_triples_simple(
        self,
        query_text: str,
        top_k: int = 10,
    ) -> List[Triple]:
        """
        Simplified version that returns only triples (without similarity scores).
        
        Args:
            query_text: A sentence or multiple sentences (string) to search for
            top_k: Number of top similar triples to return
            
        Returns:
            List of triples sorted by similarity (most similar first)
        """
        results = self.find_similar_triples(query_text, top_k=top_k)
        return [triple for triple, _ in results]
    
    def get_prioritized_entities(
        self,
        max_entities: int = 30,
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        Extract prioritized entities from the knowledge graph, grouped by label.
        Prioritizes entities with labels: INVENTION, COMPONENT, SUBSYSTEM, COMPOSITION, 
        PROCESS_STEP, METHOD, FUNCTION.
        
        Args:
            max_entities: Maximum number of entities to return (default: 30)
            
        Returns:
            Dictionary mapping label -> list of entities with name and label
            Example: {
                "INVENTION": [{"name": "display device", "label": "INVENTION"}, ...],
                "COMPONENT": [{"name": "water tank", "label": "COMPONENT"}, ...],
                ...
            }
        """
        # Priority order for labels (higher priority = lower number)
        label_priority = {
            "INVENTION": 1,
            "COMPONENT": 2,
            "SUBSYSTEM": 3,
            "COMPOSITION": 4,
            "PROCESS_STEP": 5,
            "METHOD": 6,
            "FUNCTION": 7,
        }
        
        # Collect unique entities from triples
        entity_map: Dict[str, Dict[str, str]] = {}  # name -> {name, label}
        
        for triple in self.triples:
            # Process head entity
            head_name = triple.head.name if hasattr(triple.head, 'name') else str(triple.head)
            head_label = triple.head.label if hasattr(triple.head, 'label') else "UNCLASSIFIED_ENTITY"
            
            # Normalize label to uppercase
            head_label = head_label.upper()
            
            # Only include if it's a prioritized label or if we haven't reached max_entities yet
            if head_name and head_name.strip():
                if head_name not in entity_map:
                    entity_map[head_name] = {
                        "name": head_name.strip(),
                        "label": head_label,
                    }
            
            # Process tail entity
            tail_name = triple.tail.name if hasattr(triple.tail, 'name') else str(triple.tail)
            tail_label = triple.tail.label if hasattr(triple.tail, 'label') else "UNCLASSIFIED_ENTITY"
            
            # Normalize label to uppercase
            tail_label = tail_label.upper()
            
            # Only include if it's a prioritized label or if we haven't reached max_entities yet
            if tail_name and tail_name.strip():
                if tail_name not in entity_map:
                    entity_map[tail_name] = {
                        "name": tail_name.strip(),
                        "label": tail_label,
                    }
        
        # Convert to list and prioritize
        entities_list = list(entity_map.values())
        
        # Sort by priority: prioritized labels first, then by name
        def get_priority(entity: Dict[str, str]) -> tuple:
            label = entity["label"]
            priority = label_priority.get(label, 999)  # Unprioritized labels get high priority number
            return (priority, entity["name"].lower())
        
        entities_list.sort(key=get_priority)
        
        # Take top max_entities
        entities_list = entities_list[:max_entities]
        
        # Group by label
        grouped: Dict[str, List[Dict[str, str]]] = {}
        for entity in entities_list:
            label = entity["label"]
            if label not in grouped:
                grouped[label] = []
            grouped[label].append(entity)
        
        # Sort groups by priority
        sorted_groups = {}
        for priority_label in sorted(label_priority.keys(), key=lambda x: label_priority[x]):
            if priority_label in grouped:
                sorted_groups[priority_label] = grouped[priority_label]
        
        # Add any remaining labels (non-prioritized)
        for label in sorted(grouped.keys()):
            if label not in sorted_groups:
                sorted_groups[label] = grouped[label]
        
        return sorted_groups
    
    def format_entities_for_prompt(self, max_entities: int = 30) -> str:
        """
        Format prioritized entities as a string for inclusion in LLM prompts.
        
        Args:
            max_entities: Maximum number of entities to include (default: 30)
            
        Returns:
            Formatted string with entities grouped by label
        """
        grouped_entities = self.get_prioritized_entities(max_entities=max_entities)
        
        if not grouped_entities:
            return ""
        
        lines = ["Knowledge Graph Entities (for reference):"]
        
        for label, entities in grouped_entities.items():
            lines.append(f"\n{label} entities:")
            for entity in entities:
                lines.append(f"- {entity['name']}")
        
        lines.append("\nNote: Independent claims should primarily focus on INVENTION entities.")
        lines.append("Dependent claims should focus on COMPONENT, MATERIAL, FUNCTION, or other")
        lines.append("entities that relate to their parent independent claim.")
        
        return "\n".join(lines)

