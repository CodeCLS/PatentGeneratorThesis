"""
Claim clustering system for knowledge graphs.

This module provides various strategies for clustering triples into patent claims,
distinguishing between independent (fundamental) and dependent (secondary) claims.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
import networkx as nx


@dataclass
class ClaimCluster:
    """
    Represents a claim cluster containing nodes and edges.
    
    Attributes:
        cluster_id: Unique identifier for the cluster
        nodes: Set of node IDs in this cluster
        edges: Set of edge tuples (head_id, tail_id, relation) in this cluster
        claim_type: "independent" (fundamental) or "dependent" (secondary)
        priority: Priority rank (lower = more important, for independent claims)
        metadata: Additional metadata about the cluster
    """
    cluster_id: int
    nodes: Set[str] = field(default_factory=set)
    edges: Set[Tuple[str, str, str]] = field(default_factory=set)
    claim_type: str = "dependent"  # "independent" or "dependent"
    priority: int = 999  # Lower = more important
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def size(self) -> int:
        """Return the number of edges in this cluster."""
        return len(self.edges)
    
    def node_count(self) -> int:
        """Return the number of nodes in this cluster."""
        return len(self.nodes)


class BaseClaimClusterer(ABC):
    """
    Abstract base class for claim clustering algorithms.
    
    All claim clusterers take a NetworkX graph and output a list of ClaimCluster
    objects, each labeled as "independent" (fundamental) or "dependent" (secondary).
    """
    
    def __init__(
        self, 
        min_cluster_size: int = 1, 
        max_clusters: Optional[int] = None,
        max_dependent_cluster_size: Optional[int] = None,
        forbidden_head_types: Optional[Set[str]] = None,
        forbidden_node_types: Optional[Set[str]] = None,
    ):
        """
        Initialize the clusterer.
        
        Args:
            min_cluster_size: Minimum number of edges required for a valid cluster
            max_clusters: Maximum number of independent clusters to create (None = unlimited)
            max_dependent_cluster_size: Maximum edges per dependent cluster (None = unlimited, splits large ones)
            forbidden_head_types: Set of entity types that should not appear as head nodes in claims (e.g., {"MATERIAL", "PRIOR_ART"})
            forbidden_node_types: Set of entity types that should not appear in independent claims at all (e.g., {"UNKNOWN", "PARAMETER"})
        """
        self.min_cluster_size = min_cluster_size
        self.max_clusters = max_clusters
        self.max_dependent_cluster_size = max_dependent_cluster_size
        self.forbidden_head_types = forbidden_head_types or set()
        if self.forbidden_head_types:
            # Normalize to uppercase
            self.forbidden_head_types = {t.upper() for t in self.forbidden_head_types}
        self.forbidden_node_types = forbidden_node_types or set()
        if self.forbidden_node_types:
            # Normalize to uppercase
            self.forbidden_node_types = {t.upper() for t in self.forbidden_node_types}
    
    @abstractmethod
    def cluster(self, G: nx.MultiDiGraph) -> List[ClaimCluster]:
        """
        Cluster the graph into independent and dependent claims.
        
        Args:
            G: NetworkX MultiDiGraph with nodes containing 'node_type' attribute
            
        Returns:
            List of ClaimCluster objects, sorted by priority (independent claims first)
        """
        pass
    
    def _get_node_type(self, G: nx.MultiDiGraph, node: str) -> str:
        """Get the node type from graph attributes."""
        return G.nodes[node].get("node_type", "UNKNOWN").upper()
    
    def _get_edge_relation(self, G: nx.MultiDiGraph, u: str, v: str, key: Any = 0) -> str:
        """Get the relation label from an edge."""
        if G.has_edge(u, v, key):
            return G.edges[u, v, key].get("label", "")
        return ""
    
    def _filter_forbidden_edges(
        self, 
        G: nx.MultiDiGraph, 
        edges: Set[Tuple[str, str, str]]
    ) -> Set[Tuple[str, str, str]]:
        """
        Filter out edges where the head node has a forbidden type.
        
        Args:
            G: NetworkX graph with node types
            edges: Set of edges (head, tail, relation) to filter
            
        Returns:
            Filtered set of edges
        """
        if not self.forbidden_head_types:
            return edges
        
        filtered_edges = set()
        for head, tail, relation in edges:
            head_type = self._get_node_type(G, head)
            if head_type not in self.forbidden_head_types:
                filtered_edges.add((head, tail, relation))
        
        return filtered_edges
    
    def _filter_cluster_forbidden_edges(
        self, 
        G: nx.MultiDiGraph, 
        cluster: ClaimCluster
    ) -> ClaimCluster:
        """
        Filter forbidden edges from a cluster and update nodes accordingly.
        
        Args:
            G: NetworkX graph with node types
            cluster: ClaimCluster to filter
            
        Returns:
            New ClaimCluster with forbidden edges removed
        """
        if not self.forbidden_head_types:
            return cluster
        
        filtered_edges = self._filter_forbidden_edges(G, cluster.edges)
        
        # Recalculate nodes from remaining edges
        filtered_nodes = set()
        for head, tail, _ in filtered_edges:
            filtered_nodes.add(head)
            filtered_nodes.add(tail)
        
        return ClaimCluster(
            cluster_id=cluster.cluster_id,
            nodes=filtered_nodes,
            edges=filtered_edges,
            claim_type=cluster.claim_type,
            priority=cluster.priority,
            metadata=cluster.metadata,
        )
    
    def _extract_chains_from_component(
        self,
        comp_graph: nx.MultiDiGraph,
        comp_edges: Set[Tuple[str, str, str]],
        max_chain_size: Optional[int] = None
    ) -> List[Set[Tuple[str, str, str]]]:
        """
        Extract connected chains/subgraphs from a component.
        Each chain is a connected subgraph that forms a logical dependent claim.
        Strictly enforces max_chain_size by stopping chain growth and starting new chains.
        
        Args:
            comp_graph: NetworkX graph of the component
            comp_edges: Set of all edges in the component
            max_chain_size: Maximum edges per chain (strictly enforced)
        
        Returns:
            List of edge sets, each representing a chain/subgraph
        """
        if not comp_edges:
            return []
        
        chains: List[Set[Tuple[str, str, str]]] = []
        used_edges: Set[Tuple[str, str, str]] = set()
        remaining_edges_list = list(comp_edges)
        
        # Keep extracting chains until all edges are used
        while remaining_edges_list:
            # Find the first unused edge to start a new chain
            start_edge = None
            for edge in remaining_edges_list:
                if edge not in used_edges:
                    start_edge = edge
                    break
            
            if start_edge is None:
                break  # All edges are used
            
            # Start a new chain from this edge
            chain_edges: Set[Tuple[str, str, str]] = {start_edge}
            chain_nodes: Set[str] = {start_edge[0], start_edge[1]}
            
            # Use BFS to find connected edges, building the chain
            # But stop immediately when we reach max_chain_size
            queue = deque([start_edge[0], start_edge[1]])  # Start from both head and tail
            visited_in_chain = {start_edge[0], start_edge[1]}
            
            # Continue BFS until we hit max_chain_size or run out of connected edges
            while queue:
                # Strictly enforce max_chain_size - stop immediately if reached
                if max_chain_size is not None and len(chain_edges) >= max_chain_size:
                    break
                
                current = queue.popleft()
                
                # Check all neighbors (both successors and predecessors)
                neighbors_to_check = list(set(comp_graph.successors(current)) | set(comp_graph.predecessors(current)))
                
                for neighbor in neighbors_to_check:
                    # Stop if we've reached max_chain_size (check before processing neighbor)
                    if max_chain_size is not None and len(chain_edges) >= max_chain_size:
                        break
                    
                    if neighbor in visited_in_chain:
                        continue
                    
                    # Find edges between current and neighbor
                    edges_to_add = []
                    if comp_graph.has_edge(current, neighbor):
                        for key in comp_graph[current][neighbor]:
                            edge = (current, neighbor, self._get_edge_relation(comp_graph, current, neighbor, key))
                            if edge in comp_edges and edge not in used_edges:
                                edges_to_add.append(edge)
                    if comp_graph.has_edge(neighbor, current):
                        for key in comp_graph[neighbor][current]:
                            edge = (neighbor, current, self._get_edge_relation(comp_graph, neighbor, current, key))
                            if edge in comp_edges and edge not in used_edges:
                                edges_to_add.append(edge)
                    
                    # Add edges one at a time, checking limit BEFORE each addition
                    for edge in edges_to_add:
                        # CRITICAL: Check BEFORE adding - if we're at or above limit, don't add and break
                        # Use strict check: if we have max_chain_size edges, we can't add more
                        if max_chain_size is not None:
                            if len(chain_edges) >= max_chain_size:
                                break
                            # Additional safety: if adding this edge would exceed limit, don't add it
                            if len(chain_edges) + 1 > max_chain_size:
                                break
                        
                        # Add the edge only if we haven't exceeded the limit
                        chain_edges.add(edge)
                        chain_nodes.add(neighbor)
                        if neighbor not in visited_in_chain:
                            visited_in_chain.add(neighbor)
                            queue.append(neighbor)
                        
                        # Immediate check after adding - if we exceeded, remove and break
                        if max_chain_size is not None and len(chain_edges) > max_chain_size:
                            chain_edges.remove(edge)
                            break
                
                # Break outer loop if we've reached max_chain_size
                if max_chain_size is not None and len(chain_edges) >= max_chain_size:
                    break
            
            # Final safety check: ensure we never exceed max_chain_size
            if max_chain_size is not None and len(chain_edges) > max_chain_size:
                # This should never happen, but if it does, trim to max_chain_size
                chain_edges_list = list(chain_edges)
                chain_edges = set(chain_edges_list[:max_chain_size])
            
            # Only create chain if it meets minimum size
            if len(chain_edges) >= self.min_cluster_size:
                # Final assertion: chain should never exceed max_chain_size
                if max_chain_size is not None:
                    assert len(chain_edges) <= max_chain_size, f"Chain has {len(chain_edges)} edges but max is {max_chain_size}"
                chains.append(chain_edges)
                used_edges.update(chain_edges)
            else:
                # If chain is too small, mark edges as used anyway to avoid infinite loop
                used_edges.update(chain_edges)
        
        return chains
    
    def _create_dependent_clusters(
        self, 
        remaining_edges: Set[Tuple[str, str, str]], 
        clusters: List[ClaimCluster],
        base_priority: int = 0
    ) -> None:
        """
        Create dependent clusters from remaining edges, splitting by connected chains/subgraphs.
        
        Each component is split into chains (connected subgraphs), where each chain becomes
        a separate dependent claim. This ensures that related triples stay together.
        
        Args:
            remaining_edges: Set of edges (head, tail, relation) not yet assigned
            clusters: List to append new clusters to
            base_priority: Base priority value for these dependent clusters
        """
        if not remaining_edges:
            return
        
        # Group remaining edges by connected components
        remaining_graph = nx.MultiDiGraph()
        for u, v, rel in remaining_edges:
            remaining_graph.add_edge(u, v, label=rel)
        
        for comp_id, component in enumerate(nx.weakly_connected_components(remaining_graph)):
            comp_edges = {
                (u, v, self._get_edge_relation(remaining_graph, u, v, k))
                for u, v, k in remaining_graph.edges(keys=True)
                if u in component and v in component
            }
            
            if not comp_edges:
                continue
            
            # Extract chains (connected subgraphs) from this component
            # Use max_dependent_cluster_size as max chain size if specified
            max_chain_size = self.max_dependent_cluster_size
            if max_chain_size is None:
                # If not specified, use a reasonable default to prevent huge chains
                # But this shouldn't happen if user set max_dependent_cluster_size
                max_chain_size = 50  # Fallback default
            chains = self._extract_chains_from_component(remaining_graph, comp_edges, max_chain_size)
            
            # Each chain becomes a dependent claim
            for dep_cluster_id, chain_edges in enumerate(chains):
                if len(chain_edges) >= self.min_cluster_size:
                    chain_nodes = set()
                    for u, v, _ in chain_edges:
                        chain_nodes.add(u)
                        chain_nodes.add(v)
                    
                    cluster = ClaimCluster(
                        cluster_id=len(clusters),
                        nodes=chain_nodes,
                        edges=chain_edges,
                        claim_type="dependent",
                        priority=base_priority + comp_id * 1000 + dep_cluster_id,
                        metadata={
                            "component_size": len(component),
                            "chain_size": len(chain_edges),
                            "split_by_chain": True
                        }
                    )
                    clusters.append(cluster)
    
    def _filter_clusters(self, clusters: List[ClaimCluster], G: nx.MultiDiGraph) -> List[ClaimCluster]:
        """
        Filter clusters by size and limit independent claims.
        
        Args:
            clusters: List of clusters to filter
            G: NetworkX graph (needed for filtering forbidden edges)
            
        Returns:
            Filtered list of clusters
        """
        # First, filter out forbidden edges from each cluster
        if self.forbidden_head_types:
            clusters = [self._filter_cluster_forbidden_edges(G, c) for c in clusters]
        
        # Filter by minimum size
        filtered = [c for c in clusters if c.size() >= self.min_cluster_size]
        
        # Sort by priority (independent first, then by priority value)
        independent = [c for c in filtered if c.claim_type == "independent"]
        dependent = [c for c in filtered if c.claim_type == "dependent"]
        
        independent.sort(key=lambda x: x.priority)
        dependent.sort(key=lambda x: x.priority)
        
        # Limit independent claims if specified, but keep dependent claims
        if self.max_clusters is not None:
            # Limit independent to max_clusters, but keep all dependent
            independent = independent[:self.max_clusters]
        
        return independent + dependent


class InDegreeClaimClusterer(BaseClaimClusterer):
    """
    Clusters claims based on in-degree (number of incoming edges).
    
    Strategy:
    - Nodes with high in-degree are considered fundamental (many triples point to them)
    - Clusters centered around high in-degree nodes are independent claims
    - Other clusters are dependent claims
    """
    
    def __init__(
        self,
        in_degree_threshold: float = 0.7,  # Top 70% by in-degree are fundamental
        cluster_radius: int = 2,  # How many hops to include around seed nodes
        min_cluster_size: int = 1,
        max_clusters: Optional[int] = None,
        max_dependent_cluster_size: Optional[int] = None,
    ):
        """
        Initialize the in-degree based clusterer.
        
        Args:
            in_degree_threshold: Percentile threshold for considering nodes fundamental (0-1)
            cluster_radius: Maximum distance from seed node to include in cluster
            min_cluster_size: Minimum edges per cluster
            max_clusters: Maximum number of clusters
        """
        super().__init__(min_cluster_size, max_clusters, max_dependent_cluster_size)
        self.in_degree_threshold = in_degree_threshold
        self.cluster_radius = cluster_radius
    
    def cluster(self, G: nx.MultiDiGraph) -> List[ClaimCluster]:
        """Cluster based on in-degree analysis."""
        if G.number_of_nodes() == 0:
            return []
        
        # Calculate in-degree for each node
        in_degrees = {node: G.in_degree(node) for node in G.nodes()}
        
        if not in_degrees:
            return []
        
        # Find threshold value (nodes above this are fundamental)
        sorted_degrees = sorted(in_degrees.values(), reverse=True)
        threshold_idx = int(len(sorted_degrees) * (1 - self.in_degree_threshold))
        threshold = sorted_degrees[threshold_idx] if threshold_idx < len(sorted_degrees) else sorted_degrees[-1]
        
        # Identify fundamental nodes (seed nodes for independent claims)
        fundamental_nodes = {
            node for node, degree in in_degrees.items()
            if degree >= threshold and degree > 0
        }
        
        # If no fundamental nodes found, use top nodes
        if not fundamental_nodes:
            top_n = max(1, len(in_degrees) // 10)  # Top 10%
            sorted_nodes = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)
            fundamental_nodes = {node for node, _ in sorted_nodes[:top_n]}
        
        # Build clusters around fundamental nodes
        clusters: List[ClaimCluster] = []
        used_edges: Set[Tuple[str, str, str]] = set()
        used_nodes: Set[str] = set()
        
        # Create independent claim clusters
        for priority, seed_node in enumerate(sorted(fundamental_nodes, key=lambda n: in_degrees[n], reverse=True)):
            # Skip seed node if it has forbidden type
            if self.forbidden_node_types and self._get_node_type(G, seed_node) in self.forbidden_node_types:
                continue
            cluster_nodes, cluster_edges = self._expand_cluster(G, seed_node, self.cluster_radius, for_independent=True)
            
            # Filter out already used edges
            new_edges = {e for e in cluster_edges if e not in used_edges}
            new_nodes = cluster_nodes - used_nodes
            
            if new_edges:
                cluster = ClaimCluster(
                    cluster_id=len(clusters),
                    nodes=new_nodes,
                    edges=new_edges,
                    claim_type="independent",
                    priority=priority,
                    metadata={"seed_node": seed_node, "in_degree": in_degrees[seed_node]}
                )
                clusters.append(cluster)
                used_edges.update(new_edges)
                used_nodes.update(new_nodes)
        
        # Create dependent claim clusters from remaining edges
        remaining_edges = {
            (u, v, self._get_edge_relation(G, u, v, k))
            for u, v, k in G.edges(keys=True)
            if (u, v, self._get_edge_relation(G, u, v, k)) not in used_edges
        }
        
        self._create_dependent_clusters(remaining_edges, clusters)
        
        return self._filter_clusters(clusters, G)
    
    def _expand_cluster(
        self, G: nx.MultiDiGraph, seed: str, radius: int, for_independent: bool = True
    ) -> Tuple[Set[str], Set[Tuple[str, str, str]]]:
        """
        Expand cluster from seed node using BFS up to radius hops.
        
        Args:
            G: NetworkX graph
            seed: Starting node
            radius: Maximum hops to expand
            for_independent: If True, exclude nodes with forbidden_node_types from independent claims
        """
        nodes: Set[str] = {seed}
        edges: Set[Tuple[str, str, str]] = set()
        queue = deque([(seed, 0)])  # (node, distance)
        visited = {seed}
        
        while queue:
            current, dist = queue.popleft()
            
            if dist >= radius:
                continue
            
            # Add outgoing edges
            for neighbor in G.successors(current):
                # Skip forbidden node types for independent claims
                if for_independent and self.forbidden_node_types:
                    neighbor_type = self._get_node_type(G, neighbor)
                    if neighbor_type in self.forbidden_node_types:
                        continue  # Skip this neighbor
                
                for key in G[current][neighbor]:
                    edge = (current, neighbor, self._get_edge_relation(G, current, neighbor, key))
                    edges.add(edge)
                    nodes.add(neighbor)
                    
                    if neighbor not in visited and dist + 1 < radius:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))
            
            # Add incoming edges (to capture context)
            for predecessor in G.predecessors(current):
                # Skip forbidden node types for independent claims
                if for_independent and self.forbidden_node_types:
                    predecessor_type = self._get_node_type(G, predecessor)
                    if predecessor_type in self.forbidden_node_types:
                        continue  # Skip this predecessor
                
                for key in G[predecessor][current]:
                    edge = (predecessor, current, self._get_edge_relation(G, predecessor, current, key))
                    edges.add(edge)
                    nodes.add(predecessor)
        
        return nodes, edges


class EntityTypeClaimClusterer(BaseClaimClusterer):
    """
    Clusters claims based on entity types.
    
    Strategy:
    - Nodes labeled "INVENTION" are fundamental (independent claims)
    - Clusters containing INVENTION nodes are independent claims
    - Other clusters are dependent claims
    """
    
    def __init__(
        self,
        fundamental_types: Set[str] = None,
        cluster_radius: int = 2,
        min_cluster_size: int = 1,
        max_clusters: Optional[int] = None,
        max_dependent_cluster_size: Optional[int] = None,
        forbidden_head_types: Optional[Set[str]] = None,
        forbidden_node_types: Optional[Set[str]] = None,
    ):
        """
        Initialize the entity type based clusterer.
        
        Args:
            fundamental_types: Set of entity types considered fundamental (default: {"INVENTION"})
            cluster_radius: Maximum distance from seed node to include
            min_cluster_size: Minimum edges per cluster
            max_clusters: Maximum number of clusters
            max_dependent_cluster_size: Maximum edges per dependent cluster (None = unlimited)
            forbidden_head_types: Set of entity types that should not appear as head nodes
            forbidden_node_types: Set of entity types that should not appear in independent claims
        """
        super().__init__(min_cluster_size, max_clusters, max_dependent_cluster_size, forbidden_head_types, forbidden_node_types)
        self.fundamental_types = fundamental_types or {"INVENTION"}
        self.cluster_radius = cluster_radius
    
    def cluster(self, G: nx.MultiDiGraph) -> List[ClaimCluster]:
        """Cluster based on entity types."""
        if G.number_of_nodes() == 0:
            return []
        
        # Find fundamental nodes (nodes with fundamental types)
        fundamental_nodes = {
            node for node in G.nodes()
            if self._get_node_type(G, node) in self.fundamental_types
        }
        
        clusters: List[ClaimCluster] = []
        used_edges: Set[Tuple[str, str, str]] = set()
        used_nodes: Set[str] = set()
        
        # Create independent claim clusters from fundamental nodes
        for priority, seed_node in enumerate(fundamental_nodes):
            # Skip seed node if it has forbidden type
            if self.forbidden_node_types and self._get_node_type(G, seed_node) in self.forbidden_node_types:
                continue
            cluster_nodes, cluster_edges = self._expand_cluster(G, seed_node, self.cluster_radius, for_independent=True)
            
            new_edges = {e for e in cluster_edges if e not in used_edges}
            new_nodes = cluster_nodes - used_nodes
            
            if new_edges:
                node_type = self._get_node_type(G, seed_node)
                cluster = ClaimCluster(
                    cluster_id=len(clusters),
                    nodes=new_nodes,
                    edges=new_edges,
                    claim_type="independent",
                    priority=priority,
                    metadata={"seed_node": seed_node, "seed_type": node_type}
                )
                clusters.append(cluster)
                used_edges.update(new_edges)
                used_nodes.update(new_nodes)
        
        # Create dependent clusters from remaining edges
        remaining_edges = {
            (u, v, self._get_edge_relation(G, u, v, k))
            for u, v, k in G.edges(keys=True)
            if (u, v, self._get_edge_relation(G, u, v, k)) not in used_edges
        }
        
        self._create_dependent_clusters(remaining_edges, clusters)
        
        return self._filter_clusters(clusters, G)
    
    def _expand_cluster(
        self, G: nx.MultiDiGraph, seed: str, radius: int, for_independent: bool = True
    ) -> Tuple[Set[str], Set[Tuple[str, str, str]]]:
        """
        Expand cluster from seed node using BFS.
        
        Args:
            for_independent: If True, exclude nodes with forbidden_node_types from independent claims
        """
        nodes: Set[str] = {seed}
        edges: Set[Tuple[str, str, str]] = set()
        queue = deque([(seed, 0)])
        visited = {seed}
        
        while queue:
            current, dist = queue.popleft()
            
            if dist >= radius:
                continue
            
            for neighbor in G.successors(current):
                # Skip forbidden node types for independent claims
                if for_independent and self.forbidden_node_types:
                    neighbor_type = self._get_node_type(G, neighbor)
                    if neighbor_type in self.forbidden_node_types:
                        continue  # Skip this neighbor
                
                for key in G[current][neighbor]:
                    edge = (current, neighbor, self._get_edge_relation(G, current, neighbor, key))
                    edges.add(edge)
                    nodes.add(neighbor)
                    
                    if neighbor not in visited and dist + 1 < radius:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))
            
            for predecessor in G.predecessors(current):
                # Skip forbidden node types for independent claims
                if for_independent and self.forbidden_node_types:
                    predecessor_type = self._get_node_type(G, predecessor)
                    if predecessor_type in self.forbidden_node_types:
                        continue  # Skip this predecessor
                
                for key in G[predecessor][current]:
                    edge = (predecessor, current, self._get_edge_relation(G, predecessor, current, key))
                    edges.add(edge)
                    nodes.add(predecessor)
        
        return nodes, edges


class CentralityClaimClusterer(BaseClaimClusterer):
    """
    Clusters claims based on network centrality measures.
    
    Strategy:
    - Uses betweenness, closeness, or eigenvector centrality to identify important nodes
    - High centrality nodes are fundamental (independent claims)
    - Other clusters are dependent claims
    """
    
    def __init__(
        self,
        centrality_type: str = "betweenness",  # "betweenness", "closeness", "eigenvector"
        centrality_threshold: float = 0.7,
        cluster_radius: int = 2,
        min_cluster_size: int = 1,
        max_clusters: Optional[int] = None,
        max_dependent_cluster_size: Optional[int] = None,
        forbidden_head_types: Optional[Set[str]] = None,
        forbidden_node_types: Optional[Set[str]] = None,
    ):
        """
        Initialize the centrality based clusterer.
        
        Args:
            centrality_type: Type of centrality to use
            centrality_threshold: Percentile threshold for fundamental nodes (0-1)
            cluster_radius: Maximum distance from seed node
            min_cluster_size: Minimum edges per cluster
            max_clusters: Maximum number of clusters
            max_dependent_cluster_size: Maximum edges per dependent cluster (None = unlimited)
            forbidden_head_types: Set of entity types that should not appear as head nodes
            forbidden_node_types: Set of entity types that should not appear in independent claims
        """
        super().__init__(min_cluster_size, max_clusters, max_dependent_cluster_size, forbidden_head_types, forbidden_node_types)
        self.centrality_type = centrality_type
        self.centrality_threshold = centrality_threshold
        self.cluster_radius = cluster_radius
    
    def cluster(self, G: nx.MultiDiGraph) -> List[ClaimCluster]:
        """Cluster based on centrality measures."""
        if G.number_of_nodes() == 0:
            return []
        
        # Calculate centrality
        if self.centrality_type == "betweenness":
            centrality = nx.betweenness_centrality(G)
        elif self.centrality_type == "closeness":
            centrality = nx.closeness_centrality(G)
        elif self.centrality_type == "eigenvector":
            try:
                centrality = nx.eigenvector_centrality(G, max_iter=1000)
            except:
                # Fallback to degree centrality if eigenvector fails
                centrality = dict(G.degree())
        else:
            centrality = dict(G.degree())
        
        if not centrality:
            return []
        
        # Find threshold
        sorted_values = sorted(centrality.values(), reverse=True)
        threshold_idx = int(len(sorted_values) * (1 - self.centrality_threshold))
        threshold = sorted_values[threshold_idx] if threshold_idx < len(sorted_values) else sorted_values[-1]
        
        # Identify fundamental nodes
        fundamental_nodes = {
            node for node, cent in centrality.items()
            if cent >= threshold
        }
        
        if not fundamental_nodes:
            top_n = max(1, len(centrality) // 10)
            sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
            fundamental_nodes = {node for node, _ in sorted_nodes[:top_n]}
        
        # Build clusters (similar to InDegreeClaimClusterer)
        clusters: List[ClaimCluster] = []
        used_edges: Set[Tuple[str, str, str]] = set()
        used_nodes: Set[str] = set()
        
        for priority, seed_node in enumerate(sorted(fundamental_nodes, key=lambda n: centrality[n], reverse=True)):
            # Skip seed node if it has forbidden type
            if self.forbidden_node_types and self._get_node_type(G, seed_node) in self.forbidden_node_types:
                continue
            cluster_nodes, cluster_edges = self._expand_cluster(G, seed_node, self.cluster_radius, for_independent=True)
            
            new_edges = {e for e in cluster_edges if e not in used_edges}
            new_nodes = cluster_nodes - used_nodes
            
            if new_edges:
                cluster = ClaimCluster(
                    cluster_id=len(clusters),
                    nodes=new_nodes,
                    edges=new_edges,
                    claim_type="independent",
                    priority=priority,
                    metadata={
                        "seed_node": seed_node,
                        "centrality": centrality[seed_node],
                        "centrality_type": self.centrality_type
                    }
                )
                clusters.append(cluster)
                used_edges.update(new_edges)
                used_nodes.update(new_nodes)
        
        # Create dependent clusters from remaining edges
        remaining_edges = {
            (u, v, self._get_edge_relation(G, u, v, k))
            for u, v, k in G.edges(keys=True)
            if (u, v, self._get_edge_relation(G, u, v, k)) not in used_edges
        }
        
        self._create_dependent_clusters(remaining_edges, clusters)
        
        return self._filter_clusters(clusters, G)
    
    def _expand_cluster(
        self, G: nx.MultiDiGraph, seed: str, radius: int, for_independent: bool = True
    ) -> Tuple[Set[str], Set[Tuple[str, str, str]]]:
        """
        Expand cluster from seed node using BFS.
        
        Args:
            for_independent: If True, exclude nodes with forbidden_node_types from independent claims
        """
        nodes: Set[str] = {seed}
        edges: Set[Tuple[str, str, str]] = set()
        queue = deque([(seed, 0)])
        visited = {seed}
        
        while queue:
            current, dist = queue.popleft()
            
            if dist >= radius:
                continue
            
            for neighbor in G.successors(current):
                # Skip forbidden node types for independent claims
                if for_independent and self.forbidden_node_types:
                    neighbor_type = self._get_node_type(G, neighbor)
                    if neighbor_type in self.forbidden_node_types:
                        continue  # Skip this neighbor
                
                for key in G[current][neighbor]:
                    edge = (current, neighbor, self._get_edge_relation(G, current, neighbor, key))
                    edges.add(edge)
                    nodes.add(neighbor)
                    
                    if neighbor not in visited and dist + 1 < radius:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))
            
            for predecessor in G.predecessors(current):
                # Skip forbidden node types for independent claims
                if for_independent and self.forbidden_node_types:
                    predecessor_type = self._get_node_type(G, predecessor)
                    if predecessor_type in self.forbidden_node_types:
                        continue  # Skip this predecessor
                
                for key in G[predecessor][current]:
                    edge = (predecessor, current, self._get_edge_relation(G, predecessor, current, key))
                    edges.add(edge)
                    nodes.add(predecessor)
        
        return nodes, edges


class HierarchicalClaimClusterer(BaseClaimClusterer):
    """
    Clusters claims hierarchically starting from root nodes.
    
    Strategy:
    - Nodes with no incoming edges (roots) or INVENTION type are fundamental
    - Builds clusters hierarchically from roots
    - First level clusters are independent claims
    - Deeper levels are dependent claims
    """
    
    def __init__(
        self,
        max_depth: int = 3,
        fundamental_types: Set[str] = None,
        min_cluster_size: int = 1,
        max_clusters: Optional[int] = None,
        forbidden_head_types: Optional[Set[str]] = None,
        forbidden_node_types: Optional[Set[str]] = None,
    ):
        """
        Initialize the hierarchical clusterer.
        
        Args:
            max_depth: Maximum depth to traverse from roots
            fundamental_types: Entity types considered fundamental
            min_cluster_size: Minimum edges per cluster
            max_clusters: Maximum number of clusters
            forbidden_head_types: Set of entity types that should not appear as head nodes
            forbidden_node_types: Set of entity types that should not appear in independent claims
        """
        super().__init__(min_cluster_size, max_clusters, max_dependent_cluster_size=None, forbidden_head_types=forbidden_head_types, forbidden_node_types=forbidden_node_types)
        self.max_depth = max_depth
        self.fundamental_types = fundamental_types or {"INVENTION"}
    
    def cluster(self, G: nx.MultiDiGraph) -> List[ClaimCluster]:
        """Cluster hierarchically from root nodes."""
        if G.number_of_nodes() == 0:
            return []
        
        # Find root nodes (no incoming edges) or fundamental type nodes
        root_nodes = {
            node for node in G.nodes()
            if G.in_degree(node) == 0 or self._get_node_type(G, node) in self.fundamental_types
        }
        
        if not root_nodes:
            # Fallback: use nodes with lowest in-degree
            in_degrees = {node: G.in_degree(node) for node in G.nodes()}
            min_degree = min(in_degrees.values()) if in_degrees else 0
            root_nodes = {node for node, deg in in_degrees.items() if deg == min_degree}
        
        clusters: List[ClaimCluster] = []
        used_edges: Set[Tuple[str, str, str]] = set()
        used_nodes: Set[str] = set()
        
        # Create independent clusters from root nodes
        for priority, root in enumerate(root_nodes):
            # Skip root node if it has forbidden type
            if self.forbidden_node_types and self._get_node_type(G, root) in self.forbidden_node_types:
                continue
            cluster_nodes, cluster_edges = self._build_hierarchical_cluster(G, root, 0, self.max_depth, for_independent=True)
            
            new_edges = {e for e in cluster_edges if e not in used_edges}
            new_nodes = cluster_nodes - used_nodes
            
            if new_edges:
                cluster = ClaimCluster(
                    cluster_id=len(clusters),
                    nodes=new_nodes,
                    edges=new_edges,
                    claim_type="independent",
                    priority=priority,
                    metadata={"root_node": root, "depth": self.max_depth}
                )
                clusters.append(cluster)
                used_edges.update(new_edges)
                used_nodes.update(new_nodes)
        
        # Create dependent clusters from remaining edges
        remaining_edges = {
            (u, v, self._get_edge_relation(G, u, v, k))
            for u, v, k in G.edges(keys=True)
            if (u, v, self._get_edge_relation(G, u, v, k)) not in used_edges
        }
        
        self._create_dependent_clusters(remaining_edges, clusters)
        
        return self._filter_clusters(clusters, G)
    
    def _build_hierarchical_cluster(
        self, G: nx.MultiDiGraph, root: str, current_depth: int, max_depth: int, for_independent: bool = True
    ) -> Tuple[Set[str], Set[Tuple[str, str, str]]]:
        """
        Build cluster hierarchically from root.
        
        Args:
            for_independent: If True, exclude nodes with forbidden_node_types from independent claims
        """
        if current_depth >= max_depth:
            return set(), set()
        
        nodes: Set[str] = {root}
        edges: Set[Tuple[str, str, str]] = set()
        
        # Add outgoing edges from root
        for neighbor in G.successors(root):
            # Skip forbidden node types for independent claims
            if for_independent and self.forbidden_node_types:
                neighbor_type = self._get_node_type(G, neighbor)
                if neighbor_type in self.forbidden_node_types:
                    continue  # Skip this neighbor
            
            for key in G[root][neighbor]:
                edge = (root, neighbor, self._get_edge_relation(G, root, neighbor, key))
                edges.add(edge)
                nodes.add(neighbor)
                
                # Recursively add deeper levels
                sub_nodes, sub_edges = self._build_hierarchical_cluster(
                    G, neighbor, current_depth + 1, max_depth, for_independent
                )
                nodes.update(sub_nodes)
                edges.update(sub_edges)
        
        return nodes, edges


class PageRankClaimClusterer(BaseClaimClusterer):
    """
    Clusters claims based on PageRank algorithm.
    
    Strategy:
    - Uses PageRank to identify important nodes
    - High PageRank nodes are fundamental (independent claims)
    - Other clusters are dependent claims
    """
    
    def __init__(
        self,
        pagerank_threshold: float = 0.7,
        cluster_radius: int = 2,
        damping: float = 0.85,
        min_cluster_size: int = 1,
        max_clusters: Optional[int] = None,
        max_dependent_cluster_size: Optional[int] = None,
        forbidden_head_types: Optional[Set[str]] = None,
        forbidden_node_types: Optional[Set[str]] = None,
    ):
        """
        Initialize the PageRank based clusterer.
        
        Args:
            pagerank_threshold: Percentile threshold for fundamental nodes (0-1)
            cluster_radius: Maximum distance from seed node
            damping: PageRank damping factor
            min_cluster_size: Minimum edges per cluster
            max_clusters: Maximum number of clusters
            max_dependent_cluster_size: Maximum edges per dependent cluster (None = unlimited)
            forbidden_head_types: Set of entity types that should not appear as head nodes
            forbidden_node_types: Set of entity types that should not appear in independent claims
        """
        super().__init__(min_cluster_size, max_clusters, max_dependent_cluster_size, forbidden_head_types, forbidden_node_types)
        self.pagerank_threshold = pagerank_threshold
        self.cluster_radius = cluster_radius
        self.damping = damping
    
    def cluster(self, G: nx.MultiDiGraph) -> List[ClaimCluster]:
        """Cluster based on PageRank."""
        if G.number_of_nodes() == 0:
            return []
        
        # Calculate PageRank
        try:
            pagerank = nx.pagerank(G, alpha=self.damping, max_iter=1000)
        except:
            # Fallback to degree if PageRank fails
            pagerank = {node: G.degree(node) for node in G.nodes()}
        
        if not pagerank:
            return []
        
        # Find threshold
        sorted_values = sorted(pagerank.values(), reverse=True)
        threshold_idx = int(len(sorted_values) * (1 - self.pagerank_threshold))
        threshold = sorted_values[threshold_idx] if threshold_idx < len(sorted_values) else sorted_values[-1]
        
        # Identify fundamental nodes
        fundamental_nodes = {
            node for node, pr in pagerank.items()
            if pr >= threshold
        }
        
        if not fundamental_nodes:
            top_n = max(1, len(pagerank) // 10)
            sorted_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
            fundamental_nodes = {node for node, _ in sorted_nodes[:top_n]}
        
        # Build clusters (similar to other clusterers)
        clusters: List[ClaimCluster] = []
        used_edges: Set[Tuple[str, str, str]] = set()
        used_nodes: Set[str] = set()
        
        for priority, seed_node in enumerate(sorted(fundamental_nodes, key=lambda n: pagerank[n], reverse=True)):
            # Skip seed node if it has forbidden type
            if self.forbidden_node_types and self._get_node_type(G, seed_node) in self.forbidden_node_types:
                continue
            cluster_nodes, cluster_edges = self._expand_cluster(G, seed_node, self.cluster_radius, for_independent=True)
            
            new_edges = {e for e in cluster_edges if e not in used_edges}
            new_nodes = cluster_nodes - used_nodes
            
            if new_edges:
                cluster = ClaimCluster(
                    cluster_id=len(clusters),
                    nodes=new_nodes,
                    edges=new_edges,
                    claim_type="independent",
                    priority=priority,
                    metadata={"seed_node": seed_node, "pagerank": pagerank[seed_node]}
                )
                clusters.append(cluster)
                used_edges.update(new_edges)
                used_nodes.update(new_nodes)
        
        # Create dependent clusters from remaining edges
        remaining_edges = {
            (u, v, self._get_edge_relation(G, u, v, k))
            for u, v, k in G.edges(keys=True)
            if (u, v, self._get_edge_relation(G, u, v, k)) not in used_edges
        }
        
        self._create_dependent_clusters(remaining_edges, clusters)
        
        return self._filter_clusters(clusters, G)
    
    def _expand_cluster(
        self, G: nx.MultiDiGraph, seed: str, radius: int, for_independent: bool = True
    ) -> Tuple[Set[str], Set[Tuple[str, str, str]]]:
        """
        Expand cluster from seed node using BFS.
        
        Args:
            for_independent: If True, exclude nodes with forbidden_node_types from independent claims
        """
        nodes: Set[str] = {seed}
        edges: Set[Tuple[str, str, str]] = set()
        queue = deque([(seed, 0)])
        visited = {seed}
        
        while queue:
            current, dist = queue.popleft()
            
            if dist >= radius:
                continue
            
            for neighbor in G.successors(current):
                # Skip forbidden node types for independent claims
                if for_independent and self.forbidden_node_types:
                    neighbor_type = self._get_node_type(G, neighbor)
                    if neighbor_type in self.forbidden_node_types:
                        continue  # Skip this neighbor
                
                for key in G[current][neighbor]:
                    edge = (current, neighbor, self._get_edge_relation(G, current, neighbor, key))
                    edges.add(edge)
                    nodes.add(neighbor)
                    
                    if neighbor not in visited and dist + 1 < radius:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))
            
            for predecessor in G.predecessors(current):
                # Skip forbidden node types for independent claims
                if for_independent and self.forbidden_node_types:
                    predecessor_type = self._get_node_type(G, predecessor)
                    if predecessor_type in self.forbidden_node_types:
                        continue  # Skip this predecessor
                
                for key in G[predecessor][current]:
                    edge = (predecessor, current, self._get_edge_relation(G, predecessor, current, key))
                    edges.add(edge)
                    nodes.add(predecessor)
        
        return nodes, edges


class HybridClaimClusterer(BaseClaimClusterer):
    """
    Combines multiple clustering strategies using voting/weighting.
    
    Strategy:
    - Runs multiple clusterers and combines their results
    - Nodes identified as fundamental by multiple methods are more likely independent
    - Uses weighted voting to determine claim types
    """
    
    def __init__(
        self,
        clusterers: List[BaseClaimClusterer] = None,
        weights: List[float] = None,
        min_votes: int = 2,  # Minimum clusterers that must agree
        min_cluster_size: int = 1,
        max_clusters: Optional[int] = None,
        max_dependent_cluster_size: Optional[int] = None,
        max_independent_clusters: Optional[int] = None,  # Maximum number of independent claims
        max_dependent_clusters: Optional[int] = None,    # Maximum number of dependent claims
        forbidden_head_types: Optional[Set[str]] = None,  # Entity types that shouldn't be head nodes
        forbidden_node_types: Optional[Set[str]] = None,  # Entity types that shouldn't appear in independent claims
    ):
        """
        Initialize the hybrid clusterer.
        
        Args:
            clusterers: List of clusterer instances to combine
            weights: Weights for each clusterer (default: equal weights)
            min_votes: Minimum number of clusterers that must agree on independence
            min_cluster_size: Minimum edges per cluster
            max_clusters: Maximum number of clusters (deprecated, use max_independent_clusters instead)
            max_dependent_cluster_size: Maximum edges per dependent cluster (None = unlimited)
            max_independent_clusters: Maximum number of independent claims to return (None = unlimited)
            max_dependent_clusters: Maximum number of dependent claims to return (None = unlimited)
            forbidden_head_types: Set of entity types that should not appear as head nodes in claims (e.g., {"MATERIAL", "PRIOR_ART"})
            forbidden_node_types: Set of entity types that should not appear in independent claims at all (e.g., {"UNKNOWN", "PARAMETER"})
        """
        super().__init__(min_cluster_size, max_clusters, max_dependent_cluster_size, forbidden_head_types, forbidden_node_types)
        # If clusterers are provided, use them; otherwise create defaults with forbidden_node_types
        if clusterers is None:
            self.clusterers = [
                InDegreeClaimClusterer(
                    min_cluster_size=min_cluster_size,
                    max_clusters=max_clusters,
                    max_dependent_cluster_size=max_dependent_cluster_size,
                    forbidden_head_types=forbidden_head_types,
                    forbidden_node_types=forbidden_node_types,
                ),
                EntityTypeClaimClusterer(
                    min_cluster_size=min_cluster_size,
                    max_clusters=max_clusters,
                    max_dependent_cluster_size=max_dependent_cluster_size,
                    forbidden_head_types=forbidden_head_types,
                    forbidden_node_types=forbidden_node_types,
                ),
                PageRankClaimClusterer(
                    min_cluster_size=min_cluster_size,
                    max_clusters=max_clusters,
                    max_dependent_cluster_size=max_dependent_cluster_size,
                    forbidden_head_types=forbidden_head_types,
                    forbidden_node_types=forbidden_node_types,
                ),
            ]
        else:
            self.clusterers = clusterers
        self.weights = weights or [1.0] * len(self.clusterers)
        self.min_votes = min_votes
        # Use max_independent_clusters if provided, otherwise fall back to max_clusters for backward compatibility
        self.max_independent_clusters = max_independent_clusters if max_independent_clusters is not None else max_clusters
        self.max_dependent_clusters = max_dependent_clusters
    
    def cluster(self, G: nx.MultiDiGraph) -> List[ClaimCluster]:
        """Cluster using hybrid approach."""
        if G.number_of_nodes() == 0:
            return []
        
        # Run all clusterers
        all_clusters: List[List[ClaimCluster]] = []
        for clusterer in self.clusterers:
            clusters = clusterer.cluster(G)
            all_clusters.append(clusters)
        
        # Count votes for each edge being in an independent claim
        edge_votes: Dict[Tuple[str, str, str], int] = defaultdict(int)
        edge_weights: Dict[Tuple[str, str, str], float] = defaultdict(float)
        
        for clusters, weight in zip(all_clusters, self.weights):
            for cluster in clusters:
                if cluster.claim_type == "independent":
                    for edge in cluster.edges:
                        edge_votes[edge] += 1
                        edge_weights[edge] += weight
        
        # Classify edges as independent if they have enough votes
        independent_edges = {
            edge for edge, votes in edge_votes.items()
            if votes >= self.min_votes
        }
        
        # Build final clusters
        # Group independent edges by connected components
        independent_graph = nx.MultiDiGraph()
        for u, v, rel in independent_edges:
            independent_graph.add_edge(u, v, label=rel)
        
        clusters: List[ClaimCluster] = []
        used_edges: Set[Tuple[str, str, str]] = set()
        
        # Create independent clusters
        for comp_id, component in enumerate(nx.weakly_connected_components(independent_graph)):
            comp_edges = {
                (u, v, self._get_edge_relation(independent_graph, u, v, k))
                for u, v, k in independent_graph.edges(keys=True)
                if u in component and v in component
            }
            
            if comp_edges:
                cluster = ClaimCluster(
                    cluster_id=len(clusters),
                    nodes=component,
                    edges=comp_edges,
                    claim_type="independent",
                    priority=comp_id,
                    metadata={"vote_count": min(edge_votes.get(e, 0) for e in comp_edges)}
                )
                clusters.append(cluster)
                used_edges.update(comp_edges)
        
        # Create dependent clusters from remaining edges
        remaining_edges = {
            (u, v, self._get_edge_relation(G, u, v, k))
            for u, v, k in G.edges(keys=True)
            if (u, v, self._get_edge_relation(G, u, v, k)) not in used_edges
        }
        
        # Don't filter forbidden edges BEFORE creating dependent clusters
        # This allows more clusters to be created, then we filter forbidden edges AFTER
        # This preserves cluster structure while still removing unwanted edges
        self._create_dependent_clusters(remaining_edges, clusters)
        
        # Filter forbidden edges AFTER creating clusters
        # This way we get more clusters, but they have forbidden edges removed
        if self.forbidden_head_types:
            clusters = [self._filter_cluster_forbidden_edges(G, c) for c in clusters]
        
        return self._filter_clusters_hybrid(clusters, G)
    
    def _filter_clusters_hybrid(self, clusters: List[ClaimCluster], G: nx.MultiDiGraph) -> List[ClaimCluster]:
        """Filter clusters by size and limit count for hybrid clusterer."""
        # Note: Forbidden edges are filtered AFTER creating clusters in HybridClaimClusterer
        # This allows more clusters to be created, then we remove forbidden edges from them
        
        # Filter by minimum size (after forbidden edges have been removed)
        filtered = [c for c in clusters if c.size() >= self.min_cluster_size]
        
        # Separate independent and dependent
        independent = [c for c in filtered if c.claim_type == "independent"]
        dependent = [c for c in filtered if c.claim_type == "dependent"]
        
        # Sort independent by priority (lower = more important)
        independent.sort(key=lambda x: x.priority)
        
        # Sort dependent by size (largest first) or priority
        dependent.sort(key=lambda x: (-x.size(), x.priority))
        
        # Limit independent claims if specified
        if self.max_independent_clusters is not None:
            independent = independent[:self.max_independent_clusters]
        
        # Limit dependent claims if specified
        if self.max_dependent_clusters is not None:
            dependent = dependent[:self.max_dependent_clusters]
        
        return independent + dependent

