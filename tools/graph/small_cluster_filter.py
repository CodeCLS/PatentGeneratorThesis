"""
Filter to remove small clusters of triples from knowledge graphs.
Removes isolated or small groups of triples that don't form meaningful clusters.
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Set
import networkx as nx

from tools.graph.Triple import Triple
from tools.graph.visualizer import GraphVisualizer


class SmallClusterFilter:
    """
    Filters out small clusters of triples (isolated or very small groups).
    
    Identifies connected components in the graph and removes clusters
    that are below a certain size threshold.
    """

    def __init__(
        self,
        min_cluster_size: int = 4,  # Minimum number of triples in a cluster to keep
    ):
        """
        Initialize the small cluster filter.

        Args:
            min_cluster_size: Minimum number of triples in a cluster to keep it.
                Clusters with fewer triples will be removed.
                Default: 4 (removes clusters of size 1, 2, and 3)
        """
        self.min_cluster_size = min_cluster_size

    @staticmethod
    def _entity_key(e) -> str:
        """Get stable entity key."""
        if e is None:
            return ""
        if isinstance(e, str):
            return e
        for attr in ("ref", "id", "ref_short"):
            if hasattr(e, attr):
                v = getattr(e, attr)
                if v:
                    return str(v)
        return str(e)

    def _build_graph(self, triples: List[Triple]) -> nx.Graph:
        """
        Build an undirected graph from triples to identify connected components.
        
        Args:
            triples: List of Triple objects
            
        Returns:
            NetworkX Graph with entities as nodes and triples as edges
        """
        G = nx.Graph()
        
        # Add all triples as edges
        for triple in triples:
            head_id = self._entity_key(triple.head)
            tail_id = self._entity_key(triple.tail)
            
            if not head_id or not tail_id:
                continue
            
            # Add nodes
            G.add_node(head_id)
            G.add_node(tail_id)
            
            # Add edge (undirected, so we can find connected components)
            # Store triple reference in edge data
            if G.has_edge(head_id, tail_id):
                # Multiple triples between same entities - add to edge data
                if "triples" not in G[head_id][tail_id]:
                    G[head_id][tail_id]["triples"] = []
                G[head_id][tail_id]["triples"].append(triple)
            else:
                G.add_edge(head_id, tail_id, triples=[triple])
        
        return G

    def _get_cluster_triples(
        self, 
        G: nx.Graph, 
        component: Set[str]
    ) -> List[Triple]:
        """
        Extract all triples from a connected component.
        
        Args:
            G: NetworkX graph
            component: Set of node IDs in the component
            
        Returns:
            List of Triple objects in this component
        """
        cluster_triples = []
        
        # Get all edges within this component
        subgraph = G.subgraph(component)
        for u, v, data in subgraph.edges(data=True):
            triples = data.get("triples", [])
            cluster_triples.extend(triples)
        
        return cluster_triples

    def filter_small_clusters(
        self,
        triples: List[Triple],
    ) -> Tuple[List[Triple], Dict[str, any]]:
        """
        Filter out small clusters of triples.

        Args:
            triples: List of Triple objects to filter

        Returns:
            Tuple of (filtered_triples, stats_dict)
        """
        if not triples:
            return [], {
                "input_triples": 0,
                "output_triples": 0,
                "removed_triples": 0,
                "total_clusters": 0,
                "removed_clusters": 0,
                "kept_clusters": 0,
                "min_cluster_size": self.min_cluster_size,
            }
        
        print(f"Filtering small clusters (min size: {self.min_cluster_size})...")
        print("=" * 80)
        
        # Build graph to find connected components
        G = self._build_graph(triples)
        
        # Find connected components (clusters)
        components = list(nx.connected_components(G))
        
        print(f"Found {len(components)} clusters")
        
        # Filter clusters by size
        kept_triples = []
        removed_clusters = []
        kept_clusters = []
        total_removed = 0
        
        for i, component in enumerate(components, 1):
            cluster_triples = self._get_cluster_triples(G, component)
            cluster_size = len(cluster_triples)
            
            if cluster_size < self.min_cluster_size:
                # Remove this small cluster
                removed_clusters.append({
                    "cluster_id": i,
                    "size": cluster_size,
                    "nodes": len(component),
                })
                total_removed += cluster_size
                print(f"  ❌ Cluster {i}: {cluster_size} triples, {len(component)} nodes - REMOVED")
            else:
                # Keep this cluster
                kept_triples.extend(cluster_triples)
                kept_clusters.append({
                    "cluster_id": i,
                    "size": cluster_size,
                    "nodes": len(component),
                })
                print(f"  ✅ Cluster {i}: {cluster_size} triples, {len(component)} nodes - KEPT")
        
        stats = {
            "input_triples": len(triples),
            "output_triples": len(kept_triples),
            "removed_triples": total_removed,
            "total_clusters": len(components),
            "removed_clusters": len(removed_clusters),
            "kept_clusters": len(kept_clusters),
            "min_cluster_size": self.min_cluster_size,
        }
        
        print("\n" + "=" * 80)
        print(f"✅ Small cluster filtering complete!")
        print(f"   Input: {len(triples)} triples in {len(components)} clusters")
        print(f"   Output: {len(kept_triples)} triples in {len(kept_clusters)} clusters")
        print(f"   Removed: {total_removed} triples from {len(removed_clusters)} small clusters")
        print("=" * 80)
        
        return kept_triples, stats

