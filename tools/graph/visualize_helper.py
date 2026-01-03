"""
Helper function for visualizing graphs with readable labels.
Uses the same PyVis method as GraphVisualizer.visualize_pyvis for consistency.
"""
from typing import Optional, Dict
import networkx as nx
from tools.graph.visualizer import GraphVisualizer


def visualize_nx_browser_full(
    G: nx.MultiDiGraph,
    path: str = "graph.html",
    id_to_name: Optional[Dict[str, str]] = None,
) -> None:
    """
    Visualize a NetworkX graph in the browser with readable node labels.
    Uses the same PyVis method as GraphVisualizer.visualize_pyvis for consistency.
    
    Args:
        G: NetworkX MultiDiGraph to visualize
        path: Output HTML file path
        id_to_name: Optional mapping from entity ID to display name
    """
    visualizer = GraphVisualizer()
    # Use the same method as GraphVisualizer.visualize_pyvis
    visualizer.visualize_pyvis(G, out_file=path, id_to_name=id_to_name)

