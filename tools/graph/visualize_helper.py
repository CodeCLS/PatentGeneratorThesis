"""
Helper function for visualizing graphs with readable labels.
"""
from pyvis.network import Network
import webbrowser
from tools.graph.visualizer import GraphVisualizer


def visualize_nx_browser_full(G, path="graph.html", id_to_name=None):
    """
    Visualize a NetworkX graph in the browser with readable node labels.
    
    Args:
        G: NetworkX MultiDiGraph to visualize
        path: Output HTML file path
        id_to_name: Optional mapping from entity ID to display name
    """
    visualizer = GraphVisualizer()
    id_to_name = id_to_name or {}
    
    net = Network(height="100vh", width="100%", directed=True, notebook=False)

    # Add nodes with readable labels
    for n in G.nodes():
        node_data = G.nodes[n]
        node_type = (node_data.get("node_type", "UNKNOWN") or "UNKNOWN").upper()
        color = visualizer.node_type_colors.get(node_type, visualizer.node_type_colors.get("UNKNOWN", "#bdbdbd"))
        
        # Get readable label using visualizer's method
        label = visualizer._get_readable_node_label(n, node_data, id_to_name, G)
        
        # Build title with details
        title_parts = [f"id: {n}", f"type: {node_type}"]
        if node_data.get("node_type") == "ASSERTION":
            predicate = node_data.get("predicate", "")
            category = node_data.get("category", "")
            if predicate:
                title_parts.insert(0, f"predicate: {predicate}")
            if category:
                title_parts.append(f"category: {category}")
        elif node_data.get("node_type") == "CLAIM_CONCEPT":
            kind = node_data.get("kind", "")
            breadth = node_data.get("breadth", "")
            if kind:
                title_parts.insert(0, f"kind: {kind}")
            if breadth:
                title_parts.append(f"breadth: {breadth}")
        
        net.add_node(
            n,
            label=label,
            color=color,
            shape="ellipse",
            title="<br>".join(title_parts),
        )
    
    # Add edges
    for u, v, k, d in G.edges(keys=True, data=True):
        edge_label = d.get("label", d.get("relation", ""))
        net.add_edge(u, v, label=str(edge_label), arrows="to")

    html = net.generate_html(notebook=False)
    html = html.replace(
        "<head>",
        "<head><style>html,body{height:100%;margin:0;}#mynetwork{height:100vh !important;}</style>"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    webbrowser.open(path)

