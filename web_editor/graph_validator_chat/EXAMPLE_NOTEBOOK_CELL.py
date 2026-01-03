"""
Example notebook cell for using Graph Validator Chat Interface.

Copy this into your Jupyter notebook to use the chat interface.
"""

# ============================================================================
# STEP 1: Start the Graph Validator Chat
# ============================================================================

from web_editor.graph_validator_chat import start_validator_chat
from tools.graph.visualizer import GraphVisualizer
from tools.graph.kg_gen_converter import build_id_to_name_map

# Assuming you have:
# - triples: List of Triple objects
# - G: NetworkX graph (optional)

# Build id_to_name mapping
id_to_name = build_id_to_name_map(triples)

# Start the chat interface
start_validator_chat(
    graph=G,  # Optional: NetworkX graph
    triples=triples,  # Optional: List of Triple objects
    id_to_name=id_to_name,  # Optional: Entity ID to name mapping
    port=5001,  # Port for the web interface
    open_browser=True,  # Automatically open browser
)

# The browser will open with the chat interface
# Answer questions and modify the graph through the chat
# When done, close the browser and run STEP 2 below


# ============================================================================
# STEP 2: Get Updated Graph, Triples, and Entities
# ============================================================================

from web_editor.graph_validator_chat.helper import (
    get_updated_graph,
    get_updated_triples,
    get_updated_entities,
    get_changes_summary,
    get_all_updates,
)

# Option 1: Get everything at once
updates = get_all_updates()

# Extract updated data
G_updated = updates['graph']
triples_updated = updates['triples']
entities_updated = updates['entities']
id_to_name_updated = updates['id_to_name']
changes = updates['changes']

print(f"✅ Updated graph: {G_updated.number_of_nodes()} nodes, {G_updated.number_of_edges()} edges")
print(f"✅ Updated triples: {len(triples_updated)} triples")
print(f"✅ Updated entities: {len(entities_updated)} entities")
print(f"✅ Changes: {changes}")

# Option 2: Get individual components
# G_updated = get_updated_graph()
# triples_updated = get_updated_triples()
# entities_updated = get_updated_entities()
# changes = get_changes_summary()

# Now use G_updated and triples_updated in your pipeline!
# For example:
# visualizer = GraphVisualizer()
# visualizer.visualize_pyvis(G_updated, out_file="updated_graph.html", id_to_name=id_to_name_updated)

