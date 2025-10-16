import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# ---- 1. Create the graph ----
G = nx.DiGraph()

# Example node attributes (feature vectors + importance)
G.add_node("moisture_sensor", coverage=0.83, feat=np.array([0.8, 0.9, 0.7]), importance=0.25)
G.add_node("controller", coverage=0.65, feat=np.array([0.7, 0.5, 0.8]), importance=0.4)
G.add_node("pump", coverage=0.92, feat=np.array([0.9, 0.8, 0.7]), importance=0.35)
G.add_node("water_valve", coverage=0.48, feat=np.array([0.6, 0.3, 0.4]), importance=0.2)

# Example edges with relations and weights
G.add_edge("moisture_sensor", "controller", relation="sends_data_to", weight=0.8)
G.add_edge("controller", "pump", relation="controls", weight=0.9)
G.add_edge("controller", "water_valve", relation="opens", weight=0.7)

# ---- 2. Layout and color mapping ----
pos = nx.spring_layout(G, seed=42, k=1.2)  # force-directed layout

# Node colors = coverage score
coverages = np.array([G.nodes[n]["coverage"] for n in G.nodes()])
node_colors = plt.cm.viridis(coverages)  # color map 0–1 → green-blue-yellow

# Node sizes = importance weight (scaled)
sizes = np.array([G.nodes[n]["importance"] * 2000 for n in G.nodes()])

# Edge thickness = weight
edge_weights = [G[u][v]["weight"] * 4 for u, v in G.edges()]

fig, ax = plt.subplots(figsize=(8, 6))

##test2

nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=sizes, alpha=0.9, ax=ax)
nx.draw_networkx_edges(G, pos, width=edge_weights, arrows=True, arrowstyle='-|>', alpha=0.6, ax=ax)
nx.draw_networkx_labels(G, pos, font_size=10, font_color="white", font_weight="bold", ax=ax)

edge_labels = {(u, v): G[u][v]['relation'] for u, v in G.edges()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="gray", font_size=8, ax=ax)

# ---- FIX: attach colorbar to the same Axes ----
sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0, vmax=1))
cbar = plt.colorbar(sm, ax=ax, shrink=0.7)   # <-- add ax=ax here
cbar.set_label("Coverage Strength (0–1)")

ax.set_title("Patent Knowledge Graph Example")
ax.axis("off")
plt.show()