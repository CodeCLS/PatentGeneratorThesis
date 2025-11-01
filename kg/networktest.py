import networkx as nx
import plotly.graph_objects as go

# --- Triples (Subject, Relation, Object) ---
edges = [
    ("Invention", "relates to", "DISPLAY_DEVICE"),
    ("DISPLAY_DEVICE", "for appreciation regarding", "PSEUDO_SPACE"),
    ("PSEUDO_SPACE", "includes", "Space in water"),
    ("PSEUDO_SPACE", "includes", "Space in air"),
    ("DISPLAY_DEVICE", "for appreciation of", "Creatures"),
    ("DISPLAY_DEVICE", "for appreciation of", "Objects"),
    ("Creatures_and_objects", "float in", "PSEUDO_SPACE"),
    ("Creatures_and_objects", "represented by", "FISH"),
    ("Creatures_and_objects", "represented by", "SUBMARINE"),
    ("Creatures_and_objects", "represented by", "SPACESHIP"),
    ("Creatures_and_objects", "represented by", "AIRSHIP"),
    ("Creatures_and_objects", "represented by", "BUTTERFLY"),
    ("Creatures_and_objects", "represented by", "BIRD"),
    ("INTERIOR", "with motif", "WATER"),
    ("INTERIOR", "with motif", "PSEUDO_SPACE"),
    ("INTERIOR", "provided in", "Offices"),
    ("INTERIOR", "refresh feelings of", "Residents"),
    ("INTERIOR", "serve purpose", "Refreshing feelings of occupants indoors"),
    ("AQUARIUM", "have", "Seaweeds or tropical FISH in WATER tank"),
    ("AQUARIUM", "used at", "Home (display for visual effects)"),
    ("AQUARIUM", "used at", "Shops"),
    ("AQUARIUM", "used as", "DISPLAY_DEVICE"),
    ("AQUARIUM", "used for", "Visual effects"),
    ("AQUARIUM", "keeps", "FISH"),
    ("AQUARIUM", "is a", "WATER_TANK"),
    ("AQUATIC_PLANTS", "kept in", "AQUARIUM"),
    ("LIVING_CONDITIONS", "include", "Food and WATER temperature"),
    ("Maintaining LIVING_CONDITIONS", "accompanies", "Labor cost"),
    ("Maintaining LIVING_CONDITIONS", "accompanies", "Maintenance cost"),
    ("Maintaining LIVING_CONDITIONS", "accompanies", "Other costs"),
    ("AIR_BUBBLE", "generates", "WATER_FLOW in WATER_TANK"),
    ("Environmental maintenance", "is", "Easy"),
    ("DISPLAY_DEVICE", "proposed to reduce", "Labor"),
    ("DISPLAY_DEVICE", "provided with", "Pseudo models of FISH"),
    ("Aquatic animals", "include", "FISH models"),
    ("AIR_BUBBLE", "generated from", "VENTILATION_MEMBER"),
    ("VENTILATION_MEMBER", "installed at", "WATER_TANK.BottomWall"),
    ("DISPLAY_DEVICE", "functions as", "Pseudo AQUARIUM"),
    ("DISPLAY_DEVICE", "requires", "No daily feeding"),
    ("PSEUDO_CREATURE_MODEL", "floating in", "WATER_TANK"),
    ("PSEUDO_CREATURE_MODEL", "behavior", "Unnatural due to AIR_BUBBLE adhesion"),
    ("DISPLAY_DEVICE", "generates", "WATER_FLOW in WATER_TANK"),
    ("DISPLAY_DEVICE", "generates directly", "AIR_BUBBLE in WATER_TANK"),
    ("AIR_BUBBLE", "adheres to", "PSEUDO_CREATURE_MODEL"),
    ("PSEUDO_CREATURE_MODEL", "go in", "Domains"),
    ("PSEUDO_CREATURE_MODEL", "shows", "Unnatural behavior (lying down)"),
    ("AIR_BUBBLE", "rises near", "VENTILATION_MEMBER"),
    ("PSEUDO_CREATURE_MODEL", "pushed", "Upward"),
    ("PSEUDO_CREATURE_MODEL", "pushed near", "WATER.Surface"),
    ("Behavior", "applies when replaced", "PSEUDO_CREATURE_MODEL"),
    ("Models", "include", "SUBMARINE (other than FISH)"),
    ("WATER_TANK", "not suitable for", "DISPLAY_DEVICE"),
    ("DISPLAY_DEVICE", "expresses", "PSEUDO_SPACE (non-underwater)"),
    ("Object", "provide", "WATER_TANK for appreciation"),
    ("WATER_TANK", "prevents imbalance of", "PSEUDO_CREATURE_MODEL due to AIR_BUBBLE adhesion"),
    ("WATER_TANK", "prevents lying down of", "PSEUDO_CREATURE_MODEL near WATER.Surface"),
    ("Models", "show", "More natural behavior in WATER_TANK"),
    ("WATER_TANK", "prevents sudden rising of", "PSEUDO_CREATURE_MODEL"),
    ("PSEUDO_CREATURE_MODEL", "in vicinity of", "VENTILATION_MEMBER"),
    ("WATER_TANK", "mentioned in", "Spec"),
    ("WATER_TANK", "used for", "Appreciation"),
    ("WATER_TANK", "capable of expressing", "Space"),
    ("PSEUDO_SPACE", "is", "Other than underwater space"),
    ("DISPLAY_DEVICE", "provided", "Yes"),
    ("DISPLAY_DEVICE", "used for", "Appreciation"),
    ("DISPLAY_DEVICE", "capable of expressing", "PSEUDO_SPACE"),
    ("PSEUDO_SPACE", "is", "Other than underwater space"),
]

# --- Build graph ---
G = nx.DiGraph()
for s, r, o in edges:
    G.add_edge(s, o, relation=r)

# Layout (increase 'scale' to "zoom out" the initial view)
pos = nx.spring_layout(G, seed=42, k=0.9, scale=3.0)

# --- Build edge lines ---
edge_x, edge_y = [], []
for u, v in G.edges():
    x0, y0 = pos[u]
    x1, y1 = pos[v]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    mode="lines",
    line=dict(width=1),
    hoverinfo="skip",
    name="Edges",
    showlegend=False,
)

# --- Build node scatter with hover info ---
node_x, node_y, node_text = [], [], []
for n in G.nodes():
    x, y = pos[n]
    node_x.append(x); node_y.append(y)
    deg = G.degree(n)
    node_text.append(f"{n}<br>degree: {deg}")

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode="markers+text",
    text=list(G.nodes()),
    textposition="top center",
    hoverinfo="text",
    marker=dict(size=14),
    name="Nodes"
)

# --- Edge label scatter at midpoints (toggleable in legend) ---
label_x, label_y, labels = [], [], []
for u, v, d in G.edges(data=True):
    x0, y0 = pos[u]
    x1, y1 = pos[v]
    mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    label_x.append(mx); label_y.append(my)
    labels.append(d.get("relation", ""))

edge_label_trace = go.Scatter(
    x=label_x, y=label_y,
    mode="text",
    text=labels,
    textposition="middle center",
    hoverinfo="none",
    name="Relations"  # can be toggled via legend
)

# --- Figure ---
fig = go.Figure(
    data=[edge_trace, node_trace, edge_label_trace],
    layout=go.Layout(
        title="Interactive Patent Graph (Triples with Relations)",
        showlegend=True,
        hovermode="closest",
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
)

fig.show()
