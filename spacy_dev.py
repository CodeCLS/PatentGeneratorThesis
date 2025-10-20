# pip install spacy networkx matplotlib
# python -m spacy download en_core_web_md

from __future__ import annotations
import re
from collections import Counter
from typing import List, Tuple, Set
import spacy
import networkx as nx
import matplotlib.pyplot as plt

# --------------------------
# Text (your example)
# --------------------------
TEXT = """A water tank for appreciation provided with a
water storage, a water pipe, an air bubble generating
member, wherein said water storage is provided with a
water inlet port and an opening portion and said opening
portion is so provided that the convection can be generated in said water tank for appreciation by the liquid
current from a water tank through said opening portion,
and said water storage is installed upwardly of a water
tank for appreciation and one end of said water pipe is
connected to an inlet water port of a water storage and
the other end is so installed that it is in the liquid when
the liquid is filled in a water tank for appreciation and an
air bubble generating member is so installed that it can
lead the air bubble inside of a water pipe in the peripheral portion of said other end of said water pipe and a
display device for appreciation using said water tank for
appreciation are used."""
##research

# --------------------------
# Config (extend anytime)
# --------------------------
REL_WEIGHTS = {           # influences the tree extraction
    "part-of": 3.0,
    "function-of": 2.0,
    "property-of": 1.0,
    "spatial": 1.0,
    "condition": 1.0,
}

# language cues
PART_OF_VERB_LEMMAS: Set[str] = {"provide", "include", "comprise", "contain", "equip"}
FUNCTION_VERB_LEMMAS: Set[str] = {"configure", "adapt", "operate", "enable", "allow", "cause", "generate", "lead"}
FUNCTION_ADJ_MARKERS: Set[str] = {"configured", "adapted", "operable"}
SPATIAL_PREPS: Set[str] = {"in", "on", "between", "through", "from", "to", "into", "at", "of", "with", "upwardly", "adjacent"}
CONDITION_MARKERS: Set[str] = {"wherein", "when", "if", "so that", "such that"}  # handled lightly

# --------------------------
# Utilities
# --------------------------
def clean_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("\n", " ")).strip()

def normalize_np_text(span: spacy.tokens.Span) -> str:
    # Collapse "said" → "the", trim spaces/commas; keep human-readable NP text
    txt = span.text
    txt = re.sub(r"\bsaid\s+", "the ", txt, flags=re.I)
    txt = re.sub(r"\s+", " ", txt).strip(" ,;")
    return txt

def np_head_key(np: spacy.tokens.Span) -> str:
    # a light canonical key (for root picking)
    return np.root.lemma_.lower()

def pick_root_concept(doc: spacy.tokens.Doc) -> str | None:
    c = Counter(np_head_key(ch) for ch in doc.noun_chunks)
    return c.most_common(1)[0][0] if c else None

# --------------------------
# Relation Extraction (rule-based, patent-friendly)
# --------------------------
def extract_concepts_and_relations(doc: spacy.tokens.Doc):
    """
    Returns:
      nodes: set[str] of concept names
      edges: list[tuple[src, dst, rel_type, weight]]
    """
    nodes: Set[str] = set()
    edges: List[Tuple[str, str, str, float]] = []

    noun_spans = list(doc.noun_chunks)
    for ch in noun_spans:
        nodes.add(normalize_np_text(ch))

    # 1) part-of via verbs like "provided with / including / comprising ..."
    for tok in doc:
        if tok.pos_ == "VERB" and tok.lemma_.lower() in PART_OF_VERB_LEMMAS:
            whole = None
            # subject NP as whole
            for c in tok.children:
                if c.dep_ in {"nsubj", "nsubjpass"}:
                    whole = normalize_np_text(doc[c.left_edge.i : c.right_edge.i + 1])
                    break
            # if not found, climb to a governing NP
            if not whole and tok.head.pos_ in {"NOUN", "PROPN"}:
                whole = normalize_np_text(doc[tok.head.left_edge.i : tok.head.right_edge.i + 1])
            parts = []
            for c in tok.children:
                if c.dep_ in {"dobj", "obj"}:
                    parts.append(normalize_np_text(doc[c.left_edge.i : c.right_edge.i + 1]))
                if c.dep_ == "prep":
                    for gc in c.children:
                        if gc.dep_ == "pobj":
                            parts.append(normalize_np_text(doc[gc.left_edge.i : gc.right_edge.i + 1]))
            if whole and parts:
                nodes.add(whole)
                for p in parts:
                    nodes.add(p)
                    edges.append((whole, p, "part-of", REL_WEIGHTS["part-of"]))

    # 2) function-of: adjectival "X configured/adapted/operable to VERB ..." or verbal with xcomp/to VERB
    for tok in doc:
        # adjectival marker attached to an NP: "member configured to lead ..."
        if tok.text.lower() in FUNCTION_ADJ_MARKERS and tok.dep_ in {"amod", "acl"}:
            owner_np = normalize_np_text(doc[tok.head.left_edge.i : tok.head.right_edge.i + 1])
            func = None
            for t2 in tok.subtree:
                if t2.pos_ == "VERB":
                    func = t2.lemma_.lower()
                    break
            if owner_np and func:
                nodes.add(owner_np); nodes.add(func)
                edges.append((owner_np, func, "function-of", REL_WEIGHTS["function-of"]))

        # verbal: "X VERB ... to VERB" or xcomp/ccomp
        if tok.pos_ == "VERB" and tok.lemma_.lower() in FUNCTION_VERB_LEMMAS:
            owner_np = None
            for c in tok.children:
                if c.dep_ in {"nsubj", "nsubjpass"}:
                    owner_np = normalize_np_text(doc[c.left_edge.i : c.right_edge.i + 1])
                    break
            func = None
            for c in tok.children:
                if c.dep_ in {"xcomp", "ccomp"} and c.pos_ == "VERB":
                    func = c.lemma_.lower()
                    break
            if not func:
                # scan rightward for "to VERB"
                for t2 in tok.subtree:
                    if t2.text.lower() == "to":
                        for v in t2.subtree:
                            if v.pos_ == "VERB":
                                func = v.lemma_.lower()
                                break
                        if func:
                            break
            if owner_np and func:
                nodes.add(owner_np); nodes.add(func)
                edges.append((owner_np, func, "function-of", REL_WEIGHTS["function-of"]))

    # 3) property-of: adjectival/numeric modifiers
    for ch in noun_spans:
        owner = normalize_np_text(ch)
        head = ch.root
        props = []
        for c in head.children:
            if c.dep_ == "amod":
                props.append(c.lemma_.lower())
            elif c.dep_ == "nummod":
                props.append(c.text.lower())
        for p in props:
            nodes.add(owner); nodes.add(p)
            edges.append((owner, p, "property-of", REL_WEIGHTS["property-of"]))

    # 4) spatial relations via common prepositions between NPs
    for tok in doc:
        if tok.pos_ == "ADP" and tok.lemma_.lower() in SPATIAL_PREPS:
            pobj = None
            for c in tok.children:
                if c.dep_ == "pobj":
                    pobj = normalize_np_text(doc[c.left_edge.i : c.right_edge.i + 1])
                    break
            anchor = None
            if tok.head.pos_ in {"NOUN", "PROPN"}:
                anchor = normalize_np_text(doc[tok.head.left_edge.i : tok.head.right_edge.i + 1])
            elif tok.head.pos_ == "VERB":
                # try to anchor on a nearby NP of the verb
                for c in tok.head.children:
                    if c.dep_ in {"dobj", "obj", "nsubj", "nsubjpass"}:
                        anchor = normalize_np_text(doc[c.left_edge.i : c.right_edge.i + 1])
                        break
            if anchor and pobj:
                nodes.add(anchor); nodes.add(pobj)
                edges.append((anchor, pobj, "spatial", REL_WEIGHTS["spatial"]))

    # 5) condition clauses (very light heuristic)
    for tok in doc:
        if tok.lemma_.lower() in {"wherein", "when", "if"} and tok.dep_ == "mark":
            cl_verb = tok.head
            gov_np = None
            for c in cl_verb.children:
                if c.dep_ in {"nsubj", "dobj", "obj"}:
                    gov_np = normalize_np_text(doc[c.left_edge.i : c.right_edge.i + 1])
                    break
            cond_txt = " ".join(t.text for t in cl_verb.subtree)
            cond_txt = re.sub(r"\s+", " ", cond_txt).strip(" ,;")
            if gov_np and cond_txt:
                nodes.add(gov_np); nodes.add(cond_txt)
                edges.append((gov_np, cond_txt, "condition", REL_WEIGHTS["condition"]))

    return nodes, edges

# --------------------------
# Build concept graph & tree
# --------------------------
def build_concept_graph(nodes: Set[str], edges: List[Tuple[str, str, str, float]]) -> nx.DiGraph:
    G = nx.DiGraph()
    for n in nodes:
        G.add_node(n)
    for src, dst, rel, w in edges:
        if src == dst:
            continue
        # keep the strongest weight per (src,dst)
        if G.has_edge(src, dst):
            if w > G[src][dst].get("weight", 0.0):
                G[src][dst]["weight"] = w
                G[src][dst]["rel"] = rel
        else:
            G.add_edge(src, dst, weight=w, rel=rel)
    return G
#I want to have two parts to this project 1. Information extraction from the Text into a Graph 2. Graph to a claim -- 
def derive_concept_tree(G: nx.DiGraph, root_hint: str | None = None) -> tuple[nx.DiGraph, str]:
    # choose a root
    root = root_hint
    if root is None and len(G) > 0:
        root = max(G.nodes, key=lambda n: (G.in_degree(n), G.degree(n)))
    if root is None:
        return nx.DiGraph(), ""

    # boost part-of edges so they dominate the arborescence
    H = nx.DiGraph()
    for u, v, d in G.edges(data=True):
        w = float(d.get("weight", 1.0))
        if d.get("rel") == "part-of":
            w += 5.0
        H.add_edge(u, v, weight=w, rel=d.get("rel"))

    # try maximum spanning arborescence
    try:
        from networkx.algorithms.tree.branchings import maximum_spanning_arborescence
        T = maximum_spanning_arborescence(H)
        # keep only nodes reachable from root
        reachable = nx.descendants(T, root) | {root}
        T = T.subgraph(reachable).copy()
        return T, root
    except Exception:
        # fallback: undirected MST on largest component
        UG = nx.Graph()
        for u, v, d in H.edges(data=True):
            UG.add_edge(u, v, weight=d["weight"], rel=d.get("rel", "rel"))
        if len(UG) == 0:
            return nx.DiGraph(), root
        cc = max(nx.connected_components(UG), key=len)
        MST = nx.maximum_spanning_tree(UG.subgraph(cc))
        return nx.DiGraph(MST), root

# --------------------------
# Plotting
# --------------------------
def plot_concept_graphs(G: nx.DiGraph, T: nx.DiGraph, root: str):
    if len(G) == 0:
        print("No concepts/relations found — nothing to plot.")
        return

    rel_colors = {
        "part-of": "#1f77b4",      # blue
        "function-of": "#2ca02c",  # green
        "property-of": "#ff7f0e",  # orange
        "spatial": "#9467bd",      # purple
        "condition": "#d62728",    # red
    }

    # --- 1) Multi-relation graph ---
    plt.figure(figsize=(10, 7))
    pos = nx.spring_layout(G, seed=42)
    node_opts = dict(node_size=900, node_color="#E6F2FF", edgecolors="#1f4a7f")
    edge_colors = [rel_colors.get(d.get("rel"), "#7f7f7f") for _, _, d in G.edges(data=True)]
    edge_widths = [1.0 + d.get("weight", 1.0) for _, _, d in G.edges(data=True)]
    nx.draw_networkx_nodes(G, pos, **node_opts)
    nx.draw_networkx_labels(G, pos, font_size=9)
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, arrows=True, arrowsize=15)

    from matplotlib.lines import Line2D
    legend_handles = [Line2D([0], [0], color=c, lw=3, label=r) for r, c in rel_colors.items()]
    plt.legend(handles=legend_handles, title="Relation", loc="best")
    plt.title("Concept Graph (multi-relation)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # --- 2) Tree (hierarchical) ---
    def hierarchy_pos(Gdir, root, width=1.6, vert_gap=0.35, vert_loc=0.0, xcenter=0.5, pos=None):
        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        children = list(Gdir.successors(root))
        if not children:
            return pos
        dx = width / max(len(children), 1)
        nextx = xcenter - width / 2 - dx / 2
        for child in children:
            nextx += dx
            pos[child] = (nextx, vert_loc - vert_gap)
            pos = hierarchy_pos(Gdir, child, width=dx, vert_gap=vert_gap, vert_loc=vert_loc - vert_gap,
                                xcenter=nextx, pos=pos)
        return pos

    if len(T) == 0 or root == "":
        print("Derived tree is empty or has no root.")
        return

    tpos = hierarchy_pos(T, root)
    t_edge_colors = [rel_colors.get(d.get("rel"), "#7f7f7f") for _, _, d in T.edges(data=True)]
    t_edge_widths = [1.0 + d.get("weight", 1.0) for _, _, d in T.edges(data=True)]

    plt.figure(figsize=(10, 7))
    nx.draw_networkx_nodes(T, tpos, node_size=900, node_color="#FFF7E6", edgecolors="#7f4a1f")
    nx.draw_networkx_labels(T, tpos, font_size=9)
    nx.draw_networkx_edges(T, tpos, width=t_edge_widths, edge_color=t_edge_colors, arrows=True, arrowsize=15)

    legend_handles = [Line2D([0], [0], color=c, lw=3, label=r) for r, c in rel_colors.items()]
    plt.legend(handles=legend_handles, title="Relation", loc="best")
    plt.title(f"Concept Tree (root: {root})")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# --------------------------
# Run
# --------------------------
if __name__ == "__main__":
    nlp = spacy.load("en_core_web_md")  # <--- use md/lg (parser + vectors)
    doc = nlp(clean_whitespace(TEXT))

    nodes, edges = extract_concepts_and_relations(doc)
    print(f"Concepts: {len(nodes)} | Relations: {len(edges)}")

    G = build_concept_graph(nodes, edges)
    root_hint = pick_root_concept(doc)  # e.g., "tank" or the most frequent head
    T, root = derive_concept_tree(G, root_hint=root_hint)
    print("Root concept:", root)

    # Optional: quick tree text dump
    def print_tree(T: nx.DiGraph, root: str, indent: int = 0):
        print("  " * indent + f"- {root}")
        for _, child, d in sorted(T.out_edges(root, data=True), key=lambda x: (-x[2].get("weight", 0), x[1])):
            print("  " * (indent + 1) + f"[{d.get('rel')}]")
            print_tree(T, child, indent + 2)

    if root:
        print("\nConcept Tree:")
        print_tree(T, root)

    # Plots
    plot_concept_graphs(G, T, root)
