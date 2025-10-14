"""
sentence_trees_with_cross_links.py

Builds a large NetworkX graph from text with:
  1) A dependency TREE for EACH sentence (every token is a node).
  2) Cross-sentence edges ONLY between sentence ROOTS if sentences are "connected":
       - share >= K content lemmas (configurable), OR
       - share >= 1 named entity text (exact string match), OR
       - optional adjacency link (S_i <-> S_{i+1})

Nodes:
  - One node per TOKEN INSTANCE (not lemma), so every word from the text is present.
  - Node id is "s{sent_id}_t{token.i}" to keep them unique and stable.
  - Node attrs: text, lemma, pos, tag, dep, head_id (for intra-sent tree), sent_id, is_root.

Edges:
  - Intra-sentence: head -> child (directed by default, can switch to undirected).
  - Inter-sentence: root_i <-> root_j (undirected), with reasons annotated.

Exports:
  - GEXF / GraphML (set paths below).
  - Optional matplotlib draw (skip for huge graphs).

"""

from __future__ import annotations
import itertools
from collections import defaultdict
from typing import List, Dict, Set, Tuple

import spacy
import networkx as nx
import matplotlib.pyplot as plt

# -----------------------
# Configuration
# -----------------------

TEXT = """
The present invention relates to a display device for appreciation regarding pseudo space such as
in the water or in the air and regarding creatures and/or
objects which float in the air represented by fish, submarines, spaceships, airships, butterflies, birds, and the
like.

"""

SPACY_MODEL = "en_core_web_sm"   # try 'en_core_web_trf' if you have it
DIRECTED_INTRA_SENT = True       # dependency arcs are naturally directed (head -> child)
ADD_ADJACENT_SENT_LINKS = False   # also link root_i <-> root_{i+1}
CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ"}  # for lemma-overlap heuristic
LEMMA_OVERLAP_K = 2              # min shared content lemmas between sentences to link roots
ENABLE_ENTITY_OVERLAP = True     # link roots if sentences share any exact entity text

# Exports / plotting
EXPORT_GEXF = "sentence_word_network.gexf"
EXPORT_GRAPHML = None            # e.g., "sentence_word_network.graphml"
TRY_PLOT = True                 # Large texts can be huge; set True for small demos
RANDOM_SEED = 7

# -----------------------
# Utilities
# -----------------------

def token_node_id(sent_id: int, tok_i: int) -> str:
    """Unique node id per token instance."""
    return f"s{sent_id}_t{tok_i}"

def is_root(tok) -> bool:
    """spaCy root check (dep_ == 'ROOT' or head == tok)."""
    return tok.dep_ == "ROOT" or tok.head == tok

def sent_content_lemmas(sent) -> Set[str]:
    """Lowercased lemmas for content POS in a sentence."""
    out = set()
    for tok in sent:
        if tok.is_alpha and tok.pos_ in CONTENT_POS:
            out.add(tok.lemma_.lower())
    return out

def sent_entity_strings(sent) -> Set[str]:
    """Exact surface strings for entities in a sentence (excluding trivial labels)."""
    ents = set()
    for ent in sent.ents:
        if ent.label_ not in {"CARDINAL", "ORDINAL", "DATE", "TIME"}:
            ents.add(" ".join(ent.text.split()))
    return ents

# -----------------------
# Core
# -----------------------

def build_graph_from_text(text: str) -> Tuple[nx.Graph, Dict[int, str]]:
    """
    Returns:
      G: NetworkX graph containing all token nodes, intra-sentence dependency edges,
         and cross-sentence root links.
      root_by_sent_id: mapping sent_id -> node_id of the sentence root.
    """
    nlp = spacy.load(SPACY_MODEL)
    nlp.max_length = max(nlp.max_length, len(text) + 1000)
    doc = nlp(text)

    # Build per-sentence token nodes + dependency edges (trees)
    G = nx.DiGraph() if DIRECTED_INTRA_SENT else nx.Graph()
    root_by_sent_id: Dict[int, str] = {}

    sentences = list(doc.sents)
    # Precompute sentence-level features for cross-links
    sent_lemmas = {}
    sent_ents = {}

    for sid, sent in enumerate(sentences):
        # Add nodes for all tokens in this sentence
        for tok in sent:
            nid = token_node_id(sid, tok.i)
            G.add_node(
                nid,
                text=tok.text,
                lemma=tok.lemma_,
                pos=tok.pos_,
                tag=tok.tag_,
                dep=tok.dep_,
                is_root=is_root(tok),
                sent_id=sid,
            )

        # Add dependency edges (tree inside the sentence)
        for tok in sent:
            child_id = token_node_id(sid, tok.i)
            if is_root(tok):
                root_by_sent_id[sid] = child_id
                continue
            # Only connect within-sentence arcs
            if tok.head.sent == sent:
                head_id = token_node_id(sid, tok.head.i)
                if DIRECTED_INTRA_SENT:
                    G.add_edge(head_id, child_id, kind="dep")
                else:
                    G.add_edge(head_id, child_id, kind="dep")

        # Cache features for cross-sentence decisions
        sent_lemmas[sid] = sent_content_lemmas(sent)
        sent_ents[sid] = sent_entity_strings(sent)

    # Now add cross-sentence links between roots
    # Strategy: for all sentence pairs (or just nearby), connect roots if:
    #  - entity overlap >= 1, OR
    #  - content-lemma overlap >= LEMMA_OVERLAP_K
    #  - optional adjacency backbone (sid <-> sid+1)
    # We add these as UNDIRECTED relationships between the root nodes.
    if DIRECTED_INTRA_SENT:
        # Create an undirected wrapper for cross links while keeping the main graph directed
        UG = nx.Graph()
        UG.add_nodes_from(G.nodes(data=True))
    else:
        UG = G  # already undirected

    # Adjacency links
    if ADD_ADJACENT_SENT_LINKS:
        for sid in range(len(sentences) - 1):
            r1 = root_by_sent_id.get(sid)
            r2 = root_by_sent_id.get(sid + 1)
            if r1 and r2:
                UG.add_edge(r1, r2, kind="sent_adjacent")

    # Overlap-based links (quadratic in #sentences; fine for most docs, optimize if needed)
    for i, j in itertools.combinations(range(len(sentences)), 2):
        r1 = root_by_sent_id.get(i)
        r2 = root_by_sent_id.get(j)
        if not r1 or not r2:
            continue

        reasons = []
        # Entity overlap
        if ENABLE_ENTITY_OVERLAP and (sent_ents[i] & sent_ents[j]):
            reasons.append("root_link_entity")

        # Lemma overlap
        if len(sent_lemmas[i] & sent_lemmas[j]) >= LEMMA_OVERLAP_K:
            reasons.append("root_link_lemma")

        if reasons:
            # Add a single edge with concatenated reasons
            # If an edge exists, append new reasons
            if UG.has_edge(r1, r2):
                existing = UG[r1][r2].get("kind", "")
                UG[r1][r2]["kind"] = (existing + "|" if existing else "") + "|".join(reasons)
            else:
                UG.add_edge(r1, r2, kind="|".join(reasons))

    # If we used a separate undirected graph for cross links, merge edges back
    if DIRECTED_INTRA_SENT:
        for u, v, data in UG.edges(data=True):
            if not G.has_edge(u, v) and not G.has_edge(v, u):
                # Add as undirected-ish metadata by duplicating or just one way with kind tag
                G.add_edge(u, v, kind=data.get("kind", "root_link"))

    return G, root_by_sent_id

# -----------------------
# Reporting / Plot
# -----------------------

def describe_graph(G: nx.Graph, root_by_sent_id: Dict[int, str]) -> None:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    roots = len(root_by_sent_id)
    print(f"Graph built: {n} nodes, {m} edges, {roots} sentence roots")

    # Quick stats: how many cross links vs dep edges
    kinds = defaultdict(int)
    for _, _, d in G.edges(data=True):
        kinds[d.get("kind", "unknown")] += 1
    print("Edge kinds:")
    for k, c in sorted(kinds.items(), key=lambda x: -x[1]):
        print(f"  {k:18s}: {c}")

def quick_plot(G: nx.Graph, root_by_sent_id: Dict[int, str]) -> None:
    if G.number_of_nodes() > 800:
        print("Graph too big for a quick plot; export to Gephi instead.")
        return
    # Make root nodes larger
    node_sizes = []
    for nid, data in G.nodes(data=True):
        node_sizes.append(800 if data.get("is_root") else 200)

    pos = nx.spring_layout(G, seed=RANDOM_SEED, k=None)
    plt.figure(figsize=(12, 9))
    # Draw dependency edges lighter; cross-sentence darker
    dep_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("kind") == "dep"]
    cross_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("kind") != "dep"]

    nx.draw_networkx_edges(G, pos, edgelist=dep_edges, alpha=0.15)
    nx.draw_networkx_edges(G, pos, edgelist=cross_edges, alpha=0.6, width=1.5)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, alpha=0.9)
    nx.draw_networkx_labels(
    G,
    pos,
    labels={n: G.nodes[n]["text"] for n in G.nodes()},
    font_size=8,
    verticalalignment="center",
    horizontalalignment="center"
)

    # For dense graphs, labels may clutter—toggle if needed:
    # nx.draw_networkx_labels(G, pos, labels={n: G.nodes[n]["text"] for n in G.nodes()}, font_size=8)

    plt.title("Sentence Trees with Cross-Sentence Root Links")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# -----------------------
# Main
# -----------------------

if __name__ == "__main__":
    G, roots = build_graph_from_text(TEXT)
    describe_graph(G, roots)

    if EXPORT_GEXF:
        nx.write_gexf(G, EXPORT_GEXF)
        print(f"Exported GEXF to: {EXPORT_GEXF}")
    if EXPORT_GRAPHML:
        nx.write_graphml(G, EXPORT_GRAPHML)
        print(f"Exported GraphML to: {EXPORT_GRAPHML}")

    if TRY_PLOT:
        quick_plot(G, roots)
