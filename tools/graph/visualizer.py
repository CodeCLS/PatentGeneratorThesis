"""
Graph visualization utilities using NetworkX and PyVis.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Any
import networkx as nx
from pyvis.network import Network

from tools.graph.Triple import Triple
from tools.sentence.entity import Entity


# Default labels and colors
LABELS = [
    "INVENTION", "COMPONENT", "SUBSYSTEM", "MATERIAL", "CHEMICAL", "BIOMOLECULE", "COMPOSITION",
    "PROCESS_STEP", "METHOD", "PARAMETER", "MEASUREMENT", "CONDITION", "FUNCTION", "SIGNAL",
    "CONTROL", "SOFTWARE", "HARDWARE", "FIGURE_REF", "CLAIM_ELEMENT", "PRIOR_ART",
    "UNCLASSIFIED_ENTITY", "UNKNOWN",
]

DISTINCT_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0",
    "#f032e6", "#bcf60c", "#fabebe", "#008080", "#e6beff", "#9a6324", "#fffac8",
    "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000075", "#808080", "#a9a9ff",
]

NODE_TYPE_COLORS = {t: DISTINCT_COLORS[i % len(DISTINCT_COLORS)] for i, t in enumerate(LABELS)}
NODE_TYPE_COLORS["UNKNOWN"] = "#bdbdbd"

EDGE_CLUSTER_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0",
    "#f032e6", "#bcf60c", "#fabebe", "#008080", "#e6beff", "#9a6324", "#fffac8",
    "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000075", "#808080", "#a9a9ff",
]
EDGE_COLOR_DEFAULT = "#9a9a9a"


class GraphVisualizer:
    """
    Handles graph building and visualization using NetworkX and PyVis.
    """

    def __init__(
        self,
        node_type_colors: Optional[Dict[str, str]] = None,
        edge_cluster_colors: Optional[List[str]] = None,
        edge_color_default: str = EDGE_COLOR_DEFAULT,
    ):
        """
        Initialize the visualizer.

        Args:
            node_type_colors: Mapping from node type to color hex code
            edge_cluster_colors: List of colors for cluster edges
            edge_color_default: Default color for unclustered edges
        """
        self.node_type_colors = node_type_colors or NODE_TYPE_COLORS.copy()
        self.edge_cluster_colors = edge_cluster_colors or EDGE_CLUSTER_COLORS.copy()
        self.edge_color_default = edge_color_default

    @staticmethod
    def infer_type_from_id(x: str) -> str:
        """Infer node type from entity ID."""
        if not x:
            return "UNKNOWN"
        m = re.match(r"^([A-Za-z_]+)[:\-]", str(x))
        return m.group(1).upper() if m else "UNKNOWN"

    @staticmethod
    def _entity_key(e: Entity | str | None) -> str:
        """Stable id for a node."""
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

    @staticmethod
    def _entity_name(e: Entity | None) -> str:
        """Nice display label."""
        if e is None:
            return ""
        for attr in ("name", "text", "surface", "value"):
            if hasattr(e, attr):
                v = getattr(e, attr)
                if v:
                    return str(v)
        # fall back to id
        return GraphVisualizer._entity_key(e)

    @staticmethod
    def _entity_type(e: Entity | None) -> str:
        """Type/label for coloring."""
        if e is None:
            return "UNKNOWN"
        for attr in ("label", "type", "entity_type", "kind"):
            if hasattr(e, attr):
                v = getattr(e, attr)
                if v:
                    return str(v).upper()
        return "UNKNOWN"

    @staticmethod
    def flatten_triples(x: Any) -> List[Triple]:
        """Accept List[Triple], List[List[Triple]], or a single Triple."""
        if x is None:
            return []
        if isinstance(x, Triple):
            return [x]
        if isinstance(x, list):
            out: List[Triple] = []
            for item in x:
                if item is None:
                    continue
                if isinstance(item, Triple):
                    out.append(item)
                elif isinstance(item, list):
                    out.extend([t for t in item if isinstance(t, Triple)])
            return out
        return []

    def build_graph(
        self,
        triples: List[Triple] | List[List[Triple]] | Triple,
        node_type_map: Optional[Dict[str, str]] = None,
    ) -> nx.MultiDiGraph:
        """
        Build a NetworkX MultiDiGraph from triples.

        Args:
            triples: List of Triple objects, nested list, or single Triple
            node_type_map: Optional mapping from node ID to node type

        Returns:
            NetworkX MultiDiGraph with nodes and edges
        """
        node_type_map = node_type_map or {}
        G = nx.MultiDiGraph()
        flat = self.flatten_triples(triples)

        for tr in flat:
            h_id = self._entity_key(tr.head)
            t_id = self._entity_key(tr.tail)
            r = (getattr(tr, "relation", "") or "").strip()

            if not h_id or not t_id or not r:
                continue

            h_type = (node_type_map.get(h_id) or self._entity_type(tr.head) or self.infer_type_from_id(h_id))
            t_type = (node_type_map.get(t_id) or self._entity_type(tr.tail) or self.infer_type_from_id(t_id))

            if not G.has_node(h_id):
                G.add_node(h_id, node_type=h_type)
            if not G.has_node(t_id):
                G.add_node(t_id, node_type=t_type)

            G.add_edge(h_id, t_id, label=r)

        return G

    def visualize_pyvis(
        self,
        G: nx.MultiDiGraph,
        out_file: str = "graph.html",
        id_to_name: Optional[Dict[str, str]] = None,
        cluster_attr: Optional[str] = None,
        cid_to_seedtype: Optional[Dict[int, str]] = None,
    ) -> None:
        """
        Create an interactive PyVis visualization of the graph.

        Args:
            G: NetworkX MultiDiGraph to visualize
            out_file: Output HTML file path
            id_to_name: Optional mapping from node ID to display name
            cluster_attr: Optional edge attribute name for cluster ID
            cid_to_seedtype: Optional mapping from cluster ID to seed type
        """
        id_to_name = id_to_name or {}
        net = Network(height="100vh", width="100%", directed=True, notebook=False)
        net.barnes_hut()

        # Add nodes
        for n in G.nodes():
            t = (G.nodes[n].get("node_type", "UNKNOWN") or "UNKNOWN").upper()
            color = self.node_type_colors.get(t, self.node_type_colors.get("UNKNOWN", "#bdbdbd"))
            label = id_to_name.get(n, n)

            net.add_node(
                n,
                label=label,
                color=color,
                shape="ellipse",
                title=f"name: {label}<br>id: {n}<br>type: {t}",
            )

        # Add edges
        for u, v, k, d in G.edges(keys=True, data=True):
            cid_edge = d.get(cluster_attr, -1) if cluster_attr else -1
            edge_color = (
                self.edge_cluster_colors[cid_edge % len(self.edge_cluster_colors)]
                if cid_edge != -1
                else self.edge_color_default
            )
            edge_width = 2.2 if cid_edge != -1 else 1.2

            title = d.get("label", "")
            if cluster_attr and cid_to_seedtype:
                seed_type = cid_to_seedtype.get(cid_edge, "-")
                title = f"cluster_id: {cid_edge}<br>seed_type: {seed_type}<br>relation: {d.get('label', '')}"

            net.add_edge(
                u, v,
                label=d.get("label", ""),
                arrows="to",
                color=edge_color,
                width=edge_width,
                title=title,
            )

        net.show(out_file, notebook=False)
        print(f"✅ Interactive graph written to {out_file}")

    @staticmethod
    def build_id_to_name_map(sentence_split: List[Any]) -> Dict[str, str]:
        """
        Build a mapping from entity ID to display name from sentence_split.

        Args:
            sentence_split: List of Sentence objects with entities

        Returns:
            Dictionary mapping entity ID to name
        """
        def iter_sentence_entities(sent):
            """Extract entities from a sentence object."""
            ents = getattr(sent, "entities", None)
            if ents is None:
                return []
            if isinstance(ents, dict):
                return list(ents.values())
            if isinstance(ents, list):
                return ents
            try:
                return list(ents)
            except Exception:
                return []

        id_to_name: Dict[str, str] = {}
        for sent in sentence_split:
            for ent in iter_sentence_entities(sent):
                ent_id = None
                for attr in ("ref", "id", "ref_short"):
                    if hasattr(ent, attr):
                        v = getattr(ent, attr)
                        if v:
                            ent_id = str(v)
                            break
                if not ent_id:
                    continue
                name = GraphVisualizer._entity_name(ent)
                if name and (ent_id not in id_to_name or len(name) > len(id_to_name[ent_id])):
                    id_to_name[ent_id] = name

        return id_to_name




