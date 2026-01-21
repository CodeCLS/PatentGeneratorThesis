"""
Graph visualization utilities using NetworkX and PyVis.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Any
import networkx as nx
from pyvis.network import Network

from tools.graph.data.Triple import Triple
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
        """Get entity identifier, prioritizing ref over id. Stable id for a node."""
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
        """Nice display label - uses entity.name attribute."""
        if e is None:
            return ""
        if hasattr(e, "name") and e.name:
            return str(e.name).strip()
        # Fall back to id
        return GraphVisualizer._entity_key(e)
    
    def _get_readable_node_label(
        self,
        node_id: str,
        node_data: Dict[str, Any],
        id_to_name: Dict[str, str],
        G: Optional[nx.MultiDiGraph] = None,
    ) -> str:
        """
        Get a readable label for a node based on its type and data.
        
        Args:
            node_id: The node ID
            node_data: The node's data dictionary
            id_to_name: Optional mapping from entity ID to display name
            G: Optional graph to traverse for connected entities
            
        Returns:
            A readable label string
        """
        node_type = node_data.get("node_type", "")
        
        # For assertion nodes, create a readable label from predicate and connected entities
        if node_type == "ASSERTION":
            predicate = node_data.get("predicate", "")
            category = node_data.get("category", "")
            
            # Try to get subject and object from connected nodes
            subject_label = ""
            object_label = ""
            if G:
                # Find SUBJECT edge
                for target in G.successors(node_id):
                    edge_data = G.get_edge_data(node_id, target)
                    if edge_data:
                        for key, data in edge_data.items():
                            if data.get("label") == "SUBJECT":
                                subject_label = self._get_entity_display_name(G, target, id_to_name)
                                break
                
                # Find OBJECT edge
                for target in G.successors(node_id):
                    edge_data = G.get_edge_data(node_id, target)
                    if edge_data:
                        for key, data in edge_data.items():
                            if data.get("label") == "OBJECT":
                                object_label = self._get_entity_display_name(G, target, id_to_name)
                                break
                
                # Check for literal value
                if not object_label:
                    value = node_data.get("value")
                    if value:
                        object_label = str(value)
            
            # Build label
            if subject_label and object_label:
                label = f"{subject_label} --[{predicate}]--> {object_label}"
            elif subject_label:
                label = f"{subject_label} --[{predicate}]--> ?"
            elif predicate:
                label = predicate
            else:
                label = f"Assertion: {node_id[:12]}..."
            
            # Add category if meaningful
            if category and category != "UNCLASSIFIED" and len(label) < 40:
                label = f"[{category}] {label}"
            
            # Truncate if too long
            if len(label) > 60:
                label = label[:57] + "..."
            return label
        
        # For claim concept nodes
        elif node_type == "CLAIM_CONCEPT":
            kind = node_data.get("kind", "")
            breadth = node_data.get("breadth", "")
            claim_id = node_data.get("claim_id", node_id)
            
            parts = []
            if kind:
                parts.append(kind)
            if breadth:
                parts.append(breadth)
            
            if parts:
                label = f"{' '.join(parts)} Claim"
            else:
                label = "Claim Concept"
            
            # Add short ID for uniqueness
            short_id = claim_id.split("_")[-1][:6] if "_" in claim_id else claim_id[:6]
            return f"{label} ({short_id})"
        
        # For entity nodes, use name from node data (extracted from Entity.name), then id_to_name
        else:
            # First try id_to_name mapping (built from Entity.name - most reliable)
            if node_id in id_to_name:
                name = id_to_name[node_id]
                if name:
                    return name
            
            # Then try to get name from node data (stored from Entity.name when building graph)
            if "name" in node_data:
                name = node_data["name"]
                if name and name != node_id:
                    return str(name).strip()
            
            # Fall back to node ID, but truncate if it's a long UUID
            if len(node_id) > 30:
                return f"{node_id[:27]}..."
            return node_id
    
    @staticmethod
    def _get_entity_display_name(
        G: nx.MultiDiGraph,
        entity_id: str,
        id_to_name: Dict[str, str],
    ) -> str:
        """Get display name for an entity node - uses entity.name from id_to_name or node data."""
        # First try id_to_name mapping (built from Entity.name)
        if entity_id in id_to_name:
            return id_to_name[entity_id]
        
        # Try to get from node data (stored from Entity.name when building graph)
        if G.has_node(entity_id):
            node_data = G.nodes[entity_id]
            # Direct access to name attribute (stored from Entity.name)
            if "name" in node_data:
                name = node_data["name"]
                if name:
                    return str(name)
        
        # Fall back to truncated ID
        if len(entity_id) > 20:
            return f"{entity_id[:17]}..."
        return entity_id

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
        deduplicate: bool = True,
    ) -> nx.MultiDiGraph:
        """
        Build a NetworkX MultiDiGraph from triples.

        Args:
            triples: List of Triple objects, nested list, or single Triple
            node_type_map: Optional mapping from node ID to node type
            deduplicate: If True, only add unique edges (head, tail, relation) once

        Returns:
            NetworkX MultiDiGraph with nodes and edges
        """
        node_type_map = node_type_map or {}
        G = nx.MultiDiGraph()
        flat = self.flatten_triples(triples)
        
        # Track unique edges if deduplicating
        seen_edges = set() if deduplicate else None

        for tr in flat:
            h_id = self._entity_key(tr.head)
            t_id = self._entity_key(tr.tail)
            r = (getattr(tr, "relation", "") or "").strip()

            if not h_id or not t_id or not r:
                continue

            # Check for duplicates if deduplicating
            if deduplicate:
                edge_key = (h_id, t_id, r)
                if edge_key in seen_edges:
                    continue  # Skip duplicate edge
                seen_edges.add(edge_key)

            h_type = (node_type_map.get(h_id) or self._entity_type(tr.head) or self.infer_type_from_id(h_id))
            t_type = (node_type_map.get(t_id) or self._entity_type(tr.tail) or self.infer_type_from_id(t_id))
            
            # Get entity names for node attributes
            h_name = self._entity_name(tr.head)
            t_name = self._entity_name(tr.tail)

            if not G.has_node(h_id):
                G.add_node(h_id, node_type=h_type, name=h_name)
            if not G.has_node(t_id):
                G.add_node(t_id, node_type=t_type, name=t_name)

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
            node_data = G.nodes[n]
            t = (node_data.get("node_type", "UNKNOWN") or "UNKNOWN").upper()
            color = self.node_type_colors.get(t, self.node_type_colors.get("UNKNOWN", "#bdbdbd"))
            
            # Get readable label based on node type (pass G for assertion node traversal)
            label = self._get_readable_node_label(n, node_data, id_to_name, G)
            
            # Build title with more details
            title_parts = [f"id: {n}", f"type: {t}"]
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
        Uses entity.name directly from Entity objects.

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
                # Get entity ref (primary identifier), fallback to id or ref_short
                ent_id = None
                for attr in ("ref", "id", "ref_short"):  # ref is now primary identifier
                    if hasattr(ent, attr):
                        v = getattr(ent, attr)
                        if v:
                            ent_id = str(v)
                            break
                if not ent_id:
                    continue
                
                # Use entity.name directly (not ref or other attributes)
                if hasattr(ent, "name") and ent.name:
                    name = str(ent.name).strip()
                    if name and (ent_id not in id_to_name or len(name) > len(id_to_name.get(ent_id, ""))):
                        id_to_name[ent_id] = name

        return id_to_name
    
    @staticmethod
    def build_id_to_name_map_from_triples(triples: List[Triple]) -> Dict[str, str]:
        """
        Build a mapping from entity ID to display name directly from triples.
        Uses entity.name directly from Entity objects in triples.

        Args:
            triples: List of Triple objects with Entity objects as head and tail

        Returns:
            Dictionary mapping entity ID to name
        """
        id_to_name: Dict[str, str] = {}
        
        for triple in triples:
            # Process head entity - use entity.name directly
            if hasattr(triple, "head") and triple.head:
                head_entity = triple.head
                head_id = GraphVisualizer._entity_key(head_entity)
                if hasattr(head_entity, "name") and head_entity.name:
                    name = str(head_entity.name).strip()
                    if name and (head_id not in id_to_name or len(name) > len(id_to_name.get(head_id, ""))):
                        id_to_name[head_id] = name
            
            # Process tail entity - use entity.name directly
            if hasattr(triple, "tail") and triple.tail:
                tail_entity = triple.tail
                tail_id = GraphVisualizer._entity_key(tail_entity)
                if hasattr(tail_entity, "name") and tail_entity.name:
                    name = str(tail_entity.name).strip()
                    if name and (tail_id not in id_to_name or len(name) > len(id_to_name.get(tail_id, ""))):
                        id_to_name[tail_id] = name
        
        return id_to_name

    @staticmethod
    def build_id_to_label_map(sentence_split: List[Any]) -> Dict[str, str]:
        """
        Build a mapping from entity ID to label/type from sentence_split.

        Args:
            sentence_split: List of Sentence objects with entities

        Returns:
            Dictionary mapping entity ID to label/type
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

        id_to_label: Dict[str, str] = {}
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
                # Get label from entity (label or entity_type attribute)
                label = None
                if hasattr(ent, "label"):
                    label = getattr(ent, "label")
                elif hasattr(ent, "entity_type"):
                    label = getattr(ent, "entity_type")
                if label:
                    # Use the first label we find for this ID (or keep existing if already set)
                    if ent_id not in id_to_label:
                        id_to_label[ent_id] = str(label).upper()

        return id_to_label

    def remove_singular_nodes(self, G: nx.MultiDiGraph, verbose: bool = True) -> nx.MultiDiGraph:
        """
        Remove all singular nodes (nodes with no edges) from the graph.
        
        Args:
            G: NetworkX MultiDiGraph to clean
            verbose: If True, print how many nodes were removed
            
        Returns:
            Modified graph G (nodes are removed in place)
        """
        # Find all nodes with degree 0 (no incoming or outgoing edges)
        singular_nodes = [n for n in G.nodes() if G.degree(n) == 0]
        
        if singular_nodes:
            # Remove singular nodes
            G.remove_nodes_from(singular_nodes)
            
            if verbose:
                print(f"✅ Removed {len(singular_nodes)} singular node(s) (nodes with no edges)")
        elif verbose:
            print("✅ No singular nodes found")
        
        return G




