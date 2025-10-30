import re
from typing import List, Dict, Any, Tuple, Optional
import networkx as nx
import matplotlib.pyplot as plt


class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self._warnings: List[str] = []

    # ---------- 1. converter ----------
    @staticmethod
    def _nodes_edges_to_entities_triples(
        kg_obj: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        nodes = kg_obj.get("nodes", []) or []
        edges = kg_obj.get("edges", []) or []
        entities = [
            {"eid": n.get("id"), "name": n.get("label") or n.get("id"), "type": n.get("type", "Entity")}
            for n in nodes
        ]
        triples = [
            {"s": e.get("source"), "p": e.get("predicate"), "o": e.get("target"), "evidence": e.get("evidence")}
            for e in edges
        ]
        return entities, triples

    # ---------- 2. base constructor ----------
    @classmethod
    def from_triple_dicts(
        cls,
        triples: List[Dict[str, Any]],
        entities: Optional[List[Dict[str, Any]]] = None,
    ) -> "KnowledgeGraph":
        kg = cls()
        for t in triples:
            s, p, o = t.get("s"), t.get("p"), t.get("o")
            if not (s and p and o):
                continue
            ev = t.get("evidence") or t.get("justification_sid")
            kg.graph.add_edge(s, o, predicate=p, evidence=ev)
        if entities:
            name_map = {e["eid"]: e.get("name") for e in entities if "eid" in e}
            type_map = {e["eid"]: e.get("type", "Entity") for e in entities if "eid" in e}
            nx.set_node_attributes(kg.graph, name_map, name="label")
            nx.set_node_attributes(kg.graph, type_map, name="type")
        return kg

    # ---------- 3. universal loader ----------
    @classmethod
    def from_pipeline_output(cls, output: Dict[str, Any]) -> "KnowledgeGraph":
        # case: pipeline result with kg
        if "kg" in output:
            ents, trips = cls._nodes_edges_to_entities_triples(output["kg"])
            return cls.from_triple_dicts(triples=trips, entities=ents)
        # case: just kg dict itself
        if {"nodes", "edges"} <= set(output.keys()):
            ents, trips = cls._nodes_edges_to_entities_triples(output)
            return cls.from_triple_dicts(triples=trips, entities=ents)
        # fallback: legacy shape
        ents = output.get("entities", [])
        trips = output.get("triples", [])
        return cls.from_triple_dicts(triples=trips, entities=ents)

    # ---------- 4. visualization ----------
    def draw(self, figsize=(8, 6), layout="spring", seed=42, title="Knowledge Graph"):
        pos = self._layout(layout, seed)
        plt.figure(figsize=figsize)
        nx.draw_networkx_nodes(self.graph, pos, node_size=1000)
        labels = {n: (self.graph.nodes[n].get("label") or n) for n in self.graph.nodes}
        nx.draw_networkx_labels(self.graph, pos, labels=labels, font_size=10)
        nx.draw_networkx_edges(self.graph, pos, arrowstyle="->", arrowsize=15)
        edge_labels = {(u, v): d.get("predicate", "") for u, v, d in self.graph.edges(data=True)}
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=8)
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def _layout(self, layout: str, seed: int):
        if layout == "kamada_kawai":
            return nx.kamada_kawai_layout(self.graph)
        if layout == "circular":
            return nx.circular_layout(self.graph)
        if layout == "shell":
            return nx.shell_layout(self.graph)
        return nx.spring_layout(self.graph, seed=seed)
