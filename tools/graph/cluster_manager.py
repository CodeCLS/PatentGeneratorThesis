"""
Cluster management for knowledge graphs using rule-based and semantic-based approaches.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Set, Optional, Any, Tuple
import networkx as nx
import torch
import torch.nn.functional as F

from tools.graph.visualizer import GraphVisualizer


# Default configuration
LABELS = [
    "INVENTION", "COMPONENT", "SUBSYSTEM", "MATERIAL", "CHEMICAL", "BIOMOLECULE", "COMPOSITION",
    "PROCESS_STEP", "METHOD", "PARAMETER", "MEASUREMENT", "CONDITION", "FUNCTION", "SIGNAL",
    "CONTROL", "SOFTWARE", "HARDWARE", "FIGURE_REF", "CLAIM_ELEMENT", "PRIOR_ART",
    "UNCLASSIFIED_ENTITY", "UNKNOWN",
]

SEED_TYPES = {"INVENTION", "SUBSYSTEM", "COMPONENT"}

ATTACH_ONLY = {
    "MATERIAL", "CHEMICAL", "BIOMOLECULE", "COMPOSITION",
    "PROCESS_STEP", "METHOD", "PARAMETER", "MEASUREMENT", "CONDITION",
    "FUNCTION", "SIGNAL", "CONTROL", "SOFTWARE", "HARDWARE",
    "FIGURE_REF", "PRIOR_ART", "CLAIM_ELEMENT",
    "UNCLASSIFIED_ENTITY", "UNKNOWN",
}

DEFAULT_RULES = {
    "INVENTION": {
        "hops": 2,
        "barrier_types": {"INVENTION"},
        "traverse_allow": None,
    },
    "SUBSYSTEM": {
        "hops": 1,
        "barrier_types": {"SUBSYSTEM"},
        "traverse_allow": {"INVENTION", "COMPONENT", "HARDWARE", "SOFTWARE", "CONTROL", "SIGNAL", "FUNCTION", "MATERIAL"},
    },
    "COMPONENT": {
        "hops": 1,
        "barrier_types": {"COMPONENT"},
        "traverse_allow": {"INVENTION", "SUBSYSTEM", "HARDWARE", "SOFTWARE", "MATERIAL", "FUNCTION"},
    },
    "MATERIAL": {
        "hops": 1,
        "barrier_types": {"MATERIAL"},
        "traverse_allow": {"INVENTION", "COMPONENT", "SUBSYSTEM", "CHEMICAL", "COMPOSITION"},
    },
    "CHEMICAL": {
        "hops": 1,
        "barrier_types": {"CHEMICAL"},
        "traverse_allow": {"MATERIAL", "COMPOSITION", "INVENTION", "COMPONENT", "SUBSYSTEM"},
    },
    "BIOMOLECULE": {
        "hops": 1,
        "barrier_types": {"BIOMOLECULE"},
        "traverse_allow": {"CHEMICAL", "MATERIAL", "COMPOSITION", "INVENTION", "COMPONENT", "SUBSYSTEM"},
    },
    "COMPOSITION": {
        "hops": 1,
        "barrier_types": {"COMPOSITION"},
        "traverse_allow": {"CHEMICAL", "MATERIAL", "BIOMOLECULE", "INVENTION", "COMPONENT", "SUBSYSTEM"},
    },
    "METHOD": {
        "hops": 1,
        "barrier_types": {"METHOD"},
        "traverse_allow": {"PROCESS_STEP", "CONDITION", "PARAMETER", "MEASUREMENT", "INVENTION", "COMPONENT", "SUBSYSTEM"},
    },
    "PROCESS_STEP": {
        "hops": 1,
        "barrier_types": {"PROCESS_STEP"},
        "traverse_allow": {"METHOD", "CONDITION", "PARAMETER", "MEASUREMENT", "INVENTION", "COMPONENT", "SUBSYSTEM"},
    },
    "PARAMETER": {
        "hops": 1,
        "barrier_types": {"PARAMETER"},
        "traverse_allow": {"MEASUREMENT", "CONDITION", "METHOD", "PROCESS_STEP", "INVENTION", "COMPONENT", "SUBSYSTEM"},
    },
    "MEASUREMENT": {
        "hops": 1,
        "barrier_types": {"MEASUREMENT"},
        "traverse_allow": {"PARAMETER", "CONDITION", "METHOD", "PROCESS_STEP", "INVENTION", "COMPONENT", "SUBSYSTEM"},
    },
    "CONDITION": {
        "hops": 1,
        "barrier_types": {"CONDITION"},
        "traverse_allow": {"PARAMETER", "MEASUREMENT", "METHOD", "PROCESS_STEP", "INVENTION", "COMPONENT", "SUBSYSTEM"},
    },
    "FUNCTION": {
        "hops": 1,
        "barrier_types": {"FUNCTION"},
        "traverse_allow": {"CONTROL", "SIGNAL", "SOFTWARE", "HARDWARE", "INVENTION", "COMPONENT", "SUBSYSTEM"},
    },
    "SIGNAL": {
        "hops": 1,
        "barrier_types": {"SIGNAL"},
        "traverse_allow": {"CONTROL", "SOFTWARE", "HARDWARE", "FUNCTION", "INVENTION", "COMPONENT", "SUBSYSTEM"},
    },
    "CONTROL": {
        "hops": 1,
        "barrier_types": {"CONTROL"},
        "traverse_allow": {"SIGNAL", "SOFTWARE", "HARDWARE", "FUNCTION", "INVENTION", "COMPONENT", "SUBSYSTEM"},
    },
    "SOFTWARE": {
        "hops": 1,
        "barrier_types": {"SOFTWARE"},
        "traverse_allow": {"HARDWARE", "CONTROL", "SIGNAL", "FUNCTION", "INVENTION", "COMPONENT", "SUBSYSTEM"},
    },
    "HARDWARE": {
        "hops": 1,
        "barrier_types": {"HARDWARE"},
        "traverse_allow": {"SOFTWARE", "CONTROL", "SIGNAL", "FUNCTION", "INVENTION", "COMPONENT", "SUBSYSTEM"},
    },
    "FIGURE_REF": {"hops": 1, "barrier_types": {"FIGURE_REF"}, "traverse_allow": set()},
    "PRIOR_ART": {"hops": 1, "barrier_types": {"PRIOR_ART"}, "traverse_allow": set()},
    "CLAIM_ELEMENT": {"hops": 1, "barrier_types": {"CLAIM_ELEMENT"}, "traverse_allow": {"INVENTION", "COMPONENT", "SUBSYSTEM"}},
    "UNCLASSIFIED_ENTITY": {"hops": 1, "barrier_types": set(), "traverse_allow": {"INVENTION", "COMPONENT", "SUBSYSTEM"}},
    "UNKNOWN": {"hops": 1, "barrier_types": set(), "traverse_allow": {"INVENTION", "COMPONENT", "SUBSYSTEM"}},
}

SEED_PRIORITY = [
    "COMPONENT",
    "SUBSYSTEM",
    "MATERIAL",
    "CHEMICAL",
    "BIOMOLECULE",
    "COMPOSITION",
    "SOFTWARE",
    "HARDWARE",
    "CONTROL",
    "SIGNAL",
    "FUNCTION",
    "METHOD",
    "PROCESS_STEP",
    "PARAMETER",
    "MEASUREMENT",
    "CONDITION",
    "INVENTION",
    "CLAIM_ELEMENT",
    "PRIOR_ART",
    "FIGURE_REF",
    "UNCLASSIFIED_ENTITY",
    "UNKNOWN",
]
PRIORITY_RANK = {t: i for i, t in enumerate(SEED_PRIORITY)}


class ClusterManager:
    """
    Manages cluster creation and postprocessing for knowledge graphs.
    Supports both rule-based and semantic-based clustering approaches.
    """

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        rules: Optional[Dict[str, Dict[str, Any]]] = None,
        seed_types: Optional[Set[str]] = None,
        seed_priority: Optional[List[str]] = None,
    ):
        """
        Initialize the cluster manager.

        Args:
            graph: NetworkX MultiDiGraph to cluster
            rules: Dictionary of node type rules (hops, barrier_types, traverse_allow)
            seed_types: Set of node types that can create clusters
            seed_priority: List of node types in priority order for edge assignment
        """
        self.graph = graph
        self.undirected_graph = graph.to_undirected(as_view=False)
        self.rules = rules or DEFAULT_RULES.copy()
        self.seed_types = seed_types or SEED_TYPES.copy()
        self.seed_priority = seed_priority or SEED_PRIORITY.copy()
        self.priority_rank = {t: i for i, t in enumerate(self.seed_priority)}
        self.clusters: List[Dict[str, Any]] = []

    def node_type(self, n: str) -> str:
        """Get node type from graph node."""
        return (self.graph.nodes[n].get("node_type", "UNKNOWN") or "UNKNOWN").upper()

    def nodes_within_hops_rules(
        self,
        seed: str,
        seed_type: str,
    ) -> Set[str]:
        """
        BFS traversal with rules to find nodes within hops of a seed.

        Args:
            seed: Seed node ID
            seed_type: Type of the seed node

        Returns:
            Set of node IDs within the cluster
        """
        cfg = self.rules.get(seed_type, {"hops": 1, "barrier_types": set(), "traverse_allow": None})
        max_hops = int(cfg.get("hops", 1))
        barrier_types = {t.upper() for t in (cfg.get("barrier_types") or set())}
        traverse_allow = cfg.get("traverse_allow", None)
        if traverse_allow is not None:
            traverse_allow = {t.upper() for t in traverse_allow}

        seen = {seed}
        q = deque([(seed, 0)])

        while q:
            cur, d = q.popleft()
            if d >= max_hops:
                continue

            cur_t = self.node_type(cur)

            # Do not expand outward through barriers (except seed itself at d=0)
            if d > 0 and cur_t in barrier_types:
                continue

            # If traverse_allow is specified, only expand outward through allowed types
            if d > 0 and traverse_allow is not None and cur_t not in traverse_allow:
                continue

            for nb in self.undirected_graph.neighbors(cur):
                if nb not in seen:
                    seen.add(nb)
                    q.append((nb, d + 1))

        return seen

    def create_rule_based_clusters(self) -> List[Dict[str, Any]]:
        """
        Create clusters using rule-based BFS traversal.

        Returns:
            List of cluster dictionaries with cluster_id, seed, seed_type, and nodes
        """
        # Find all seed nodes
        seeds = [(n, self.node_type(n)) for n in self.graph.nodes() if self.node_type(n) in self.rules]
        print(f"Seed nodes: {len(seeds)}")

        # Build clusters (one per seed)
        clusters = []
        cid = 0
        for s, st in seeds:
            ns = self.nodes_within_hops_rules(s, st)
            clusters.append({
                "cluster_id": cid,
                "seed": s,
                "seed_type": st,
                "nodes": ns,
            })
            cid += 1

        print(f"Clusters (no merging): {len(clusters)}")
        self.clusters = clusters
        return clusters

    def assign_edges_to_clusters(
        self,
        clusters: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[int, str]:
        """
        Assign edges to clusters based on priority.

        Args:
            clusters: Optional list of clusters (uses self.clusters if None)

        Returns:
            Dictionary mapping cluster_id to seed_type
        """
        clusters = clusters or self.clusters
        if not clusters:
            return {}

        cid_to_seedtype = {c["cluster_id"]: c.get("seed_type", "UNKNOWN") for c in clusters}

        node_to_cids = defaultdict(list)
        for c in clusters:
            for n in c["nodes"]:
                node_to_cids[n].append(c["cluster_id"])

        def edge_cluster_id(u: str, v: str) -> int:
            cu, cv = node_to_cids.get(u, []), node_to_cids.get(v, [])
            if not cu or not cv:
                return -1
            inter = set(cu).intersection(cv)
            if not inter:
                return -1

            # Pick best by priority rank, then cid
            def key(cid):
                st = cid_to_seedtype.get(cid, "UNKNOWN")
                return (self.priority_rank.get(st, 10_000), cid)

            return min(inter, key=key)

        # Annotate edges with cluster color + cluster_id
        for u, v, k, d in self.graph.edges(keys=True, data=True):
            cide = edge_cluster_id(u, v)
            d["cluster_id"] = cide

        return cid_to_seedtype

    @staticmethod
    def induced_edge_count(G: nx.MultiDiGraph, nodes: Set[str]) -> int:
        """Count edges within a set of nodes."""
        if not nodes:
            return 0
        nodes = set(nodes)
        cnt = 0
        for u, v, k in G.edges(keys=True):
            if u in nodes and v in nodes:
                cnt += 1
        return cnt

    def postprocess_clusters(
        self,
        min_delete_edges: int = 10,
        min_merge_edges: int = 30,
        max_iters: int = 5,
        max_path_search: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Iteratively postprocess clusters by deleting small ones and merging medium ones.

        Args:
            min_delete_edges: Minimum edges to keep a cluster (below this = delete)
            min_merge_edges: Minimum edges to keep a cluster independent (below this = merge)
            max_iters: Maximum iterations
            max_path_search: Maximum path length for finding nearest INVENTION cluster

        Returns:
            List of postprocessed clusters
        """
        clusters = self.clusters.copy()

        for it in range(max_iters):
            cid_to_cluster = {c["cluster_id"]: c for c in clusters}

            invention_cids = [c["cluster_id"] for c in clusters if c.get("seed_type") == "INVENTION"]
            if not invention_cids:
                print("⚠️ No INVENTION clusters found. Cannot merge-to-invention. Stopping.")
                break

            # Recompute edge counts each iteration
            cid_to_edges = {
                cid: self.induced_edge_count(self.graph, cid_to_cluster[cid]["nodes"])
                for cid in cid_to_cluster
            }

            deleted = {cid for cid, e in cid_to_edges.items() if e < min_delete_edges}
            medium = [cid for cid, e in cid_to_edges.items() if min_delete_edges <= e < min_merge_edges]

            print(f"\nITER {it+1}")
            print(f"  clusters: {len(clusters)}")
            print(f"  delete: {len(deleted)}")
            print(f"  merge: {len(medium)}")

            if not deleted and not medium:
                print("  ✅ stable (no small clusters left)")
                break

            # Build seed lookup for inventions + pick fallback target
            inv_seed_to_cid = {cid_to_cluster[cid]["seed"]: cid for cid in invention_cids}
            inv_seeds = set(inv_seed_to_cid.keys())
            largest_inv = max(invention_cids, key=lambda cid: cid_to_edges.get(cid, 0))

            def nearest_invention_cid(start_seed: str):
                # If starting seed is itself an invention seed
                if start_seed in inv_seed_to_cid:
                    return inv_seed_to_cid[start_seed], 0

                seen = {start_seed}
                q = deque([(start_seed, 0)])
                while q:
                    cur, d = q.popleft()
                    if max_path_search is not None and d > max_path_search:
                        continue
                    if cur in inv_seeds:
                        return inv_seed_to_cid[cur], d
                    for nb in self.undirected_graph.neighbors(cur):
                        if nb not in seen:
                            seen.add(nb)
                            q.append((nb, d + 1))
                return largest_inv, float("inf")  # fallback: force merge

            merged_into = {}
            for cid in medium:
                if cid in deleted:
                    continue
                c = cid_to_cluster[cid]
                target_cid, dist = nearest_invention_cid(c["seed"])
                cid_to_cluster[target_cid]["nodes"] |= set(c["nodes"])
                merged_into[cid] = target_cid

            removed = deleted | set(merged_into.keys())

            # Keep clusters not removed
            clusters_kept = []
            for c in clusters:
                if c["cluster_id"] in removed:
                    continue
                clusters_kept.append(cid_to_cluster[c["cluster_id"]])

            # Reindex cluster ids 0..N-1 for cleanliness
            clusters2 = []
            for new_id, c in enumerate(clusters_kept):
                c2 = dict(c)
                c2["cluster_id"] = new_id
                clusters2.append(c2)

            clusters = clusters2

        self.clusters = clusters
        return clusters

    def create_semantic_clusters(
        self,
        encoder: Any,
        sim_threshold: float = 0.75,
    ) -> Tuple[List[Set[str]], Dict[str, int]]:
        """
        Create clusters using semantic similarity of path embeddings.

        Args:
            encoder: TextEncoder instance with embed_text_mean method
            sim_threshold: Cosine similarity threshold for clustering

        Returns:
            Tuple of (list of cluster node sets, node to cluster_id mapping)
        """
        def cosine(emb_a: torch.Tensor, emb_b: torch.Tensor) -> float:
            """Compute cosine similarity between two 1D embeddings."""
            emb_a = emb_a.view(-1)
            emb_b = emb_b.view(-1)
            a_norm = F.normalize(emb_a, dim=0)
            b_norm = F.normalize(emb_b, dim=0)
            return torch.dot(a_norm, b_norm).item()

        def edge_to_text(G, u, v, data) -> str:
            """Build a textual representation of one triple (edge)."""
            head_txt = G.nodes[u].get("label", str(u))
            tail_txt = G.nodes[v].get("label", str(v))
            rel_text = data.get("relation", data.get("label", ""))
            text = (
                f"{head_txt} {head_txt} "
                f"{rel_text} "
                f"{tail_txt} {tail_txt}"
            )
            return text

        def get_first_edge_info(G, node):
            """Finds the first incident edge to node, builds its text, and embeds it."""
            # Outgoing edges
            for nbr in G.neighbors(node):
                for key, data in G[node][nbr].items():
                    triple_text = edge_to_text(G, node, nbr, data)
                    seed_emb = encoder.embed_text_mean(triple_text)
                    return triple_text, seed_emb

            # Incoming edges if directed
            if G.is_directed():
                for pred in G.predecessors(node):
                    for key, data in G[pred][node].items():
                        triple_text = edge_to_text(G, pred, node, data)
                        seed_emb = encoder.embed_text_mean(triple_text)
                        return triple_text, seed_emb

            return None, None

        def recursive_traversing(G, node, seed_embedding, sim_threshold, visited, cluster_nodes, path_texts):
            """Depth-first traversal using path embeddings."""
            if node in visited:
                return

            visited.add(node)
            cluster_nodes.add(node)

            # Successors
            for nbr in G.neighbors(node):
                for key, data in G[node][nbr].items():
                    triple_text = edge_to_text(G, node, nbr, data)
                    new_path_texts = path_texts + [triple_text]
                    combined_text = " . ".join(new_path_texts)
                    path_emb = encoder.embed_text_mean(combined_text)
                    sim = cosine(path_emb, seed_embedding)

                    if sim >= sim_threshold and nbr not in visited:
                        recursive_traversing(
                            G, nbr, seed_embedding, sim_threshold, visited, cluster_nodes, new_path_texts
                        )

            # Predecessors (for directed graphs)
            if G.is_directed():
                for pred in G.predecessors(node):
                    for key, data in G[pred][node].items():
                        triple_text = edge_to_text(G, pred, node, data)
                        new_path_texts = path_texts + [triple_text]
                        combined_text = " . ".join(new_path_texts)
                        path_emb = encoder.embed_text_mean(combined_text)
                        sim = cosine(path_emb, seed_embedding)

                        if sim >= sim_threshold and pred not in visited:
                            recursive_traversing(
                                G, pred, seed_embedding, sim_threshold, visited, cluster_nodes, new_path_texts
                            )

        visited = set()
        clusters: List[Set[str]] = []
        node2cluster: Dict[str, int] = {}
        cluster_id = 0

        for node in self.graph.nodes():
            if node in visited:
                continue

            seed_text, seed_emb = get_first_edge_info(self.graph, node)
            if seed_emb is None:
                continue

            cluster_nodes = set()
            initial_path_texts = [seed_text]

            recursive_traversing(
                self.graph,
                node,
                seed_emb,
                sim_threshold,
                visited,
                cluster_nodes,
                initial_path_texts,
            )

            if not cluster_nodes:
                continue

            clusters.append(cluster_nodes)
            for n in cluster_nodes:
                node2cluster[n] = cluster_id

            cluster_id += 1

        return clusters, node2cluster

    def attach_to_main_clusters(
        self,
        main_cluster_ids: List[int],
        encoder: Any,
        attach_threshold: float = 0.75,
    ) -> Dict[Tuple[str, str, Any], Optional[int]]:
        """
        Attach non-main edges to the nearest main cluster based on embedding similarity.

        Args:
            main_cluster_ids: List of main cluster IDs
            encoder: TextEncoder instance
            attach_threshold: Cosine similarity threshold for attachment

        Returns:
            Dictionary mapping (u, v, key) to main cluster ID or None
        """
        def cosine(emb_a: torch.Tensor, emb_b: torch.Tensor) -> float:
            emb_a = emb_a.view(-1)
            emb_b = emb_b.view(-1)
            a_norm = F.normalize(emb_a, dim=0)
            b_norm = F.normalize(emb_b, dim=0)
            return torch.dot(a_norm, b_norm).item()

        # Compute centroids for main clusters
        cluster_centroids = {}
        for cid in main_cluster_ids:
            embs = []
            for (u, v, key, data) in self.graph.edges(keys=True, data=True):
                if data.get("cluster") == cid:
                    emb = data.get("embedding")
                    if emb is not None:
                        if isinstance(emb, list):
                            emb = torch.tensor(emb)
                        embs.append(emb)

            if embs:
                stacked = torch.stack(embs, dim=0)
                cluster_centroids[cid] = stacked.mean(dim=0)

        # Attach non-main edges
        edge2parent_claim = {}
        for (u, v, key, data) in self.graph.edges(keys=True, data=True):
            edge_cluster = data.get("cluster")

            if edge_cluster in main_cluster_ids:
                continue

            emb = data.get("embedding")
            if emb is None:
                edge2parent_claim[(u, v, key)] = None
                continue

            if isinstance(emb, list):
                emb = torch.tensor(emb)

            best_cid = None
            best_sim = -1.0

            for mid, centroid in cluster_centroids.items():
                sim = cosine(emb, centroid)
                if sim > best_sim:
                    best_sim = sim
                    best_cid = mid

            if best_cid is not None and best_sim >= attach_threshold:
                edge2parent_claim[(u, v, key)] = best_cid
                data["parent_claim"] = best_cid
            else:
                edge2parent_claim[(u, v, key)] = None
                data["parent_claim"] = None

        return edge2parent_claim




