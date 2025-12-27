"""
FAISS-based relation merging for knowledge graph triples.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
import faiss

from tools.graph.Triple import Triple
from tools.graph.visualizer import GraphVisualizer


class FAISSEdgeMerger:
    """
    Merges similar relations between the same head and tail entities using FAISS.
    """

    def __init__(
        self,
        sim_threshold: float = 0.8,
        embed_dim: int = 256,
        ngram: int = 3,
        keep: str = "first",
    ):
        """
        Initialize the FAISS edge merger.

        Args:
            sim_threshold: Cosine similarity threshold for merging (0.0-1.0)
            embed_dim: Dimension of relation embeddings
            ngram: N-gram size for hash embedding
            keep: Strategy for choosing representative relation ("first", "shortest", "longest")
        """
        self.sim_threshold = sim_threshold
        self.embed_dim = embed_dim
        self.ngram = ngram
        self.keep = keep

    @staticmethod
    def _entity_key_any(x) -> str:
        """Extract entity key from Entity object or string."""
        if x is None:
            return ""
        if isinstance(x, str):
            return x
        for attr in ("ref", "id", "ref_short"):
            if hasattr(x, attr):
                v = getattr(x, attr)
                if v:
                    return str(v)
        return str(x)

    @staticmethod
    def _relation_norm(s: str) -> str:
        """Normalize relation string."""
        s = (s or "").strip().lower()
        s = re.sub(r"\s+", " ", s)
        return s

    def _hash_embed(self, text: str) -> np.ndarray:
        """
        Create a relation embedding using a stable bag-of-hashes vector.

        Args:
            text: Relation text to embed

        Returns:
            Normalized embedding vector
        """
        t = f" {text} "
        v = np.zeros(self.embed_dim, dtype=np.float32)
        if not text:
            return v
        for i in range(len(t) - self.ngram + 1):
            g = t[i:i+self.ngram]
            h = (hash(g) & 0xFFFFFFFF) % self.embed_dim
            v[h] += 1.0
        # L2 normalize for cosine via inner product
        n = np.linalg.norm(v)
        if n > 0:
            v /= n
        return v

    def merge_relations(
        self,
        triples: List[Triple],
    ) -> Tuple[List[Triple], Dict[str, any]]:
        """
        Merge similar relations between the same head and tail entities.

        For each (head, tail) pair, clusters relation strings by cosine similarity
        >= sim_threshold, then merges each cluster into 1 triple.

        Args:
            triples: List of Triple objects to merge

        Returns:
            Tuple of (merged_triples, stats_dict)
        """
        # Group triples by (head_id, tail_id)
        by_pair = defaultdict(list)
        for tr in triples or []:
            h = self._entity_key_any(getattr(tr, "head", None))
            t = self._entity_key_any(getattr(tr, "tail", None))
            r = self._relation_norm(getattr(tr, "relation", ""))
            if not h or not t or not r:
                continue
            by_pair[(h, t)].append(tr)

        merged_triples: List[Triple] = []
        merged_count = 0
        kept_count = 0

        for (h_id, t_id), trs in by_pair.items():
            # Unique relation strings for this pair
            rels = []
            for tr in trs:
                rels.append(self._relation_norm(tr.relation))
            # Keep mapping relation->example triple objects
            rel_to_trs = defaultdict(list)
            for tr in trs:
                rel_to_trs[self._relation_norm(tr.relation)].append(tr)

            uniq_rels = list(rel_to_trs.keys())
            if len(uniq_rels) == 1:
                # Nothing to merge
                merged_triples.append(rel_to_trs[uniq_rels[0]][0])
                kept_count += 1
                continue

            # Embed relations
            X = np.stack([self._hash_embed(r) for r in uniq_rels]).astype(np.float32)
            # Already normalized
            index = faiss.IndexFlatIP(self.embed_dim)
            index.add(X)

            # Union-find to cluster relations by similarity >= threshold
            parent = list(range(len(uniq_rels)))

            def find(a):
                while parent[a] != a:
                    parent[a] = parent[parent[a]]
                    a = parent[a]
                return a

            def union(a, b):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra

            # Search all neighbors (k = all) for this pair
            sims, idxs = index.search(X, len(uniq_rels))
            for i in range(len(uniq_rels)):
                for jpos, j in enumerate(idxs[i]):
                    if j < 0 or j == i:
                        continue
                    if sims[i][jpos] >= self.sim_threshold:
                        union(i, int(j))

            # Build clusters
            clusters = defaultdict(list)
            for i in range(len(uniq_rels)):
                clusters[find(i)].append(i)

            # Pick representative relation string per cluster and emit one triple
            for _, members in clusters.items():
                member_rels = [uniq_rels[i] for i in members]

                if self.keep == "shortest":
                    rep_rel = min(member_rels, key=len)
                elif self.keep == "longest":
                    rep_rel = max(member_rels, key=len)
                else:
                    rep_rel = member_rels[0]

                # Pick a base triple to copy head/tail objects from
                base_tr = rel_to_trs[member_rels[0]][0]

                # Create a merged Triple
                merged_triples.append(
                    Triple(head=base_tr.head, relation=rep_rel, tail=base_tr.tail)
                )

                # Stats
                if len(members) > 1:
                    merged_count += (len(members) - 1)
                kept_count += 1

        stats = {
            "pairs": len(by_pair),
            "out_triples": len(merged_triples),
            "merged_relations_removed": merged_count,
            "kept_clusters": kept_count,
            "sim_threshold": self.sim_threshold,
            "embed_dim": self.embed_dim,
            "ngram": self.ngram,
        }
        return merged_triples, stats

