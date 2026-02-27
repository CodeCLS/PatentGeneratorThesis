"""
FAISS-based relation merging for knowledge graph triples.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import faiss

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from tools.graph.data.Triple import Triple
from tools.graph.visualizer import GraphVisualizer


class FAISSEdgeMerger:
    """
    Merges similar relations between the same head and tail entities using FAISS.
    """

    def __init__(
        self,
        sim_threshold: float = 0.8,
        model_name: str = "all-MiniLM-L6-v2",
        embed_dim: Optional[int] = None,
        ngram: int = 3,
        keep: str = "first",
    ):
        """
        Initialize the FAISS edge merger.

        Args:
            sim_threshold: Cosine similarity threshold for merging (0.0-1.0)
            model_name: Name of the SentenceTransformer model to use
            embed_dim: Dimension of relation embeddings (optional, derived from model if not provided)
            ngram: N-gram size for fallback hash embedding (if neural model not available)
                - Controls how the relation text is broken into character sequences for embedding
                - Example: ngram=3 means 3-character sequences ("abc", "bcd", "cde" for "abcde")
                - Higher values (e.g., 4-5) capture longer patterns, lower values (e.g., 2-3) are more flexible
                - Default: 3 (good balance between specificity and generalization)
            keep: Strategy for choosing representative relation when multiple similar relations are merged
                - "first": Keep the first relation encountered (default)
                - "shortest": Keep the shortest relation string (most concise)
                - "longest": Keep the longest relation string (most descriptive)
        """
        self.sim_threshold = sim_threshold
        self.model_name = model_name
        self._model: Optional[Any] = None
        self._embedding_dim = embed_dim
        self.ngram = ngram
        self.keep = keep

    def _get_model(self):
        """Lazy-load the sentence transformer model."""
        if self._model is None and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self._model = SentenceTransformer(self.model_name)
                # If embed_dim not set, derive from model
                if self._embedding_dim is None:
                    self._embedding_dim = self._model.get_sentence_embedding_dimension()
            except Exception as e:
                print(f"⚠️ Error loading SentenceTransformer '{self.model_name}': {e}")
                self._model = None
        
        # If model loading failed or not available, use fallback dimension
        if self._embedding_dim is None:
            self._embedding_dim = 256
            
        return self._model

    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Create a relation embedding using neural model or bag-of-hashes fallback.

        Args:
            text: Relation text to embed

        Returns:
            Normalized embedding vector
        """
        model = self._get_model()
        
        if model is not None:
            # Neural embedding
            v = model.encode(text, normalize_embeddings=True)
            return np.asarray(v, dtype=np.float32)
        else:
            # Fallback to bag-of-hashes
            return self._hash_embed(text)

    def _hash_embed(self, text: str) -> np.ndarray:
        """
        Fallback: Create a relation embedding using a stable bag-of-hashes vector.
        Used if sentence-transformers is not available.
        """
        t = f" {text} "
        v = np.zeros(self._embedding_dim, dtype=np.float32)
        if not text:
            return v
        for i in range(len(t) - self.ngram + 1):
            g = t[i:i+self.ngram]
            h = (hash(g) & 0xFFFFFFFF) % self._embedding_dim
            v[h] += 1.0
        # L2 normalize for cosine via inner product
        n = np.linalg.norm(v)
        if n > 0:
            v /= n
        return v

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

    def merge_relations(
        self,
        triples: List[Triple],
    ) -> Tuple[List[Triple], Dict[str, any]]:
        """
        Merge similar relations between the same head and tail entities.
        
        IMPORTANT: This method ONLY merges relations (edge labels), NOT nodes.
        Nodes (entities) are preserved exactly as they appear in the input triples.
        Only relation strings between the same (head, tail) pair are merged.

        For each (head, tail) pair, clusters relation strings by cosine similarity
        >= sim_threshold, then merges each cluster into 1 triple.

        Args:
            triples: List of Triple objects to merge

        Returns:
            Tuple of (merged_triples, stats_dict)
        """
        # Ensure model is initialized to get embedding dimension
        self._get_model()
        
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

            # Embed relations using the neural model (or fallback)
            X = np.stack([self._get_embedding(r) for r in uniq_rels]).astype(np.float32)
            # Already normalized
            index = faiss.IndexFlatIP(self._embedding_dim)
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

                # Create a merged Triple with the same head/tail entities, but merged relation
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
            "model_name": self.model_name,
            "embed_dim": self._embedding_dim,
        }
        return merged_triples, stats




