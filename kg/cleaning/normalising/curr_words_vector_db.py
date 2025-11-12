# pip install faiss-cpu
import faiss
import json
import numpy as np
from typing import Dict, List, Callable, Optional

EmbedFn = Callable[[str], np.ndarray]

class WordVectorDB:
    """
    Manages lemma embeddings + FAISS indices per POS.
    Handles build/search/save/load so other code stays simple.
    """
    def __init__(self, embed_fn: EmbedFn, dim: int, lowercase: bool = True):
        self.embed = embed_fn
        self.dim = dim
        self.lowercase = lowercase
        self.indices: Dict[str, faiss.Index] = {}   # POS -> FAISS index
        self.labels:  Dict[str, List[str]] = {}     # POS -> lemmas in index order
        self._built = False

    def _emb(self, w: str) -> np.ndarray:
        v = np.asarray(self.embed(w), dtype="float32").reshape(-1)
        if v.shape[0] != self.dim:
            raise ValueError(f"Embed dim {v.shape[0]} != expected {self.dim}")
        return v

    def build(self, canon_by_pos: Dict[str, List[str]] | List[str], index_type: str = "flat_ip"):
        """
        canon_by_pos: dict like {"VERB":[...], "NOUN":[...]} or a single list for 'ALL'.
        index_type: 'flat_ip' (exact), 'ivf_flat', or 'hnsw'
        """
        if isinstance(canon_by_pos, list):
            canon_by_pos = {"ALL": canon_by_pos}

        for pos, lemmas in canon_by_pos.items():
            pos = pos.upper()
            canon = [l.lower() if self.lowercase else l for l in lemmas]
            vecs = np.vstack([self._emb(x) for x in canon]).astype("float32")
            faiss.normalize_L2(vecs)

            if index_type == "flat_ip":
                index = faiss.IndexFlatIP(self.dim)
            elif index_type == "ivf_flat":
                nlist = max(64, int(np.sqrt(len(canon))) * 4)
                quant = faiss.IndexFlatIP(self.dim)
                index = faiss.IndexIVFFlat(quant, self.dim, nlist, faiss.METRIC_INNER_PRODUCT)
                index.train(vecs)
                index.nprobe = min(32, nlist)
            elif index_type == "hnsw":
                index = faiss.IndexHNSWFlat(self.dim, 32)
                index.hnsw.efConstruction = 200
                index.hnsw.efSearch = 64
            else:
                raise ValueError("index_type must be flat_ip | ivf_flat | hnsw")

            index.add(vecs)
            self.indices[pos] = index
            self.labels[pos]  = canon

        self._built = True

    def search(self, pos: str, query_vec: np.ndarray, k: int = 1):
        if not self._built:
            raise RuntimeError("VectorDB not built. Call build() first.")
        key = pos.upper() if pos.upper() in self.indices else "ALL"
        q = np.asarray(query_vec, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(q)
        sims, idxs = self.indices[key].search(q, k)
        # For IP with normalized vectors: sims == cosine similarity
        results = [(self.labels[key][int(i)], float(s)) for i, s in zip(idxs[0], sims[0])]
        return results

    def save(self, path_prefix: str, meta: Optional[dict] = None):
        meta = dict(meta or {})
        meta.update({"dim": self.dim, "lowercase": self.lowercase, "pos_keys": list(self.indices.keys())})
        for pos, index in self.indices.items():
            faiss.write_index(index, f"{path_prefix}.{pos}.faiss")
            with open(f"{path_prefix}.{pos}.labels.json", "w", encoding="utf-8") as f:
                json.dump(self.labels[pos], f, ensure_ascii=False)
        with open(f"{path_prefix}.meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

    def load(self, path_prefix: str):
        with open(f"{path_prefix}.meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.dim = meta["dim"]
        self.lowercase = meta["lowercase"]
        self.indices.clear(); self.labels.clear()
        for pos in meta["pos_keys"]:
            self.indices[pos] = faiss.read_index(f"{path_prefix}.{pos}.faiss")
            with open(f"{path_prefix}.{pos}.labels.json", "r", encoding="utf-8") as f:
                self.labels[pos] = json.load(f)
        self._built = True
