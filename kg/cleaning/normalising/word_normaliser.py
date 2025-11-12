import numpy as np
from typing import Dict, List, Callable, Optional
from kg.cleaning.normalising.curr_words_vector_db import WordVectorDB

class WordNormaliser:
    def __init__(
        self,
        embed_fn: Callable[[str], np.ndarray],
        dim: int,
        threshold: float = 0.75,
        lowercase: bool = True,
    ):
        self.db = WordVectorDB(embed_fn=embed_fn, dim=dim, lowercase=lowercase)
        self.threshold = threshold
        # optional exact overrides
        self.alias: Dict[str, Dict[str, str]] = {"VERB": {}, "NOUN": {}, "ADJ": {}, "ALL": {}}

    def fit(self, canon_by_pos: Dict[str, List[str]] | List[str], index_type: str = "flat_ip"):
        self.db.build(canon_by_pos, index_type=index_type)

    def normalise_token(self, token: str, pos: str = "ALL") -> str:
        t = token.lower() if self.db.lowercase else token

        # 1. Exact alias rule first
        if t in self.alias.get(pos.upper(), {}):
            return self.alias[pos.upper()][t]
        if t in self.alias["ALL"]:
            return self.alias["ALL"][t]

        # 2. Embed and search
        vec = self.db._emb(t)
        hits = self.db.search(pos, vec, k=1)
        best_lemma, best_sim = hits[0]

        # 3. If similar enough → use canonical
        if best_sim >= self.threshold:
            return best_lemma

        # 4. Otherwise → treat as new word, add it to the index
        index_key = pos.upper() if pos.upper() in self.db.indices else "ALL"
        index = self.db.indices[index_key]
        index.add(vec.reshape(1, -1))                     # add new vector
        self.db.labels[index_key].append(t)               # store label alongside
        return t


    def normalise_tokens(self, tokens: List[str], pos_tags: Optional[List[str]] = None) -> List[str]:
        out = []
        for i, tok in enumerate(tokens):
            pos = pos_tags[i] if pos_tags else "ALL"
            out.append(self.normalise_token(tok, pos))
        return out
