from typing import Any, Dict, Tuple, Optional
import numpy as np
import networkx as nx
import torch
from torch import nn
from torch_geometric.nn.kge import TransE


class KGECreator:
    """
    Build a TransE KGE (via PyTorch Geometric) from a NetworkX MultiDiGraph.

    Responsibilities:
    - map nodes & relations -> integer IDs
    - hold TransE model + training loop
    - compute triple scores / embeddings
    - set edge weights on the original graph
    - compute a minimum spanning tree from those weights
    """

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        embedding_dim: int = 128,
        margin: float = 1.0,
        device: Optional[str] = None,
    ):
        self.graph = graph
        self.embedding_dim = embedding_dim
        self.margin = margin

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Mappings
        self.entity_to_id: Dict[Any, int] = {}
        self.id_to_entity: Dict[int, Any] = {}
        self.relation_to_id: Dict[str, int] = {}
        self.id_to_relation: Dict[int, str] = {}

        # Triple tensors (will be filled by _build_triples)
        self.h: Optional[torch.Tensor] = None
        self.r: Optional[torch.Tensor] = None
        self.t: Optional[torch.Tensor] = None

        # Map each edge (u, v, key) -> index into h/r/t
        self.edge_to_idx: Dict[Tuple[Any, Any, Any], int] = {}

        # KGE model
        self.model: Optional[TransE] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

        # Build triples from graph
        self._build_triples()
        self._init_model()

    # ------------------------------------------------------------------
    # 1. Graph -> integer triples
    # ------------------------------------------------------------------
    def _get_ent_id(self, node: Any) -> int:
        if node not in self.entity_to_id:
            idx = len(self.entity_to_id)
            self.entity_to_id[node] = idx
            self.id_to_entity[idx] = node
        return self.entity_to_id[node]

    def _get_rel_id(self, rel: str) -> int:
        if rel not in self.relation_to_id:
            idx = len(self.relation_to_id)
            self.relation_to_id[rel] = idx
            self.id_to_relation[idx] = rel
        return self.relation_to_id[rel]

    def _build_triples(self) -> None:
        triples = []
        self.edge_to_idx.clear()

        for idx, (u, v, key, data) in enumerate(self.graph.edges(keys=True, data=True)):
            rel = data.get("role") or data.get("label") or "related_to"
            h_id = self._get_ent_id(u)
            r_id = self._get_rel_id(rel)
            t_id = self._get_ent_id(v)

            triples.append((h_id, r_id, t_id))
            self.edge_to_idx[(u, v, key)] = idx

        if not triples:
            raise ValueError("Graph has no edges; cannot build KGE.")

        triples = np.array(triples, dtype=np.int64)
        self.h = torch.from_numpy(triples[:, 0]).to(self.device)
        self.r = torch.from_numpy(triples[:, 1]).to(self.device)
        self.t = torch.from_numpy(triples[:, 2]).to(self.device)

    # ------------------------------------------------------------------
    # 2. Init TransE model
    # ------------------------------------------------------------------
    def _init_model(self) -> None:
        num_nodes = len(self.entity_to_id)
        num_relations = len(self.relation_to_id)

        self.model = TransE(
            num_nodes,              # positional
            num_relations,          # positional
            self.embedding_dim,     # positional: embedding_dim
            self.margin,            # positional: margin
            1                       # norm (default = 1)
        ).to(self.device)


        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    # ------------------------------------------------------------------
    # 3. Train TransE (very simple loop with tail-corruption)
    # ------------------------------------------------------------------
    def train(self, epochs: int = 10, batch_size: int = 512) -> None:
        if self.model is None or self.optimizer is None:
            raise RuntimeError("Model not initialized.")

        n = self.h.size(0)
        criterion = nn.MarginRankingLoss(margin=self.margin)

        for epoch in range(epochs):
            self.model.train()
            perm = torch.randperm(n, device=self.device)
            total_loss = 0.0

            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                h = self.h[idx]
                r = self.r[idx]
                t = self.t[idx]

                # negative sampling: corrupt tail
                t_neg = t[torch.randperm(t.size(0))]

                self.optimizer.zero_grad()

                # lower score = more plausible
                pos_score = self.model(h, r, t)        # [B]
                neg_score = self.model(h, r, t_neg)    # [B]

                # We want pos_score + margin < neg_score
                y = torch.ones_like(pos_score)
                loss = criterion(-pos_score, -neg_score, y)

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * h.size(0)

            avg_loss = total_loss / n

    # ------------------------------------------------------------------
    # 4. Triple score / importance
    # ------------------------------------------------------------------
    def _triple_score_ids(self, h_id: int, r_id: int, t_id: int) -> float:
        if self.model is None:
            raise RuntimeError("Model not initialized.")

        self.model.eval()
        with torch.no_grad():
            h = torch.tensor([h_id], device=self.device)
            r = torch.tensor([r_id], device=self.device)
            t = torch.tensor([t_id], device=self.device)
            score = self.model(h, r, t)[0].item()  # scalar
        return score  # lower = more plausible

    def triple_importance_ids(self, h_id: int, r_id: int, t_id: int) -> float:
        """
        Convert TransE score to an importance in [0,1]:
        higher TransE score -> closer to 1.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized.")

        self.model.eval()
        with torch.no_grad():
            h = torch.tensor([h_id], device=self.device)
            r = torch.tensor([r_id], device=self.device)
            t = torch.tensor([t_id], device=self.device)
            score = self.model(h, r, t)[0].item()  # scalar

        # Simple sigmoid squashing to [0,1]
        # (You could also do a global min-max, but sigmoid is local & cheap)
        importance = 1.0 / (1.0 + np.exp(-score))
        return float(importance)


    def triple_importance_edge(self, u: Any, v: Any, key: Any, weight_attr: str = "weight") -> float:
        return float(self.graph[u][v][key].get(weight_attr, 1.0))


    # ------------------------------------------------------------------
    # 5. Set weights on the original graph
    # ------------------------------------------------------------------
    def set_weights_from_kge(self, weight_attr: str = "weight") -> None:
        """
        Compute TransE scores for all triples, then min-max normalize
        them to [0,1] and store as edge weights.

        Higher score => higher importance.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized.")

        self.model.eval()
        with torch.no_grad():
            # scores shape: [num_edges]
            scores = self.model(self.h, self.r, self.t).cpu().numpy()

        s_min = float(scores.min())
        s_max = float(scores.max())

        if s_max == s_min:
            # all scores identical, just set 1.0 (or 0.5 if you prefer)
            imps = np.ones_like(scores, dtype=float)
        else:
            # scale to [0,1]: lowest score -> 0, highest -> 1
            imps = (scores - s_min) / (s_max - s_min)

        # assign back to edges
        for (u, v, key), idx in self.edge_to_idx.items():
            imp = float(imps[idx])
            self.graph[u][v][key][weight_attr] = imp

    def set_weights_from_dict(
        self,
        weights: Dict[Tuple[Any, Any, Any], float],
        weight_attr: str = "weight",
    ) -> None:
        for (u, v, key), w in weights.items():
            if self.graph.has_edge(u, v, key):
                self.graph[u][v][key][weight_attr] = float(w)

    def set_weights_from_callable(
        self,
        score_fn,
        weight_attr: str = "weight",
    ) -> None:
        """
        score_fn(u, v, data_dict) -> float
        """
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            data[weight_attr] = float(score_fn(u, v, data))
    def score(self) -> Dict[Tuple[Any, Any, Any], float]:
        """
        Compute a KGE-based importance score for every edge in the graph.

        Returns
        -------
        Dict[(u, v, key), float]
            Mapping from edge (u, v, key) to importance in [0, 1],
            where higher = more plausible/important according to TransE.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized.")

        if self.h is None or self.r is None or self.t is None:
            raise RuntimeError("Triples not built.")

        self.model.eval()
        with torch.no_grad():
            # Raw TransE scores: higher = more plausible
            scores = self.model(self.h, self.r, self.t).cpu().numpy()

        s_min = float(scores.min())
        s_max = float(scores.max())

        if s_max == s_min:
            importances = np.ones_like(scores, dtype=float)
        else:
            # Normalize to [0,1]: lowest score -> 0, highest -> 1
            importances = (scores - s_min) / (s_max - s_min)

        edge_scores: Dict[Tuple[Any, Any, Any], float] = {}
        for (u, v, key), idx in self.edge_to_idx.items():
            edge_scores[(u, v, key)] = float(importances[idx])

        return edge_scores
    def predict_tails(
    self,
    head: Any,
    relation: str,
    top_k: int = 10,
    ):
        """
        Given a head entity and a relation, rank all possible tail entities.

        Parameters
        ----------
        head : Any
            Node label in your original graph (e.g. "Alice").
        relation : str
            Relation label (e.g. "friend_of").
        top_k : int
            How many top tails to return.

        Returns
        -------
        List[Tuple[Any, float]]
            List of (tail_entity_label, score), sorted by descending plausibility.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized.")

        if head not in self.entity_to_id:
            raise ValueError(f"Unknown head entity: {head}")
        if relation not in self.relation_to_id:
            raise ValueError(f"Unknown relation: {relation}")

        h_id = self.entity_to_id[head]
        r_id = self.relation_to_id[relation]

        num_entities = len(self.entity_to_id)

        self.model.eval()
        with torch.no_grad():
            # Build batches: (h, r, all possible t)
            h_batch = torch.full(
                (num_entities,),
                h_id,
                dtype=torch.long,
                device=self.device,
            )
            r_batch = torch.full(
                (num_entities,),
                r_id,
                dtype=torch.long,
                device=self.device,
            )
            t_batch = torch.arange(num_entities, device=self.device, dtype=torch.long)

            scores = self.model(h_batch, r_batch, t_batch)  # shape: [num_entities]

            # Higher score = more plausible
            scores = scores.cpu().numpy()

        # Get top-k indices
        top_k = min(top_k, num_entities)
        top_idx = np.argsort(-scores)[:top_k]  # sort descending

        # Map back to entity labels
        results = []
        for idx in top_idx:
            tail_label = self.id_to_entity[int(idx)]
            tail_score = float(scores[idx])
            results.append((tail_label, tail_score))

        return results



    # ------------------------------------------------------------------
    # 6. Minimum spanning tree (on undirected projection)
    # ------------------------------------------------------------------
    def minimum_spanning_tree(self, weight_attr: str = "weight") -> nx.Graph:
        """
        Compute MST on an undirected projection of the graph:
        - collapse parallel edges using the smallest weight.
        - returns a simple undirected Graph.
        """
        undirected = nx.Graph()
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            w = float(data.get(weight_attr, 1.0))
            if undirected.has_edge(u, v):
                if w < undirected[u][v].get(weight_attr, float("inf")):
                    undirected[u][v][weight_attr] = w
            else:
                undirected.add_edge(u, v, **{weight_attr: w})

        mst = nx.minimum_spanning_tree(undirected, weight=weight_attr)
        return mst
