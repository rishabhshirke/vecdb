import numpy as np
import math
import random
from typing import Dict, List, Optional, Set, Tuple
from ..distance import get_metric


class HNSWIndex:
    """
    Hierarchical Navigable Small World graph index.

    How it works:
    - Vectors are nodes in a multi-layer graph.
    - Layer 0 holds ALL nodes; each higher layer holds a random subset.
    - Insertion: assign a random max layer, then greedily connect to nearest
      neighbors starting from the top layer down to layer 0.
    - Search: enter at the top, greedily descend toward the query, then do
      a beam search on layer 0 to find the final k results.

    Key params:
      M              — max edges per node per layer (16 is standard)
      ef_construction — beam width during insertion (higher = better recall, slower build)
      ef             — beam width during search (higher = better recall, slower query)
    """

    def __init__(self, metric: str = "cosine", M: int = 16, ef_construction: int = 200, ef: int = 50):
        self.metric_fn = get_metric(metric)
        self.M = M
        self.M0 = 2 * M          # layer 0 gets more connections
        self.ef_construction = ef_construction
        self.ef = ef
        self.ml = 1 / math.log(M) # level normalization factor

        self._vectors: Dict[str, np.ndarray] = {}
        # graphs[layer][node_id] = set of neighbor ids
        self._graphs: List[Dict[str, Set[str]]] = []
        self._entry_point: Optional[str] = None
        self._max_layer: int = -1

    def _score(self, a: str, b: str) -> float:
        return self.metric_fn(self._vectors[a], self._vectors[b])

    def _random_level(self) -> int:
        return int(-math.log(random.random()) * self.ml)

    def add(self, id: str, vector: np.ndarray):
        self._vectors[id] = vector
        level = self._random_level()

        # Extend graph layers if needed
        while len(self._graphs) <= level:
            self._graphs.append({})

        # Initialize the node as a key in every layer it belongs to BEFORE
        # any edge insertion — this prevents KeyError when a neighbor tries
        # to add a back-edge to a node that hasn't been keyed yet.
        for lyr in range(level + 1):
            self._graphs[lyr].setdefault(id, set())

        # First node — becomes entry point
        if self._entry_point is None:
            self._entry_point = id
            self._max_layer = level
            return

        ep = [self._entry_point]

        # From top layer down to level+1: greedy descent (narrow beam, width=1)
        for lyr in range(self._max_layer, level, -1):
            ep = [n for n, _ in self._search_layer(vector, ep, ef=1, layer=lyr)]

        # From min(level, max_layer) down to 0: insert with ef_construction beam
        for lyr in range(min(level, self._max_layer), -1, -1):
            candidates = self._search_layer(vector, ep, ef=self.ef_construction, layer=lyr)
            max_conn = self.M0 if lyr == 0 else self.M
            neighbors = candidates[:max_conn]

            self._graphs[lyr][id] = set(n for n, _ in neighbors)
            for neighbor_id, _ in neighbors:
                # setdefault guards against neighbors that somehow lack a key
                self._graphs[lyr].setdefault(neighbor_id, set()).add(id)
                if len(self._graphs[lyr][neighbor_id]) > max_conn:
                    self._graphs[lyr][neighbor_id] = self._select_neighbors(
                        neighbor_id, self._graphs[lyr][neighbor_id], max_conn, lyr
                    )
            ep = [n for n, _ in candidates]

        if level > self._max_layer:
            self._max_layer = level
            self._entry_point = id

    def _search_layer(self, query: np.ndarray, entry_points: List[str], ef: int, layer: int) -> List[Tuple[str, float]]:
        """Beam search on a single layer. Returns top-ef candidates sorted by score desc."""
        visited: Set[str] = set(entry_points)
        # candidates = min-heap by negative score (we want max score)
        # Use lists and sort for clarity (readability > micro-perf here)
        candidates = [(id, self.metric_fn(query, self._vectors[id])) for id in entry_points]
        dynamic_list = list(candidates)

        while candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            best_id, best_score = candidates.pop(0)

            # Worst in our result list
            worst_score = min(s for _, s in dynamic_list)
            if best_score < worst_score:
                break

            for neighbor in self._graphs[layer].get(best_id, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    score = self.metric_fn(query, self._vectors[neighbor])
                    if len(dynamic_list) < ef or score > worst_score:
                        dynamic_list.append((neighbor, score))
                        candidates.append((neighbor, score))
                        if len(dynamic_list) > ef:
                            dynamic_list.sort(key=lambda x: x[1], reverse=True)
                            dynamic_list = dynamic_list[:ef]

        dynamic_list.sort(key=lambda x: x[1], reverse=True)
        return dynamic_list

    def _select_neighbors(self, node_id: str, candidates: Set[str], max_conn: int, layer: int) -> Set[str]:
        scored = [(n, self._score(node_id, n)) for n in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return set(n for n, _ in scored[:max_conn])

    def remove(self, id: str):
        if id not in self._vectors:
            return
        del self._vectors[id]
        for layer_graph in self._graphs:
            neighbors = layer_graph.pop(id, set())
            for neighbor in neighbors:
                layer_graph.get(neighbor, set()).discard(id)
        # Reset entry point if needed
        if self._entry_point == id:
            self._entry_point = next(iter(self._vectors), None)

    def update(self, id: str, vector: np.ndarray):
        # Rebuild: remove old, re-insert with new vector
        self.remove(id)
        self.add(id, vector)

    def search(self, query: np.ndarray, k: int) -> List[Tuple[str, float]]:
        if not self._vectors or self._entry_point is None:
            return []
        ep = [self._entry_point]
        for lyr in range(self._max_layer, 0, -1):
            results = self._search_layer(query, ep, ef=1, layer=lyr)
            ep = [results[0][0]] if results else ep
        results = self._search_layer(query, ep, ef=max(self.ef, k), layer=0)
        return results[:k]

    def __len__(self):
        return len(self._vectors)
