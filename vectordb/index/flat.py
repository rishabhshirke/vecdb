import numpy as np
from typing import List, Tuple
from ..distance import get_metric


class FlatIndex:
    """Exact brute-force search. Compares query against every stored vector."""

    def __init__(self, metric: str = "cosine"):
        self.metric_fn = get_metric(metric)
        self._ids: List[str] = []
        self._vectors: List[np.ndarray] = []

    def add(self, id: str, vector: np.ndarray):
        self._ids.append(id)
        self._vectors.append(vector)

    def remove(self, id: str):
        if id in self._ids:
            idx = self._ids.index(id)
            self._ids.pop(idx)
            self._vectors.pop(idx)

    def update(self, id: str, vector: np.ndarray):
        if id in self._ids:
            idx = self._ids.index(id)
            self._vectors[idx] = vector

    def search(self, query: np.ndarray, k: int) -> List[Tuple[str, float]]:
        if not self._ids:
            return []
        scores = [(self._ids[i], self.metric_fn(query, v)) for i, v in enumerate(self._vectors)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def __len__(self):
        return len(self._ids)
