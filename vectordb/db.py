import numpy as np
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

from .storage import Storage
from .filter import matches
from .index.flat import FlatIndex
from .index.hnsw import HNSWIndex


class Collection:
    def __init__(self, name: str, dim: int, metric: str = "cosine", index_type: str = "hnsw"):
        self.name = name
        self.dim = dim
        self.metric = metric
        self.storage = Storage()

        if index_type == "hnsw":
            self.index = HNSWIndex(metric=metric)
        else:
            self.index = FlatIndex(metric=metric)

    def insert(self, id: str, vector: List[float], metadata: dict = {}):
        vec = self._validate(vector)
        if self.storage.exists(id):
            raise ValueError(f"ID '{id}' already exists. Use update instead.")
        self.storage.insert(id, vec, metadata)
        self.index.add(id, vec)

    def update(self, id: str, vector: Optional[List[float]] = None, metadata: Optional[dict] = None):
        if not self.storage.exists(id):
            raise KeyError(f"ID '{id}' not found.")
        if vector is not None:
            vec = self._validate(vector)
            self.storage.update_vector(id, vec)
            self.index.update(id, vec)
        if metadata is not None:
            self.storage.update_metadata(id, metadata)

    def delete(self, id: str):
        if not self.storage.exists(id):
            raise KeyError(f"ID '{id}' not found.")
        self.storage.delete(id)
        self.index.remove(id)

    def get(self, id: str) -> dict:
        rec = self.storage.get(id)
        if rec is None:
            raise KeyError(f"ID '{id}' not found.")
        return {"id": rec.id, "vector": rec.vector.tolist(), "metadata": rec.metadata}

    def search(
        self,
        query: List[float],
        k: int = 10,
        filters: Optional[dict] = None,
        include_vector: bool = False,
    ) -> List[dict]:
        vec = self._validate(query)

        # Over-fetch when filtering to have enough candidates after filtering
        fetch_k = k * 10 if filters else k
        raw = self.index.search(vec, fetch_k)

        results = []
        for id, score in raw:
            rec = self.storage.get(id)
            if rec is None:
                continue
            if filters and not matches(rec.metadata, filters):
                continue
            entry = {"id": id, "score": score, "metadata": rec.metadata}
            if include_vector:
                entry["vector"] = rec.vector.tolist()
            results.append(entry)
            if len(results) == k:
                break

        return results

    def count(self) -> int:
        return self.storage.count()

    def _validate(self, vector: List[float]) -> np.ndarray:
        vec = np.array(vector, dtype=np.float32)
        if vec.ndim != 1 or len(vec) != self.dim:
            raise ValueError(f"Expected vector of dim {self.dim}, got {len(vec)}")
        return vec


class VectorDB:
    def __init__(self, persist_dir: Optional[str] = None):
        self._collections: Dict[str, Collection] = {}
        self.persist_dir = persist_dir

    def create_collection(self, name: str, dim: int, metric: str = "cosine", index_type: str = "hnsw") -> Collection:
        if name in self._collections:
            raise ValueError(f"Collection '{name}' already exists.")
        col = Collection(name, dim, metric, index_type)
        self._collections[name] = col
        return col

    def get_collection(self, name: str) -> Collection:
        if name not in self._collections:
            raise KeyError(f"Collection '{name}' not found.")
        return self._collections[name]

    def delete_collection(self, name: str):
        if name not in self._collections:
            raise KeyError(f"Collection '{name}' not found.")
        del self._collections[name]

    def list_collections(self) -> List[dict]:
        return [
            {"name": n, "count": c.count(), "dim": c.dim, "metric": c.metric}
            for n, c in self._collections.items()
        ]

    def save(self):
        if not self.persist_dir:
            raise RuntimeError("persist_dir not set.")
        os.makedirs(self.persist_dir, exist_ok=True)
        with open(os.path.join(self.persist_dir, "db.pkl"), "wb") as f:
            pickle.dump(self._collections, f)

    def load(self):
        if not self.persist_dir:
            raise RuntimeError("persist_dir not set.")
        path = os.path.join(self.persist_dir, "db.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No saved DB at {path}")
        with open(path, "rb") as f:
            self._collections = pickle.load(f)
