import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


class VectorRecord:
    __slots__ = ("id", "vector", "metadata")

    def __init__(self, id: str, vector: np.ndarray, metadata: dict):
        self.id = id
        self.vector = vector
        self.metadata = metadata


class Storage:
    def __init__(self):
        self._records: Dict[str, VectorRecord] = {}

    def insert(self, id: str, vector: np.ndarray, metadata: dict):
        self._records[id] = VectorRecord(id, vector, metadata)

    def get(self, id: str) -> Optional[VectorRecord]:
        return self._records.get(id)

    def delete(self, id: str) -> bool:
        if id in self._records:
            del self._records[id]
            return True
        return False

    def update_vector(self, id: str, vector: np.ndarray):
        if id in self._records:
            self._records[id].vector = vector

    def update_metadata(self, id: str, metadata: dict):
        if id in self._records:
            self._records[id].metadata = metadata

    def exists(self, id: str) -> bool:
        return id in self._records

    def all_ids(self) -> List[str]:
        return list(self._records.keys())

    def count(self) -> int:
        return len(self._records)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._records, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            self._records = pickle.load(f)
