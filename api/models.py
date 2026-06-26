from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class CreateCollectionRequest(BaseModel):
    name: str
    dim: int
    metric: str = "cosine"
    index_type: str = "hnsw"


class InsertRequest(BaseModel):
    id: str
    vector: List[float]
    metadata: Dict[str, Any] = {}


class InsertManyRequest(BaseModel):
    vectors: List[InsertRequest]


class UpdateRequest(BaseModel):
    vector: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None


class SearchRequest(BaseModel):
    vector: List[float]
    k: int = 10
    filters: Optional[Dict[str, Any]] = None
    include_vector: bool = False


class SearchResult(BaseModel):
    id: str
    score: float
    metadata: Dict[str, Any]
    vector: Optional[List[float]] = None
