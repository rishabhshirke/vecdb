import numpy as np


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    # Negative so higher score = more similar (consistent with cosine/dot)
    return float(-np.linalg.norm(a - b))


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


METRICS = {
    "cosine": cosine,
    "euclidean": euclidean,
    "dot": dot_product,
}


def get_metric(name: str):
    if name not in METRICS:
        raise ValueError(f"Unknown metric '{name}'. Choose from: {list(METRICS)}")
    return METRICS[name]
