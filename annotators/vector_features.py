# aethermind-interpretation/annotators/vector_features.py

import numpy as np
from typing import List

def summarize_vectors(vectors: list) -> dict:
    # If input is list of dicts with 'x', extract the 'x' arrays
    if vectors and isinstance(vectors[0], dict) and "x" in vectors[0]:
        vectors = [v["x"] for v in vectors]
    arr = np.array(vectors)
    return {
      "mean":  arr.mean(axis=0).tolist(),
      "std":   arr.std(axis=0).tolist(),
      "max":   arr.max(axis=0).tolist(),
      "min":   arr.min(axis=0).tolist(),
    }

def detect_spikes(vectors: list, thresh: float=1.5) -> list:
    if vectors and isinstance(vectors[0], dict) and "x" in vectors[0]:
        vectors = [v["x"] for v in vectors]
    arr = np.array(vectors)
    std = arr.std(0)
    std[std == 0] = 1.0  # Prevent division by zero
    zs = (arr - arr.mean(0)) / std
    return sorted({i for i, row in enumerate(zs) if abs(row).max() > thresh})