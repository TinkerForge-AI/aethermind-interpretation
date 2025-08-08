# Common utilities for working with event data and embeddings in Aethermind.
from __future__ import annotations
import json, os
from typing import Any, Dict, List
import numpy as np

# Directory for storing artifacts (e.g., intermediate files, cached embeddings)
ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def load_events(path: str) -> List[Dict[str, Any]]:
    """
    Load a list of event dicts from a JSON file.
    Raises ValueError if the file does not contain a list.
    """
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of events.")
    return data

def save_events(path: str, events: List[Dict[str, Any]]) -> None:
    """
    Save a list of event dicts to a JSON file, pretty-printed.
    """
    with open(path, "w") as f:
        json.dump(events, f, indent=2)

def get_in(d: Dict, path: List[str], default=None):
    """
    Safely traverse a nested dict using a list of keys.
    Returns the value at the path, or default if any key is missing.
    Example: get_in(event, ["annotations", "vision", "tags"])
    """
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def bool_to_float(x) -> float:
    """
    Convert a boolean or truthy value to float (1.0 or 0.0).
    Used for embedding binary features.
    """
    return 1.0 if bool(x) else 0.0

def safe_float(x, default=0.0) -> float:
    """
    Convert x to float, returning default if conversion fails.
    Useful for robust feature extraction from noisy data.
    """
    try:
        return float(x)
    except Exception:
        return float(default)

def ensure_event_id(e: Dict[str, Any]) -> None:
    """
    Assert that the event dict has an 'event_id' key.
    Raises ValueError if missing. Used to enforce pipeline integrity.
    """
    if "event_id" not in e:
        raise ValueError("Expected 'event_id' from Step 0. Run normalize first.")

def listify(x, length=None, fill=0.0):
    """
    Ensure x is a list. If x is None, returns a list of 'fill' values of given length.
    If x is a list and length is specified, pads or truncates to that length.
    Used for feature vector normalization.
    """
    if x is None:
        return [fill] * (length or 0)
    if isinstance(x, list):
        if length is not None:
            if len(x) >= length:
                return x[:length]
            return x + [fill] * (length - len(x))
        return x
    return [x]
