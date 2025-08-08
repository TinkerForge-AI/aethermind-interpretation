# aethermind-interpretation/embeddings/build_text_view.py

# Utilities for building compact, human-readable text views of annotated events.
from __future__ import annotations
from typing import Dict, Any, List
from .common import get_in

def _join_labels(items: List[Dict[str, Any]], key="label", limit=10) -> str:
    """
    Extracts and joins up to `limit` labels from a list of dicts.
    Cleans up labels by removing numeric/category prefixes and replacing underscores.
    Used to summarize tag lists (e.g., vision or audio labels).
    Example input: [{"label": "131,/m/032n05,Whale_vocalization"}, ...]
    Output: "Whale vocalization, ..."
    """
    out = []
    for it in items[:limit]:
        lab = it.get(key)
        if isinstance(lab, str) and lab:
            # Remove leading numeric/category prefixes like "131,/m/032n05,Whale vocalization"
            if "," in lab:
                lab = lab.split(",")[-1].strip()
            out.append(lab.replace("_", " "))
    return ", ".join(out)

def event_to_text(e: Dict[str, Any]) -> str:
    """
    Build a compact, human-readable summary string for an annotated event.
    This is used for cheap, stable text views (e.g., for embedding or debugging).
    Extracts key annotation fields: vision tags, audio mood, music flag, scene type, top audio labels, and speech transcript.
    Speech is truncated to 120 chars to avoid exploding vocab size.
    Example output:
      "tags: whale vocalization ; mood: happy ; is_music: True ; scene: ocean ; audio: whale ; speech: Hello world..."
    """
    ann = e.get("annotations", {})  # All annotators write their results here
    tags = _join_labels(ann.get("vision_tags", []))  # Vision tag summary
    mood = get_in(e, ["annotations", "audio_moods", "mood"], "unknown")  # Audio mood label
    is_music = get_in(e, ["annotations", "audio_moods", "is_music"], False)  # Is music detected
    scene = ann.get("scene_type", "unknown")  # Scene classification
    speech = ann.get("speech_transcript", "") or ""  # Speech transcript
    top_aud = _join_labels(get_in(e, ["annotations", "audio_moods", "top_labels"], []))  # Top audio labels

    parts = []
    if tags:
        parts.append(f"tags: {tags}")
    parts.append(f"mood: {mood}")
    parts.append(f"is_music: {bool(is_music)}")
    parts.append(f"scene: {scene}")
    if top_aud:
        parts.append(f"audio: {top_aud}")
    if speech.strip():
        # Truncate speech to avoid very long outputs
        s = speech.strip().replace("\n", " ")
        if len(s) > 120:
            s = s[:117] + "..."
        parts.append(f"speech: {s}")

    # Join all non-empty parts with semicolons
    return " ; ".join(parts)
