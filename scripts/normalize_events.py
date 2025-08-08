#!/usr/bin/env python3
import json, os, re, hashlib, argparse
from typing import Any, Dict, List, Tuple

SESSION_RE = re.compile(r"chunks/([^/]+)/")

def extract_session_id(video_path: str) -> str:
    """
    Pulls session_id from a path like:
    'chunks/session_20250805_162657/chunk_0000.mp4'
    """
    m = SESSION_RE.search(video_path.replace("\\", "/"))
    if m:
        return m.group(1)
    # Fallback: directory name above the file
    return os.path.basename(os.path.dirname(video_path))

def canon_time(x: float) -> str:
    """
    Canonicalize a timestamp (float seconds) to a string rounded to ms.
    """
    try:
        return f"{float(x):.3f}"
    except Exception:
        return "0.000"

def make_event_uid(session_id: str, video_basename: str, start_s: float, end_s: float) -> str:
    # Human-readable uid (not hashed). Useful for debugging.
    return f"{session_id}|{video_basename}|{canon_time(start_s)}-{canon_time(end_s)}"

def stable_event_id(uid: str) -> str:
    # Short stable ID (12-byte blake2b hex)
    return hashlib.blake2b(uid.encode("utf-8"), digest_size=12).hexdigest()

def ensure_defaults(e: Dict[str, Any]) -> None:
    # Normalize a few fields so downstream code can rely on them.
    e.setdefault("valence", "unknown")
    e.setdefault("source", "perception")
    e.setdefault("annotations", {})
    e["annotations"].setdefault("vision_tags", [])
    e["annotations"].setdefault("audio_moods", {"mood":"unknown","loudness":0.0,"is_music":False,"confidence":0.0,"top_labels":[]})
    e["annotations"].setdefault("motion_analysis", {"movement_type":"unknown","motion_intensity":0.0,"directional_bias":"none"})
    e["annotations"].setdefault("time_of_day", "unknown")
    e["annotations"].setdefault("speech_transcript", "")
    e["annotations"].setdefault("scene_type", "unknown")
    e.setdefault("actions", [])
    e.setdefault("vectors", [])
    # Make sure timestamps are floats
    e["start"] = float(e.get("start", 0.0))
    e["end"]   = float(e.get("end",   e["start"] + 2.0))

def process_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for e in events:
        ensure_defaults(e)
        video_path = e.get("video_path", "")
        session_id = extract_session_id(video_path)
        video_basename = os.path.basename(video_path) or "unknown.mp4"

        uid = make_event_uid(session_id, video_basename, e["start"], e["end"])
        eid = stable_event_id(uid)

        e["session_id"] = session_id
        e["event_uid"]  = uid           # human readable (keep for debugging)
        e["event_id"]   = eid           # stable ID to use everywhere

        out.append(e)
    return out

def main():
    ap = argparse.ArgumentParser(description="Normalize events and add stable IDs.")
    ap.add_argument("input_json", help="Path to events JSON (list of event dicts).")
    ap.add_argument("-o", "--output_json", help="Output path. Default: <input>.normalized.json")
    args = ap.parse_args()

    with open(args.input_json, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise SystemExit("Input JSON must be a list of events.")

    out = process_events(data)

    out_path = args.output_json or args.input_json.replace(".json", ".normalized.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"✅ Wrote {len(out)} events → {out_path}")

if __name__ == "__main__":
    main()
