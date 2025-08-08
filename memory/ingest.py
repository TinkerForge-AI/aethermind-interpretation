# aethermind-interpretation/memory/ingest.py
from __future__ import annotations
import json, argparse
from typing import Any, Dict
from .db import connect

def _row_from_event(e: Dict[str, Any]) -> Dict[str, Any]:
    ann = e.get("annotations", {}) or {}
    emb = e.get("embeddings", {}) or {}
    return {
        "event_id": e["event_id"],
        "session_id": e.get("session_id"),
        "start_ts": float(e.get("start", 0.0)),
        "end_ts": float(e.get("end", 0.0)),
        "scene_type": ann.get("scene_type", "unknown"),
        "is_event": bool(e.get("is_event", False)),
        "valence": e.get("valence", "unknown"),
        "video_path": e.get("video_path", ""),
        "audio_path": e.get("audio_path", ""),
        "text_view": e.get("text_view", ""),
        "embeddings_T": emb.get("T", None),
        "embeddings_S": emb.get("S", None),
        "embeddings_F": emb.get("F", None),
        "raw": json.dumps(e),
    }

def main():
    ap = argparse.ArgumentParser(description="Ingest embedded events JSON into DuckDB.")
    ap.add_argument("embedded_json", help="Path to session_events.embedded.json")
    args = ap.parse_args()

    events = json.load(open(args.embedded_json, "r"))
    if not isinstance(events, list):
        raise SystemExit("Input must be a list of events with embeddings/text_view.")

    rows = [_row_from_event(e) for e in events]

    con = connect()
    con.executemany("""
        INSERT OR REPLACE INTO events
        (event_id, session_id, start_ts, end_ts, scene_type, is_event, valence,
         video_path, audio_path, text_view, embeddings_T, embeddings_S, embeddings_F, raw)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [tuple(r.values()) for r in rows])

    cnt = con.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    print(f"✅ Ingest complete. events table now has {cnt} rows.")

if __name__ == "__main__":
    main()
