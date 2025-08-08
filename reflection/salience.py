from __future__ import annotations
import argparse, json, math
from typing import Dict, Any, List
import numpy as np
from numpy.linalg import norm
from memory.db import connect

"""
“Was this moment worth keeping?”

We will combine 4 signals:

    Novelty: is this event different from the last few events?
    Action intensity: did the agent do much (mouse/keys)?
    Affect: strong audio mood/music confidence?
    Eventness: your is_event flag.

This will translate to a formula like:

    salience = 0.35*novelty + 0.25*action + 0.25*affect + 0.15*eventness

"""

def _load_events_for_salience() -> List[Dict[str, Any]]:
    con = connect(read_only=False)
    # Ensure column exists
    con.execute("ALTER TABLE events ADD COLUMN IF NOT EXISTS salience DOUBLE")
    # Pull minimal fields + raw for action/affect details
    df = con.execute("""
        SELECT event_id, session_id, start_ts, end_ts, is_event, embeddings_F, raw
        FROM events
        WHERE embeddings_F IS NOT NULL
        ORDER BY session_id, start_ts
    """).fetchdf()
    # Convert rows to dicts
    out = []
    for _, r in df.iterrows():
        e = {
            "event_id": r["event_id"],
            "session_id": r["session_id"],
            "start_ts": float(r["start_ts"]),
            "end_ts": float(r["end_ts"]),
            "is_event": bool(r["is_event"]),
            "F": np.array(r["embeddings_F"], dtype=np.float32),
            "raw": json.loads(r["raw"]),
        }
        out.append(e)
    return out

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = norm(a) + 1e-9
    nb = norm(b) + 1e-9
    return float((a @ b) / (na * nb))

def _action_intensity(raw: Dict[str, Any], start_ts: float, end_ts: float) -> float:
    actions = raw.get("actions", []) or []
    dur = max(end_ts - start_ts, 1e-6)
    rate = len(actions) / dur  # events per second
    # squashing: >1.5/s ~ high
    return max(0.0, min(1.0, rate / 1.5))

def _affect(raw: Dict[str, Any]) -> float:
    aud = (raw.get("annotations") or {}).get("audio_moods") or {}
    conf = float(aud.get("confidence", 0.0) or 0.0)
    is_music = 1.0 if aud.get("is_music", False) else 0.0
    # simple blend; conf is 0..1 already
    return max(0.0, min(1.0, 0.7*conf + 0.3*is_music))

def compute_salience_per_session(events: List[Dict[str, Any]], window:int=5) -> List[Dict[str, Any]]:
    """
    events: sorted by time within a single session.
    novelty = 1 - max cosine with last `window` events (uses F embeddings).
    """
    # Normalize F once
    for e in events:
        f = e["F"]
        e["F"] = f / (norm(f) + 1e-9)

    for i, e in enumerate(events):
        # novelty
        if i == 0:
            novelty = 1.0
        else:
            start = max(0, i - window)
            sims = [ _cosine(e["F"], events[j]["F"]) for j in range(start, i) ]
            novelty = 1.0 - max(sims)  # 0 if identical to recent, 1 if orthogonal

        action = _action_intensity(e["raw"], e["start_ts"], e["end_ts"])
        affect = _affect(e["raw"])
        eventness = 1.0 if e["is_event"] else 0.0

        sal = 0.35*novelty + 0.25*action + 0.25*affect + 0.15*eventness
        e["salience"] = float(max(0.0, min(1.0, sal)))
    return events

def main():
    ap = argparse.ArgumentParser(description="Compute salience per event and store in DuckDB.")
    ap.add_argument("--window", type=int, default=5, help="novelty window size (previous events)")
    args = ap.parse_args()

    rows = _load_events_for_salience()
    # group by session
    by_sess: Dict[str, List[Dict[str, Any]]] = {}
    for e in rows:
        by_sess.setdefault(e["session_id"], []).append(e)

    # compute per session
    updates = []
    for sess, evs in by_sess.items():
        evs_sorted = sorted(evs, key=lambda x: x["start_ts"])
        evs_scored = compute_salience_per_session(evs_sorted, window=args.window)
        for e in evs_scored:
            updates.append((e["salience"], e["event_id"]))

    con = connect(read_only=False)
    con.executemany("UPDATE events SET salience = ? WHERE event_id = ?", updates)
    # quick report
    r = con.execute("SELECT AVG(salience), MIN(salience), MAX(salience) FROM events").fetchone()
    print(f"✅ Salience updated. avg={r[0]:.3f} min={r[1]:.3f} max={r[2]:.3f}")

# python3 -m reflection.salience --window 5
if __name__ == "__main__":
    main()
