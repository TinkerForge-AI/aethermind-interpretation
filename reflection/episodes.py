from __future__ import annotations
import argparse, json, hashlib
from typing import Dict, Any, List, Tuple
import numpy as np
from numpy.linalg import norm
from memory.db import connect

"""
This module:

Creates episodes table if missing.
Walks events in time order, merging when rules pass; otherwise starts a new episode.
Stores: episode_id, session_id, start_ts, end_ts, scene_type, num_events, salience_mean, 
    tags (top-N labels), summary (tiny template), embeddings_F (mean vector).
Writes events.episode_id for back-reference.
"""

EP_SCHEMA = r"""
CREATE TABLE IF NOT EXISTS episodes (
  episode_id TEXT PRIMARY KEY,
  session_id TEXT,
  start_ts DOUBLE,
  end_ts DOUBLE,
  scene_type TEXT,
  num_events INTEGER,
  salience_mean DOUBLE,
  tags TEXT[],
  summary TEXT,
  embeddings_F DOUBLE[]
);
"""

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = norm(a) + 1e-9
    nb = norm(b) + 1e-9
    return float((a @ b) / (na * nb))

def _tags_from_raw(raw: Dict[str, Any], limit:int=10) -> List[str]:
    ann = raw.get("annotations") or {}
    tags = ann.get("vision_tags", []) or []
    labs = []
    for t in tags[:limit]:
        lab = t.get("label")
        if isinstance(lab, str) and lab:
            # strip any leading "id,mid,label"
            if "," in lab:
                lab = lab.split(",")[-1].strip()
            labs.append(lab.lower().replace(" ", "_"))
    return labs

def _jaccard(a: List[str], b: List[str]) -> float:
    A, B = set(a), set(b)
    if not A and not B: return 1.0
    if not A or not B: return 0.0
    return len(A & B) / max(1, len(A | B))

def _episode_id(session_id: str, start_ts: float, end_ts: float) -> str:
    key = f"{session_id}|{start_ts:.3f}-{end_ts:.3f}"
    return hashlib.blake2b(key.encode(), digest_size=12).hexdigest()

def _load_events_for_merge():
    con = connect(read_only=False)
    # migrations
    con.execute("ALTER TABLE events ADD COLUMN IF NOT EXISTS episode_id TEXT")
    con.execute(EP_SCHEMA)
    df = con.execute("""
        SELECT event_id, session_id, start_ts, end_ts, scene_type, salience, embeddings_F, raw
        FROM events
        WHERE embeddings_F IS NOT NULL
        ORDER BY session_id, start_ts
    """).fetchdf()
    events = []
    for _, r in df.iterrows():
        events.append({
            "event_id": r["event_id"],
            "session_id": r["session_id"],
            "start_ts": float(r["start_ts"]),
            "end_ts": float(r["end_ts"]),
            "scene_type": r["scene_type"] or "unknown",
            "salience": float(r["salience"]) if r["salience"] is not None else 0.0,
            "F": np.array(r["embeddings_F"], dtype=np.float32),
            "raw": json.loads(r["raw"]),
        })
    return events

def _summarize(scene: str, tags: List[str], speech_present: bool, affect_str: str) -> str:
    bits = []
    if scene and scene != "unknown": bits.append(scene.replace("_"," "))
    if tags: bits.append("tags: " + ", ".join(tags[:3]))
    if affect_str: bits.append(affect_str)
    if speech_present: bits.append("speech present")
    return "; ".join(bits) if bits else "episode"

def _affect_str(raw: Dict[str, Any]) -> str:
    aud = (raw.get("annotations") or {}).get("audio_moods") or {}
    mood = (aud.get("mood") or "").lower()
    if mood and mood != "unknown":
        return f"mood: {mood}"
    return ""

def _speech_present(raw: Dict[str, Any]) -> bool:
    s = (raw.get("annotations") or {}).get("speech_transcript") or ""
    return bool(s.strip())

def merge_session(events: List[Dict[str, Any]],
                  gap_max: float = 3.0,
                  cos_min: float = 0.85,
                  jaccard_min: float = 0.40) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]]]:
    """
    Returns:
      episodes: list of episode dicts
      mapping: list of (event_id, episode_id)
    """
    eps = []
    mapping = []
    if not events: return eps, mapping

    # precompute normalized F and tags
    for e in events:
        e["F"] = e["F"] / (norm(e["F"]) + 1e-9)
        e["tags"] = _tags_from_raw(e["raw"])
        e["speech_present"] = _speech_present(e["raw"])
        e["affect_str"] = _affect_str(e["raw"])

    cur = [events[0]]
    for e in events[1:]:
        prev = cur[-1]
        same_scene = (e["scene_type"] == prev["scene_type"])
        small_gap  = (e["start_ts"] - prev["end_ts"]) <= gap_max
        cos_ok     = (_cosine(e["F"], prev["F"]) >= cos_min)
        jac_ok     = (_jaccard(e["tags"], prev["tags"]) >= jaccard_min)

        if same_scene and small_gap and cos_ok and jac_ok:
            cur.append(e)
        else:
            eps.append(_finalize_episode(cur))
            cur = [e]
    eps.append(_finalize_episode(cur))

    # build mapping
    for ep in eps:
        for eid in ep["_members"]:
            mapping.append((eid, ep["episode_id"]))
    # drop internal field
    for ep in eps:
        del ep["_members"]
    return eps, mapping

def _finalize_episode(members: List[Dict[str, Any]]) -> Dict[str, Any]:
    sess = members[0]["session_id"]
    scene = members[0]["scene_type"]
    st = members[0]["start_ts"]
    en = members[-1]["end_ts"]
    eid = _episode_id(sess, st, en)
    F_mean = np.mean([m["F"] for m in members], axis=0).astype(float).tolist()
    sal_mean = float(np.mean([m["salience"] for m in members])) if members else 0.0

    # aggregate tags by frequency
    tag_counts: Dict[str, int] = {}
    speech_present = False
    affect = ""
    for m in members:
        speech_present = speech_present or m["speech_present"]
        if not affect: affect = m["affect_str"]
        for t in m["tags"]:
            tag_counts[t] = tag_counts.get(t, 0) + 1
    top_tags = [t for t, _ in sorted(tag_counts.items(), key=lambda kv: (-kv[1], kv[0]))][:8]

    summary = _summarize(scene, top_tags, speech_present, affect)

    return {
        "episode_id": eid,
        "session_id": sess,
        "start_ts": st,
        "end_ts": en,
        "scene_type": scene,
        "num_events": len(members),
        "salience_mean": sal_mean,
        "tags": top_tags,
        "summary": summary,
        "embeddings_F": F_mean,
        "_members": [m["event_id"] for m in members],
    }

def main():
    ap = argparse.ArgumentParser(description="Merge events into episodes and store in DuckDB.")
    ap.add_argument("--gap", type=float, default=3.0, help="max allowed gap (s) between events")
    ap.add_argument("--cos", type=float, default=0.85, help="min cosine sim between consecutive events")
    ap.add_argument("--jaccard", type=float, default=0.40, help="min Jaccard overlap of tag sets")
    args = ap.parse_args()

    events = _load_events_for_merge()
    # group by session & scene for stability
    by_sess: Dict[str, List[Dict[str, Any]]] = {}
    for e in events:
        by_sess.setdefault(e["session_id"], []).append(e)

    all_eps = []
    mapping = []
    for sess, evs in by_sess.items():
        evs_sorted = sorted(evs, key=lambda x: x["start_ts"])
        eps, map_ = merge_session(evs_sorted, gap_max=args.gap, cos_min=args.cos, jaccard_min=args.jaccard)
        all_eps.extend(eps)
        mapping.extend(map_)

    con = connect(read_only=False)
    # insert episodes
    con.executemany("""
        INSERT OR REPLACE INTO episodes
        (episode_id, session_id, start_ts, end_ts, scene_type, num_events, salience_mean, tags, summary, embeddings_F)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [(
        ep["episode_id"], ep["session_id"], ep["start_ts"], ep["end_ts"], ep["scene_type"],
        ep["num_events"], ep["salience_mean"], ep["tags"], ep["summary"], ep["embeddings_F"]
    ) for ep in all_eps])

    # back-fill events.episode_id
    con.executemany("UPDATE events SET episode_id = ? WHERE event_id = ?", [(epid, eid) for (eid, epid) in mapping])

    print(f"✅ Episodes written: {len(all_eps)}; events mapped: {len(mapping)}")
    # quick peek
    peek = con.execute("SELECT episode_id, session_id, scene_type, num_events, summary FROM episodes ORDER BY start_ts LIMIT 5").fetchall()
    for row in peek:
        print(row)

# python3 -m reflection.episodes --gap 3.0 --cos 0.85 --jaccard 0.40
if __name__ == "__main__":
    main()
