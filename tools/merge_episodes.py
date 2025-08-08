# tools/merge_episodes.py
from __future__ import annotations
import argparse, json, math, os, sys, uuid, sqlite3
import numpy as np
from typing import List, Dict, Any, Tuple
from memory.db import connect  # your existing helper

# ---------- helpers

def _b2f(arr):
    """Convert DuckDB DOUBLE[] array (Python list) to np.ndarray (float32)."""
    return np.array(arr, dtype=np.float32) if arr is not None else None

def _f2b(arr: np.ndarray) -> bytes:
    return np.asarray(arr, dtype=np.float32).tobytes()

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None: return 0.0
    da = np.linalg.norm(a); db = np.linalg.norm(b)
    if da == 0 or db == 0: return 0.0
    return float(a.dot(b) / (da * db))

def jaccard(tags_a: List[str], tags_b: List[str]) -> float:
    sa, sb = set(tags_a or []), set(tags_b or [])
    if not sa and not sb: return 1.0
    inter = len(sa & sb); union = len(sa | sb)
    return inter / union if union else 0.0

def agg_tags(member_tags: List[List[dict]], weights: List[float], topk:int=10) -> List[str]:
    from collections import Counter
    c = Counter()
    for tags, w in zip(member_tags, weights):
        for t in (tags or []):
            if isinstance(t, dict) and "label" in t:
                c[t["label"]] += w
            elif isinstance(t, str):
                c[t] += w
    return [t for t,_ in c.most_common(topk)]

def summarize(scene_type: str, tags: List[str], mood: str, speech_sample: str, motion: str) -> str:
    # Keep it deterministic + cheap; you can swap with LLM later
    parts = []
    if scene_type: parts.append(f"In {scene_type}")
    if mood: parts.append(f"{mood} mood")
    if speech_sample: parts.append(f'heard “{speech_sample}”')
    if motion: parts.append(motion)
    if tags: parts.append("objects: " + ", ".join(tags[:5]))
    return "; ".join(parts) + "."

def tiny_thought(scene_type: str, tags: List[str], mood: str) -> Tuple[str,str,float]:
    # 15–40 chars; mood -> valence guess
    txt = (f"{mood or 'neutral'} {scene_type or 'scene'}").strip()
    txt = txt[:40]
    valence = {'tense':'-', 'sad':'-', 'angry':'-', 'calm':'0', 'neutral':'0',
               'bright':'+', 'happy':'+', 'excited':'+'}.get((mood or '').lower(), '0')
    return txt, valence, 0.6  # constant confidence for now

def safe_strip(val):
    # If val is a string, strip it
    if isinstance(val, str):
        return val.strip()
    # If val is a list, join and strip
    if isinstance(val, list):
        return " ".join(str(x) for x in val).strip()
    # If val is a dict, join values and strip
    if isinstance(val, dict):
        return " ".join(str(x) for x in val.values()).strip()
    return ""

def should_merge(prev: Dict[str,Any], cur: Dict[str,Any]) -> bool:
    # Heuristics from the plan
    if prev['scene_type'] != cur['scene_type']: return False
    if (cur['start'] - prev['end']) > 3.0: return False
    if jaccard(prev['tags'], cur['tags']) < 0.4: return False
    if cosine(prev['F'], cur['F']) < 0.85: return False
    return True

# ---------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-id", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    con = connect(read_only=False)
    # con.row_factory = sqlite3.Row  # <-- REMOVE THIS LINE

    # ---- QUERY EVENTS (edit column names here if yours differ)
    rows = con.execute("""
      SELECT event_id, session_id, start_ts, end_ts, scene_type,
             raw->'annotations'->'tags' AS tags,
             embeddings_F,
             raw->'annotations'->'speech_transcript' AS speech_transcript,
             raw->'annotations'->'audio_moods' AS audio_mood,
             raw->'annotations'->'motion_analysis' AS motion_descriptor
      FROM events
      WHERE session_id = ?
      ORDER BY start_ts ASC
    """, [args.session_id]).fetchall()
    # DuckDB returns tuples; access by index or use .fetchdf() for DataFrame

    if not rows:
        print("No events for that session.")
        return

    evs = []
    for r in rows:
        evs.append({
            "event_id": r[0],
            "start": float(r[2]),
            "end": float(r[3]),
            "scene_type": r[4] or "",
            "tags": json.loads(r[5] or "[]"),
            "F": _b2f(r[6]),
            "mood": safe_strip(json.loads(r[8] or '""')),
            "motion": safe_strip(json.loads(r[9] or '""')),
            "speech": safe_strip(json.loads(r[7] or '""')),
        })

    episodes = []
    cur = None
    members = []

    def flush_episode():
        nonlocal cur, members
        if not members: return
        ep_id = uuid.uuid4().hex[:12]  # or use your own ep-id generator
        start = members[0]["start"]
        end   = members[-1]["end"]
        # weights: simple duration per event
        w = [m["end"] - m["start"] for m in members]
        w = [max(x, 1e-3) for x in w]
        W = sum(w)
        w_norm = [x / W for x in w]

        # aggregate tags
        tags = agg_tags([m["tags"] for m in members], w_norm, topk=12)

        # mean F-embed
        Fs = np.stack([m["F"] for m in members if m["F"] is not None], axis=0) if any(m["F"] is not None for m in members) else None
        Fm = Fs.mean(axis=0) if Fs is not None else None

        # summary + thought (pull a short speech sample if any)
        sample_speech = ""
        for m in members:
            if m["speech"]:
                sample_speech = m["speech"].splitlines()[0][:40]
                break
        mood = next((m["mood"] for m in members if m["mood"]), "")
        motion = next((m["motion"] for m in members if m["motion"]), "")
        summary = summarize(members[0]["scene_type"], tags, mood, sample_speech, motion)
        thought_text, valence, conf = tiny_thought(members[0]["scene_type"], tags, mood)

        episodes.append({
            "episode_id": ep_id,
            "start": start, "end": end,
            "scene_type": members[0]["scene_type"],
            "tags": tags,
            "F": Fm,
            "summary": summary,
            "thought_text": thought_text,
            "valence": valence,
            "confidence": conf,
            "member_ids": [m["event_id"] for m in members],
        })
        # reset
        cur, members = None, []

    for e in evs:
        if cur is None:
            cur = e
            members = [e]
            continue
        if should_merge(cur, e):
            members.append(e)
            cur = e
        else:
            flush_episode()
            cur = e
            members = [e]
    flush_episode()

    print(f"Merged into {len(episodes)} episode(s).")

    if args.dry_run:
        for i, ep in enumerate(episodes, 1):
            print(f"[{i}] {ep['episode_id']} {ep['scene_type']} {ep['start']:.2f}-{ep['end']:.2f}  tags={ep['tags'][:5]}  summary={ep['summary']}")
        return

    # ---- WRITE TO DB
    with con:
        for ep in episodes:
            # Only include columns that exist in your table
            con.execute("""
                INSERT INTO episodes (
                    episode_id, start_ts, end_ts, caption, tags_json, f_embed,
                    summary, thought_text, valence_guess, confidence
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (episode_id) DO UPDATE SET
                    start_ts=EXCLUDED.start_ts,
                    end_ts=EXCLUDED.end_ts,
                    caption=EXCLUDED.caption,
                    tags_json=EXCLUDED.tags_json,
                    f_embed=EXCLUDED.f_embed,
                    summary=EXCLUDED.summary,
                    thought_text=EXCLUDED.thought_text,
                    valence_guess=EXCLUDED.valence_guess,
                    confidence=EXCLUDED.confidence
            """, [
                ep["episode_id"],
                ep["start"],
                ep["end"],
                f"{ep['scene_type']} — {ep['summary']}",  # caption
                json.dumps(ep["tags"]),                   # tags_json
                _f2b(ep["F"]) if ep["F"] is not None else None,  # f_embed
                ep["summary"],
                ep["thought_text"],
                ep["valence"],
                ep["confidence"]
            ])
        for ord_i, ev_id in enumerate(ep["member_ids"]):
            con.execute("""
            INSERT OR REPLACE INTO episode_events (episode_id, event_id, ord)
            VALUES (?, ?, ?)
            """, [ep["episode_id"], ev_id, ord_i])

    print("✅ Episodes written to DB.")

# python3 -m tools.merge_episodes --session-id session_20250805_162657 --dry-run
if __name__ == "__main__":
    main()
