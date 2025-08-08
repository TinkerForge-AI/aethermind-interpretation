from __future__ import annotations
import argparse, json
from typing import Dict, Any, List
from memory.db import connect

def _short_speech(raw: Dict[str, Any], maxlen=80) -> str:
    s = ((raw.get("annotations") or {}).get("speech_transcript") or "").strip()
    if not s: return ""
    s = s.replace("\n", " ")
    return s if len(s) <= maxlen else s[:maxlen-1] + "…"

def _top_tags(raw: Dict[str, Any], k=3) -> List[str]:
    tags = ((raw.get("annotations") or {}).get("vision_tags") or [])[:k]
    out = []
    for t in tags:
        lab = t.get("label")
        if isinstance(lab, str) and lab:
            if "," in lab:
                lab = lab.split(",")[-1].strip()
            out.append(lab.lower())
    return out

def _mood_bits(raw: Dict[str, Any]) -> str:
    aud = (raw.get("annotations") or {}).get("audio_moods") or {}
    mood = (aud.get("mood") or "").lower()
    is_music = aud.get("is_music", False)
    bits = []
    if mood and mood != "unknown": bits.append(mood)
    if is_music: bits.append("music")
    return " ".join(bits)

def _motion_bit(raw: Dict[str, Any]) -> str:
    ma = (raw.get("annotations") or {}).get("motion_analysis") or {}
    mt = (ma.get("movement_type") or "unknown").lower()
    if mt != "unknown": return mt
    return ""

def caption_event(raw: Dict[str, Any]) -> str:
    ann = raw.get("annotations") or {}
    scene = (ann.get("scene_type") or "unknown").replace("_"," ")
    tags = _top_tags(raw, k=3)
    mood = _mood_bits(raw)
    motion = _motion_bit(raw)
    speech = _short_speech(raw)

    parts = []
    if scene != "unknown": parts.append(f"in {scene}")
    if tags: parts.append("see " + ", ".join(tags))
    if mood: parts.append(mood)
    if motion: parts.append(f"motion: {motion}")
    if speech: parts.append(f'speech: "{speech}"')
    return "; ".join(parts) if parts else "moment"

def caption_episode(summary:str, num_events:int) -> str:
    # Use the stored summary as backbone; add length for context.
    if summary:
        return f"{summary} — {num_events} events"
    return f"episode of {num_events} events"

def main():
    ap = argparse.ArgumentParser(description="Write human-readable captions for events and episodes.")
    args = ap.parse_args()

    con = connect(read_only=False)
    # Add columns if missing
    con.execute("ALTER TABLE events   ADD COLUMN IF NOT EXISTS caption_event TEXT")
    con.execute("ALTER TABLE episodes ADD COLUMN IF NOT EXISTS caption TEXT")

    # Events: build from raw JSON
    ev = con.execute("SELECT event_id, raw FROM events").fetchall()
    ev_updates = []
    for eid, raw_json in ev:
        raw = json.loads(raw_json)
        ev_updates.append((caption_event(raw), eid))
    con.executemany("UPDATE events SET caption_event = ? WHERE event_id = ?", ev_updates)

    # Episodes: derive from summary + num_events
    ep = con.execute("SELECT episode_id, summary, num_events FROM episodes").fetchall()
    ep_updates = []
    for epid, summ, n in ep:
        ep_updates.append((caption_episode(summ or "", int(n or 0)), epid))
    con.executemany("UPDATE episodes SET caption = ? WHERE episode_id = ?", ep_updates)

    # Quick peek
    sample = con.execute("SELECT event_id, caption_event FROM events LIMIT 3").fetchall()
    print("✅ Captions written. Sample events:", sample)

# python3 -m explain.captions
if __name__ == "__main__":
    main()
