from __future__ import annotations
import os, argparse, textwrap
from memory.db import connect

def esc(s:str) -> str:
    return s.replace("|","\\|") if isinstance(s,str) else s

def main():
    ap = argparse.ArgumentParser(description="Generate a Markdown Session Journal with human-readable captions.")
    ap.add_argument("--session-id", required=True)
    ap.add_argument("--out", help="Output .md path", default=None)
    ap.add_argument("--with-frames", action="store_true", help="Include keyframe image paths (no extraction here)")
    args = ap.parse_args()

    con = connect(read_only=True)

    # Episodes in order
    eps = con.execute("""
        SELECT episode_id, start_ts, end_ts, scene_type, num_events, caption
        FROM episodes
        WHERE session_id = ?
        ORDER BY start_ts
    """,[args.session_id]).fetchall()

    if not eps:
        raise SystemExit(f"No episodes for session {args.session_id}")

    # Events per episode
    evs = con.execute("""
        SELECT event_id, episode_id, start_ts, end_ts, scene_type, caption_event, text_view, video_path, audio_path
        FROM events
        WHERE session_id = ?
        ORDER BY start_ts
    """,[args.session_id]).fetchall()

    # Group events
    by_ep = {}
    for row in evs:
        by_ep.setdefault(row[1], []).append(row)

    out_path = args.out or f"reports/session_{args.session_id}.md"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w") as f:
        f.write(f"# Session Journal — {args.session_id}\n\n")
        f.write("> Human-readable timeline aligning text, visuals, and audio.\n\n")

        for (epid, st, en, scene, n, cap) in eps:
            f.write(f"## Episode {epid[:8]} — [{st:.3f}, {en:.3f}] — scene: {esc(scene)}\n\n")
            f.write(f"**Summary:** {esc(cap or '')}\n\n")
            f.write("| Time | Event ID | Caption | Scene | Video | Audio |\n")
            f.write("|---:|---|---|---|---|---|\n")
            for (eid, _epid, est, een, escene, ecap, textv, vpath, apath) in by_ep.get(epid, []):
                vid = esc(vpath or "")
                aud = esc(apath or "")
                f.write(f"| [{est:.3f}, {een:.3f}] | `{eid[:8]}` | {esc(ecap or '')} | {esc(escene)} | `{vid}` | `{aud}` |\n")
            f.write("\n")

    print(f"✅ Wrote {out_path}")
    print("Open it in your editor or GitHub to browse the timeline.")

# python3 -m reports.journal --session-id session_20250805_162657
if __name__ == "__main__":
    main()
