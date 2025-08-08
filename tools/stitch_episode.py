# aethermind-interpretation/tools/stitch_episode.py
from __future__ import annotations
import os, argparse, tempfile, subprocess, sys, shlex
from memory.db import connect

#
# Why this script exists
# ----------------------
# We take a sequence of event-aligned media items (video + optional audio)
# and produce a single MP4. Historical issues included:
#   - Using the concat demuxer with mismatched codecs => broken PTS, drift.
#   - Copying audio on the final pass => lost/broken audio in the output.
#   - Treating missing external audio as "no audio" even when the video
#     already contained an audio stream => silent output.
#
# Fixes implemented:
#   1) We use the concat FILTER (not demuxer) which recomputes PTS.
#   2) We always re-encode audio on the final pass (AAC/48k/2ch).
#   3) For segments without an explicit audio file, we probe the video for an
#      embedded audio stream and use it if present; otherwise we synthesize
#      silence via anullsrc so the timeline stays consistent (and the output
#      always has an audio track).
#   4) We verify the final file actually contains an audio stream; if not,
#      we exit with a non-zero code so upstream pipelines can catch it.


def _resolve_episode_id(epid_or_prefix: str) -> str:
    con = connect(read_only=True)
    # exact match first
    row = con.execute("SELECT episode_id FROM episodes WHERE episode_id = ?", [epid_or_prefix]).fetchone()
    if row: return row[0]
    # prefix match
    rows = con.execute("SELECT episode_id FROM episodes WHERE episode_id LIKE ? LIMIT 2", [epid_or_prefix + "%"]).fetchall()
    if not rows:
        sys.exit(f"No episode_id matching '{epid_or_prefix}'.")
    if len(rows) > 1:
        sys.exit(f"Prefix '{epid_or_prefix}' is ambiguous: {', '.join(r[0][:12] for r in rows)}")
    return rows[0][0]

def _resolve_path(p: str) -> str:
    if not p: return ""
    if os.path.isabs(p) and os.path.exists(p):
        return p
    root = os.environ.get("AETHERMIND_MEDIA_ROOT", "")
    if root:
        cand = os.path.normpath(os.path.join(root, p))
        if os.path.exists(cand): return cand
        if p.startswith("chunks/"):
            cand2 = os.path.normpath(os.path.join(root, p))
            if os.path.exists(cand2): return cand2
    cand = os.path.abspath(p)
    if os.path.exists(cand): return cand
    return ""  # not found

def _run(cmd):
    print("▶", " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True)

def _has_audio_stream(path: str) -> bool:
    """Return True if ffprobe detects at least one audio stream in the file.

    We intentionally keep this lightweight and resilient. If ffprobe fails for
    any reason, we assume no audio to avoid false-positives.
    """
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=index",
            "-of", "csv=p=0",
            path
        ], stderr=subprocess.STDOUT)
        return bool(out.strip())
    except Exception:
        return False

def _assert_output_has_audio(path: str) -> None:
    """Raise SystemExit if the produced file has no audio stream.

    This is a safety check so we never silently ship a video with no audio
    when the pipeline expects one. It also helps catch regressions quickly.
    """
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=index,codec_name,channels,sample_rate",
            "-of", "json",
            path
        ], stderr=subprocess.STDOUT).decode("utf-8", errors="ignore")
        if '"streams"' not in out or '[]' in out:
            print(f"❌ Output has no audio streams: {path}", file=sys.stderr)
            sys.exit(1)
        else:
            print("✅ Audio stream detected in output.")
    except subprocess.CalledProcessError as e:
        print(f"❌ ffprobe failed on output: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    ap = argparse.ArgumentParser(description="Concat an episode's event videos+audio into one mp4.")
    ap.add_argument("--episode-id", required=True, help="Full ID or unique prefix")
    ap.add_argument("--out", required=True, help="Output mp4 path")
    ap.add_argument("--burn-caption", action="store_true", help="Burn episode caption into the final video")
    args = ap.parse_args()

    epid = _resolve_episode_id(args.episode_id)

    con = connect(read_only=True)
    rows = con.execute("""
        SELECT e.video_path, e.audio_path, ep.caption
        FROM events e
        JOIN episodes ep ON e.episode_id = ep.episode_id
        WHERE e.episode_id = ?
        ORDER BY e.start_ts
    """,[epid]).fetchall()
    if not rows:
        sys.exit("No videos for that episode (no events joined).")

    caption = rows[0][2] or ""

    # Resolve files
    pairs = []
    missing = 0
    for (vpath, apath, _) in rows:
        v = _resolve_path(vpath or "")
        a = _resolve_path(apath or "")
        if not v:
            print(f"⚠️  Missing video file: {vpath}", file=sys.stderr)
            missing += 1
            continue
        pairs.append((v, a))  # a may be "", we’ll handle it
    if not pairs:
        sys.exit("No usable video files after resolving paths. Set AETHERMIND_MEDIA_ROOT.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with tempfile.TemporaryDirectory() as td:

        # Refactored approach
        # -------------------
        # We prepare inputs for the concat filter as pairs of [video][audio] for
        # each segment. For audio, prefer explicit audio_path if present; else use
        # the video's embedded audio if it exists; else synthesize silence.
        input_cmds: list[str] = []     # flat ffmpeg -i args
        filter_inputs: list[str] = []  # e.g. ["[0:v][1:a]", "[2:v][2:a]", ...]

        # Track the current global input index as we append -i arguments.
        in_idx = 0
        for (v, a) in pairs:
            # Always add the video as an input.
            input_cmds.extend(["-i", v])
            v_idx = in_idx
            in_idx += 1

            # Decide the audio source for this segment.
            if a:  # explicit audio file given
                input_cmds.extend(["-i", a])
                a_idx = in_idx
                in_idx += 1
            else:
                if _has_audio_stream(v):
                    # Use the video's own embedded audio (same input index).
                    a_idx = v_idx
                else:
                    # No audio provided and no embedded audio; synthesize silence.
                    # Note: lavfi sources are added as normal inputs and will get
                    # their own input index.
                    input_cmds.extend(["-f", "lavfi", "-i",
                                       "anullsrc=channel_layout=stereo:sample_rate=48000"])
                    a_idx = in_idx
                    in_idx += 1

            # Collect the pair references for concat.
            filter_inputs.append(f"[{v_idx}:v][{a_idx}:a]")

        n_segments = len(pairs)

        # Build the filter_complex graph:
        #   - concat the N segments' V/A into [v][a]
        #   - resample/normalize audio timeline to avoid initial offset
        filter_complex = (
            f"{''.join(filter_inputs)}"
            f"concat=n={n_segments}:v=1:a=1[v][a];"
            f"[a]aresample=async=1:first_pts=0[aout]"
        )
        if args.burn_caption:
            safe_caption = caption.replace(":", r"\:").replace("'", r"\\'")
            drawtext = f"drawtext=text='{safe_caption}':x=20:y=20:fontsize=24:fontcolor=white:box=1:boxcolor=0x00000088"
            filter_complex += f";[v]{drawtext}[vout]"
            map_v = ["-map", "[vout]"]
        else:
            map_v = ["-map", "[v]"]

        # Compose final ffmpeg command. Key points:
        #   - -fflags +genpts helps regenerate presentation timestamps.
        #   - We re-encode video (x264) and audio (AAC 48kHz stereo).
        #   - +faststart moves moov atom to the front for web playback.
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-hide_banner", "-fflags", "+genpts",
            *input_cmds,
            "-filter_complex", filter_complex,
            *map_v, "-map", "[aout]",
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2",
            "-movflags", "+faststart",
            args.out
        ]
        _run(ffmpeg_cmd)

        # Verify the output truly contains an audio stream.
        _assert_output_has_audio(args.out)

    print(f"✅ Wrote {args.out}")

    # Example usage (run from repo root):
    #   python3 -m tools.stitch_episode \
    #     --episode-id b58311d49d10a04a0a6db340 \
    #     --out out/episode_b58311d49d10a04a0a6db340.mp4 \
    #     --burn-caption
if __name__ == "__main__":
    main()