# pipeline/nightly.py
from __future__ import annotations
import argparse, os, sys, json, glob, tempfile, subprocess, shlex
from typing import List, Optional
from pathlib import Path

# Load .env early (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("[WARN] python-dotenv not installed; .env file will not be loaded.", file=sys.stderr)

MEM_DIR = os.environ.get("AETHERMIND_MEM_DIR")
EMB_ARTIFACT_DIR = os.environ.get("AETHERMIND_EMB_ARTIFACT_DIR")
MEDIA_ROOT = os.environ.get("AETHERMIND_MEDIA_ROOT")

print(f"[DEBUG] MEM_DIR: {MEM_DIR}")
print(f"[DEBUG] EMB_ARTIFACT_DIR: {EMB_ARTIFACT_DIR}")
print(f"[DEBUG] MEDIA_ROOT: {MEDIA_ROOT}")

# ---------------- utils ----------------

def run(cmd: List[str], cwd: Optional[str] = None) -> None:
    env = os.environ.copy()
    print("▶", " ".join(shlex.quote(c) for c in cmd))
    print(f"[DEBUG] run() from: {os.getcwd()}  cwd={cwd}")
    print(f"[DEBUG] ENV AETHERMIND_MEM_DIR={env.get('AETHERMIND_MEM_DIR')}")
    print(f"[DEBUG] ENV AETHERMIND_MEDIA_ROOT={env.get('AETHERMIND_MEDIA_ROOT')}")
    print(f"[DEBUG] ENV AETHERMIND_EMB_ARTIFACT_DIR={env.get('AETHERMIND_EMB_ARTIFACT_DIR')}")
    subprocess.run(cmd, check=True, cwd=cwd, env=env)

def ensure_env():
    if not os.environ.get("AETHERMIND_MEM_DIR"):
        print("ℹ️  AETHERMIND_MEM_DIR not set; using default under memory/store/", file=sys.stderr)
    if not os.environ.get("AETHERMIND_EMB_ARTIFACT_DIR"):
        print("ℹ️  AETHERMIND_EMB_ARTIFACT_DIR not set; using embeddings/artifacts/", file=sys.stderr)
    if not os.environ.get("AETHERMIND_MEDIA_ROOT"):
        print("ℹ️  AETHERMIND_MEDIA_ROOT not set; video tools will rely on relative paths.", file=sys.stderr)

def expand_inputs(patterns: List[str]) -> List[Path]:
    files: List[Path] = []
    for p in patterns:
        files.extend([Path(x) for x in glob.glob(p)])
    files = [f for f in files if f.suffix == ".json"]
    if not files:
        sys.exit("No input JSON files matched. Pass paths to *normalized* event JSONs.")
    return sorted(files)

def concat_events_to_tmp(json_paths: List[Path]) -> Path:
    all_events = []
    for p in json_paths:
        with open(p, "r") as f:
            data = json.load(f)
        if not isinstance(data, list):
            sys.exit(f"{p} is not a list of events.")
        all_events.extend(data)
    td = tempfile.TemporaryDirectory()
    out = Path(td.name) / "all_events.normalized.concat.json"
    with open(out, "w") as f:
        json.dump(all_events, f)
    out._td_ref = td  # keep tempdir alive
    return out

def get_session_ids_from_db() -> List[str]:
    code = r"""
from memory.db import connect
con = connect(True)
rows = con.execute("SELECT DISTINCT session_id FROM events WHERE session_id IS NOT NULL ORDER BY 1").fetchall()
print("\n".join([r[0] for r in rows if r[0]]))
"""
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env=os.environ.copy())
    return [line.strip() for line in res.stdout.splitlines() if line.strip()]

def csv_list(arg: Optional[str]) -> List[str]:
    if not arg: return []
    return [x.strip() for x in arg.split(",") if x.strip()]

# ---------------- modes / defaults ----------------

def apply_mode_defaults(ap: argparse.Namespace) -> None:
    mode = ap.mode or "default"

    def set_if_none(name: str, value):
        if getattr(ap, name) is None:
            setattr(ap, name, value)

    if mode == "recall":
        set_if_none("merge_gap_sec", 5.0)
        set_if_none("merge_cos_min", 0.80)
        set_if_none("merge_jaccard_min", 0.30)
        set_if_none("min_episode_events", 2)
        set_if_none("keep_salience_min", 0.30)
        set_if_none("episode_min_sec", 0.0)
        set_if_none("episode_max_sec", 3600.0)
    elif mode == "precision":
        set_if_none("merge_gap_sec", 2.5)
        set_if_none("merge_cos_min", 0.88)
        set_if_none("merge_jaccard_min", 0.45)
        set_if_none("min_episode_events", 4)
        set_if_none("keep_salience_min", 0.50)
        set_if_none("episode_min_sec", 6.0)
        set_if_none("episode_max_sec", 300.0)
    else:  # default
        set_if_none("merge_gap_sec", 3.0)
        set_if_none("merge_cos_min", 0.85)
        set_if_none("merge_jaccard_min", 0.40)
        set_if_none("min_episode_events", 3)

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(
        description="Nightly pipeline: fit/embed/ingest/index/salience/episodes/captions/journals (+Recall/Precision + Stitch All)"
    )
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="One or more paths/globs to *normalized* event JSON files (Step 0 output).")

    # Artifacts
    ap.add_argument("--refit", action="store_true",
                    help="Refit text/structured/fusion artifacts on the concatenation of all inputs.")
    ap.add_argument("--skip-fit", action="store_true",
                    help="Skip fit even if artifacts are missing (not recommended on first run).")

    # Indexing
    ap.add_argument("--rebuild-index", action="store_true",
                    help="Force rebuild of HNSW index after ingest (default: rebuild).")
    ap.add_argument("--no-index", action="store_true",
                    help="Skip index rebuild (not typical).")

    # Salience
    ap.add_argument("--salience-window", type=int, default=5,
                    help="Temporal window size used by reflection.salience (default: 5).")

    # Episode merging thresholds
    ap.add_argument("--mode", choices=["default", "recall", "precision"], default="default",
                    help="Preset threshold profile. Explicit flags override these defaults.")
    ap.add_argument("--merge-gap-sec", type=float, dest="merge_gap_sec",
                    help="Max gap (seconds) between events for merging.")
    ap.add_argument("--merge-cos-min", type=float, dest="merge_cos_min",
                    help="Min cosine(F) between neighbors to merge.")
    ap.add_argument("--merge-jaccard-min", type=float, dest="merge_jaccard_min",
                    help="Min Jaccard(tags) between neighbors to merge.")

    # Stitching selection & filters
    ap.add_argument("--stitch-all", action="store_true",
                    help="Stitch ALL episodes that pass filters (use with care).")
    ap.add_argument("--stitch-top-k", type=int, default=0,
                    help="If >0, stitch top-K episodes by salience_mean (after filters).")
    ap.add_argument("--stitch-dir", type=str, default="out",
                    help="Directory to write stitched MP4s (default: out).")

    ap.add_argument("--min-episode-events", type=int, dest="min_episode_events",
                    help="Minimum events per episode to be eligible.")
    ap.add_argument("--keep-salience-min", type=float, dest="keep_salience_min",
                    help="Minimum salience_mean for episodes to be eligible.")
    ap.add_argument("--episode-min-sec", type=float, dest="episode_min_sec",
                    help="Minimum episode duration (seconds) for selection.")
    ap.add_argument("--episode-max-sec", type=float, dest="episode_max_sec",
                    help="Maximum episode duration (seconds) for selection.")
    ap.add_argument("--require-speech", action="store_true",
                    help="Only select episodes containing at least one event with non-empty speech transcript.")
    ap.add_argument("--scene-include", type=str,
                    help="Comma-separated list; only select episodes with scene_type in this list.")
    ap.add_argument("--scene-exclude", type=str,
                    help="Comma-separated list; exclude episodes with scene_type in this list.")
    ap.add_argument("--ban-tags", type=str,
                    help='Comma-separated list; exclude episodes whose tags_json contains any of these (substring match).')

    # Reports
    ap.add_argument("--make-journals", action="store_true",
                    help="Generate Markdown journals for all sessions seen in inputs.")
    ap.add_argument("--report", action="store_true",
                    help="Print a tiny summary report after episode generation.")
    args = ap.parse_args()

    # Mode defaults
    apply_mode_defaults(args)

    ensure_env()
    print(f"[DEBUG] main() in: {os.getcwd()}")
    print(f"[DEBUG] MODE: {args.mode}  merge_gap={args.merge_gap_sec}  cos_min={args.merge_cos_min}  jaccard_min={args.merge_jaccard_min}")
    print(f"[DEBUG] SELECTION: min_events={args.min_episode_events}  keep_salience_min={args.keep_salience_min}  "
          f"ep_sec=[{args.episode_min_sec},{args.episode_max_sec}]  require_speech={args.require_speech}")

    in_files = expand_inputs(args.inputs)

    # 0) Optional: refit artifacts
    if args.refit:
        concat_path = concat_events_to_tmp(in_files)
        run([sys.executable, "-m", "embeddings.fit", str(concat_path)])
    else:
        need_artifacts = not Path("embeddings/artifacts/tfidf.joblib").exists()
        if need_artifacts and not args.skip_fit:
            run([sys.executable, "-m", "embeddings.fit", str(in_files[0])])

    # 1) Embed each input → write *.embedded.json next to it
    embedded_paths: List[Path] = []
    for nf in in_files:
        out = nf.with_name(nf.stem.replace(".normalized", "") + ".embedded.json")
        run([sys.executable, "-m", "embeddings.embed", str(nf), "-o", str(out)])
        embedded_paths.append(out)

    # 2) Ingest each embedded file into the DB
    for ef in embedded_paths:
        run([sys.executable, "-m", "memory.ingest", str(ef)])

    # 3) Build (or rebuild) vector index
    if not args.no_index:
        run([sys.executable, "-m", "memory.build_index"])
    elif args.rebuild_index:
        run([sys.executable, "-m", "memory.build_index"])

    # 4) Salience
    run([sys.executable, "-m", "reflection.salience", "--window", str(args.salience_window)])

    # 5) Episodes (pass through the merge thresholds)
    run([
        sys.executable, "-m", "reflection.episodes",
        "--gap", str(args.merge_gap_sec),
        "--cos", str(args.merge_cos_min),
        "--jaccard", str(args.merge_jaccard_min),
    ])
    print("[DEBUG] Episodes completed!")

    # 6) Captions
    run([sys.executable, "-m", "explain.captions"])
    print("✅ Captions completed!")

    # Optional quick report
    if args.report:
        code = r"""
from memory.db import connect
con = connect(True)
print("---- Episodes summary ----")
row = con.execute("SELECT COUNT(*) FROM episodes").fetchone()
print("episodes:", row[0])
print("\nby scene_type (top 10):")
for r in con.execute("SELECT scene_type, COUNT(*) c FROM episodes GROUP BY 1 ORDER BY c DESC LIMIT 10").fetchall():
    print(f"{r[0] or 'unknown':24s} {r[1]}")
print("\nduration histogram (s):")
for r in con.execute(
    "SELECT ROUND(end_ts - start_ts, 0) AS dur, COUNT(*) FROM episodes GROUP BY 1 ORDER BY 1"
).fetchall():
    print(f"{int(r[0]):4d}s  {r[1]}")
"""
        subprocess.run([sys.executable, "-c", code], check=True, env=os.environ.copy())

    # 7) Journals (optional)
    if args.make_journals:
        sess_ids = get_session_ids_from_db()
        for sid in sess_ids:
            out_md = Path("reports") / f"session_{sid}.md"
            out_md.parent.mkdir(parents=True, exist_ok=True)
            run([sys.executable, "-m", "reports.journal", "--session-id", sid, "--out", str(out_md)])

    # 8) Stitch episodes (ALL or TOP-K), applying selection filters first
    if args.stitch_all or args.stitch_top_k > 0:
        include_scenes = csv_list(args.scene_include)
        exclude_scenes = csv_list(args.scene_exclude)
        ban_tags = csv_list(args.ban_tags)

        code = rf"""
from memory.db import connect
con = connect(True)

min_ev = {int(args.min_episode_events)}
keep_sal_min = {args.keep_salience_min if args.keep_salience_min is not None else -1.0}
min_sec = {args.episode_min_sec if args.episode_min_sec is not None else 0.0}
max_sec = {args.episode_max_sec if args.episode_max_sec is not None else 999999.0}
require_speech = {1 if args.require_speech else 0}

include_scenes = {include_scenes!r}
exclude_scenes = {exclude_scenes!r}
ban_tags = {ban_tags!r}

filters = []
filters.append(f"e.num_events >= {{min_ev}}")
filters.append(f"(e.end_ts - e.start_ts) BETWEEN {{min_sec}} AND {{max_sec}}")
if keep_sal_min >= 0:
    filters.append(f"e.salience_mean >= {{keep_sal_min}}")
if include_scenes:
    inc = ",".join("'" + s.replace("'", "''") + "'" for s in include_scenes)
    filters.append(f"e.scene_type IN ({{inc}})")
if exclude_scenes:
    exc = ",".join("'" + s.replace("'", "''") + "'" for s in exclude_scenes)
    filters.append(f"e.scene_type NOT IN ({{exc}})")
if ban_tags:
    for t in ban_tags:
        esc = t.replace("'", "''")
        filters.append(f"e.tags_json NOT LIKE '%\"{{esc}}\"%'")

where_clause = " AND ".join(filters) if filters else "1=1"

base_sql = f'''
SELECT
  e.episode_id,
  e.salience_mean,
  e.num_events,
  e.scene_type,
  (e.end_ts - e.start_ts) AS dur
FROM episodes e
WHERE {{where_clause}}
'''

if require_speech:
    base_sql = f'''
    SELECT * FROM ({{base_sql}})
    WHERE episode_id IN (
      SELECT e2.episode_id
      FROM episode_events ee2
      JOIN events ev2 ON ev2.event_id = ee2.event_id
      JOIN episodes e2 ON e2.episode_id = ee2.episode_id
      GROUP BY e2.episode_id
      HAVING SUM(CASE WHEN COALESCE(json_extract_string(ev2.raw, '$.annotations.speech_transcript'), '') <> '' THEN 1 ELSE 0 END) > 0
    )
    '''

order_sql = " ORDER BY salience_mean DESC, dur DESC"
limit_sql = "" if {1 if args.stitch_all else 0} else " LIMIT {args.stitch_top_k}"

sql = base_sql + order_sql + limit_sql
rows = con.execute(sql).fetchall()
print("\n".join([r[0] for r in rows]))
"""
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env=os.environ.copy())
        ep_ids = [line.strip() for line in res.stdout.splitlines() if line.strip()]

        out_dir = Path(args.stitch_dir); out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[DEBUG] Stitching {len(ep_ids)} episode(s) to: {out_dir.resolve()}")
        for eid in ep_ids:
            out_mp4 = out_dir / f"episode_{eid[:8]}_with_audio.mp4"
            run([
                sys.executable, "-m", "tools.stitch_episode",
                "--episode-id", eid,
                "--out", str(out_mp4),
                "--burn-caption",
            ])

    print("✅ Nightly pipeline complete.")

# Example (Recall mode, stitch ALL):
# python3 -m pipeline.nightly \
#   --mode recall \
#   --inputs "../aethermind-perception/chunks/session_2025*/session_events.normalized.json" \
#   --report --make-journals \
#   --stitch-all --min-episode-events 2 \
#   --keep-salience-min 0.30 --episode-min-sec 0 --episode-max-sec 600

# python3 -m pipeline.nightly \
#   --inputs "../aethermind-perception/chunks/session_2025*/session_events.normalized.json" \
#   --mode recall --stitch-all

if __name__ == "__main__":
    main()
