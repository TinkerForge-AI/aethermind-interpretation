import subprocess
import os
import sys

def run(cmd, check=True):
    print(f"[PIPELINE] Running: {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, check=check)
    if result.returncode != 0:
        print(f"[ERROR] Step failed: {' '.join(str(c) for c in cmd)}", file=sys.stderr)
        sys.exit(result.returncode)

def main(session_folder):
    # Paths
    raw_events = os.path.join(session_folder, "session_events.json")
    normalized_events = os.path.join(session_folder, "session_events.normalized.json")
    embedded_events = os.path.join(session_folder, "session_events.embedded.json")

    # 1. Normalize events
    run([
        "python3", "scripts/normalize_events.py",
        raw_events,
        "-o", normalized_events
    ])

    # 2. Annotate events (first pass: all annotators except vector features)
    run([
        "python3", "run_interpretation.py",
        normalized_events
    ])

    # 3. Annotate events (second pass: only vector features)
    run([
        "python3", "run_interpretation.py",
        normalized_events,
        "--only_vector_features"
    ])

    # ^^^ turn off these top 3 if you just need to write things to DuckDB!

    # 4. Embed events
    run([
        "python3", "-m", "embeddings.embed",
        normalized_events,
        "-o", embedded_events
    ])

    # 5. Ingest embedded events into DuckDB
    run([
        "python3", "-m", "memory.ingest",
        embedded_events
    ])

    # 6. Run nightly pipeline (episodes, journals, stitching)
    run([
        "python3", "-m", "pipeline.nightly",
        "--inputs", normalized_events,
        "--make-journals",
        "--stitch-all",
        "--min-episode-events", "2",
        "--keep-salience-min", "0.30",
        "--episode-min-sec", "0",
        "--episode-max-sec", "600"
    ])

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 run_pipeline.py <session_folder>")
        sys.exit(1)
    main(sys.argv[1])