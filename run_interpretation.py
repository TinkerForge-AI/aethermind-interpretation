# run_interpretation.py

# RUN NORMALIZE EVENTS FIRST!
# python3 scripts/normalize_events.py \
#   ../aethermind-perception/aethermind_perception/chunks/ \
#   session_20250808_220541/session_events.json \
#   -o ../aethermind-perception/aethermind_perception/chunks/ \
#   session_20250808_220541/session_events.json

# test file is in: 
# ../aethermind-perception/sessions/session_20250808_220541/session_events.json

# real TEST!
# ../aethermind-perception/sessions/session_20250808_220541/session_events.json

import os
import re
import json
from pathlib import Path

from annotators.vision_tags import tag_visual_scene, log_unseen_tags
from annotators.audio_moods import detect_audio_mood
from annotators.motion_analysis import analyze_motion
from annotators.time_of_day import estimate_time_of_day
from annotators.speech_recognizer import transcribe_audio
from annotators.scene_classifier import estimate_scene
from annotators.vector_features import summarize_vectors, detect_spikes

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

def load_events(path):
    with open(path, 'r') as f:
        return json.load(f)


def save_events(events, path):
    with open(path, 'w') as f:
        json.dump(events, f, indent=2)

def profanity_filter(text):
    env_profanities = os.environ.get("PROFANITIES")
    if env_profanities:
        profanities = [w.strip() for w in env_profanities.split(",") if w.strip()]
    else:
        profanities = [
            "badword", "anotherbadword", "yetanotherbadword"
        ]
    def repl(match):
        return "*" * len(match.group(0))
    for word in profanities:
        pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
        text = pattern.sub(repl, text)
    return text

def annotate_events(event_file, only_vector_features=False):
    events = load_events(event_file)

    for idx, event in enumerate(events):
        video_path = Path(MEDIA_ROOT) / event["video_path"]
        audio_path = Path(MEDIA_ROOT) / event["audio_path"]
        print(f"\n[Info] Processing event {idx+1}/{len(events)}: {video_path}")
        if not video_path.exists():
            print(f"[Warn] Skipping missing video: {video_path}")
            continue

        try:
            print("  [Step] Running vision tag annotator...")
            vision_tags = tag_visual_scene(video_path)
            log_unseen_tags(vision_tags)
            unique = {}
            for t in vision_tags:
                if t["label"] not in unique or unique[t["label"]] < t["score"]:
                    unique[t["label"]] = t["score"]
            cleaned = [{"label": l, "score": s} for l, s in unique.items()]
            # sort by score desc
            cleaned.sort(key=lambda x: -x["score"])
            event.setdefault("annotations", {})["vision_tags"] = cleaned[:5]
            print("    [Done] Vision tags annotated.")

            print("  [Step] Running audio moods annotator...")
            audio_moods = detect_audio_mood(audio_path)
            event["annotations"]["audio_moods"] = audio_moods
            print("    [Done] Audio moods annotated.")

            print("  [Step] Running motion analysis annotator...")
            motion = analyze_motion(video_path)
            event["annotations"]["motion_analysis"] = motion
            print("    [Done] Motion analysis annotated.")

            print("  [Step] Running time of day annotator...")
            time_of_day = estimate_time_of_day(video_path)
            event["annotations"]["time_of_day"] = time_of_day
            print("    [Done] Time of day annotated.")

            print("  [Step] Running speech recognizer annotator...")
            transcript = transcribe_audio(audio_path)
            # filtered_transcript = profanity_filter(transcript)
            event["annotations"]["speech_transcript"] = transcript # filtered_transcript
            print("    [Done] Speech transcript annotated (profanity filtered).")

            print("  [Step] Running scene classifier annotator...")
            scene = estimate_scene(video_path)
            event["annotations"]["scene_type"] = scene
            print("    [Done] Scene type annotated.")

            # Vector features summarizer (always runs if only_vector_features is True)
            print("  [Step] Running vector features summarizer...")
            raw_vectors = event.get("vectors", [])
            event.setdefault("annotations", {})["vector_summary"] = summarize_vectors(raw_vectors)
            event["annotations"]["vector_spikes"]  = detect_spikes(raw_vectors)
            print("    [Done] Vector features summarized.")

        except Exception as e:
            print(f"[Error] Failed to annotate {video_path}: {e}")

    save_events(events, event_file)
    print(f"[Done] Annotated {len(events)} events.")


if __name__ == "__main__":

    import argparse
    import subprocess
    parser = argparse.ArgumentParser()
    parser.add_argument("event_file", type=str, help="Path to the JSON file containing detected events.")
    parser.add_argument("--embed_out", type=str, default=None, help="Path to save embedded events JSON (optional)")
    parser.add_argument("--only_vector_features", action="store_true", help="Run only the vector features annotator.")
    args = parser.parse_args()

    annotate_events(args.event_file, only_vector_features=args.only_vector_features)

    # Automatically run embeddings.embed after annotation
    # python3 run_interpretation.py <event_file.json> [--embed_out <output_file.json>]
    embed_out = args.embed_out or (os.path.splitext(args.event_file)[0] + ".embedded.json")
    print(f"[Info] Running embeddings.embed on {args.event_file} -> {embed_out}")
    result = subprocess.run([
        "python3", "-m", "embeddings.embed", args.event_file, "-o", embed_out
    ], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"[Error] embeddings.embed failed: {result.stderr}")
    else:
        print(f"[Done] Embedded events written to {embed_out}")
