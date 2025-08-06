
# run_interpretation.py

# test file is in: 
# ../aethermind-perception/sessions/session_20250801_135701/session_events.json

# real TEST!
# ../aethermind-perception/sessions/session_20250801_170832/session_events.json

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

def annotate_events(event_file):
    events = load_events(event_file)

    for idx, event in enumerate(events):
        video_path = Path("../aethermind-perception") / event["video_path"]
        audio_path = Path("../aethermind-perception") / event["audio_path"]
        print(f"\n[Info] Processing event {idx+1}/{len(events)}: {video_path}")
        if not video_path.exists():
            print(f"[Warn] Skipping missing video: {video_path}")
            continue

        try:
            # print("  [Step] Running vision tag annotator...")
            # vision_tags = tag_visual_scene(video_path)
            # log_unseen_tags(vision_tags)
            # unique = {}
            # for t in vision_tags:
            #     if t["label"] not in unique or unique[t["label"]] < t["score"]:
            #         unique[t["label"]] = t["score"]
            # cleaned = [{"label": l, "score": s} for l, s in unique.items()]
            # # sort by score desc
            # cleaned.sort(key=lambda x: -x["score"])
            # event["annotations"]["vision_tags"] = cleaned[:5]
            # print("    [Done] Vision tags annotated.")

            print("  [Step] Running audio moods annotator...")
            audio_moods = detect_audio_mood(audio_path)
            event["annotations"]["audio_moods"] = audio_moods
            print("    [Done] Audio moods annotated.")

            # print("  [Step] Running motion analysis annotator...")
            # motion = analyze_motion(video_path)
            # event["annotations"]["motion_analysis"] = motion
            # print("    [Done] Motion analysis annotated.")

            # print("  [Step] Running time of day annotator...")
            # time_of_day = estimate_time_of_day(video_path)
            # event["annotations"]["time_of_day"] = time_of_day
            # print("    [Done] Time of day annotated.")

            # print("  [Step] Running speech recognizer annotator...")
            # transcript = transcribe_audio(audio_path)
            # filtered_transcript = profanity_filter(transcript)
            # event["annotations"]["speech_transcript"] = filtered_transcript
            # print("    [Done] Speech transcript annotated (profanity filtered).")

            # print("  [Step] Running scene classifier annotator...")
            # scene = estimate_scene(video_path)
            # event["annotations"]["scene_type"] = scene
            # print("    [Done] Scene type annotated.")

            # print("  [Step] Running vector features summarizer...")
            # raw_vectors = event.get("vectors", [])
            # summary = summarize_vectors(raw_vectors)
            # spikes   = detect_spikes(raw_vectors)
            # event["annotations"]["vector_summary"] = summary
            # event["annotations"]["vector_spikes"]  = spikes
            # print("    [Done] Vector features summarized.")

        except Exception as e:
            print(f"[Error] Failed to annotate {video_path}: {e}")

    save_events(events, event_file)
    print(f"[Done] Annotated {len(events)} events.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("event_file", type=str, help="Path to the JSON file containing detected events.")
    args = parser.parse_args()

    annotate_events(args.event_file)
