from annotators.vision_tags import tag_visual_scene
from annotators.audio_moods import analyze_audio_mood
from annotators.motion_analysis import describe_motion

def interpret_event(event):
    annotations = {}

    annotations.update(tag_visual_scene(event["video_path"]))
    annotations.update(analyze_audio_mood(event["audio_path"]))
    annotations.update(describe_motion(event["raw_motion"]))

    event["annotations"] = annotations
    event["valence"] = annotations.get("valence", "unknown")
    return event
