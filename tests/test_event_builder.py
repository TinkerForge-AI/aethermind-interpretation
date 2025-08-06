import pytest
from hsm.event_builder import make_core_event
from hsm.core_event import CoreEvent, Tag, Motion, AudioMood, TextOverlay, BackgroundNoise, Association


def test_minimal_core_event():
    raw = {
        "id": "evt1",
        "time_span": {"start": 0.0, "end": 1.0},
    }
    evt = make_core_event(raw)
    assert isinstance(evt, CoreEvent)
    assert evt.id == "evt1"
    assert evt.source == "perception"
    assert evt.time_span == {"start": 0.0, "end": 1.0}
    # defaults
    assert evt.confidence == 0.0
    assert evt.vision_tags == []
    assert evt.audio_tags == []
    assert evt.speech_transcript is None
    assert evt.scene_type is None
    assert evt.motion is None
    assert evt.audio_mood is None
    assert evt.text_overlays == []
    assert evt.background_noise == []
    assert len(evt.associations) == 5
    for assoc in evt.associations:
        assert assoc.rough_confidence == 0.5
        # pointers should be None
        assert assoc.vision_ref is None
        assert assoc.audio_ref is None


def test_full_core_event():
    raw = {
        "id": "evt2",
        "time_span": {"start": 1.0, "end": 2.0},
        "confidence": 0.8,
        "vision_tags": [{"label": "cat", "score": 0.9}],
        "audio_tags": [{"label": "meow", "score": 0.7}],
        "speech_transcript": "hello world",
        "scene_type": "indoor",
        "motion": {"raw_motion": 0.1, "raw_energy": 0.2, "motion_trend": "increasing"},
        "audio_mood": {"mood": "happy", "loudness": 0.5, "is_music": False, "pitch": "mid"},
        "text_overlays": [{"text": "Hi", "bounding_box": [0, 0, 10, 10], "confidence": 0.95}],
        "background_noise": [{"type": "wind", "confidence": 0.4}],
    }
    evt = make_core_event(raw)
    assert evt.id == "evt2"
    assert evt.confidence == 0.8
    assert isinstance(evt.vision_tags[0], Tag)
    assert evt.vision_tags[0].label == "cat"
    assert isinstance(evt.motion, Motion)
    assert evt.motion.motion_trend == "increasing"
    assert isinstance(evt.audio_mood, AudioMood)
    assert evt.audio_mood.mood == "happy"
    assert isinstance(evt.text_overlays[0], TextOverlay)
    assert evt.text_overlays[0].confidence == 0.95
    assert isinstance(evt.background_noise[0], BackgroundNoise)


def test_missing_time_span_uses_defaults():
    raw = {"id": "evt3"}
    evt = make_core_event(raw)
    assert evt.time_span == {"start": 0.0, "end": 0.0}
