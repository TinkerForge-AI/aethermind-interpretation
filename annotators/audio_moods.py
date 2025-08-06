# interpretation/audio_moods.py

import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import openl3
from pathlib import Path

# ——————————————————————————————————————————————————————————————————————
# 1) YamNet for broad sound-class inference (incl. Music vs Speech)
# ——————————————————————————————————————————————————————————————————————
_YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"
_YAMNET = None
_CLASS_NAMES = None

def _load_yamnet():
    global _YAMNET, _CLASS_NAMES
    if _YAMNET is None:
        _YAMNET = hub.load(_YAMNET_HANDLE)
        # retrieve the class map file that YamNet provides
        class_map_path = _YAMNET.class_map_path().numpy().decode("utf-8")
        _CLASS_NAMES = open(class_map_path, "r").read().splitlines()

def run_yamnet(y: np.ndarray, sr: int):
    _load_yamnet()
    # YamNet expects a mono float32 waveform at 16 kHz
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000
    scores, embeddings, spectrogram = _YAMNET(y)
    # average the frame-level scores
    mean_scores = tf.reduce_mean(scores, axis=0).numpy()
    top_idxs = np.argsort(mean_scores)[::-1][:3]
    top = [{"label": _CLASS_NAMES[i], "score": float(mean_scores[i])} for i in top_idxs]
    # determine if “music” is present
    is_music = any("music" in t["label"].lower() for t in top)
    return {"top_labels": top, "is_music": is_music}

# ——————————————————————————————————————————————————————————————————————
# 2) OpenL3 for rich embeddings → small mood classifier
# ——————————————————————————————————————————————————————————————————————
# (OpenL3 will download its small CNN + linear model automatically)
openl3_model = None

def run_openl3_embedding(audio_path):
    audio, sr = librosa.load(audio_path, sr=None, mono=False)
    emb, ts = openl3.get_audio_embedding(audio, sr, content_type="music")
    # For now just average to a single vector—downstream you can fit a tiny classifier
    return np.mean(emb, axis=0)

# ——————————————————————————————————————————————————————————————————————
# 3) Main wrapper
# ——————————————————————————————————————————————————————————————————————
def detect_audio_mood(audio_path: Path) -> dict:
    # load file
    y, sr = librosa.load(str(audio_path), sr=None, mono=True)

    # RMS loudness
    rms = float(np.mean(librosa.feature.rms(y=y)))

    # YAMNet inference
    yam = run_yamnet(y, sr)

    # OpenL3 embedding → placeholder “mood” guess via simple stats
    emb = run_openl3_embedding(audio_path)
    # e.g. use the first principal component to guess “tense” vs “calm”
    pc1 = float(np.mean(emb))  
    mood = "tense" if pc1 > 0.0 else "calm"

    return {
        "mood": mood,
        "loudness": rms,
        "is_music": yam["is_music"],
        "confidence": float(np.max([t["score"] for t in yam["top_labels"]])),
        "top_labels": yam["top_labels"],
    }
