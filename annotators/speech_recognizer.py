# annotators/speech_recognizer.py

import whisper
from pathlib import Path

# Load model once
_model = whisper.load_model("tiny")  # Consider "base" if accuracy is too low

def transcribe_audio(audio_path: Path) -> str | None:
    if not audio_path.exists():
        return None

    try:
        result = _model.transcribe(str(audio_path), fp16=False)
        text = result.get("text", "").strip()
        return text if text else None
    except Exception as e:
        print(f"[speech_recognizer] Error: {e}")
        return None
