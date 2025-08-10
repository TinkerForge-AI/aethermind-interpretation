# aethermind-interpretation/embeddings/fit.py

# Fit embedding models for text and structured event features, saving artifacts for downstream use.
from __future__ import annotations
import os, json, argparse
from typing import Any, Dict, List
import numpy as np
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import StandardScaler

from .common import ARTIFACT_DIR, load_events, get_in, bool_to_float, safe_float
from .build_text_view import event_to_text

def build_structured_features(e: Dict[str, Any]) -> List[float]:
    """
    Extracts a fixed-length numeric feature vector from a single event dict.
    Combines vector summary stats, audio/motion features, event scores, and hashed scene/mood labels.
    Pads/truncates vector stats to 9 elements each for consistency.
    Returns a list of floats suitable for structured embedding.
    """
    ann = e.get("annotations", {})
    vsum = ann.get("vector_summary", {}) or {}
    mean = vsum.get("mean", []) or []
    std  = vsum.get("std", [])  or []
    mx   = vsum.get("max", [])  or []
    mn   = vsum.get("min", [])  or []

    # Pad/truncate each vector stat to 9 elements
    mean = (mean + [0.0]*9)[:9]
    std  = (std  + [0.0]*9)[:9]
    mx   = (mx   + [0.0]*9)[:9]
    mn   = (mn   + [0.0]*9)[:9]

    # Scalar features from event and annotations
    raw_motion = safe_float(e.get("raw_motion", 0.0))
    raw_energy = safe_float(e.get("raw_energy", 0.0))
    event_score = safe_float(e.get("event_score", 0.0))
    is_event = 1.0 if e.get("is_event", False) else 0.0

    aud = ann.get("audio_moods", {}) or {}
    loudness = safe_float(aud.get("loudness", 0.0))
    is_music = bool_to_float(aud.get("is_music", False))
    confidence = safe_float(aud.get("confidence", 0.0))

    # Motion analysis features
    motion = ann.get("motion_analysis", {}) or {}
    motion_intensity = safe_float(motion.get("motion_intensity", 0.0))

    # Action rate: number of actions per second
    actions = e.get("actions", []) or []
    dur = max(safe_float(e.get("end", 0.0)) - safe_float(e.get("start", 0.0)), 1e-6)
    action_rate = len(actions) / dur

    # Hash scene/mood labels to stable floats in [0,1)
    scene = (ann.get("scene_type") or "unknown").lower()
    mood = (aud.get("mood") or "unknown").lower()
    scene_h = (hash(scene) % 997) / 997.0
    mood_h  = (hash(mood)  % 997) / 997.0

    feats: List[float] = []
    feats += mean + std + mx + mn             # 36 elements
    feats += [raw_motion, raw_energy]         # 38
    feats += [event_score, is_event]          # 40
    feats += [loudness, is_music, confidence] # 43
    feats += [motion_intensity, action_rate]  # 45
    feats += [scene_h, mood_h]                # 47
    return feats
    return feats


def main():
    """
    Main entry point for fitting text and structured embedding models.
    Loads events, builds text and structured features, fits models, and saves artifacts for downstream use.
    Command-line arguments:
      events_json: Path to normalized events JSON (with event_id)
      --max_text_features: Max TF-IDF vocab size
      --t_dim: Target text embedding dimension
      --s_dim: Target structured embedding dimension
      --f_dim: Target fused embedding dimension
    """
    ap = argparse.ArgumentParser(description="Fit text and structured embedding artifacts.")
    ap.add_argument("events_json", help="Path to normalized events JSON (with event_id).")
    ap.add_argument("--max_text_features", type=int, default=2000)
    ap.add_argument("--t_dim", type=int, default=128)
    ap.add_argument("--s_dim", type=int, default=64)
    ap.add_argument("--f_dim", type=int, default=128)
    args = ap.parse_args()

    events = load_events(args.events_json)

    # ---------- TEXT PIPELINE ----------
    # Build compact text views for each event and fit TF-IDF vectorizer
    texts = [event_to_text(e) for e in events]
    try:
        tfidf = TfidfVectorizer(
            max_features=args.max_text_features,
            ngram_range=(1,2),
            lowercase=True,
            strip_accents="unicode",
            min_df=2,
            max_df=0.9,
            sublinear_tf=True
        )
        X_tfidf = tfidf.fit_transform(texts)
    except ValueError as ve:
        print(f"[WARN] TF-IDF fit failed: {ve}\nTrying fallback min_df=1, max_df=1.0...")
        tfidf = TfidfVectorizer(
            max_features=args.max_text_features,
            ngram_range=(1,2),
            lowercase=True,
            strip_accents="unicode",
            min_df=1,
            max_df=1.0,
            sublinear_tf=True
        )
        X_tfidf = tfidf.fit_transform(texts)

    # Reduce text features to t_dim using TruncatedSVD
    tsvd = TruncatedSVD(n_components=min(args.t_dim, X_tfidf.shape[1]-1) if X_tfidf.shape[1] > 1 else 1, random_state=42)
    T_emb = tsvd.fit_transform(X_tfidf)   # [N_events, t_dim]

    # ---------- STRUCTURED PIPELINE ----------
    # Build numeric feature vectors and fit StandardScaler + PCA
    X_struct = np.array([build_structured_features(e) for e in events], dtype=np.float32)  # [N_events, D_struct]
    s_scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = s_scaler.fit_transform(X_struct)

    s_pca = PCA(n_components=min(args.s_dim, Xs.shape[1]), random_state=42)
    S_emb = s_pca.fit_transform(Xs)       # [N_events, s_dim]

    # ---------- FUSION ----------
    # Concatenate text and structured embeddings, then reduce to f_dim
    X_fuse = np.concatenate([T_emb, S_emb], axis=1)  # [N_events, t_dim + s_dim]
    f_scaler = StandardScaler(with_mean=True, with_std=True)
    Xf = f_scaler.fit_transform(X_fuse)

    # Ensure n_components for PCA is valid: must be <= min(n_samples, n_features)
    n_samples, n_features = Xf.shape
    max_f_dim = min(args.f_dim, n_samples, n_features)
    if max_f_dim < 1:
        raise ValueError(f"Cannot fit PCA: not enough samples/features (samples={n_samples}, features={n_features})")
    f_pca = PCA(n_components=max_f_dim, random_state=42)
    F_emb = f_pca.fit_transform(Xf)       # [N_events, f_dim]

    # ---------- SAVE ARTIFACTS ----------
    # Save all fitted models and metadata for downstream use
    dump(tfidf, os.path.join(ARTIFACT_DIR, "tfidf.joblib"))
    dump(tsvd,  os.path.join(ARTIFACT_DIR, "tsvd_text128.joblib"))
    dump(s_scaler, os.path.join(ARTIFACT_DIR, "s_scaler.joblib"))
    dump(s_pca,    os.path.join(ARTIFACT_DIR, "s_pca64.joblib"))
    dump(f_scaler, os.path.join(ARTIFACT_DIR, "f_scaler.joblib"))
    dump(f_pca,    os.path.join(ARTIFACT_DIR, "f_pca128.joblib"))

    meta = {
        "t_dim": T_emb.shape[1],
        "s_dim": S_emb.shape[1],
        "f_dim": F_emb.shape[1],
        "max_text_features": args.max_text_features
    }
    with open(os.path.join(ARTIFACT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Fit complete. Artifacts written to {ARTIFACT_DIR}")
    print(f"   dims: T={meta['t_dim']} S={meta['s_dim']} F={meta['f_dim']} (may be < targets if data small)")

# python3 -m embeddings.fit \
# ../aethermind-perception/aethermind_perception/chunks/session_20250808_220541/session_events.normalized.json  \
# --t_dim 64 --s_dim 32 --f_dim 64
if __name__ == "__main__":
    main()
