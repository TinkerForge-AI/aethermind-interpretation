
# Embedding pipeline for annotated events: produces text, structured, and fused embeddings.
from __future__ import annotations
import os, json, argparse
from typing import Any, Dict, List
import numpy as np
from joblib import load

# Local imports: utilities for event loading, saving, and feature extraction
from .common import ARTIFACT_DIR, load_events, save_events, ensure_event_id
from .build_text_view import event_to_text
from .fit import build_structured_features

def _load_artifacts(artifact_dir: str):
    """
    Load all fitted model artifacts needed for embedding:
      - tfidf: text vectorizer
      - tsvd: SVD for text embedding
      - s_scaler: scaler for structured features
      - s_pca: PCA for structured embedding
      - f_scaler: scaler for fused features
      - f_pca: PCA for fused embedding
    """
    tfidf = load(os.path.join(artifact_dir, "tfidf.joblib"))
    tsvd  = load(os.path.join(artifact_dir, "tsvd_text128.joblib"))
    s_scaler = load(os.path.join(artifact_dir, "s_scaler.joblib"))
    s_pca    = load(os.path.join(artifact_dir, "s_pca64.joblib"))
    f_scaler = load(os.path.join(artifact_dir, "f_scaler.joblib"))
    f_pca    = load(os.path.join(artifact_dir, "f_pca128.joblib"))
    return tfidf, tsvd, s_scaler, s_pca, f_scaler, f_pca

def main():
    """
    Main CLI entry for embedding events.
    Loads fitted models, transforms events into text/structured/fused embeddings, and writes results to JSON.
    Steps:
      1. Load fitted models (TF-IDF, SVD, scalers, PCA) from ARTIFACT_DIR or user-specified dir.
      2. Load annotated events from input JSON.
      3. For each event:
         - Build compact text view and store as 'text_view'
         - Transform to text embedding (T)
         - Build structured feature vector and transform to structured embedding (S)
         - Concatenate T and S, transform to fused embedding (F)
      4. Write all embeddings back to each event under 'embeddings' key.
      5. Save output JSON (default: <input>.embedded.json)
    """
    ap = argparse.ArgumentParser(description="Embed events with T/S/F and write text_view.")
    ap.add_argument("events_json", help="Path to normalized events JSON (with event_id).")
    ap.add_argument("-o", "--output_json", help="Output path. Default: <input>.embedded.json")
    ap.add_argument("--artifacts", help="Path to artifacts dir; default uses embeddings/artifacts",
                    default=os.environ.get("AETHERMIND_EMB_ARTIFACT_DIR", ARTIFACT_DIR))
    args = ap.parse_args()

    # Load fitted models for text, structured, and fusion embeddings
    tfidf, tsvd, s_scaler, s_pca, f_scaler, f_pca = _load_artifacts(args.artifacts)
    events = load_events(args.events_json)

    # --- TEXT VIEW ---
    # Build compact text views for each event and store in 'text_view'
    # TF - IDF = term frequency - inverse doc frequency
    texts = []
    for e in events:
        ensure_event_id(e)  # Validate event has event_id
        tv = event_to_text(e)
        e["text_view"] = tv
        texts.append(tv)

    # --- TEXT EMBEDDING ---
    # Transform text views to TF-IDF + SVD embedding
    X_tfidf = tfidf.transform(texts)           # [N_events, Vocab_size]
    T_emb = tsvd.transform(X_tfidf)            # [N_events, Tdim]

    # --- STRUCTURED EMBEDDING ---
    # Build numeric feature vectors and transform to PCA embedding
    X_struct = np.array([build_structured_features(e) for e in events], dtype=np.float32)
    S_emb = s_pca.transform(s_scaler.transform(X_struct))  # [N_events, Sdim]

    # --- FUSION EMBEDDING ---
    # Concatenate T and S, then transform to fused PCA embedding
    X_fuse = np.concatenate([T_emb, S_emb], axis=1)        # [N_events, Tdim + Sdim]
    F_emb = f_pca.transform(f_scaler.transform(X_fuse))    # [N_events, Fdim]

    # --- WRITE EMBEDDINGS BACK ---
    for i, e in enumerate(events):
        e.setdefault("embeddings", {})
        e["embeddings"]["T"] = T_emb[i].round(6).tolist()  # Text embedding
        e["embeddings"]["S"] = S_emb[i].round(6).tolist()  # Structured embedding
        e["embeddings"]["F"] = F_emb[i].round(6).tolist()  # Fused embedding

    # --- SAVE OUTPUT ---
    out_path = args.output_json or args.events_json.replace(".json", ".embedded.json")
    save_events(out_path, events)
    print(f"✅ Embedded {len(events)} events → {out_path}")
    print(f"   Included fields: text_view, embeddings.T/S/F")

# python3 -m embeddings.embed \
# ../aethermind-perception/aethermind_perception/chunks/session_20250808_220541/session_events.json \
# -o ../aethermind-perception/aethermind_perception/chunks/session_20250808_220541/session_events.embedded.json
if __name__ == "__main__":
    main()
