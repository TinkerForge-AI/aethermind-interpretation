# aethermind-interpretation/memory/search.py
from __future__ import annotations
import os, json, argparse
from typing import List, Tuple, Optional
import numpy as np
import hnswlib
from joblib import load
from .db import connect, MEM_DIR

INDEX_PATH = os.path.join(MEM_DIR, "f_index.hnsw")
IDMAP_PATH = os.path.join(MEM_DIR, "idmap.json")
ART_DIR = os.environ.get("AETHERMIND_EMB_ARTIFACT_DIR",
                         os.path.join(os.path.dirname(__file__), "..", "embeddings", "artifacts"))

def _load_index():
    with open(IDMAP_PATH, "r") as f:
        mapping = json.load(f)
    ids = mapping["ids"]
    con = connect(read_only=True)
    dim = con.execute("""
        SELECT len(embeddings_F)
        FROM events
        WHERE embeddings_F IS NOT NULL
        LIMIT 1
    """).fetchone()[0]
    index = hnswlib.Index(space='cosine', dim=dim)
    index.load_index(INDEX_PATH)
    index.set_ef(200)
    return index, ids

def _get_event_vector(event_id: str) -> np.ndarray:
    con = connect(read_only=True)
    row = con.execute("SELECT embeddings_F FROM events WHERE event_id = ?", [event_id]).fetchone()
    if not row or row[0] is None:
        raise SystemExit(f"No embeddings_F for event_id={event_id}")
    v = np.array(row[0], dtype=np.float32)
    v = v / (np.linalg.norm(v) + 1e-9)
    return v

def _text_to_F(text: str) -> np.ndarray:
    tfidf = load(os.path.join(ART_DIR, "tfidf.joblib"))
    tsvd  = load(os.path.join(ART_DIR, "tsvd_text128.joblib"))
    f_scaler = load(os.path.join(ART_DIR, "f_scaler.joblib"))
    f_pca    = load(os.path.join(ART_DIR, "f_pca128.joblib"))
    T = tsvd.transform(tfidf.transform([text]))
    f_in_dim = f_scaler.mean_.shape[0]
    s_dim = f_in_dim - T.shape[1]
    if s_dim < 0:
        raise SystemExit("Artifact mismatch: T dim larger than fusion input.")
    S = np.zeros((1, s_dim), dtype=np.float32)
    Xf = np.concatenate([T, S], axis=1)
    F = f_pca.transform(f_scaler.transform(Xf)).astype(np.float32)
    F = F / (np.linalg.norm(F) + 1e-9)
    return F[0]

def _apply_filters(candidates: List[Tuple[int, float]], ids: List[str],
                   scene: Optional[str], session: Optional[str],
                   t_start: Optional[float], t_end: Optional[float],
                   k: int):
    con = connect(read_only=True)
    kept = []
    for lab, sim in candidates:
        eid = ids[lab]
        conds, params = ["event_id = ?"], [eid]
        if scene:
            conds.append("lower(scene_type) = ?")
            params.append(scene.lower())
        if session:
            conds.append("session_id = ?")
            params.append(session)
        if t_start is not None:
            conds.append("start_ts >= ?")
            params.append(float(t_start))
        if t_end is not None:
            conds.append("end_ts <= ?")
            params.append(float(t_end))
        sql = "SELECT event_id, scene_type, session_id, start_ts, end_ts FROM events WHERE " + " AND ".join(conds)
        row = con.execute(sql, params).fetchone()
        if row:
            kept.append((eid, sim, row[1], row[2], row[3], row[4]))
        if len(kept) >= k:
            break
    return kept

def main():
    ap = argparse.ArgumentParser(description="Search similar events via HNSW index on F embeddings.")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--event-id", help="Query by existing event_id")
    mode.add_argument("--text", help="Query by free text (token-style works best; e.g., 'scene_catacomb tag_vendor')")
    ap.add_argument("--scene", help="Filter: scene_type (exact match)")
    ap.add_argument("--session-id", help="Filter: session_id")
    ap.add_argument("--start", type=float, help="Filter: start_ts >= (epoch seconds)")
    ap.add_argument("--end", type=float, help="Filter: end_ts <= (epoch seconds)")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--probe", type=int, default=100, help="ANN candidate pool before filters")
    args = ap.parse_args()

    index, ids = _load_index()
    q = _get_event_vector(args.event_id) if args.event_id else _text_to_F(args.text)

    labels, dists = index.knn_query(q, k=max(args.k*5, args.probe))
    labs = labels[0].tolist()
    sims = (1.0 - dists[0]).tolist()

    candidates = list(zip(labs, sims))
    results = _apply_filters(candidates, ids, args.scene, args.session_id, args.start, args.end, args.k)

    for eid, sim, scene, sess, st, en in results:
        print(f"{sim:.4f}  {eid}  scene={scene}  session={sess}  [{st:.3f},{en:.3f}]")

if __name__ == "__main__":
    main()
