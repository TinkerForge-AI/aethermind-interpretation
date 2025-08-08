
# Build or rebuild a fast nearest-neighbor index (HNSW) for fused event embeddings.
from __future__ import annotations
import os, json, argparse
import numpy as np
import hnswlib
from .db import connect, MEM_DIR

# Paths for index and id mapping
INDEX_PATH = os.path.join(MEM_DIR, "f_index.hnsw")
IDMAP_PATH = os.path.join(MEM_DIR, "idmap.json")

def main():
    """
    CLI entry to build a HNSW index for fast nearest-neighbor search over fused event embeddings (embeddings_F).
    Steps:
      1. Connect to DuckDB and fetch all events with embeddings_F.
      2. Normalize vectors for cosine similarity.
      3. Map string event_ids to integer labels for HNSW.
      4. Build and save HNSW index and id mapping.
    Args:
      --M: HNSW graph degree (controls index connectivity)
      --efC: HNSW construction parameter (tradeoff: speed vs. accuracy)
    """
    ap = argparse.ArgumentParser(description="Build (or rebuild) HNSW index on embeddings_F.")
    ap.add_argument("--M", type=int, default=16, help="HNSW M (graph degree)")
    ap.add_argument("--efC", type=int, default=200, help="HNSW efConstruction")
    args = ap.parse_args()

    # Connect to DuckDB and fetch event embeddings
    con = connect(read_only=False)
    df = con.execute("""
        SELECT event_id, embeddings_F FROM events
        WHERE embeddings_F IS NOT NULL
    """).fetchdf()

    if df.empty:
        raise SystemExit("No embeddings_F found in DB. Ingest embedded events first.")

    # Build arrays for HNSW
    ids = df["event_id"].tolist()
    vecs = np.array(df["embeddings_F"].tolist(), dtype=np.float32)
    dim = vecs.shape[1]

    # Normalize vectors for cosine similarity search
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    vecs = vecs / norms

    # Map string event_ids to integer labels for HNSW
    id2label = {i: idx for idx, i in enumerate(ids)}
    labels = np.arange(len(ids)).astype(np.int64)

    # Build HNSW index
    index = hnswlib.Index(space='cosine', dim=dim)
    index.init_index(max_elements=len(ids), ef_construction=args.efC, M=args.M)
    index.add_items(vecs, labels)
    index.set_ef(200)

    # Save index and id mapping
    index.save_index(INDEX_PATH)
    with open(IDMAP_PATH, "w") as f:
        json.dump({"ids": ids}, f)

    print(f"✅ Built HNSW index with {len(ids)} items @ dim={dim}")
    print(f"   Saved index → {INDEX_PATH}")
    print(f"   Saved idmap → {IDMAP_PATH}")

# python3 -m memory.build_index --M 16 --efC 200
if __name__ == "__main__":
    main()
