# aethermind-interpretation/memory/db.py
from __future__ import annotations
import os, duckdb

MEM_DIR = os.environ.get("AETHERMIND_MEM_DIR",
                         os.path.join(os.path.dirname(__file__), "store"))
os.makedirs(MEM_DIR, exist_ok=True)
DB_PATH = os.path.join(MEM_DIR, "memory.duckdb")

SCHEMA_SQL = r"""
CREATE TABLE IF NOT EXISTS events (
  event_id TEXT PRIMARY KEY,
  session_id TEXT,
  start_ts DOUBLE,
  end_ts DOUBLE,
  scene_type TEXT,
  is_event BOOLEAN,
  valence TEXT,
  video_path TEXT,
  audio_path TEXT,
  text_view TEXT,
  embeddings_T DOUBLE[],
  embeddings_S DOUBLE[],
  embeddings_F DOUBLE[],
  raw JSON
);
"""

def connect(read_only: bool=False) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(DB_PATH, read_only=read_only)
    con.execute("PRAGMA threads=4;")
    if not read_only:
        con.execute(SCHEMA_SQL)
    return con
