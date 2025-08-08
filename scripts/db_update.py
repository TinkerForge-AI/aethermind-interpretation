import duckdb
import json
import os

MEM_DIR = os.environ.get("AETHERMIND_MEM_DIR", os.path.join(os.path.dirname(__file__), "../memory/store"))
DB_PATH = os.path.join(MEM_DIR, "memory.duckdb")
con = duckdb.connect(DB_PATH)

con.execute("ALTER TABLE episodes ADD COLUMN IF NOT EXISTS session_id TEXT;")
con.close()