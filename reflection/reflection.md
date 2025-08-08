# How to use Step 3 end-to-end:

After embedding + ingesting + building the index (Step 2):

```bash
Copy
Edit
# Salience
python3 -m reflection.salience --window 5

# Episodes
python3 -m reflection.episodes --gap 3.0 --cos 0.85 --jaccard 0.40
```

Query your episodes quickly:

```bash
python3 - <<'PY'
from memory.db import connect
con = connect(True)
print("episodes:", con.execute("SELECT COUNT(*) FROM episodes").fetchone()[0])
rows = con.execute("SELECT episode_id, num_events, summary FROM episodes ORDER BY start_ts LIMIT 10").fetchall()
for r in rows: print(r)
PY
```
See an episode’s members:

```bash
python3 - <<'PY'
from memory.db import connect
con = connect(True)
print("episodes:", con.execute("SELECT COUNT(*) FROM episodes").fetchone()[0])
rows = con.execute("SELECT episode_id, num_events, summary FROM episodes ORDER BY start_ts LIMIT 10").fetchall()
for r in rows: print(r)
PY  
```

See an episode’s members:

```bash
python3 - <<'PY'
from memory.db import connect
con = connect(True)
ep = con.execute("SELECT episode_id FROM episodes ORDER BY start_ts LIMIT 1").fetchone()[0]
rows = con.execute("SELECT event_id, start_ts, end_ts FROM events WHERE episode_id = ? ORDER BY start_ts", [ep]).fetchall()
print("episode:", ep, "members:", len(rows))
for r in rows: print(r)
PY
```