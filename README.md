Aethermind Pipeline: Recommended Execution Order
> **Human-in-the-Loop (HITL) Note:** Episodes are built from any subset of input streams (video, audio, keyboard, mouse). Not all episodes will have complete media; only those with valid video and audio paths are stitched for human review. All episodes are stored in the database for downstream analysis.

1. Normalize Events
Prepare raw perception event JSONs for annotation and embedding.

From `scripts/` folder run:

```
python3 normalize_events.py ../../aethermind-perception/aethermind_perception/chunks/ \  
  session_20250808_220541/session_events.json -o ../../aethermind-perception/ \ 
  aethermind_perception/  chunks/session_20250808_220541/session_events.normalized.json
```

2. Annotate Events
Run all perception annotators (vision, audio, motion, etc.) on the normalized events.

From the repo root, run:



2. Fit Embedding Artifacts
Fit text and fusion embedding models on the normalized events (only needed if artifacts are missing or you want to refit).

```
python3 -m embeddings.fit ../aethermind-perception/aethermind_perception/chunks/session_20250808_220541/session_events.json
```

3. Embed Events
Generate vector embeddings for each normalized event.

```
python3 -m embeddings.embed \
  ../aethermind-perception/aethermind_perception/chunks/session_20250808_220541/session_events.json \
  -o ../aethermind-perception/aethermind_perception/chunks/session_20250808_220541/session_events.embedded.json
```

4. Ingest Embedded Events
Insert embedded events into the DuckDB database.

```
python3 -m memory.ingest ../aethermind-perception/aethermind_perception/chunks/session_20250808_220541/session_events.embedded.json
```

5. Run Nightly Pipeline
Orchestrate episode construction, salience, journals, and stitching.
This step will call reflection, episode merging, and reporting.

```
python3 -m pipeline.nightly \
  --inputs "../aethermind-perception/aethermind_perception/chunks/session_20250808_220541/session_events.normalized.json" \
  --make-journals --stitch-top-k 3 --min-episode-events 3
```

## DuckDB tables:

┌────────────────┐
│      name      │
│    varchar     │
├────────────────┤
│ episode_events │
│ episodes       │
│ events         │
│ seed_event_map │
│ seeds          │
│ symbol_members │
│ symbol_sizes   │
│ symbols        │
└────────────────┘

Database Record Insertion Summary

The Aethermind pipeline uses DuckDB to store and manage semantic event and episode data. Records are inserted into the database at several key stages:

1. Event Ingestion

Module: memory.ingest
Process: After perception events are annotated and embedded, each event is inserted into the events table.
Schema: Typical fields include event_id, session_id, timestamp, annotations, and vector embeddings.

2. Episode Construction

Module: reflection.episodes
Process: Episodes are constructed from sequences of events and inserted into the episodes table.
Schema: The table includes fields such as:
  episode_id (primary key)
  session_id
  start_ts, end_ts
  caption, tags_json, f_embed, summary
Additional fields: thought_text, valence_guess, confidence
Note: The code ensures the schema matches the database, serializing complex fields (e.g., tags, embeddings) as JSON or BLOBs.

3. Schema Alignment

The pipeline checks and migrates the database schema as needed, adding missing columns before inserts to maintain fidelity and prevent errors.

4. Bulk Insert Pattern

Records are typically inserted using bulk operations (executemany) for efficiency.
Example (Python):

```python
  con.executemany(
      "INSERT OR REPLACE INTO episodes (episode_id, session_id, start_ts, end_ts, caption, tags_json, f_embed, summary, thought_text, valence_guess, confidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
      episode_tuples
  )
```

5. Error Handling

The pipeline logs and handles errors during insertion, including schema mismatches and missing columns, to ensure robust operation.