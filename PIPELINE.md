# Aethermind Interpretation Pipeline

This document describes the recommended end-to-end pipeline for processing perception events and building hierarchical semantic memory (HSM) in Aethermind.  
It covers all steps, expected inputs/outputs, and integration points across the three main repos.

---

## **Pipeline Overview**

1. **Chunking & Event Detection**  
   *(in `aethermind-perception`)*  
   - Chunk raw video/audio into segments.
   - Detect events and save as `session_events.json`.

2. **Vectorization & Merge**  
   *(in `aethermind-perception`)*  
   - Extract feature vectors from chunks.
   - Merge vectors into events, producing `session_events_with_vectors.json`.

3. **Normalization**  
   *(in `aethermind-interpretation`)*  
   - Standardize event schema and fields.
   - Output: `session_events.normalized.json`
   ```bash
   python3 scripts/normalize_events.py <session_events.json> -o <session_events.normalized.json>
   ```

4. **Annotation (First Pass)**  
   - Run all annotators except vector features.
   - Output: updated `session_events.normalized.json`
   ```bash
   python3 run_interpretation.py <session_events.normalized.json>
   ```

5. **Annotation (Second Pass: Vector Features)**  
   - Summarize vector features after all other annotations.
   - Output: updated `session_events.normalized.json`
   ```bash
   python3 run_interpretation.py <session_events.normalized.json> --only_vector_features
   ```

6. **Embedding**  
   - Embed annotated events for semantic memory.
   - Output: `session_events.embedded.json`
   ```bash
   python3 -m embeddings.embed <session_events.normalized.json> -o <session_events.embedded.json>
   ```

7. **Memory Ingestion**  
   - Ingest embedded events into DuckDB for querying and semantic processing.
   ```bash
   python3 -m memory.ingest <session_events.embedded.json>
   ```

8. **Nightly Pipeline (Episodes, Journals, Stitching)**  
   - Run episode construction, journaling, and video stitching.
   - Output: stitched episodes, reports, and journals.
   ```bash
   python3 -m pipeline.nightly \
     --inputs <session_events.normalized.json> \
     --make-journals --stitch-all --min-episode-events 2 \
     --keep-salience-min 0.30 --episode-min-sec 0 --episode-max-sec 600
   ```

---

## **Best Practices**

- **Always keep both raw and normalized event files** for reproducibility.
- **Run annotation in two passes** to ensure vector features are summarized after feature extraction.
- **Use the same DuckDB file** for all ingestion and querying steps.
- **Check logs for [DEBUG] and [Error] messages** to diagnose pipeline issues.
- **Update `.env` variables** for correct media and memory paths.

---

## **Troubleshooting**

- **DuckDB lock errors:**  
  Ensure only one process writes to the DB at a time. Kill conflicting processes if needed.
- **Missing events in stitching:**  
  Check that all episodes have valid video/audio paths and that selection filters are not too strict.
- **Empty vectors in annotation:**  
  Confirm feature extraction annotators run before vector summarization.

---

## **Integration Points**

- **Perception repo:** Chunking, event detection, vectorization.
- **Interpretation repo:** Normalization, annotation, embedding, memory, HSM.
- **Central orchestration:** Use a master script or Makefile to run all steps in sequence.

---

## **Example Master Script**

See `scripts/run_full_interpretation.py` for a recommended orchestration script.

---

## **References**

- See `README.md` for component details.
- See `.github/copilot-instructions.md` for architecture and coding patterns.

---

If you encounter issues or need to update the pipeline, document changes here for team