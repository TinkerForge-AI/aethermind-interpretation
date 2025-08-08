(The file `/home/dchisholm125/Desktop/repos-local/aethermind-interpretation/.github/copilot-instructions.md` exists, but is empty)
# Copilot Instructions for Aethermind Interpretation

## Big Picture Architecture

- **Purpose**: This repo implements the perception annotation and hierarchical semantic memory (HSM) pipeline for Aethermind.
- **Major Components**:
	- `run_interpretation.py`: Orchestrates annotation of perception events (video/audio), calling all annotators and writing annotated event JSON.
	- `annotators/`: Modular annotators for vision, audio, motion, time, speech, and scene classification. Each annotator updates the `annotations` dict in the event.
	- `interpreters/`: Contains interpreter logic for transforming annotated events.
	- `hsm/`: Hierarchical semantic memory logic, including the authoritative `CoreEvent` schema, event merging, and semantic level construction.
	- `utils/`: Utility functions for video/audio processing.

## Data Flow

- **Input**: Perception events (video/audio chunks) as JSON, with raw and annotated fields.
- **Annotation**: Each annotator updates `event["annotations"]` with its results.
- **Event Construction**: `hsm/event_builder.py` maps the `annotations` dict to the strict `CoreEvent` dataclass.
- **Merging**: `hsm/merge_utils.py` merges events using similarity metrics and deduplication logic, producing higher-level semantic summaries.

## Developer Workflows

- **Run Perception Pipeline**:
	```bash
	python run_interpretation.py <event_file.json>
	```
- **Run HSM Merge**:
	```bash
	python -m hsm.run_hsm <session_events.json> --max-level N --window-size W
	```
- **Testing**:
	- Use `pytest` with `PYTHONPATH=.` to run tests in `tests/`.
- **Schema Validation**:
	- Use Pydantic models in `hsm/schemas.py` for validating event shape.

## Project-Specific Patterns

- **CoreEvent Schema**: All semantic events use the strict dataclass in `hsm/core_event.py`. Tag lists must have `label` and `score: float`.
- **Associations Deduplication**: When merging events, associations are deduplicated by type and pointer fields (see `hsm/merge_utils.py`).
- **Vector Features**: Vector summarization expects a list of dicts with an `x` key; always extract `x` before passing to numpy.
- **Error Handling**: Annotator steps are wrapped in try/except with clear error logging.
- **Debugging**: Use `[DEBUG]` print statements for merge scores, tag sets, and vector inputs to trace pipeline behavior.

## Integration Points

- **Dependencies**: Relies on `librosa`, `openl3`, `tensorflow`, `torch`, and `tfhub` for perception and embedding.
- **Modular Annotators**: Add new annotators in `annotators/` and update orchestration in `run_interpretation.py`.
- **Centralized Merging**: All event merging and semantic logic is in `hsm/`.

## Examples

- See `hsm/event_builder.py` for mapping raw event dicts to CoreEvent.
- See `hsm/merge_utils.py` for deduplication and similarity logic.
- See `annotators/vector_features.py` for correct vector input handling.

---

If any section is unclear or missing, please provide feedback for further refinement.
