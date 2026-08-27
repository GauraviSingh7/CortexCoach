# Where this project stands, and what to pick up next

Short handover note. Full per-model detail lives in
`coaching_main/docs/known-gaps.md` — read that before touching the models.

## What this is

Real-time coaching-session analyser. A FastAPI backend transcribes a
conversation, scores each turn (GROW phase, engagement, emotion, sarcasm,
digression, VAK learning style), streams the results over a WebSocket, and
builds a report when the session stops. A Streamlit dashboard consumes it.
There is also a newer React app in `coaching_main/frontend-web/`.

Everything described here is on `main` — the refactor and model-integration
work was merged in PR #4. Clone and work from `main`.

## Run it in five minutes, no API keys needed

```bash
cd coaching_main
pip install -r requirements.txt

# terminal 1
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

# terminal 2
streamlit run frontend/streamlit_app.py --server.port 8501
```

Open http://localhost:8501 and click **🔁 Replay sample session**. That
feeds a bundled 40-turn transcript (`tests/data/sample_session.json`)
through the real analysis pipeline — same code path as a live session,
no AssemblyAI or Gemini credentials required. It runs ~18s, then says
playback is complete; press **⏹️ Stop Session** to generate the report.

Live and file modes do need keys. Copy `coaching_main/.env.example` to
`.env` and fill in `ASSEMBLYAI_API_KEY` (required for those modes) and
`GEMINI_API_KEY` (optional — it only rewrites prose; the numbers are
computed locally either way).

Tests: `python -m pytest tests -q` from `coaching_main` (22 pass).
Ignore `test_file_mode.py` / `test_wav_file.py` at the top level — stale
manual scripts with a hardcoded path from another machine.

## State of play

A reviewer filed 14 findings. Ten are fixed and have regression tests
named after them in `tests/test_regressions.py` — GROW percentages summing
to 100%, even speaker splits, sarcasm detection, digression markers,
listening scores, theme false positives, and honest model-status
reporting among them.

**The four that remain are all the same problem: no trained model
actually runs.** All four score with documented rule-based heuristics
instead. This is not a wiring bug — each adapter detects its artifact and
flips itself from `heuristic` to `trained` with no code change. The
artifacts are missing or unusable:

| Model | Blocked by | To unblock |
| --- | --- | --- |
| Sarcasm | Keras tokenizer from training was never shipped, so text can't be mapped to its 30k vocab | Drop `tokenizer.pkl` or `word_index.json` into `models/sarcasm_detection/`. **No retraining** |
| VAK | Config, vocab, tokenizer, label encoder all present — weights absent | Find `model.safetensors` / `pytorch_model.bin`. **No retraining if it exists** |
| Emotion | `model_weight.pth` is a 40-feature *audio* graph-conv net; it cannot score text, which is the only input the pipeline carries | Train a text emotion classifier, or supply the training-time graph construction + audio extractor |
| Engagement | Real RandomForest over 9744 audio features, but the extraction is undocumented *and* `AudioChunk.audio_data` is never populated anywhere | Populate audio in `audio_processor.py` / `file_audio_processor.py`, and get the feature extractor |

Nothing degrades silently: `GET /model-status`, the startup logs and the
dashboard panel all report per-model state, artifacts found/missing, and
a specific blocking reason. Every heuristic value is labelled as such in
the report via `analysis_sources`.

## Suggested order of work

1. **Chase the two missing files first** — the sarcasm tokenizer and the
   VAK weights. Both are pure retrieval, no training, and each one flips
   a model to `trained` immediately. Best effort-to-payoff in the repo.
2. **Decide on emotion.** Either commission a text classifier, or accept
   text-only lexicon scoring as a documented limitation. Don't start here
   without that decision — it's the only item that genuinely needs a
   model trained.
3. **Audio plumbing** (`AudioChunk.audio_data`) if engagement or
   audio-based emotion matter. This is real work and unlocks nothing on
   its own — the engagement feature extractor is still needed alongside it.
4. **Digression via embeddings.** Currently marker-based and deliberately
   so; token overlap produced false positives on two thirds of a clean
   session. Cosine similarity against a rolling topic centroid would do it
   properly. Lowest priority, clearly scoped.

## Things worth knowing before you change anything

- **Replay mode is the test harness.** Any pipeline change should be
  checked by replaying the sample session and diffing the report — it is
  deterministic and exercises the full path.
- **Metrics are computed locally and are authoritative.** Gemini only
  rewrites `key_insights`, `recommendations` and `transcript_summary`;
  `_merge_narrative` in `backend/core/orchestrator.py` deliberately keeps
  it away from the numbers. Keep it that way.
- **Diarization merges consecutive same-speaker segments**, which can
  silently halve the turn count on real audio. The processor logs a
  warning when it sees this — check for it before trusting per-turn
  metrics from an audio session.
- Streamlit reruns the whole script on every interaction. `streamlit_app.py`
  calls `main()` unconditionally, not behind an `if __name__ == "__main__"`
  guard, for that reason.
