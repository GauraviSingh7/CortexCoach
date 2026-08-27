# Known gaps

What is still blocked, why, and exactly what would unblock it. Everything
here is reported at runtime through `GET /model-status`, in the startup
logs, and in the dashboard's Model Status panel — nothing in this list
fails silently any more.

---

## 1. Emotion recognition — no usable trained model

**Status:** `heuristic` — text lexicon in `backend/analysis/emotion.py`.

`models/emotion_recognition/model_weight.pth` is a state dict for a graph
convolution network:

```
conv1.lin.weight (128, 40)    conv2.lin.weight (128, 128)
fc1.weight       (128, 128)   fc2.weight       (6, 128)
```

That is a **40-feature audio** model with **6** classes. It cannot score
text, which is the only input the pipeline currently carries.

**To unblock, one of:**

- supply a **text** emotion classifier (`text_emotion_model.pkl` +
  optional `vectorizer.pkl` + `label_encoder.pkl`, or a HuggingFace
  checkpoint with `model.safetensors`) — the adapter in
  `models/emotion_recognition/inference.py` picks either up automatically; or
- supply all three of: `torch-geometric`, the **graph construction used at
  training time** (undocumented), and a real 40-dimensional audio feature
  extractor — plus the audio-plumbing fix in item 4.

---

## 2. Sarcasm detection — model present, tokenizer missing

**Status:** `heuristic` — rules in `backend/analysis/sarcasm.py`.

`models/sarcasm_detection/model_lstm.pkl` is a pickled Keras functional
model: `Embedding(30000, 128) → BiLSTM → GlobalMaxPool → Dense → sigmoid`,
input shape `(None, 20)`. It loads, but the Keras `Tokenizer` /
`word_index` from training was never shipped, so text cannot be mapped
onto its 30k-token vocabulary. Feeding it arbitrary ids would produce
confident nonsense.

**To unblock:** drop the training tokenizer into
`models/sarcasm_detection/` as either `tokenizer.pkl` (a pickled
`keras.preprocessing.text.Tokenizer`) or `word_index.json`, and uncomment
`tensorflow` in `requirements.txt`. The adapter detects either file and
switches itself to `trained` with no code change.

---

## 3. VAK learning style — weights absent entirely

**Status:** `heuristic` — keyword scoring in `backend/analysis/vak.py`.

`models/vak_inference/` contains a complete BERT sequence-classification
setup *except* the weights:

| Present | Missing |
| --- | --- |
| `config.json` (3 labels) | `model.safetensors` |
| `vocab.txt` (30522 tokens) | `pytorch_model.bin` |
| `tokenizer_config.json` | |
| `label_encoder.pkl` | |

Neither this repository nor the older copy on OneDrive contains the
weights file, so this needs to come from whoever trained the model.

**To unblock:** drop `model.safetensors` (or `pytorch_model.bin`) into
`models/vak_inference/`. The adapter loads it automatically.

> **Note on label order.** The previous code assumed
> `probs[0]→visual, probs[1]→auditory, probs[2]→kinesthetic`. The label
> encoder is alphabetical — `Auditory, Kinesthetic, Visual` — so that
> mapping was wrong and would have produced plausible-looking but
> scrambled results the moment weights arrived. Labels are now read from
> the encoder.

---

## 4. Engagement — trained model is audio-only and nothing carries audio

**Status:** `heuristic` — keyword scoring in
`models/interest_detection/inference.py`.

`engagement_pipeline.pkl` loads successfully and holds a real trained
`RandomForestClassifier` over **9744 features** (MFCC-scale, classes
`High / Medium / Low Engagement`), plus its `StandardScaler` and
`LabelEncoder`.

It has never actually been used. The pickle stores the classifier under
`.model`, but `_predict_from_text` reads `.text_model`, so every call fell
through to the keyword path — producing plausible numbers that were never
model output. This is why engagement *looked* like the one working model.

Two things block it independently:

1. `AudioChunk.audio_data` is **never populated anywhere in the codebase**
   — not in live mode, not in file mode. No audio model can be reached.
2. The 9744-feature extraction used at training time is undocumented and
   not reconstructible from the artifact.

**To unblock:** populate `AudioChunk.audio_data` in
`backend/models/audio_processor.py` and `file_audio_processor.py`, and
supply the feature extractor used at training time.

---

## 5. Digression detection is marker-based, not semantic

**Status:** working, but deliberately limited.

Explicit discourse markers ("by the way", "random, but", "did you
watch…") are authoritative and flag a turn. Lexical overlap against the
recent topic window is kept as an **advisory** score capped below the
reporting threshold.

This is a considered limit, not an oversight. Natural coaching dialogue
has high vocabulary turnover — a perfectly on-topic turn routinely shares
almost no content words with the turns before it — so overlap alone
cannot separate genuine drift from ordinary variety. Using it as a flag
produced false positives on roughly two thirds of a clean test session.

**To improve:** score drift with sentence embeddings (cosine similarity
against a rolling topic centroid) rather than token overlap.

---

## 6. Diarization quality is an input risk, not a code bug

`transcript.utterances` merges *consecutive segments by the same diarized
speaker*. When the diarizer cannot separate two voices, adjacent turns
collapse into one utterance and the turn count silently halves.

The pipeline no longer hides this: `FileAudioProcessor` logs a warning
when only one speaker is found, or when four or more consecutive
utterances come from the same speaker. Role assignment itself is now
robust to which label the diarizer happens to pick (see
`backend/models/speaker_router.py`), but merged turns cannot be recovered
after the fact — check the warning before trusting per-turn metrics.
