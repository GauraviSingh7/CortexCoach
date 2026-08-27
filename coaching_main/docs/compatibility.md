# Version compatibility

Constraints that exist for a reason. Each one records a failure that was
actually hit, so that a future "let's just upgrade everything" does not
quietly reintroduce it.

Checked against the working environment on 2026-08-28: Windows 11,
Python 3.12, global install (no virtualenv).

## Hard constraints

### `assemblyai>=0.65.0,<2.0` — live capture breaks on 1.0 without the fix

The 1.0 release removed `assemblyai.extras`, where `MicrophoneStream` lived.
`audio_processor.py` used to call `aai.extras.MicrophoneStream(...)`; on
1.0.0 that raises `AttributeError: module 'assemblyai' has no attribute
'extras'` in the capture thread. The session still started, AssemblyAI still
connected, and the log read `Session terminated: 0s audio processed` — a
live session that captured nothing while reporting itself healthy.

This is why live mode "worked two months ago and not today": the pin was
`>=0.44.0`, which resolved to a 0.6x release then and to 1.0.0 later. Nothing
in the repo changed.

`audio_processor.py` now supplies its own PyAudio frame source, so both the
0.6x and 1.x lines work. The upper bound is there because the streaming API
has now moved twice.

### `scikit-learn>=1.6.1` — model pickles were written with 1.6.1

`models/interest_detection/engagement_pipeline.pkl` unpickles into
`RandomForestClassifier`, `StandardScaler` and `LabelEncoder`. Loading it on
1.5.1 works but emits `InconsistentVersionWarning` for each estimator, and
sklearn does not guarantee cross-version unpickling.

### `google-generativeai>=0.8.0` — required for `gemini-2.5-flash`

Older releases only expose `gemini-1.5-*`; the model name in
`gemini_analyzer.py` fails to resolve below 0.8.

### `numpy==1.26.4` — pinned below 2.0

`torch` 2.4.x and the shipped pickles were built against the 1.x ABI.

## Audio capture: PyAudio host-API behaviour

Not a version constraint, but the same class of trap, and it cost more time
than any of the above. On this hardware the three host APIs each fail
differently, and none of them is the obvious default:

| Host API | Device | Opens at 16 kHz | Paced correctly | Delivers audio |
| --- | --- | --- | --- | --- |
| MME (PyAudio's default!) | `[1]` Microphone Array | yes | yes | **no — pure silence** |
| DirectSound | `[4]`, `[5]` | yes | **no** | yes |
| WASAPI | `[9]` | **no** (`Invalid sample rate`) | — | — |

Two separate traps:

1. **PyAudio's global default input is the MME endpoint**, which opens
   happily and then returns nothing but zeroes forever. Indistinguishable
   from a working microphone in a quiet room.
2. **The DirectSound endpoints do not pace reads.** `read()` returns
   instantly whether or not new audio has arrived, *and*
   `get_read_available()` reports `0` forever. Reading them in a loop either
   spins a core and grows the SDK's send queue without bound (~24 MB/s
   observed, backend reached 2 GB RSS and the event loop stalled) or, if you
   gate on `get_read_available()`, captures nothing at all.

`audio_processor.py` therefore uses **callback-driven capture**
(`stream_callback=`), which PortAudio paces correctly on every device here,
with a bounded queue so a stalled consumer drops frames rather than growing
memory. Device selection walks the host APIs in preference order, discards
candidates that refuse 16 kHz or return silence, and reports what it settled
on.

## Environment drift

The installed environment has diverged from the `==` pins throughout — for
example `fastapi` 0.128.0 against a pinned 0.104.1, `streamlit` 1.41.1
against 1.28.1, `pydantic` 2.10.6 against 2.5.0. Everything works on the
newer versions, so the pins were left alone rather than forcing a downgrade,
but a clean `pip install -r requirements.txt` would install something quite
different from what is being run. Worth reconciling deliberately.

`chromadb==0.4.18` is listed but **not installed**; session history is not
persisted and the orchestrator logs `ChromaDB storage unavailable` at
startup.

There is no virtualenv — packages live in the user/global site-packages.

## System audio is not captured, and cannot be with PyAudio alone

Live capture reads a **microphone**. It hears a video playing on another
tab only if that audio comes out of the speakers loudly enough for the mic
to pick up — on headphones it hears nothing at all.

PyAudio exposes no WASAPI loopback, so there is no in-process way to tap
what the machine is playing. To capture system audio, one of:

- **Enable Stereo Mix** (Windows Sound → Recording → show disabled
  devices). It then appears in the sidebar's microphone picker like any
  other input. Present on this machine as device `[24]` but currently
  disabled.
- **Install a virtual cable** (VB-CABLE or similar) and route playback
  through it.
- **Switch to `PyAudioWPatch`**, a PyAudio fork that does expose WASAPI
  loopback. Drop-in for the import, but a Windows-only dependency.

## AssemblyAI streaming: the diarization field is `speaker_label`

`streaming.v3.TurnEvent` carries **`speaker_label`**. There is no
`speaker_id` on it — reading that name returns `None` for every turn, and
role assignment then silently falls back to guessing from each utterance's
wording, which makes one speaker look like two and vice versa. Confirmed
against the installed SDK: `TurnEvent` fields are `type, turn_order,
turn_is_formatted, end_of_turn, transcript, end_of_turn_confidence, words,
language_code, language_confidence, speaker_label`.
