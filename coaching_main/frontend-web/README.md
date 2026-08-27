# AI Coaching Observer — web dashboard

React + Vite + TypeScript replacement for the Streamlit dashboard.

## Why this exists

Streamlit re-runs the entire script on every update. The old dashboard
polled with `time.sleep(1.5); st.rerun()`, shuttled WebSocket messages
from a background thread through a `queue.Queue` into `session_state`,
and rebuilt every chart and the whole transcript on each tick.

This app subscribes to `/ws/feedback` with a native WebSocket and
dispatches each message into a reducer, so:

- turns arrive **pushed**, with no polling interval;
- only the components bound to the changed slice re-render;
- the transcript appends instead of being rebuilt, and auto-scroll pauses
  while you are reading back through earlier turns.

## Running it

Two processes. From `coaching_main/`:

```bash
# 1. backend
python -m uvicorn backend.main:app --port 8000

# 2. dashboard
cd frontend-web
npm install
npm run dev          # http://localhost:5173
```

Vite proxies `/api/*` and `/ws/*` to `localhost:8000`, so the browser
sees a single origin and there is no CORS to configure.

### No API keys? Use replay mode

The backend can run the real analysis pipeline over a stored transcript
with no AssemblyAI or Gemini credentials:

```bash
curl -X POST localhost:8000/session/start \
  -H 'Content-Type: application/json' \
  -d '{"session_type":"replay","transcript_path":"tests/data/sample_session.json"}'
```

Everything downstream of the transcript is the production code path, so
this exercises speaker routing, GROW classification, sarcasm, digression
and report generation exactly as a live session would.

## Verifying

```bash
npm run verify     # typecheck + headless render checks + production build
```

`npm run smoke` renders every component to a string under Node. Effects
do not run, so nothing touches the network, but each component body,
prop access and branch on the initial path executes — which catches the
runtime errors a typecheck cannot.

## Layout

```
src/
  types.ts                 wire types, mirroring the FastAPI schemas
  lib/
    api.ts                 REST client (throws BackendError)
    ws.ts                  WebSocket hook with reconnect backoff
    format.ts              metric / percent / dominant-emotion helpers
  store/session.ts         reducer + derived live stats
  components/
    Header.tsx             model health at a glance
    ControlPanel.tsx       start live / upload file / stop
    StatsBanner.tsx        GROW, engagement, sarcasm, drift
    LiveTranscript.tsx     appends; smart auto-scroll
    Suggestions.tsx        prompts for the latest turn
    GrowTimeline.tsx       phase distribution (Recharts)
    EmotionChart.tsx       emotional journey (Recharts)
    ModelStatusPanel.tsx   trained vs heuristic, with blocking reasons
    SessionReport.tsx      final report
    ui/primitives.tsx      Card, Metric, Badge, SourceBadge, Button
```

## Notes on the stack

- **No shadcn/ui.** The dashboard needs six primitives; vendoring a
  component library and its Radix dependency tree for that is more
  surface than value. `ui/primitives.tsx` is 170 lines.
- **No TanStack Query.** Live data arrives over the WebSocket, and the
  REST surface is four endpoints called at well-defined moments. A query
  cache would not earn its place here.
- **Dark theme only.** A monitoring dashboard reads better dark, and
  committing to one palette means every chart, border and badge is tuned
  against a known background.

## Provenance is a first-class concern

Every metric the backend computes carries a source: `model`, `heuristic`
or `unavailable`. The UI renders that as a `SourceBadge` next to the
value, and `ModelStatusPanel` shows exactly why any degraded model is not
running. Three of the four shipped models cannot load — see
`../docs/known-gaps.md` — and the dashboard says so rather than
presenting heuristic output as model output.
