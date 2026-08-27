# AI Coach Observer - Real-time Coaching Session Dashboard

A comprehensive real-time dashboard system for monitoring and analyzing coaching sessions using the GROW model. Features live audio transcription, emotion recognition, engagement tracking, and AI-powered coaching analysis.

## 🏗️ Project Structure

```
coaching_main/
├── backend/              # FastAPI backend server
│   ├── analysis/        # GROW, emotion, sarcasm, digression, VAK, themes
│   ├── core/            # Orchestrator, turn processing, broadcasting
│   ├── models/          # Audio capture, inference, speaker routing, storage
│   ├── reporting/       # Local report assembly
│   ├── schemas/         # Data models
│   └── main.py          # FastAPI application entry point
├── frontend/            # Streamlit dashboard
│   ├── streamlit_app.py # Entry point and layout
│   └── ui/              # Views: api, session, controls, live, report, panels
├── frontend-web/        # React + Vite + TypeScript dashboard (alternative UI)
├── models/              # Pre-trained ML models
│   ├── emotion_recognition/
│   ├── interest_detection/
│   ├── sarcasm_detection/
│   └── vak_inference/
├── docs/                # known-gaps.md, compatibility.md
├── tests/               # Regression suite + sample transcript
├── reports/             # Generated session reports (JSON)
├── requirements.txt     # Python dependencies
├── PLAN.md              # Handover notes / what to pick up next
└── README.md            # This file
```

## 🚀 Quick Start

### Prerequisites

1. **Python 3.11+** (recommended)
2. **API Keys:**
   - AssemblyAI API key — required for live and file sessions only
   - Google Gemini API key — optional; without it reports are built by the
     local analyzer (Gemini only rewrites prose, never the computed metrics)
   - Neither is needed for **replay mode** (see Quick Demo below)
3. **System Dependencies:**
   - PyAudio requires audio system libraries
   - On Windows: Usually included with Python packages
   - On Linux: `sudo apt-get install portaudio19-dev python3-pyaudio`
   - On macOS: `brew install portaudio`

### Installation

1. **Navigate to project directory:**
   ```bash
   cd coaching_main
   ```

2. **Create virtual environment (recommended):**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   
   Create a `.env` file in the `coaching_main` directory:
   ```env
   ASSEMBLYAI_API_KEY=your_assemblyai_api_key_here
   GEMINI_API_KEY=your_gemini_api_key_here
   ```
   
   Or export them in your shell:
   ```bash
   # Windows (PowerShell):
   $env:ASSEMBLYAI_API_KEY="your_key"
   $env:GEMINI_API_KEY="your_key"
   
   # Linux/macOS:
   export ASSEMBLYAI_API_KEY="your_key"
   export GEMINI_API_KEY="your_key"
   ```

## 🎯 Running the Application

The application consists of two parts that need to run simultaneously:

### 1. Backend Server (FastAPI)

Open a **first terminal** and run:

```bash
cd coaching_main

# Activate virtual environment if using one
# Windows: venv\Scripts\activate
# Linux/macOS: source venv/bin/activate

# Run the backend server
python -m backend.main

# Or using uvicorn directly:
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

The backend will start on `http://localhost:8000`

**Verify it's running:**
- Visit: `http://localhost:8000/docs` (FastAPI interactive docs)
- Visit: `http://localhost:8000/health` (health check)

### 2. Frontend Dashboard (Streamlit)

Open a **second terminal** and run:

```bash
cd coaching_main

# Activate virtual environment if using one
# Windows: venv\Scripts\activate
# Linux/macOS: source venv/bin/activate

# Run Streamlit app
streamlit run frontend/streamlit_app.py

# Or if streamlit is in PATH:
streamlit run frontend/streamlit_app.py --server.port 8501
```

The dashboard will open automatically in your browser at `http://localhost:8501`

## ⚡ Quick Demo — No API Keys Needed

With both servers running, click **🔁 Replay sample session** in the sidebar.
It feeds a bundled 40-turn transcript (`tests/data/sample_session.json`)
through the real analysis pipeline — the same code path as a live session,
with no AssemblyAI or Gemini credentials required. It runs for about 18
seconds, then reports that playback is complete; press **⏹️ Stop Session**
to generate the report.

This is the fastest way to confirm the whole system works.

## 📋 Usage Guide

### Starting a Session

1. **Ensure both servers are running** (backend on :8000, frontend on :8501)
2. **Open the Streamlit dashboard** in your browser
3. **Check audio devices** (optional) - use the sidebar to view available microphones
4. **Click "▶️ Start Session"** in the sidebar
5. **Begin speaking** - the system will:
   - Capture audio from your microphone
   - Transcribe speech in real-time
   - Analyze emotions, engagement, and coaching quality
   - Display live feedback in the dashboard

### Monitoring the Session

The dashboard shows:
- **Real-time Transcription**: Live speech-to-text with speaker identification
- **Feedback Metrics**: Engagement scores, coaching quality, GROW phase tracking
- **Visualizations**: Charts showing emotional trajectory and engagement over time
- **Session Statistics**: Duration, chunks processed, active status

### Stopping a Session

1. Click **"⏹️ Stop Session"** in the sidebar
2. Wait for the final report to generate
3. Review the comprehensive session analysis

## 🔧 Configuration

### Audio Device Selection

If you have multiple microphones, pick one under **🎤 Microphone** in the
sidebar. Left on *Auto*, the backend walks the available host APIs and skips
devices that refuse 16 kHz or return silence — PyAudio's default device is not
always a working one. See `docs/compatibility.md`.

- List devices via: `GET http://localhost:8000/devices/audio`
- Only the **microphone** is captured. System/tab audio needs a loopback
  device (e.g. enable Stereo Mix); see `docs/compatibility.md`.

### Backend Configuration

Configuration currently lives alongside the code it affects rather than in a
central settings module (`backend/config/settings.py` is a placeholder and is
empty):

- **Session persistence** — `CHROMADB_PERSIST_DIR` env var (default `./.chromadb`)
- **API keys** — `.env`, see Installation above
- **Analysis thresholds** — module constants, e.g. `DIGRESSION_THRESHOLD` in
  `backend/analysis/digression.py`
- **Backend/WebSocket URLs used by the dashboard** — `frontend/ui/api.py`

See `docs/compatibility.md` for dependency constraints and the audio-device
notes that live capture depends on.

## 🐛 Troubleshooting

### Common Issues

1. **"API keys not configured" error:**
   - Ensure `.env` file exists with correct keys
   - Or export environment variables before starting

2. **PyAudio installation fails:**
   - Install system audio libraries (see Prerequisites)
   - On Windows, try: `pip install pipwin` then `pipwin install pyaudio`

3. **"Cannot connect to backend" error:**
   - Verify backend is running on port 8000
   - Check firewall settings
   - Ensure no other application is using port 8000

4. **No audio captured:**
   - Check microphone permissions in system settings
   - Verify correct audio device is selected
   - Test microphone with another application

5. **AssemblyAI connection errors:**
   - Verify API key is correct and has credits
   - Check internet connection
   - Review AssemblyAI account status

### Debug Mode

Run with debug logging:
```bash
# Backend
uvicorn backend.main:app --log-level debug

# Frontend
streamlit run frontend/streamlit_app.py --logger.level=debug
```

## 📊 API Endpoints

- `GET /` - API info (name, version, docs link)
- `GET /health` - Health status
- `POST /session/start` - Start a session (`live`, `file` or `replay`)
- `POST /session/start/file` - Start a session from an uploaded audio file
- `POST /session/stop` - Stop current session and get report
- `GET /session/status` - Session status, including capture warnings
- `GET /devices/audio` - List available audio input devices
- `GET /model-status` - Per-model state, and why any model is degraded
- `WS /ws/feedback` - WebSocket for real-time feedback

See `http://localhost:8000/docs` for interactive API documentation.

## 🏭 Production Deployment

For production deployment:

1. **Use production ASGI server:**
   ```bash
   uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 4
   ```

2. **Configure CORS properly:**
   - Edit `backend/main.py` to specify allowed origins

3. **Set secure environment variables:**
   - Use environment variable management (AWS Secrets, etc.)
   - Never commit API keys to version control

4. **Use reverse proxy:**
   - Set up Nginx/Apache in front of the FastAPI server
   - Configure SSL/TLS certificates

## 📝 Notes

- The system uses AssemblyAI's streaming.v3 API for real-time transcription
- Models are loaded on first use - initial startup may take a moment
- ChromaDB stores each session's final report; data persists in `.chromadb/`
  (override with `CHROMADB_PERSIST_DIR`). On the **first** stop it downloads an
  ~80 MB embedding model, so that request can take a couple of minutes — this
  is why the frontend allows a generous stop timeout. Subsequent runs are fast.
  If `chromadb` is not installed the API still runs; sessions simply are not
  persisted, and the backend logs `ChromaDB storage unavailable` at startup.
- All ML models are stored in the `models/` directory


## DFD
<img width="732" height="847" alt="image" src="https://github.com/user-attachments/assets/e653b626-d958-47a3-acfe-ce2a1a474085" />
---
