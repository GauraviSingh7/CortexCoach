# Quick Start Guide - Running the Application

## Prerequisites Check

Before starting, ensure you have:
- ✅ Python 3.11+ installed
- ✅ AssemblyAI API key
- ✅ Gemini API key
- ✅ Microphone connected and working

## Step-by-Step Instructions

### Step 1: Set Up Environment Variables

**Option A: Create .env file (Recommended)**
```bash
# In coaching_main directory, create .env file:
ASSEMBLYAI_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
```

**Option B: Export in terminal**
```bash
# Windows PowerShell:
$env:ASSEMBLYAI_API_KEY="your_key"
$env:GEMINI_API_KEY="your_key"

# Linux/macOS:
export ASSEMBLYAI_API_KEY="your_key"
export GEMINI_API_KEY="your_key"
```

### Step 2: Install Dependencies

```bash
cd coaching_main
pip install -r requirements.txt
```

**Note:** If PyAudio fails on Windows, try:
```bash
pip install pipwin
pipwin install pyaudio
```

### Step 3: Start Backend Server

Open **Terminal 1**:
```bash
cd coaching_main
python -m backend.main
```

Wait for: `INFO:     Uvicorn running on http://0.0.0.0:8000`

✅ Verify: Open http://localhost:8000/docs in browser

### Step 4: Start Frontend Dashboard

Open **Terminal 2**:
```bash
cd coaching_main
streamlit run frontend/streamlit_app.py
```

✅ Browser should open automatically at http://localhost:8501

### Step 5: Use the Application

1. In the Streamlit dashboard sidebar, click **"▶️ Start Session"**
2. Speak into your microphone
3. Watch real-time transcription and feedback appear
4. Click **"⏹️ Stop Session"** when done

## Troubleshooting

**Backend won't start?**
- Check if port 8000 is available
- Verify API keys are set correctly
- Check terminal for error messages

**Frontend can't connect?**
- Ensure backend is running on port 8000
- Check browser console for errors
- Verify `API_BASE_URL` in streamlit_app.py matches backend

**No audio captured?**
- Check microphone permissions
- Verify device is working in system settings
- Try selecting different device via API: GET /devices/audio

**Import errors?**
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt` again
- Check Python version: `python --version` (should be 3.11+)

## Quick Commands Reference

```bash
# Start backend
python -m backend.main

# Start frontend  
streamlit run frontend/streamlit_app.py

# Check backend health
curl http://localhost:8000/health

# View API docs
# Open: http://localhost:8000/docs
```

## Running Both in One Command (Windows)

Create `start.bat`:
```batch
@echo off
start "Backend" cmd /k "python -m backend.main"
timeout /t 3
start "Frontend" cmd /k "streamlit run frontend/streamlit_app.py"
```

## Running Both in One Command (Linux/macOS)

Create `start.sh`:
```bash
#!/bin/bash
python -m backend.main &
sleep 3
streamlit run frontend/streamlit_app.py
```

Make executable: `chmod +x start.sh`

---

**Need help?** Check the full README.md for detailed documentation.
