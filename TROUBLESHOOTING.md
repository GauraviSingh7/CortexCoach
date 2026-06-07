# macOS Setup Guide

## Prerequisites

Install Homebrew if it is not already installed:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Install PortAudio:

```bash
brew install portaudio
```

---

## Create Virtual Environment

Python 3.11 is recommended for maximum compatibility.

```bash
python3 -m venv venv
source venv/bin/activate
```

Upgrade pip:

```bash
pip install --upgrade pip
```

Install project dependencies:

```bash
pip install -r requirements.txt
```

Install AssemblyAI extras:

```bash
pip install "assemblyai[extras]"
```

---

## macOS Certificate Fix (Required)

AssemblyAI Streaming uses secure WebSocket connections.

On macOS, Python installations from python.org may not have certificates configured automatically.

Run:

```bash
open "/Applications/Python 3.12/Install Certificates.command"
```

If using Python 3.11:

```bash
open "/Applications/Python 3.11/Install Certificates.command"
```

Verify connectivity:

```bash
python - <<'PY'
import requests

r = requests.get("https://api.assemblyai.com")
print(r.status_code)
PY
```

Expected result:

```text
404
```

A `404` response confirms SSL certificates are working correctly.

---

## PortAudio / PyAudio Troubleshooting

If startup fails with:

```text
AssemblyAIExtrasNotInstalledError
```

or:

```text
ImportError: pyaudio._portaudio
```

verify PyAudio can load:

```bash
python -c "import pyaudio; print('PyAudio OK')"
```

Expected:

```text
PyAudio OK
```

### If PyAudio Cannot Find PortAudio

You may see:

```text
Library not loaded:
.../portaudio/lib/libportaudio.2.dylib
```

Create a symlink to the PortAudio library:

```bash
mkdir -p ~/portaudio/lib

ln -sf \
$(brew --prefix portaudio)/lib/libportaudio.2.dylib \
~/portaudio/lib/libportaudio.2.dylib
```

If using the bundled PortAudio source tree:

```bash
mkdir -p ~/portaudio/lib

ln -sf \
<PROJECT_ROOT>/portaudio/lib/.libs/libportaudio.2.dylib \
~/portaudio/lib/libportaudio.2.dylib
```

Verify:

```bash
python -c "import pyaudio; print('PyAudio OK')"
```

---

## Verify AssemblyAI Installation

```bash
python - <<'PY'
import assemblyai as aai
import pyaudio

print("AssemblyAI:", aai.__version__)
print("PyAudio OK")
PY
```

Expected output:

```text
AssemblyAI: <version>
PyAudio OK
```

---

## Verify AssemblyAI Streaming

```bash
python - <<'PY'
import assemblyai as aai

stream = aai.extras.MicrophoneStream()
print("MicrophoneStream OK")
PY
```

Expected output:

```text
MicrophoneStream OK
```

---

## Start Backend

```bash
python backend/main.py
```

or

```bash
uvicorn backend.main:app --reload
```

depending on your local setup.

---

## Common Errors

### SSL Certificate Verification Failed

Error:

```text
[SSL: CERTIFICATE_VERIFY_FAILED]
```

Fix:

```bash
open "/Applications/Python 3.12/Install Certificates.command"
```

Verify:

```bash
python - <<'PY'
import requests
print(requests.get("https://api.assemblyai.com").status_code)
PY
```

Expected:

```text
404
```

---

### PyAudio Import Error

Error:

```text
Could not import the PyAudio C module
```

Fix:

```bash
brew install portaudio

pip uninstall pyaudio
pip install --no-cache-dir pyaudio
```

---

### AssemblyAIExtrasNotInstalledError

Fix:

```bash
pip install "assemblyai[extras]"
```

and verify:

```bash
python -c "import pyaudio"
```

runs successfully.

---

### PortAudio Library Not Found

Error:

```text
ImportError: dlopen(.../pyaudio/_portaudio.so)
Library not loaded:
~/portaudio/lib/libportaudio.2.dylib
```

Fix:

```bash
mkdir -p ~/portaudio/lib

ln -sf \
<PROJECT_ROOT>/portaudio/lib/.libs/libportaudio.2.dylib \
~/portaudio/lib/libportaudio.2.dylib
```

Verify:

```bash
python -c "import pyaudio; print('PyAudio OK')"
```

Expected:

```text
PyAudio OK
```

---

## Recommended Python Version

For the most stable experience on macOS:

```text
Python 3.11.x
```

Python 3.12 may work, but audio-related dependencies (PyAudio, PortAudio, and some streaming SDKs) are generally more reliable on Python 3.11.