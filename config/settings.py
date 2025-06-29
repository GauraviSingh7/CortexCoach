from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models" / "saved_models"
CHROMA_DB_PATH = BASE_DIR / "chroma_db"

# Secrets
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# WebSocket
WEBSOCKET_HOST = "localhost"
WEBSOCKET_PORT = 8765

# Streamlit
STREAMLIT_HOST = "localhost"
STREAMLIT_PORT = 8501

# Logging
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s [%(levelname)s] %(message)s",
    "file": BASE_DIR / "logs" / "app.log"
}
(BASE_DIR / "logs").mkdir(parents=True, exist_ok=True)

# Sessions
SESSION_CONFIG = {
    "timeout_minutes": 20
}

MODEL_PATHS = {
    'facial_emotion': MODEL_DIR / "emotion_model.h5",
    'sarcasm': MODEL_DIR / "model_lstm.pkl",
    'vak': MODEL_DIR / "vak_model"  # Not .pkl
}

class Settings:
    def __init__(self):
        self.GEMINI_API_KEY = GEMINI_API_KEY
        self.CHROMA_DB_PATH = str(CHROMA_DB_PATH)
        self.MODEL_PATHS = MODEL_PATHS
        self.MODEL_DIR = str(MODEL_DIR)
        self.SESSION_CONFIG = SESSION_CONFIG

def get_settings():
    return Settings()