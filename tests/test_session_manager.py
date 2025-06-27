from core.session_manager import SessionManager
from unittest.mock import MagicMock
from core.coaching_rag_system import CoachingRAGSystem
from models.inference_pipeline import MultimodalInferencePipeline

def test_session_manager_creation():
    system = CoachingRAGSystem("fake", "./test_chroma")
    pipeline = MultimodalInferencePipeline({
        'facial_emotion': 'models/saved_models/emotion_model.h5',
        'sarcasm': 'models/saved_models/best_sarcasm_model.h5',
        'vak': 'models/saved_models/label_encoder.pkl'
    })
    manager = SessionManager(system, pipeline, timeout_minutes=5)
    assert manager.coaching_system is system
