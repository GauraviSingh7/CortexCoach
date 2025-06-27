import pytest
import warnings
from core.coaching_rag_system import CoachingRAGSystem
from core.session_manager import SessionManager
from models.inference_pipeline import MultimodalInferencePipeline

# Suppress coroutine not awaited warning for background cleanup task
warnings.filterwarnings("ignore", category=RuntimeWarning)

@pytest.fixture(scope="module")
def coaching_system():
    return CoachingRAGSystem(
        gemini_api_key="fake-key",  # Use a dummy key for test
        chroma_persist_dir="./test_chroma"
    )

@pytest.fixture(scope="module")
def inference_pipeline():
    return MultimodalInferencePipeline({
        'facial_emotion': 'models/saved_models/emotion_model.h5',
        'sarcasm': 'models/saved_models/best_sarcasm_model.h5',
        'vak': 'models/saved_models/label_encoder.pkl'
    })

@pytest.fixture(scope="module")
def session_manager(coaching_system, inference_pipeline):
    return SessionManager(
        coaching_system=coaching_system,
        inference_pipeline=inference_pipeline,
        timeout_minutes=5
    )

def test_session_creation(session_manager):
    assert session_manager is not None
    assert isinstance(session_manager, SessionManager)

def test_coaching_system_template_loading(coaching_system):
    # Should have pre-populated templates
    assert coaching_system.template_collection.count() >= 4

def test_pipeline_model_status(inference_pipeline):
    status = inference_pipeline.get_model_status()
    assert isinstance(status, dict)
    assert all(model in status for model in ['facial_emotion', 'sarcasm', 'vak'])
