import os
import pytest
from core.coaching_rag_system import CoachingRAGSystem
from models.inference_pipeline import MultimodalInferencePipeline

def test_rag_initialization():
    api_key = os.getenv("GEMINI_API_KEY", "fake-key")  # Use dummy if not in CI
    chroma_path = "chroma_db"

    rag = CoachingRAGSystem(
        gemini_api_key=api_key,
        chroma_persist_dir=chroma_path
    )
    
    assert rag.context_collection is not None
    assert rag.template_collection is not None
    assert rag.response_collection is not None

def test_model_loading():
    model_paths = {
        'facial_emotion': 'models/saved_models/facial_emotion_model.h5',
        'sarcasm': 'models/saved_models/sarcasm_model.pkl',
        'vak': 'models/saved_models/vak_model.pkl'
    }
    pipeline = MultimodalInferencePipeline(model_paths)
    status = pipeline.get_model_status()

    assert all(isinstance(v, bool) for v in status.values())
