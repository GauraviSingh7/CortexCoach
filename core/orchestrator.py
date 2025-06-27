from core.coaching_rag_system import CoachingRAGSystem
from core.session_manager import SessionManager
from models.inference_pipeline import MultimodalInferencePipeline
from config.settings import GEMINI_API_KEY, CHROMA_DB_PATH, MODEL_DIR
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class SystemOrchestrator:
    """Central orchestrator for initializing and validating all subsystems"""

    def __init__(self):
        self.rag = None
        self.session_manager = None
        self.model_pipeline = None

    def initialize_all(self):
        logger.info("🔧 Initializing CoachingRAGSystem...")
        self.rag = CoachingRAGSystem(
            gemini_api_key=GEMINI_API_KEY,
            chroma_persist_dir=str(CHROMA_DB_PATH)
        )

        logger.info("📦 Initializing models...")
        self.model_pipeline = MultimodalInferencePipeline({
            'facial_emotion': str(MODEL_DIR / "facial_emotion_model.h5"),
            'sarcasm': str(MODEL_DIR / "sarcasm_model.pkl"),
            'vak': str(MODEL_DIR / "vak_model.pkl")
        })

        logger.info("🧠 Initializing SessionManager...")
        self.session_manager = SessionManager(
            coaching_system=self.rag
        )

        return {
            "rag": self.rag,
            "session_manager": self.session_manager,
            "model_pipeline": self.model_pipeline
        }

    def check_health(self):
        assert self.rag.template_collection.count() > 0, "❌ No templates loaded in Chroma"
        model_status = self.model_pipeline.get_model_status()
        for name, ok in model_status.items():
            assert ok, f"❌ Model '{name}' failed to load"
        logger.info("✅ All systems initialized and healthy.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    orchestrator = SystemOrchestrator()
    orchestrator.initialize_all()
    orchestrator.check_health()
