import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image
import io

# Local imports - using relative imports within app
from .layout.sidebar import render_sidebar, render_phase_tips
from .layout.coach_interface import render_coaching_interface, render_analytics_panel, render_multimodal_inputs
from .layout.setup import render_session_setup
from .logic.session import (
    initialize_session_state, start_session, end_session,
    reset_session_state, check_session_health, get_session_summary
)
from .logic.respond import handle_user_message, generate_coaching_response
from .logic.context import create_multimodal_context, analyze_text_features, process_facial_emotion, process_voice_emotion
from .logic.analytics import (
    update_analytics, get_session_duration, 
    get_phase_progression, calculate_engagement_score, get_dominant_emotion_from_context
)
from .logic.grow import check_phase_completion, transition_to_phase, query_llm_boolean
from .logic.vark import update_learning_style_via_model
from .utils.helpers import normalize_text_input

# Core system imports
from core.coaching_rag_system import CoachingRAGSystem, GROWPhase
from models.multimodal_processor import MultimodalContextProcessor
from config.settings import get_settings, MODEL_PATHS

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="AI Coaching Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

class StreamlitCoachingApp:
    def __init__(self):
        self.settings = get_settings()
        self.start_time = time.time()
        self.user_id = f"user_{uuid.uuid4().hex[:8]}"
        self.session_id = f"session_{uuid.uuid4().hex[:8]}"
        self._setup_logging()

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        )

    def run(self):
        st.set_page_config(page_title="AI Coach", layout="wide")
        st.title("🧠 AI Coaching Assistant")

        initialize_session_state(self)
        self._load_components()

        # Show session setup if no active session
        if not st.session_state.session_id:
            render_session_setup(self)
        else:
            render_sidebar(self)
            render_coaching_interface(self)

    def _load_components(self):
        if st.session_state.coaching_system is None:
            st.session_state.coaching_system = CoachingRAGSystem(
                gemini_api_key=self.settings.GEMINI_API_KEY,
                chroma_persist_dir=self.settings.CHROMA_DB_PATH
            )

        if st.session_state.multimodal_processor is None:
            st.session_state.multimodal_processor = MultimodalContextProcessor(
                model_paths=MODEL_PATHS
            )

    # Delegate methods to logic modules
    def analyze_text_features(self, text: str) -> Dict[str, Any]:
        return analyze_text_features(self, text)

    def handle_user_message(self, message: str):
        handle_user_message(self, message)

    def generate_coaching_response(self, user_input: str, initial: bool = False):
        generate_coaching_response(self, user_input, initial)

    def create_multimodal_context(self, user_input: str):
        return create_multimodal_context(self, user_input)

    def start_session(self, user_name: str, user_goal: str):
        start_session(self, user_name, user_goal)

    def end_session(self):
        end_session(self)

    def check_phase_completion(self, message: str):
        check_phase_completion(self, message)

    def transition_to_phase(self, new_phase: GROWPhase):
        transition_to_phase(self, new_phase)

    def query_llm_boolean(self, prompt: str) -> str:
        return query_llm_boolean(self, prompt)

    def update_learning_style_via_model(self, user_text: str):
        update_learning_style_via_model(self, user_text)

    def update_analytics(self, context):
        update_analytics(self, context)

    def get_session_duration(self) -> int:
        return get_session_duration(self)

    def get_phase_progression(self) -> Dict:
        return get_phase_progression(self)

    def calculate_engagement_score(self) -> float:
        return calculate_engagement_score(self)

    def get_dominant_emotion_from_context(self, context):
        return get_dominant_emotion_from_context(self, context)

    def process_facial_emotion(self, image_data) -> Optional[Dict]:
        return process_facial_emotion(self, image_data)

    def process_voice_emotion(self, audio_data) -> Optional[Dict]:
        return process_voice_emotion(self, audio_data)

    def render_phase_tips(self):
        render_phase_tips(self)

    def render_analytics_panel(self):
        render_analytics_panel(self)

    def render_multimodal_inputs(self):
        render_multimodal_inputs(self)

    def render_sidebar_analytics(self):
        from app.layout.sidebar import render_sidebar_analytics
        render_sidebar_analytics(self)
        
    def reset_session_state(self):
        reset_session_state(self)

    def check_session_health(self) -> bool:
        return check_session_health(self)

    def get_session_summary(self) -> Dict:
        return get_session_summary(self)
