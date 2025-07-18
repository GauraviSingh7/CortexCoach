import streamlit as st
import numpy as np
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from PIL import Image
import io
from core.coaching_rag_system import MultimodalContext, VARKType
from app.ui.components import EmotionAnalysisComponent

logger = logging.getLogger(__name__)

def create_multimodal_context(self, user_input: str) -> MultimodalContext:
    """Create multimodal context from current state"""

    from app.ui.components import EmotionAnalysisComponent  # just to be explicit here

    # Use averaged emotion from last 40 frames
    facial_emotion = EmotionAnalysisComponent.compute_average_emotion(
        st.session_state.get("emotion_history", [])
    )
    logger.info(f"[CTX] Averaged facial emotion: {facial_emotion}")


    voice_emotion = {}  # Extend later if needed

    # Determine dominant VARK type
    dominant_vark = max(st.session_state.vark_profile, key=st.session_state.vark_profile.get)
    vark_confidence = st.session_state.vark_profile[dominant_vark]

    # Interest level (last 5 turns)
    interest_level = (
        np.mean(st.session_state.interest_levels[-5:]) 
        if st.session_state.interest_levels else 0.5
    )

    # Text features (sarcasm, sentiment, digression)
    text_features = self.analyze_text_features(user_input)

    goal_clarity_score = st.session_state.goal_clarity_score

    logger.info(f"[CTX] Current Phase: {st.session_state.current_phase}, Input: {user_input}")
    logger.info(f"[CTX] Using existing clarity score: {goal_clarity_score}")

    return MultimodalContext(
        user_id=st.session_state.user_id,
        session_id=st.session_state.session_id,
        timestamp=datetime.now(),
        grow_phase=st.session_state.current_phase,
        utterance=user_input,

        facial_emotion=facial_emotion,
        voice_emotion=voice_emotion,
        text_sentiment=text_features.get('sentiment', {}),

        vark_type=VARKType(dominant_vark),
        vark_confidence=vark_confidence,

        sarcasm_detected=text_features.get('sarcasm_detected', False),
        sarcasm_confidence=text_features.get('sarcasm_confidence', 0.0),
        interest_level=interest_level,
        digression_score=0.0, 

        conversation_turn=st.session_state.conversation_turn,
        previous_phase_completion=len(st.session_state.phase_history) > 0,
        goal_clarity_score=goal_clarity_score,
        system_instruction=""
    )
    
def analyze_text_features(self, text: str) -> Dict[str, Any]:
    if not hasattr(st.session_state, 'multimodal_processor'):
        st.error("Multimodal processor not loaded.")
        return {
            'sarcasm_detected': False,
            'sarcasm_confidence': 0.0,
            'text_sentiment': {},
            'vark_type': "visual",
            'vark_confidence': 0.25
        }

    if isinstance(text, list):
        text = " ".join(map(str, text))
    elif not isinstance(text, str):
        text = str(text)

    if not text.strip():
        return {}

    # Get real output from model
    result = st.session_state.multimodal_processor.analyze_text(text)

    # Ensure sarcasm fields are present
    result.setdefault('sarcasm_detected', False)
    result.setdefault('sarcasm_confidence', 0.0)

    return result



def process_facial_emotion(self, image_data) -> Optional[Dict]:
    """Process facial emotion from camera input"""
    
    try:
        if st.session_state.multimodal_processor:
            # Convert image data to format expected by processor
            image_bytes = image_data.getvalue()
            image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image_array = np.array(image_pil)
            return st.session_state.multimodal_processor.process_facial_emotion(image_array)
    except Exception as e:
        logger.error(f"Facial emotion processing error: {e}")
    
    return None

def process_voice_emotion(self, audio_data) -> Optional[Dict]:
    """Process voice emotion from audio input"""
    
    try:
        if st.session_state.multimodal_processor:
            return st.session_state.multimodal_processor.process_voice_emotion(audio_data)
    except Exception as e:
        logger.error(f"Voice emotion processing error: {e}")
    
    return None