import streamlit as st
import numpy as np
from datetime import datetime
from core.coaching_rag_system import MultimodalContext, VARKType, GROWPhase
from typing import Any, Dict, List, Optional, Tuple, Union


def update_analytics(self, context: MultimodalContext):
        """Update analytics data"""
        
        # Update emotion history
        emotion_data = {
            'timestamp': context.timestamp,
            'facial_emotion': context.facial_emotion,
            'voice_emotion': context.voice_emotion,
            'dominant_emotion': self.get_dominant_emotion_from_context(context)
        }
        st.session_state.emotion_history.append(emotion_data)
        
        # Update interest levels
        st.session_state.interest_levels.append(context.interest_level)
        
        # Update VARK profile (adaptive learning)
        if context.vark_confidence > 0.7:
            # Strengthen the detected VARK type
            current_weight = st.session_state.vark_profile[context.vark_type.value]
            st.session_state.vark_profile[context.vark_type.value] = min(1.0, current_weight + 0.1)
            
            # Normalize weights
            total_weight = sum(st.session_state.vark_profile.values())
            for key in st.session_state.vark_profile:
                st.session_state.vark_profile[key] /= total_weight

def get_dominant_emotion_from_context(self, context: MultimodalContext) -> str:
        """Get dominant emotion from context"""
        return st.session_state.coaching_system._get_dominant_emotion(
            context.facial_emotion, context.voice_emotion
        )
    
def get_session_duration(self) -> int:
    """Get session duration in minutes"""
    if st.session_state.conversation_history:
        start_time = st.session_state.conversation_history[0]['timestamp']
        duration = datetime.now() - start_time
        return int(duration.total_seconds() / 60)
    return 0

def get_phase_progression(self) -> Dict:
    """Get phase progression data"""
    phases = [GROWPhase.GOAL, GROWPhase.REALITY, GROWPhase.OPTIONS, GROWPhase.WILL]
    current_phase = st.session_state.current_phase
    if isinstance(current_phase, str):
        current_phase = GROWPhase(current_phase)
    elif not isinstance(current_phase, GROWPhase):
        current_phase = GROWPhase.GOAL

    current_index = phases.index(current_phase)
    
    return {
        'current_phase': current_phase.value,
        'progress_percentage': ((current_index + 1) / len(phases)) * 100,
        'completed_phases': len(st.session_state.phase_history),
        'total_phases': len(phases)
    }

def calculate_engagement_score(self) -> float:
    """Calculate overall engagement score"""
    if not st.session_state.interest_levels:
        return 0.0
    
    return np.mean(st.session_state.interest_levels)