import streamlit as st
import uuid
import time
from datetime import datetime
from core.coaching_rag_system import GROWPhase, VARKType
import logging

logger = logging.getLogger(__name__)

def initialize_session_state(self):
    """Initialize Streamlit session state with improved state management"""
    
    # Core session data
    if 'session_id' not in st.session_state:
        st.session_state.session_id = None
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'coaching_system' not in st.session_state:
        st.session_state.coaching_system = None
    if 'multimodal_processor' not in st.session_state:
        st.session_state.multimodal_processor = None
    
    # GROW Model state
    if 'current_phase' not in st.session_state:
        st.session_state.current_phase = GROWPhase.GOAL
    if 'phase_history' not in st.session_state:
        st.session_state.phase_history = []
    if 'goal_clarity_score' not in st.session_state:
        st.session_state.goal_clarity_score = 0.0
    
    # Conversation state
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'conversation_turn' not in st.session_state:
        st.session_state.conversation_turn = 0
    
    # Multimodal data
    if 'emotion_history' not in st.session_state:
        st.session_state.emotion_history = []
    if 'vark_profile' not in st.session_state:
        st.session_state.vark_profile = {'visual': 0.25, 'auditory': 0.25, 'reading': 0.25, 'kinesthetic': 0.25}
    if 'interest_levels' not in st.session_state:
        st.session_state.interest_levels = []
    
    # UI state (removed show_analytics)
    if 'camera_enabled' not in st.session_state:
        st.session_state.camera_enabled = False
    if 'audio_enabled' not in st.session_state:
        st.session_state.audio_enabled = False
    
    # CRITICAL: Response generation flags with better defaults
    if 'waiting_for_response' not in st.session_state:
        st.session_state.waiting_for_response = False
    if 'ui_locked' not in st.session_state:
        st.session_state.ui_locked = False
    
    # Additional state for message processing reliability
    if 'last_message_id' not in st.session_state:
        st.session_state.last_message_id = None
    if 'processing_message' not in st.session_state:
        st.session_state.processing_message = False

def start_session(self, user_name: str, user_goal: str):
    """Start a new coaching session with improved error handling"""

    try:
        # Prevent multiple simultaneous session starts
        if st.session_state.get('waiting_for_response', False):
            st.warning("Please wait for the current operation to complete")
            return
        
        # Set processing state
        st.session_state.waiting_for_response = True
        st.session_state.ui_locked = True
        
        # Identifiers
        session_id = f"session_{int(time.time())}"
        user_id = user_name.lower().replace(" ", "_") if user_name else f"user_{int(time.time())}"

        # Core state
        st.session_state.session_id = session_id
        st.session_state.user_id = user_id
        st.session_state.current_phase = GROWPhase.GOAL
        st.session_state.conversation_turn = 0
        st.session_state.phase_history = []
        st.session_state.conversation_history = []

        # Initialize metrics
        st.session_state.interest_levels = []
        st.session_state.vark_profile = {
            'visual': 0.25, 'auditory': 0.25, 'reading': 0.25, 'kinesthetic': 0.25
        }

        # Evaluate initial goal clarity
        clarity = st.session_state.coaching_system.evaluate_goal_clarity(user_goal)
        st.session_state.goal_clarity_score = clarity
        logger.info(f"[INIT] Goal clarity from goal input: {clarity}")

        # Add initial emotion data
        st.session_state.emotion_history = [{
            'timestamp': datetime.now(),
            'facial_emotion': {},
            'voice_emotion': {},
            'dominant_emotion': 'neutral'
        }]

        # Save user goal with unique message ID
        message_id = f"msg_{int(time.time() * 1000)}"
        st.session_state.last_message_id = message_id
        
        st.session_state.conversation_history.append({
            'role': 'user',
            'content': user_goal,
            'timestamp': datetime.now(),
            'phase': GROWPhase.GOAL.value,
            'message_id': message_id
        })

        # Trigger optional VARK update
        if hasattr(self, "update_learning_style_via_model"):
            self.update_learning_style_via_model(user_goal)

        logger.info(f"🚀 Started session: {session_id} for user: {user_id}")

        # Generate initial response
        self.generate_coaching_response(user_goal, initial=True)
        
        st.success("🎯 Coaching session started successfully!")

    except Exception as e:
        st.error(f"Failed to start session: {e}")
        logger.error(f"Session start error: {e}")
        # Reset flags on error
        st.session_state.waiting_for_response = False
        st.session_state.ui_locked = False
        st.rerun()

def end_session(self):
    """End the current coaching session with proper cleanup"""
    
    try:
        # Prevent ending session while processing
        if st.session_state.get('waiting_for_response', False):
            st.warning("Cannot end session while processing. Please wait.")
            return
        
        # Store session summary before clearing (optional)
        session_summary = {
            'session_id': st.session_state.get('session_id'),
            'total_turns': st.session_state.get('conversation_turn', 0),
            'final_phase': st.session_state.get('current_phase'),
            'goal_clarity': st.session_state.get('goal_clarity_score', 0.0),
            'end_time': datetime.now()
        }
        
        logger.info(f"📋 Session summary: {session_summary}")
        
        # Reset all session state (removed show_analytics from reset_keys)
        reset_keys = [
            'session_id', 'user_id', 'current_phase', 'conversation_history',
            'conversation_turn', 'phase_history', 'emotion_history', 
            'interest_levels', 'goal_clarity_score', 'waiting_for_response',
            'ui_locked', 'last_message_id', 'processing_message'
        ]
        
        for key in reset_keys:
            if key in st.session_state:
                if key == 'current_phase':
                    st.session_state[key] = GROWPhase.GOAL
                elif key in ['conversation_history', 'phase_history', 'emotion_history', 'interest_levels']:
                    st.session_state[key] = []
                elif key in ['conversation_turn', 'goal_clarity_score']:
                    st.session_state[key] = 0 if 'turn' in key else 0.0
                else:
                    st.session_state[key] = None if key in ['session_id', 'user_id', 'last_message_id'] else False
        
        # Reset VARK profile to default
        st.session_state.vark_profile = {
            'visual': 0.25, 'auditory': 0.25, 'reading': 0.25, 'kinesthetic': 0.25
        }
        
        # Clear multimodal settings (removed show_analytics)
        if 'camera_enabled' in st.session_state:
            st.session_state.camera_enabled = False
        if 'audio_enabled' in st.session_state:
            st.session_state.audio_enabled = False
        
        logger.info("✅ Session ended successfully - all state cleared")
        st.success("👋 Session ended. Thank you for using the coaching system!")
        
        # Force UI refresh to show cleared state
        st.rerun()
        
    except Exception as e:
        st.error(f"Error ending session: {e}")
        logger.error(f"Session end error: {e}")
        
        # Emergency reset - clear critical flags even if other cleanup fails
        st.session_state.waiting_for_response = False
        st.session_state.ui_locked = False
        st.session_state.session_id = None
        
        st.rerun()

def reset_session_state(self):
    """Complete reset of all session state variables"""
    
    # Get all keys to reset (removed show_analytics)
    keys_to_reset = [
        'session_id', 'user_id', 'coaching_system', 'multimodal_processor',
        'current_phase', 'phase_history', 'goal_clarity_score',
        'conversation_history', 'conversation_turn', 'emotion_history',
        'vark_profile', 'interest_levels', 'camera_enabled', 'audio_enabled',
        'waiting_for_response', 'ui_locked',
        'last_message_id', 'processing_message'
    ]
    
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
    
    # Reinitialize with defaults
    self.initialize_session_state()
    
    logger.info("🔄 Complete session state reset performed")

def check_session_health(self):
    """Check and repair session state if needed"""
    
    try:
        # Check for corrupted state
        required_keys = ['session_id', 'current_phase', 'conversation_history']
        missing_keys = [key for key in required_keys if key not in st.session_state]
        
        if missing_keys:
            logger.warning(f"Missing session keys detected: {missing_keys}")
            self.initialize_session_state()
            return False
        
        # Check for stuck states
        if (st.session_state.get('waiting_for_response', False) and 
            st.session_state.get('ui_locked', False)):
            
            # Check if stuck for too long (could implement timestamp check)
            logger.warning("Detected potentially stuck UI state")
            
            # Could add logic here to detect genuinely stuck states
            # For now, just log the warning
        
        return True
        
    except Exception as e:
        logger.error(f"Session health check failed: {e}")
        return False

def get_session_summary(self):
    """Get current session summary for analytics or export"""
    
    try:
        conversation_history = st.session_state.get('conversation_history', [])
        
        # Calculate session metrics
        user_messages = [msg for msg in conversation_history if msg['role'] == 'user']
        assistant_messages = [msg for msg in conversation_history if msg['role'] == 'assistant']
        
        # Phase distribution
        phase_counts = {}
        for msg in conversation_history:
            phase = msg.get('phase', 'goal')
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
        
        # Session duration (if we have timestamps)
        session_duration = None
        if conversation_history:
            start_time = conversation_history[0].get('timestamp')
            end_time = conversation_history[-1].get('timestamp')
            if start_time and end_time:
                session_duration = (end_time - start_time).total_seconds()
        
        summary = {
            'session_id': st.session_state.get('session_id'),
            'user_id': st.session_state.get('user_id'),
            'total_messages': len(conversation_history),
            'user_messages': len(user_messages),
            'assistant_messages': len(assistant_messages),
            'current_phase': st.session_state.get('current_phase'),
            'phase_distribution': phase_counts,
            'goal_clarity_score': st.session_state.get('goal_clarity_score', 0.0),
            'session_duration_seconds': session_duration,
            'final_vark_profile': st.session_state.get('vark_profile', {}),
            'average_interest_level': (
                sum(st.session_state.get('interest_levels', [])) / 
                len(st.session_state.get('interest_levels', [1]))
            ) if st.session_state.get('interest_levels') else 0.0
        }
        
        return summary
        
    except Exception as e:
        logger.error(f"Error generating session summary: {e}")
        return {
            'session_id': st.session_state.get('session_id'),
            'error': str(e)
        }