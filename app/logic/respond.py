import streamlit as st
import logging
from datetime import datetime
from core.coaching_rag_system import GROWPhase

logger = logging.getLogger(__name__)

def handle_user_message(self, message: str):
    """Handle user message input with robust state management"""

    if not message.strip():
        return

    # CRITICAL: Set response generation flags IMMEDIATELY
    st.session_state.waiting_for_response = True
    st.session_state.ui_locked = True

    try:
        logger.info(f"🔄 Processing user message: {message[:50]}...")
        
        # Add user message to conversation history
        st.session_state.conversation_history.append({
            'role': 'user',
            'content': message,
            'timestamp': datetime.now(),
            'phase': st.session_state.current_phase.value
        })

        # Increment turn count
        st.session_state.conversation_turn += 1

        # Update goal clarity score BEFORE checking phase completion
        if st.session_state.current_phase == GROWPhase.GOAL:
            try:
                goal_clarity_score = st.session_state.coaching_system.evaluate_goal_clarity(message)
                st.session_state.goal_clarity_score = goal_clarity_score
                logger.info(f"🎯 Updated Goal clarity score: {goal_clarity_score}")
            except Exception as e:
                logger.warning(f"Goal clarity evaluation failed: {e}")
                st.session_state.goal_clarity_score = st.session_state.get('goal_clarity_score', 0.0)

        # Check if current phase should transition BEFORE generating response
        try:
            self.check_phase_completion(message)
        except Exception as e:
            logger.warning(f"Phase completion check failed: {e}")

        # Update VARK profile dynamically every N turns
        if st.session_state.conversation_turn % 5 == 0:
            try:
                self.update_learning_style_via_model(message)
            except Exception as e:
                logger.warning(f"VARK update failed: {e}")

        # Generate coaching response - this is the critical part
        self.generate_coaching_response(message)

        logger.info(f"✅ Message processed successfully")
        logger.info(f"📈 Goal clarity score: {st.session_state.goal_clarity_score}")
        logger.info(f"🔄 Current phase: {st.session_state.current_phase.value}")
        logger.info(f"👥 Conversation turn: {st.session_state.conversation_turn}")

    except Exception as e:
        st.error(f"Error processing message: {e}")
        logger.error(f"Message processing error: {e}")
        # Always reset flags on error
        st.session_state.waiting_for_response = False
        st.session_state.ui_locked = False
        st.rerun()

def generate_coaching_response(self, user_input: str, initial: bool = False):
    """Generate coaching response using the RAG system with robust state management"""

    try:
        logger.info(f"🤖 Generating coaching response (initial={initial})")
        
        # Ensure we're in response generation mode (unless this is initial call)
        if not initial:
            st.session_state.waiting_for_response = True
            st.session_state.ui_locked = True

        # Create multimodal context
        context = self.create_multimodal_context(user_input)

        # Extract dominant emotion for logging and UI
        dominant_emotion = self.get_dominant_emotion_from_context(context)
        logger.info(f"[EMOTION] Dominant emotion passed to coach: {dominant_emotion}")

        # Generate response using the coaching system
        coaching_response = st.session_state.coaching_system.generate_coaching_response(context)

        if not coaching_response or not coaching_response.strip():
            logger.warning("Empty response generated, using fallback")
            coaching_response = "I understand. Could you tell me more about that?"

        # Add to conversation history
        response_entry = {
            'role': 'assistant',
            'content': coaching_response,
            'timestamp': datetime.now(),
            'phase': st.session_state.current_phase.value,
            'context_data': {
                'dominant_emotion': dominant_emotion,
                'interest_level': getattr(context, 'interest_level', 0.5),
                'vark_type': getattr(context, 'vark_type', 'visual').value if hasattr(getattr(context, 'vark_type', 'visual'), 'value') else str(getattr(context, 'vark_type', 'visual')),
                'sarcasm_detected': getattr(context, 'sarcasm_detected', False),
                'sarcasm_confidence': float(getattr(context, 'sarcasm_confidence', 0.0) or 0.0),
                'digression_score': getattr(context, 'digression_score', 0.0)
            }
        }

        logger.info(
            f"🎭 Sarcasm data: detected={response_entry['context_data']['sarcasm_detected']}, "
            f"confidence={response_entry['context_data']['sarcasm_confidence']:.2f}"
        )

        st.session_state.conversation_history.append(response_entry)

        # Update analytics
        try:
            self.update_analytics(context)
        except Exception as e:
            logger.warning(f"Analytics update failed: {e}")

        logger.info(f"✅ Response generated successfully: {coaching_response[:50]}...")

    except Exception as e:
        st.error(f"Failed to generate response: {e}")
        logger.error(f"Response generation error: {e}")

        # Add fallback response to prevent conversation from breaking
        fallback_response = {
            'role': 'assistant',
            'content': "I apologize, but I encountered an issue generating my response. Could you please rephrase your message?",
            'timestamp': datetime.now(),
            'phase': st.session_state.current_phase.value
        }
        st.session_state.conversation_history.append(fallback_response)

    finally:
        # CRITICAL: Always reset flags and rerun, regardless of success or failure
        st.session_state.waiting_for_response = False
        st.session_state.ui_locked = False

        logger.info("🔓 Response generation flags cleared, triggering rerun")
        st.rerun()
