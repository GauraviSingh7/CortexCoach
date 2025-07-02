import streamlit as st
import logging
from datetime import datetime
from core.coaching_rag_system import GROWPhase

logger = logging.getLogger(__name__)


def check_phase_transition(self):
        """Check if phase transition should occur"""
        turn = st.session_state.conversation_turn
        phase = st.session_state.current_phase

        if turn < 2:
            return  # Don’t transition too early

        if phase == GROWPhase.GOAL:
            if st.session_state.goal_clarity_score >= 0.7:
                self.transition_to_phase(GROWPhase.REALITY)

        elif phase == GROWPhase.REALITY:
            if turn >= 6:  # You can add LLM check here too later
                self.transition_to_phase(GROWPhase.OPTIONS)

        elif phase == GROWPhase.OPTIONS:
            if turn >= 9:  # Add LLM check for decision confidence later
                self.transition_to_phase(GROWPhase.WILL)

    
def transition_to_phase(self, new_phase: GROWPhase):
    """Transition to a new GROW phase"""
    current_phase = st.session_state.current_phase

    # Avoid transitioning to the same phase
    if new_phase == current_phase:
        logger.warning("⚠️ Attempted to transition to the same phase.")
        return

    # Check if phase already completed (avoid duplicates)
    for phase_entry in st.session_state.phase_history:
        if isinstance(phase_entry, dict) and phase_entry.get('phase') == current_phase:
            logger.warning(f"⚠️ Phase {current_phase.value} already marked complete.")
            return

    # Count user turns in current phase
    turns_in_current_phase = sum(
        1 for msg in st.session_state.conversation_history
        if msg['role'] == 'user' and msg.get('phase') == current_phase.value
    )

    # Append completed phase to history
    st.session_state.phase_history.append({
        'phase': current_phase,
        'completed_at': datetime.now(),
        'turns_spent': turns_in_current_phase,
        'completed': True
    })

    # Update phase state
    st.session_state.current_phase = new_phase

    # Generate phase transition message
    phase_messages = {
        GROWPhase.REALITY: "Great! Now let's explore your current situation. What's happening right now?",
        GROWPhase.OPTIONS: "Perfect! Now let's brainstorm some options. What are different ways you could approach this?",
        GROWPhase.WILL: "Excellent! Now let's create your action plan. What specific steps will you take?"
    }

    transition_message = phase_messages.get(new_phase, f"Moving to {new_phase.value.title()} phase")

    # Add transition message to conversation history
    st.session_state.conversation_history.append({
        'role': 'assistant',
        'content': f"🎯 **Phase Transition:** {transition_message}",
        'timestamp': datetime.now(),
        'phase': new_phase.value,
        'is_transition': True
    })

    logger.info(f"🚦 Successfully transitioned: {current_phase.value.upper()} → {new_phase.value.upper()}")
    st.success(f"✅ Moved to {new_phase.value.title()} phase!")
    
    # Force UI refresh
    st.rerun()

def check_phase_completion(self, message: str):
            """Check whether the current GROW phase should transition."""
            current_phase = st.session_state.current_phase
            current_turn = st.session_state.conversation_turn
            
            logger.info(f"🔄 [PHASE CHECK] Current Phase: {current_phase.value}")
            logger.info(f"💬 [USER MESSAGE] {message}")
            logger.info(f"🎯 [CLARITY SCORE] {st.session_state.goal_clarity_score}")
            logger.info(f"📊 [TURNS] Conversation Turn: {current_turn}")

            # ✅ GOAL → REALITY
            if current_phase == GROWPhase.GOAL:
                clarity_score = st.session_state.goal_clarity_score
                logger.info(f"🧠 [GOAL] Current Goal Clarity Score: {clarity_score}")

                # Check if goal is clear enough and we have enough turns
                if clarity_score >= 0.75 and current_turn >= 2:
                    logger.info("✅ [GOAL PHASE COMPLETE] Transitioning to REALITY phase")
                    self.transition_to_phase(GROWPhase.REALITY)
                    return
                else:
                    logger.info(f"⏳ [GOAL PHASE NOT READY] Clarity: {clarity_score}/0.75, Turns: {current_turn}/2")

            # ✅ REALITY → OPTIONS
            elif current_phase == GROWPhase.REALITY:
                # Check if user has described their current situation
                if current_turn >= 4:  # Allow at least 2 turns in REALITY phase
                    prompt = f"""Analyze if the user has described their current situation, challenges, or obstacles.
                    
                    User message: "{message}"
                    
                    Respond with only 'yes' or 'no'."""
                    
                    result = self.query_llm_boolean(prompt)
                    logger.info(f"🤖 [REALITY CHECK] LLM Response: {result}")
                    
                    if "yes" in result.lower():
                        logger.info("✅ [REALITY PHASE COMPLETE] Transitioning to OPTIONS phase")
                        self.transition_to_phase(GROWPhase.OPTIONS)
                        return
                    else:
                        logger.info("⏳ [REALITY PHASE NOT READY] Current situation not fully explored")

            # ✅ OPTIONS → WILL
            elif current_phase == GROWPhase.OPTIONS:
                # Check if user has generated options
                if current_turn >= 6:  # Allow at least 2 turns in OPTIONS phase
                    prompt = f"""Analyze if the user has proposed any options, solutions, or ideas for moving forward.
                    
                    User message: "{message}"
                    
                    Respond with only 'yes' or 'no'."""
                    
                    result = self.query_llm_boolean(prompt)
                    logger.info(f"🧪 [OPTIONS CHECK] LLM Response: {result}")
                    
                    if "yes" in result.lower():
                        logger.info("✅ [OPTIONS PHASE COMPLETE] Transitioning to WILL phase")
                        self.transition_to_phase(GROWPhase.WILL)
                        return
                    else:
                        logger.info("⏳ [OPTIONS PHASE NOT READY] No clear options identified")

            # ✅ WILL → Completion
            elif current_phase == GROWPhase.WILL:
                # Check if user has committed to action
                if current_turn >= 8:  # Allow at least 2 turns in WILL phase
                    prompt = f"""Analyze if the user has committed to concrete action steps or made specific commitments.
                    
                    User message: "{message}"
                    
                    Respond with only 'yes' or 'no'."""
                    
                    result = self.query_llm_boolean(prompt)
                    logger.info(f"✅ [WILL CHECK] LLM Response: {result}")
                    
                    if "yes" in result.lower():
                        logger.info("🎉 [SESSION COMPLETE] Coaching session completed")
                        st.success("🎉 Coaching session completed! You've made great progress.")
                        # You could add session completion logic here
                        self.end_session()
                        return
                    else:
                        logger.info("⏳ [WILL PHASE NOT READY] No clear commitment detected")



def query_llm_boolean(self, prompt: str) -> str:
    """Query LLM for boolean response"""
    try:
        # Use the existing coaching system's Gemini model
        response = st.session_state.coaching_system.gemini_model.generate_content(prompt)
        result = response.text.lower().strip()
        logger.info(f"🤖 LLM Boolean Response: {result}")
        return result
    except Exception as e:
        logger.error(f"LLM boolean check failed: {e}")
        return "no"