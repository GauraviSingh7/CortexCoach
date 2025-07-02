import streamlit as st
import plotly.express as px
from core.coaching_rag_system import GROWPhase
from ui.components import GROWPhaseTracker, VARKStyleAnalyzer, SessionMetricsComponent

def render_sidebar(self):
    """Render sidebar with controls and analytics"""
    
    with st.sidebar:
        st.header("🎛️ Controls")

        # --- Removed waiting spinner and disabling logic ---

        # Session controls
        if st.session_state.session_id:
            st.success(f"Session: {st.session_state.session_id[:8]}...")

            st.button(
                "End Session", 
                type="secondary", 
                key=f"end_session_button_{st.session_state.session_id}"
            )
        
        st.divider()
        
        # Multimodal settings (no disabling now)
        st.subheader("📹 Input Settings")
        camera_enabled = st.checkbox(
            "Enable Camera", 
            value=st.session_state.get('camera_enabled', False),
            key=f"camera_checkbox_{st.session_state.get('session_id', 'default')}"
        )
        
        audio_enabled = st.checkbox(
            "Enable Microphone", 
            value=st.session_state.get('audio_enabled', False),
            key=f"audio_checkbox_{st.session_state.get('session_id', 'default')}"
        )
        
        # Update session state
        st.session_state.camera_enabled = camera_enabled
        st.session_state.audio_enabled = audio_enabled

        st.divider()
        
        # Phase tips (always visible)
        self.render_phase_tips()
        
        st.divider()

        # Analytics (always rendered)
        try:
            self.render_sidebar_analytics()
        except Exception as e:
            st.error(f"Error rendering analytics: {e}")


def render_phase_tips(self):
    """Render phase tips in sidebar"""
    st.subheader("💡 Phase Tips")
    
    phase_guidance = {
        GROWPhase.GOAL: {
            "title": "🎯 Goal Setting",
            "tips": [
                "Be specific about your desired outcome",
                "Make your goal measurable",
                "Set a realistic timeframe"
            ]
        },
        GROWPhase.REALITY: {
            "title": "🔍 Current Reality",
            "tips": [
                "Be honest about where you are now",
                "Identify available resources",
                "Acknowledge obstacles"
            ]
        },
        GROWPhase.OPTIONS: {
            "title": "💡 Explore Options",
            "tips": [
                "Think creatively",
                "Consider all possibilities",
                "Don't judge ideas yet"
            ]
        },
        GROWPhase.WILL: {
            "title": "✅ Way Forward",
            "tips": [
                "Choose specific actions",
                "Set deadlines",
                "Identify accountability measures"
            ]
        }
    }

    current_phase = st.session_state.get('current_phase', GROWPhase.GOAL)
    if isinstance(current_phase, str):
        current_phase = GROWPhase(current_phase)
    elif not isinstance(current_phase, GROWPhase):
        current_phase = GROWPhase.GOAL

    current_guidance = phase_guidance.get(current_phase, phase_guidance[GROWPhase.GOAL])
    
    st.markdown(f"**{current_guidance['title']}**")
    for tip in current_guidance["tips"]:
        st.markdown(f"• {tip}")

def render_sidebar_analytics(self):
    """Render analytics in sidebar"""

    st.subheader("📊 Session Analytics")

    try:
        # --- VARK Profile ---
        vark_profile = st.session_state.get("vark_profile", {})
        if vark_profile and all(k in vark_profile for k in ["visual", "auditory", "reading", "kinesthetic"]):
            VARKStyleAnalyzer().render_mini_profile(vark_profile)

            # Show dominant learning style
            dominant = max(vark_profile, key=vark_profile.get)
            st.markdown(f"**Dominant Style:** `{dominant.capitalize()}`")
        else:
            st.info("VARK profile will be available after a few exchanges.")

        # --- Interest Level Trend ---
        interest_data = st.session_state.get("interest_levels", [])
        if len(interest_data) >= 2:  # Need at least 2 data points for a trend
            fig = px.line(
                y=interest_data[-10:],  # Last 10 points
                title="Interest Level Trend",
                labels={"y": "Interest Level", "index": "Turn"},
                height=200
            )
            fig.update_layout(
                showlegend=False, 
                margin=dict(l=0, r=0, t=30, b=0),
                xaxis_title="Recent Turns",
                yaxis_title="Interest"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Interest level trend will appear after more conversation.")

        # --- Session Metrics ---
        current_phase = st.session_state.get("current_phase", GROWPhase.GOAL)
        phase_value = current_phase.value if hasattr(current_phase, 'value') else str(current_phase)
        
        SessionMetricsComponent().render_mini_metrics({
            "total_turns": st.session_state.get("conversation_turn", 0),
            "current_phase": phase_value,
            "goal_clarity": round(st.session_state.get("goal_clarity_score", 0.0), 2)
        })
        
    except Exception as e:
        st.error(f"Error in analytics rendering: {e}")
        st.info("Analytics will be restored shortly.")