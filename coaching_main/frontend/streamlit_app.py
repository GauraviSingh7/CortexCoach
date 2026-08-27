"""
AI Coaching Observer - dashboard entry point.

Layout and wiring only. The views live in ``frontend/ui/``:

    ui/api.py        REST + WebSocket transport
    ui/session.py    session lifecycle and message routing
    ui/controls.py   header, controls, settings sidebar
    ui/live.py       live monitoring views
    ui/analytics.py  charts
    ui/report.py     final session report
    ui/panels.py     model status, GROW coverage, provenance
"""

import sys
import time
from pathlib import Path

import streamlit as st

# Streamlit puts the script's own directory on sys.path; add its parent too
# so `ui.*` resolves however the app is launched.
_HERE = Path(__file__).resolve().parent
for _path in (str(_HERE), str(_HERE.parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

st.set_page_config(
    page_title="AI Coaching Observer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

from ui import api
from ui.api import WebSocketClient
from ui.controls import render_control_panel, render_header, render_settings
from ui.live import render_real_time_feedback
from ui.panels import render_model_status
from ui.report import render_session_report

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

if 'session_active' not in st.session_state:
    st.session_state.session_active = False
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'feedback_data' not in st.session_state:
    st.session_state.feedback_data = []
if 'websocket_connected' not in st.session_state:
    st.session_state.websocket_connected = False
if 'current_grow_phase' not in st.session_state:
    st.session_state.current_grow_phase = "Reality"
if 'current_engagement' not in st.session_state:
    st.session_state.current_engagement = 0.5
if 'current_learning_style' not in st.session_state:
    st.session_state.current_learning_style = "Insufficient Data"
if 'current_digression' not in st.session_state:
    st.session_state.current_digression = 0.0
if 'current_sarcasm' not in st.session_state:
    st.session_state.current_sarcasm = 0.0
if 'sarcasm_detected' not in st.session_state:
    st.session_state.sarcasm_detected = False
# Set when a replay/file source runs out, so the 1.5s refresh loop stops
# instead of re-polling a session that has nothing left to say.
if 'playback_finished' not in st.session_state:
    st.session_state.playback_finished = False
if 'playback_error' not in st.session_state:
    st.session_state.playback_error = None
if 'ws_client' not in st.session_state:
    st.session_state.ws_client = WebSocketClient()
# Transcript architecture: only final turns go into transcript_history;
# in-progress text lives in current_utterances keyed by speaker.
if 'transcript_history' not in st.session_state:
    st.session_state.transcript_history = []
if 'current_utterances' not in st.session_state:
    st.session_state.current_utterances = {'coach': '', 'coachee': ''}


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point"""
    
    render_header()
    render_control_panel()
    render_settings()
    
    if st.session_state.session_active:
        render_real_time_feedback()
        
        # Auto-refresh, unless playback has run out
        if (
            st.session_state.get("auto_refresh", True)
            and not st.session_state.playback_finished
        ):
            time.sleep(1.5)
            st.rerun()
    else:
        render_session_report()

        if 'final_report' not in st.session_state:
            with st.expander("🔬 Model Status", expanded=True):
                render_model_status(api.get_model_status())

        if 'final_report' not in st.session_state:
            st.markdown("""
            ## 🚀 Getting Started
            
            1. **Start Session**: Click "▶️ Start Session" in the sidebar
            2. **Monitor Live**: Watch real-time GROW phases, engagement, digression, sarcasm, and suggestions
            3. **Stop Session**: Click "⏹️ Stop Session" to generate comprehensive report
            
            ### 📊 Live Features
            - **GROW Phase Tracking**: See current coaching phase in real-time
            - **Engagement Monitoring**: Track coachee interest level
            - **Topic Focus**: Monitor conversation digression (staying on-topic)
            - **Sarcasm Detection**: Identify passive-aggression and resistance
            - **Learning Style Detection**: Identify VAK preferences
            - **AI Suggestions**: Get instant coaching advice
            """)


# Streamlit executes this module top to bottom on every rerun, so the
# entry point is called unconditionally rather than behind an
# `if __name__ == "__main__"` guard - under the test harness the module
# is not __main__, and the guard silently rendered an empty page.
main()
