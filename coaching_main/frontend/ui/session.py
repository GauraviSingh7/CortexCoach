"""Session lifecycle and WebSocket message routing for the dashboard."""

import time

import streamlit as st

from ui import api
from ui.api import API_BASE_URL, BackendError


def _reset_session_ui() -> None:
    """Clear every trace of the previous session before a new one starts."""
    st.session_state.pop('final_report', None)
    st.session_state.feedback_data = []
    st.session_state.transcript_history = []
    st.session_state.current_utterances = {'coach': '', 'coachee': ''}
    st.session_state.current_grow_phase = "Uncertain"
    st.session_state.current_engagement = 0.0
    st.session_state.current_learning_style = "Insufficient Data"
    st.session_state.current_digression = 0.0
    st.session_state.current_sarcasm = 0.0
    st.session_state.sarcasm_detected = False
    st.session_state.playback_finished = False
    st.session_state.playback_error = None


def start_session():
    """Start a new coaching session."""
    try:
        with st.spinner("Starting session..."):
            device_index = st.session_state.get("mic_device_index")
            data = api.start_session("live", device_index=device_index)
            _reset_session_ui()
            st.session_state.session_id = data["session_id"]
            st.session_state.session_active = True

            if not st.session_state.ws_client.connected:
                if st.session_state.ws_client.connect():
                    time.sleep(1)
                    st.session_state.websocket_connected = True

            st.success(f"✅ Session started: {st.session_state.session_id[:8]}...")
            return True
    except BackendError as exc:
        st.error(f"❌ Failed to start session: {exc}")
    except Exception as exc:
        st.error(
            f"❌ Cannot reach the backend at {API_BASE_URL}. Is it running? ({exc})"
        )
    return False


def start_replay_session(transcript_path: str = "tests/data/sample_session.json"):
    """Replay a stored transcript through the real analysis pipeline.

    Needs no AssemblyAI or Gemini credentials, so the dashboard can be
    driven end to end without keys.
    """
    try:
        with st.spinner("Starting replay..."):
            data = api.start_session("replay", transcript_path=transcript_path)
            _reset_session_ui()
            st.session_state.session_id = data["session_id"]
            st.session_state.session_active = True

            if not st.session_state.ws_client.connected:
                if st.session_state.ws_client.connect():
                    time.sleep(1)
                    st.session_state.websocket_connected = True

            st.success("Replaying the sample session...")
            return True
    except BackendError as exc:
        st.error(f"Could not start replay: {exc}")
    except Exception as exc:
        st.error(f"Could not reach the backend at {API_BASE_URL} ({exc})")
    return False


def stop_session():
    """Stop the session and fetch the report."""
    try:
        with st.spinner("Stopping session and generating report..."):
            report = api.stop_session()
            st.session_state.session_active = False
            st.session_state.websocket_connected = False
            st.success("✅ Session stopped successfully")
            return report
    except BackendError as exc:
        st.error(f"❌ Failed to stop session: {exc}")
    except Exception as exc:
        st.error(f"❌ Error stopping session: {exc}")
    return None


def get_session_status():
    return api.get_session_status()


def process_real_time_updates():
    """Route WebSocket messages into partial (live) or final (history) buckets."""
    new_data = []

    if 'ws_client' not in st.session_state:
        return new_data

    for msg in st.session_state.ws_client.get_messages():
        msg_type = msg.get('type', 'final')
        speaker  = msg.get('speaker', 'coach')

        if msg_type == 'playback_complete':
            # A finite source (replay/file) has fed its last turn. The
            # session stays active - stopping it is what builds the
            # report - so this only ends the live streaming state.
            st.session_state.playback_finished = True
            st.session_state.playback_error = msg.get('error')
            st.session_state.current_utterances = {'coach': '', 'coachee': ''}
            continue

        if msg_type == 'partial':
            # Update the live streaming bubble; do NOT add to history
            st.session_state.current_utterances[speaker] = msg.get('transcript', '')
            continue

        # --- final message ---
        # Clear the streaming bubble for this speaker
        st.session_state.current_utterances[speaker] = ''

        # Add to both legacy feedback_data (for analytics) and transcript_history
        st.session_state.feedback_data.append(msg)
        st.session_state.transcript_history.append(msg)
        new_data.append(msg)

        if 'grow_phase' in msg:
            st.session_state.current_grow_phase = msg['grow_phase'].get('phase', 'Reality')
        if 'engagement_score' in msg:
            st.session_state.current_engagement = msg['engagement_score']
        if 'learning_style' in msg:
            st.session_state.current_learning_style = msg['learning_style']
        if 'digression_level' in msg:
            st.session_state.current_digression = msg['digression_level']
        if 'sarcasm_score' in msg:
            st.session_state.current_sarcasm = msg['sarcasm_score']
        if 'sarcasm_detected' in msg:
            st.session_state.sarcasm_detected = msg['sarcasm_detected']

    return new_data
