"""Header, session controls and the settings sidebar."""

import streamlit as st

from ui import api
from ui.api import API_BASE_URL, WS_URL
from ui.session import (
    get_session_status,
    start_replay_session,
    start_session,
    stop_session,
)


def render_header():
    """Render the main header"""
    st.title("🎯 AI Coaching Observer Dashboard")
    st.markdown("Real-time analysis and feedback for coaching sessions")
    
    if st.session_state.session_active:
        st.success(f"🟢 **Session Active** | ID: {st.session_state.session_id}")
    else:
        st.info("🔴 **No Active Session**")


def render_control_panel():
    """Render the session control panel"""
    st.sidebar.header("📋 Session Control")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("▶️ Start Session", disabled=st.session_state.session_active, type="primary"):
            if start_session():
                st.rerun()
            
    with col2:
        if st.button("⏹️ Stop Session", disabled=not st.session_state.session_active):
            report = stop_session()
            if report:
                st.session_state.final_report = report
            st.rerun()
    
    st.sidebar.caption(
        "No API keys? Replay the bundled 40-turn transcript through the "
        "same analysis pipeline."
    )
    if st.sidebar.button(
        "🔁 Replay sample session", disabled=st.session_state.session_active
    ):
        if start_replay_session():
            st.rerun()

    # Session status
    if st.session_state.session_active:
        status = get_session_status()
        if status:
            st.sidebar.metric("Chunks Processed", status.get('chunks_processed', 0))
            ws_connected = st.session_state.ws_client.connected if 'ws_client' in st.session_state else False
            st.sidebar.metric("WebSocket", "🟢 Connected" if ws_connected else "🔴 Disconnected")

            # The backend is the authority here: the WebSocket announcement
            # is missed if the socket dropped, but the status poll is not.
            if status.get("source_finished"):
                st.session_state.playback_finished = True

            # A live session whose microphone died still reports active.
            if status.get("capture_error"):
                st.sidebar.error(
                    "🎤 Audio capture stopped: " + str(status["capture_error"])
                )
            elif status.get("capture_warning"):
                st.sidebar.warning("🔇 " + str(status["capture_warning"]))

        if st.session_state.get("playback_finished"):
            st.sidebar.info(
                "✅ Playback complete — press **Stop Session** to generate "
                "the report."
            )
    else:
        # The backend can still hold a session this browser session has lost
        # track of - a page reload, a Streamlit restart, or a second tab.
        # Without this the Stop button stays disabled and every Start fails
        # with "session already running", which is unrecoverable from the UI.
        orphan = get_session_status()
        if orphan and orphan.get("active"):
            st.sidebar.warning(
                f"A {orphan.get('session_type') or 'live'} session is already "
                "running on the backend, started outside this page."
            )
            if st.sidebar.button("↩️ Take over / stop it"):
                st.session_state.session_id = orphan.get("session_id")
                st.session_state.session_active = True
                report = stop_session()
                if report:
                    st.session_state.final_report = report
                st.rerun()

        try:
            healthy = api.check_health()
            if healthy:
                st.sidebar.success("✅ Backend Connected")
            else:
                st.sidebar.error("❌ Backend Unhealthy")
        except:
            st.sidebar.error("❌ Backend Offline")


def render_settings():
    """Render settings panel"""
    st.sidebar.header("⚙️ Settings")
    
    with st.sidebar.expander("🔧 API Configuration"):
        st.text_input("Backend URL", value=API_BASE_URL, disabled=True)
        st.text_input("WebSocket URL", value=WS_URL, disabled=True)
        
        if st.button("🔄 Reconnect WebSocket"):
            if st.session_state.ws_client.connect():
                st.success("WebSocket reconnected!")
    
    with st.sidebar.expander("🎤 Microphone"):
        devices = api.get_audio_devices()
        if not devices:
            st.caption("Backend reported no input devices.")
        else:
            labels = ["Auto (best available)"] + [
                f"[{d['index']}] {d['name'][:40]} @ {d['sample_rate']}Hz"
                for d in devices
            ]
            choice = st.selectbox(
                "Input device", labels, index=0,
                help="If nothing is transcribed, the default device is "
                     "probably returning silence - pick another here.",
            )
            st.session_state.mic_device_index = (
                None if choice == labels[0]
                else devices[labels.index(choice) - 1]["index"]
            )

    with st.sidebar.expander("🔬 Model Diagnostics"):
        if st.button("Check model status"):
            st.session_state.model_status = api.get_model_status()
        status = st.session_state.get("model_status")
        if status:
            degraded = status.get("degraded", [])
            if degraded:
                st.warning(
                    f"{status.get('trained_count', 0)}/{status.get('total_count', 0)} "
                    f"trained. Degraded: {', '.join(degraded)}"
                )
            else:
                st.success("All models running trained weights")

    with st.sidebar.expander("🎨 Display Settings"):
        st.checkbox("Auto-refresh data", value=True, key="auto_refresh")
        if st.session_state.get("auto_refresh", True) and st.session_state.session_active:
            st.info("🔄 Auto-refreshing every 1.5 seconds")
