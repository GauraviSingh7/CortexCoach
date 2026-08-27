"""Header, session controls and the settings sidebar."""

import streamlit as st

from ui import api
from ui.api import API_BASE_URL, WS_URL
from ui.session import get_session_status, start_session, stop_session


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
    
    # Session status
    if st.session_state.session_active:
        status = get_session_status()
        if status:
            st.sidebar.metric("Chunks Processed", status.get('chunks_processed', 0))
            ws_connected = st.session_state.ws_client.connected if 'ws_client' in st.session_state else False
            st.sidebar.metric("WebSocket", "🟢 Connected" if ws_connected else "🔴 Disconnected")
    else:
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
