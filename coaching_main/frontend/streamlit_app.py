"""
AI Coaching Observer - Complete Streamlit Frontend Dashboard
WITH SARCASM DETECTION SUPPORT
Real-time stats display with GROW phase, engagement, digression, sarcasm, and learning style
FULLY CORRECTED VERSION - Ready to use
"""

import streamlit as st
import requests
import websocket
import json
import threading
import time
import html
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import queue

# Page configuration - MUST BE FIRST
st.set_page_config(
    page_title="AI Coaching Observer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CONFIGURATION
# =============================================================================

API_BASE_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws/feedback"

# =============================================================================
# WEBSOCKET CLIENT CLASS - DEFINE BEFORE SESSION STATE
# =============================================================================

class WebSocketClient:
    def __init__(self):
        self.ws = None
        self.connected = False
        self.message_queue = queue.Queue()  # Thread-safe queue
        
    def connect(self):
        """Connect to WebSocket for real-time updates"""
        try:
            self.ws = websocket.WebSocketApp(
                WS_URL,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
                on_open=self.on_open
            )
            threading.Thread(target=self.ws.run_forever, daemon=True).start()
            self.connected = True
            return True
        except Exception as e:
            print(f"WebSocket connection failed: {e}")
            return False
    
    def on_open(self, ws):
        self.connected = True
        print("✅ WebSocket connected")
            
    def on_message(self, ws, message):
        """Handle incoming WebSocket messages - thread-safe"""
        try:
            feedback = json.loads(message)
            self.message_queue.put(feedback)
            print(f"📥 Received feedback: {feedback.get('speaker')} - {feedback.get('transcript', '')[:50]}")
        except Exception as e:
            print(f"Error processing message: {e}")
            
    def on_error(self, ws, error):
        print(f"WebSocket error: {error}")
        
    def on_close(self, ws, close_status_code, close_msg):
        self.connected = False
        print("❌ WebSocket closed")
    
    def get_messages(self):
        """Get all pending messages from queue"""
        messages = []
        try:
            while not self.message_queue.empty():
                messages.append(self.message_queue.get_nowait())
        except queue.Empty:
            pass
        return messages

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
if 'real_time_data' not in st.session_state:
    st.session_state.real_time_data = queue.Queue()
if 'current_grow_phase' not in st.session_state:
    st.session_state.current_grow_phase = "Reality"
if 'current_engagement' not in st.session_state:
    st.session_state.current_engagement = 0.5
if 'current_learning_style' not in st.session_state:
    st.session_state.current_learning_style = "Unknown"
if 'current_digression' not in st.session_state:
    st.session_state.current_digression = 0.0
if 'current_sarcasm' not in st.session_state:
    st.session_state.current_sarcasm = 0.0
if 'sarcasm_detected' not in st.session_state:
    st.session_state.sarcasm_detected = False
if 'ws_client' not in st.session_state:
    st.session_state.ws_client = WebSocketClient()
# Transcript architecture: only final turns go into transcript_history;
# in-progress text lives in current_utterances keyed by speaker.
if 'transcript_history' not in st.session_state:
    st.session_state.transcript_history = []
if 'current_utterances' not in st.session_state:
    st.session_state.current_utterances = {'coach': '', 'coachee': ''}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def start_session():
    """Start a new coaching session"""
    try:
        with st.spinner("Starting session..."):
            response = requests.post(
                f"{API_BASE_URL}/session/start",
                json={"session_type": "live"},
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                # Wipe all prior-session UI state so a new session starts clean.
                # No insights/recommendations/transcripts/metric chips from the
                # previous session may bleed into this one.
                st.session_state.pop('final_report', None)
                st.session_state.feedback_data = []
                st.session_state.transcript_history = []
                st.session_state.current_utterances = {'coach': '', 'coachee': ''}
                st.session_state.current_grow_phase = "Uncertain"
                st.session_state.current_engagement = 0.0
                st.session_state.current_learning_style = "Unknown"
                st.session_state.current_digression = 0.0
                st.session_state.current_sarcasm = 0.0
                st.session_state.sarcasm_detected = False

                st.session_state.session_id = data["session_id"]
                st.session_state.session_active = True

                # Connect WebSocket
                if not st.session_state.ws_client.connected:
                    if st.session_state.ws_client.connect():
                        time.sleep(1)
                        st.session_state.websocket_connected = True
                        
                st.success(f"✅ Session started: {st.session_state.session_id[:8]}...")
                return True
            else:
                error_detail = response.text
                try:
                    error_json = response.json()
                    error_detail = error_json.get("detail", error_detail)
                except:
                    pass
                st.error(f"❌ Failed to start session: {error_detail}")
                return False
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to backend. Ensure it's running on http://localhost:8000")
        return False
    except Exception as e:
        st.error(f"❌ Error starting session: {str(e)}")
        return False

def stop_session():
    """Stop the current session and get report"""
    try:
        with st.spinner("Stopping session and generating report..."):
            response = requests.post(f"{API_BASE_URL}/session/stop", timeout=60)
            if response.status_code == 200:
                report = response.json()
                st.session_state.session_active = False
                st.session_state.websocket_connected = False
                st.success("✅ Session stopped successfully")
                return report
            else:
                st.error(f"❌ Failed to stop session: {response.text}")
                return None
    except Exception as e:
        st.error(f"❌ Error stopping session: {e}")
        return None

def get_session_status():
    """Get current session status"""
    try:
        response = requests.get(f"{API_BASE_URL}/session/status", timeout=2)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def process_real_time_updates():
    """Route WebSocket messages into partial (live) or final (history) buckets."""
    new_data = []

    if 'ws_client' not in st.session_state:
        return new_data

    for msg in st.session_state.ws_client.get_messages():
        msg_type = msg.get('type', 'final')
        speaker  = msg.get('speaker', 'coach')

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

# =============================================================================
# UI COMPONENTS
# =============================================================================

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
            response = requests.get(f"{API_BASE_URL}/health", timeout=2)
            if response.status_code == 200:
                st.sidebar.success("✅ Backend Connected")
            else:
                st.sidebar.error("❌ Backend Unhealthy")
        except:
            st.sidebar.error("❌ Backend Offline")

def render_live_stats_banner():
    """Render prominent live statistics banner with SARCASM"""
    st.markdown("### 📊 Live Session Stats")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        phase = st.session_state.current_grow_phase
        phase_emoji = {"Goal": "🎯", "Reality": "🔍", "Options": "💡", "Way Forward": "🚀", "Uncertain": "❓"}.get(phase, "📍")
        st.metric("GROW Phase", f"{phase_emoji} {phase}", help="Current phase in GROW model")
    
    with col2:
        engagement = st.session_state.current_engagement
        engagement_pct = int(engagement * 100)
        color = "🟢" if engagement > 0.6 else "🟡" if engagement > 0.3 else "🔴"
        delta = f"{engagement_pct-50}%" if engagement != 0.5 else None
        st.metric("Engagement", f"{color} {engagement_pct}%", delta=delta, help="Coachee engagement")
    
    with col3:
        # Topic Focus (inverse of digression)
        digression = st.session_state.current_digression
        focus_score = 1 - digression
        focus_pct = int(focus_score * 100)
        
        if digression < 0.3:
            focus_icon = "🟢"
            focus_label = "Focused"
        elif digression < 0.6:
            focus_icon = "🟡"
            focus_label = "Drifting"
        else:
            focus_icon = "🔴"
            focus_label = "Off-Topic"
        
        st.metric(
            "Topic Focus", 
            f"{focus_icon} {focus_pct}%",
            delta=focus_label,
            help="Conversation focus (lower digression = better)"
        )
    
    with col4:
        # NEW: SARCASM INDICATOR
        sarcasm = st.session_state.current_sarcasm
        sarcasm_pct = int(sarcasm * 100)
        
        if sarcasm < 0.3:
            sarcasm_icon = "🟢"
            sarcasm_label = "Genuine"
        elif sarcasm < 0.6:
            sarcasm_icon = "🟡"
            sarcasm_label = "Possibly Sarcastic"
        else:
            sarcasm_icon = "😏"
            sarcasm_label = "Sarcastic"
        
        st.metric(
            "Tone Authenticity",
            f"{sarcasm_icon} {100-sarcasm_pct}%",
            delta=sarcasm_label,
            help="Detects sarcasm/passive-aggression"
        )
    
    with col5:
        style = st.session_state.current_learning_style
        style_emoji = {"Visual": "👁️", "Auditory": "👂", "Kinesthetic": "✋"}.get(style.split('(')[0].strip(), "❓")
        st.metric("Learning Style", f"{style_emoji} {style}", help="VAK learning preference")
    
    with col6:
        total = len(st.session_state.feedback_data)
        st.metric("Interactions", total, help="Total turns processed")

def render_real_time_feedback():
    """Render real-time feedback section"""
    st.header("🔄 Real-Time Monitoring Dashboard")
    
    process_real_time_updates()
    
    if not st.session_state.feedback_data:
        st.info("⏳ Waiting for session data... Speak into your microphone to see real-time transcription.")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            status = get_session_status()
            if status:
                st.metric("Chunks Processed", status.get('chunks_processed', 0))
        with col2:
            ws_status = "🟢 Connected" if st.session_state.ws_client.connected else "🔴 Disconnected"
            st.metric("WebSocket", ws_status)
        with col3:
            st.metric("Digression", "0%")
        with col4:
            st.metric("Sarcasm", "0%")
        return
    
    # Live Stats Banner
    render_live_stats_banner()
    
    # Sarcasm Alert
    if st.session_state.sarcasm_detected and st.session_state.current_sarcasm > 0.4:
        st.error(f"😏 **Sarcasm Detected!** The current tone may indicate frustration, resistance, or passive-aggression. (Score: {st.session_state.current_sarcasm:.0%})")
    
    # Digression Alert
    digression = st.session_state.current_digression
    if digression > 0.6:
        st.error(f"⚠️ **Conversation is drifting off-topic** (Digression: {digression:.0%})")
    elif digression > 0.4:
        st.warning(f"💡 **Topic focus decreasing** (Digression: {digression:.0%})")
    
    st.markdown("---")
    
    # Split transcript — full width
    st.subheader("💬 Live Conversation")
    render_live_transcript_compact()

    st.subheader("💡 AI Coaching Suggestions")
    render_latest_suggestions()
    
    st.markdown("---")
    st.subheader("📈 Analytics & Trends")
    render_analytics_dashboard()
    
    st.markdown("---")
    col_grow, col_emotions = st.columns([1, 1])
    
    with col_grow:
        render_grow_phases()
    
    with col_emotions:
        render_emotion_tracking()

def _utterance_card(msg: dict, streaming: bool = False) -> str:
    """Return an HTML card for one finalized (or in-progress) utterance."""
    ts        = datetime.fromtimestamp(msg['timestamp']).strftime("%H:%M:%S") if 'timestamp' in msg else ""
    # Escape user-facing text before interpolation — transcripts can contain
    # `<`, `>`, `&` which would otherwise break the surrounding HTML layout.
    transcript = html.escape(msg.get('transcript', '—'))
    digression = msg.get('digression_level', 0.0)
    sarcasm    = msg.get('sarcasm_score', 0.0)
    speaker    = msg.get('speaker', 'coach')
    speaker_id = msg.get('speaker_id')

    if speaker == 'coach':
        border, label_color = '#1565C0', '#1565C0'
        grow   = msg.get('grow_phase', {}).get('phase', '')
        meta   = f"GROW: <b>{grow}</b> &nbsp;·&nbsp; Engagement: {msg.get('engagement_score', 0):.0%}" if grow else ""
    else:
        border, label_color = '#6A1B9A', '#6A1B9A'
        primary_emotion = (
            max(msg.get('emotion_trend', {}).items(), key=lambda x: x[1])[0].title()
            if msg.get('emotion_trend') else ""
        )
        meta = f"Emotion: <b>{primary_emotion}</b> &nbsp;·&nbsp; Interest: {msg.get('engagement_score', 0):.0%}" if primary_emotion else ""

    id_badge   = (f'<span style="font-size:10px;border:1px solid {border};border-radius:3px;'
                  f'padding:0 3px;opacity:.6;margin-left:4px;">Spk {speaker_id.replace("SPEAKER_","")}</span>'
                  if speaker_id else "")
    focus_dot  = "🟢" if digression < 0.3 else ("🟡" if digression < 0.6 else "🔴")
    sarc_badge = "😏" if sarcasm > 0.6 else ("🤨" if sarcasm > 0.4 else "")
    streaming_indicator = ('<span style="display:inline-block;width:8px;height:8px;border-radius:50%;'
                           'background:#4CAF50;margin-left:6px;animation:pulse 1s infinite;"></span>'
                           if streaming else "")

    bg = "#f0f7ff" if not streaming else "#e8f5e9"
    return f"""
    <div style="background:{bg};border-left:4px solid {border};border-radius:6px;
                padding:10px 12px;margin:5px 0;">
        <div style="display:flex;align-items:center;margin-bottom:5px;">
            {id_badge}{streaming_indicator}
            <span style="margin-left:auto;font-size:11px;color:#999;">{focus_dot}{sarc_badge} {ts}</span>
        </div>
        <div style="font-size:14px;line-height:1.5;color:#212121;">{transcript}</div>
        {'<div style="margin-top:4px;font-size:11px;color:#666;">' + meta + '</div>' if meta else ''}
    </div>"""


def render_live_transcript_compact():
    """Split-column transcript: Coach left, Coachee right. Partials stream live."""
    history  = st.session_state.transcript_history[-40:]
    current  = st.session_state.current_utterances  # {'coach': str, 'coachee': str}

    coach_msgs   = [m for m in history if m.get('speaker') == 'coach']
    coachee_msgs = [m for m in history if m.get('speaker') == 'coachee']

    col_coach, col_coachee = st.columns(2)

    def render_column(msgs, speaker, live_text, col):
        icon  = "🎯" if speaker == "coach" else "👤"
        label = "Coach" if speaker == "coach" else "Coachee"
        color = "#1565C0" if speaker == "coach" else "#6A1B9A"
        with col:
            st.markdown(
                f'<div style="text-align:center;padding:6px;background:{"#e3f2fd" if speaker=="coach" else "#f3e5f5"};'
                f'border-radius:6px;margin-bottom:8px;">'
                f'<span style="font-weight:700;font-size:15px;color:{color};">{icon} {label}</span>'
                f'<span style="font-size:12px;color:#666;margin-left:8px;">({len(msgs)} turns)</span>'
                f'</div>',
                unsafe_allow_html=True
            )
            # Newest-first: streaming bubble (in-progress utterance) on top,
            # then finalized messages reversed so the most recent appears next.
            cards = ""
            if live_text:
                cards += _utterance_card(
                    {'speaker': speaker, 'transcript': live_text},
                    streaming=True
                )
            cards += "".join(_utterance_card(m) for m in reversed(msgs[-15:]))
            if not cards:
                cards = '<div style="color:#aaa;text-align:center;padding:20px;font-size:13px;">Waiting…</div>'
            st.html(f'<div style="max-height:520px;overflow-y:auto;padding:4px;">{cards}</div>')

    render_column(coach_msgs,   'coach',   current.get('coach',   ''), col_coach)
    render_column(coachee_msgs, 'coachee', current.get('coachee', ''), col_coachee)

def render_latest_suggestions():
    """Render latest AI coaching suggestions"""
    if not st.session_state.feedback_data:
        st.info("No suggestions yet")
        return
    
    latest = st.session_state.feedback_data[-1]
    suggestions = latest.get('suggestions', [])
    
    if suggestions:
        for suggestion in suggestions:
            # Highlight sarcasm-related suggestions
            if "sarcasm" in suggestion.lower() or "😏" in suggestion:
                st.error(f"🚨 {suggestion}")
            else:
                st.success(f"💡 {suggestion}")
    else:
        st.info("✅ Coaching is on track")
    
    # GROW phase guidance
    phase = st.session_state.current_grow_phase
    phase_guidance = {
        "Goal": "Focus: Help coachee clarify what they want to achieve",
        "Reality": "Focus: Explore the current situation and obstacles",
        "Options": "Focus: Brainstorm possible solutions together",
        "Way Forward": "Focus: Commit to specific actions and next steps"
    }
    
    if phase in phase_guidance:
        st.info(f"📌 {phase_guidance[phase]}")

def render_analytics_dashboard():
    """Render analytics dashboard with charts including sarcasm"""
    if len(st.session_state.feedback_data) < 2:
        st.info("Need more data points for analytics...")
        return
    
    df = pd.DataFrame(st.session_state.feedback_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df['focus_score'] = 1 - df['digression_level']
    df['authenticity_score'] = 1 - df.get('sarcasm_score', 0)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        fig = px.line(df, x='timestamp', y='engagement_score', color='speaker',
                     title='Engagement Over Time',
                     color_discrete_map={'coach': '#1f77b4', 'coachee': '#ff7f0e'})
        fig.update_layout(height=300)
        st.plotly_chart(fig, width="stretch", key="engagement_line")

    with col2:
        fig = px.line(df, x='timestamp', y='focus_score',
                     title='Topic Focus (Higher = Better)',
                     color_discrete_sequence=['#2ca02c'])
        fig.add_hline(y=0.7, line_dash="dash", line_color="green",
                     annotation_text="Good Focus")
        fig.update_layout(height=300)
        st.plotly_chart(fig, width="stretch", key="focus_line")

    with col3:
        # NEW: Sarcasm tracking
        if 'sarcasm_score' in df.columns:
            fig = px.line(df, x='timestamp', y='authenticity_score',
                         title='Tone Authenticity (Higher = Better)',
                         color_discrete_sequence=['#9467bd'])
            fig.add_hline(y=0.7, line_dash="dash", line_color="green",
                         annotation_text="Authentic")
            fig.add_hline(y=0.4, line_dash="dash", line_color="orange",
                         annotation_text="Possibly Sarcastic")
            fig.update_layout(height=300)
            st.plotly_chart(fig, width="stretch", key="authenticity_line")

    with col4:
        avg_engagement = df.groupby('speaker')['engagement_score'].mean()
        fig = px.bar(x=avg_engagement.index, y=avg_engagement.values,
                    title='Avg Engagement by Speaker', color=avg_engagement.index,
                    color_discrete_map={'coach': '#1f77b4', 'coachee': '#ff7f0e'})
        fig.update_layout(height=300)
        st.plotly_chart(fig, width="stretch", key="avg_engagement_bar")

def render_grow_phases():
    """Render GROW model phase tracking"""
    st.subheader("🎯 GROW Model Phases")
    
    if not st.session_state.feedback_data:
        st.info("No GROW phase data available yet...")
        return
    
    grow_data = []
    for feedback in st.session_state.feedback_data:
        if 'grow_phase' in feedback:
            grow_data.append({
                'timestamp': datetime.fromtimestamp(feedback['timestamp']),
                'phase': feedback['grow_phase']['phase'],
                'confidence': feedback['grow_phase']['confidence']
            })
    
    if not grow_data:
        st.info("No GROW phase data processed...")
        return
    
    df_grow = pd.DataFrame(grow_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        phase_counts = df_grow['phase'].value_counts()
        fig = px.pie(values=phase_counts.values, names=phase_counts.index,
                    title='GROW Phase Distribution')
        st.plotly_chart(fig, width="stretch", key="grow_pie")

    with col2:
        fig = px.scatter(df_grow, x='timestamp', y='phase', size='confidence',
                        title='GROW Phase Timeline', color='confidence',
                        color_continuous_scale='Viridis')
        fig.update_layout(height=400)
        st.plotly_chart(fig, width="stretch", key="grow_timeline")

def render_emotion_tracking():
    """Render emotion tracking visualization"""
    st.subheader("😊 Emotional Journey")
    
    if not st.session_state.feedback_data:
        st.info("No emotion data available yet...")
        return
    
    emotion_data = []
    for feedback in st.session_state.feedback_data:
        timestamp = datetime.fromtimestamp(feedback['timestamp'])
        speaker = feedback['speaker']
        emotions = feedback.get('emotion_trend', {})
        
        for emotion, score in emotions.items():
            emotion_data.append({
                'timestamp': timestamp,
                'speaker': speaker,
                'emotion': emotion,
                'score': score
            })
    
    if not emotion_data:
        st.info("No emotion data processed yet...")
        return
    
    df_emotions = pd.DataFrame(emotion_data)
    
    fig = px.line(df_emotions, x='timestamp', y='score', color='emotion',
                 facet_col='speaker', title='Emotional Trends Over Time')
    fig.update_layout(height=400)
    st.plotly_chart(fig, width="stretch", key="emotion_facets")

def render_session_report():
    """Render final session report"""
    st.header("📋 Session Report")

    # Never render a report while a new session is in progress — even if a
    # stale final_report somehow lingers, the live view owns the screen.
    if st.session_state.session_active:
        return

    if 'final_report' not in st.session_state:
        st.info("Complete a session to generate a report...")
        return
    
    report = st.session_state.final_report.get('report', st.session_state.final_report)
    
    st.subheader(f"Session: {report.get('session_id', 'Unknown')}")
    st.write(f"Duration: {report.get('duration_minutes', 0):.1f} minutes")
    
    col1, col2, col3 = st.columns(3)

    eff = report.get('coaching_effectiveness') or {}
    def _fmt(metric_key):
        v = eff.get(metric_key)
        return f"{v:.2f}" if isinstance(v, (int, float)) and v > 0 else "Not Available"

    with col1:
        st.metric("Overall Effectiveness", _fmt('overall'))
    with col2:
        st.metric("Questioning Quality", _fmt('questioning'))
    with col3:
        st.metric("Listening Quality", _fmt('listening'))

    # Surface the wired-in sarcasm & digression rollups
    sarc = report.get('sarcasm_summary') or {}
    dig  = report.get('digression_summary') or {}
    if sarc or dig:
        st.subheader("🔎 Conversation Signals")
        sc1, sc2 = st.columns(2)
        with sc1:
            if sarc:
                st.write(f"**Sarcasm detected:** {sarc.get('count_detected', 0)} of {sarc.get('total_evaluated', 0)} turns "
                         f"(avg score {sarc.get('average_score', 0):.2f}, peak {sarc.get('max_score', 0):.2f})")
                if sarc.get('by_type'):
                    st.write("Types: " + ", ".join(f"{k}={v}" for k, v in sarc['by_type'].items()))
            else:
                st.write("**Sarcasm:** Not Available")
        with sc2:
            if dig:
                st.write(f"**Off-topic moments:** {dig.get('off_topic_moments', 0)} of {dig.get('total_evaluated', 0)} turns "
                         f"(avg {dig.get('average_score', 0):.2f}, peak {dig.get('max_score', 0):.2f})")
            else:
                st.write("**Digression:** Not Available")

    # Learning style (real VAK if available, else "Insufficient Data")
    vak = report.get('learning_style_analysis') or {}
    if vak:
        st.subheader("👁️👂✋ Learning Style (VAK)")
        v1, v2, v3 = st.columns(3)
        v1.metric("Visual", f"{vak.get('visual', 0):.0%}")
        v2.metric("Auditory", f"{vak.get('auditory', 0):.0%}")
        v3.metric("Kinesthetic", f"{vak.get('kinesthetic', 0):.0%}")
    else:
        st.subheader("👁️👂✋ Learning Style (VAK)")
        st.info("Insufficient Data")
    
    st.subheader("🔍 Key Insights")
    for insight in report.get('key_insights', []):
        st.write(f"• {insight}")
    
    st.subheader("💡 Recommendations")
    for rec in report.get('recommendations', []):
        st.write(f"• {rec}")
    
    st.subheader("📝 Summary")
    st.write(report.get('transcript_summary', 'No summary available'))
    
    if st.button("📥 Download Report"):
        report_json = json.dumps(report, indent=2, default=str)
        st.download_button(
            label="Download JSON Report",
            data=report_json,
            file_name=f"coaching_report_{report.get('session_id', 'unknown')}.json",
            mime="application/json"
        )

def render_settings():
    """Render settings panel"""
    st.sidebar.header("⚙️ Settings")
    
    with st.sidebar.expander("🔧 API Configuration"):
        st.text_input("Backend URL", value=API_BASE_URL, disabled=True)
        st.text_input("WebSocket URL", value=WS_URL, disabled=True)
        
        if st.button("🔄 Reconnect WebSocket"):
            if st.session_state.ws_client.connect():
                st.success("WebSocket reconnected!")
    
    with st.sidebar.expander("🎨 Display Settings"):
        st.checkbox("Auto-refresh data", value=True, key="auto_refresh")
        if st.session_state.get("auto_refresh", True) and st.session_state.session_active:
            st.info("🔄 Auto-refreshing every 1.5 seconds")

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
        
        # Auto-refresh
        if st.session_state.get("auto_refresh", True):
            time.sleep(1.5)
            st.rerun()
    else:
        render_session_report()
        
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

if __name__ == "__main__":
    main()
