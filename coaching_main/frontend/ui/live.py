"""Live session views: stats banner, transcript and suggestions."""

import html
from datetime import datetime

import streamlit as st

from ui.analytics import (
    render_analytics_dashboard,
    render_emotion_tracking,
    render_grow_phases,
)
from ui.session import get_session_status, process_real_time_updates


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

    if st.session_state.get("playback_finished"):
        error = st.session_state.get("playback_error")
        if error:
            st.error(
                "⚠️ **Playback stopped early:** " + str(error)
                + " — press **⏹️ Stop Session** in the sidebar to "
                "close the session."
            )
        else:
            turns = len(st.session_state.feedback_data)
            st.success(
                f"🏁 **Playback complete** — {turns} turns analysed. "
                "Press **⏹️ Stop Session** in the sidebar to generate the "
                "report. Live updates have stopped."
            )

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
        border = '#1565C0'
        grow   = msg.get('grow_phase', {}).get('phase', '')
        meta   = f"GROW: <b>{grow}</b> &nbsp;·&nbsp; Engagement: {msg.get('engagement_score', 0):.0%}" if grow else ""
    else:
        border = '#6A1B9A'
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
