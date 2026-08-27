"""Final session report view."""

import json

import streamlit as st

from ui.panels import (
    render_grow_coverage,
    render_model_status,
    render_provenance,
)


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
    
    if st.session_state.get("report_is_stale"):
        st.warning(
            "⚠️ This report is from an earlier session - the backend had no "
            "session running when it was requested."
        )

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
    
    coverage = report.get('grow_coverage') or {}
    if coverage:
        st.subheader("🎯 GROW Coverage")
        render_grow_coverage(coverage)

    st.subheader("🔍 Key Insights")
    for insight in report.get('key_insights', []):
        st.write(f"• {insight}")
    
    st.subheader("💡 Recommendations")
    for rec in report.get('recommendations', []):
        st.write(f"• {rec}")
    
    st.subheader("📝 Summary")
    st.write(report.get('transcript_summary', 'No summary available'))
    
    render_provenance(report.get('analysis_sources') or {})

    model_status = report.get('model_status') or {}
    if model_status:
        with st.expander("🔬 Which models produced these numbers?", expanded=False):
            render_model_status(model_status)

    if st.button("📥 Download Report"):
        report_json = json.dumps(report, indent=2, default=str)
        st.download_button(
            label="Download JSON Report",
            data=report_json,
            file_name=f"coaching_report_{report.get('session_id', 'unknown')}.json",
            mime="application/json"
        )
