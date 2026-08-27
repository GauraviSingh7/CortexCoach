"""Analytics charts built from the accumulated feedback stream."""

from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st


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
