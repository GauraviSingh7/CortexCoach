import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
import json
import uuid

# Add project root to path for imports
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.coaching_rag_system import GROWPhase, VARKType
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class EmotionAnalysisComponent:
    """Component for displaying emotion analysis data"""
    
    def __init__(self):
        self.emotion_colors = {
            'happy': '#2E8B57',      # Sea Green
            'sad': '#4169E1',        # Royal Blue
            'angry': '#DC143C',      # Crimson
            'surprise': '#FF8C00',   # Dark Orange
            'fear': '#9370DB',       # Medium Purple
            'disgust': '#8B4513',    # Saddle Brown
            'neutral': '#708090'     # Slate Gray
        }
    
    def update_real_time_emotion_chart(self, chart_placeholder, text_placeholder, emotion_data: Dict[str, float]):
        """Update a persistent chart and status text separately"""
        if not emotion_data:
            chart_placeholder.empty()
            text_placeholder.info("No emotion data available")
            return

        emotions = list(emotion_data.keys())
        values = list(emotion_data.values())

        fig = go.Figure(data=[
            go.Bar(
                x=values,
                y=emotions,
                orientation='h',
                marker_color=[self.emotion_colors.get(emotion, '#808080') for emotion in emotions],
                text=[f'{value:.2f}' for value in values],
                textposition='auto'
            )
        ])

        fig.update_layout(
            title="😊 Real-Time Facial Emotion",
            xaxis_title="Confidence",
            yaxis_title="Emotions",
            height=300,
            margin=dict(l=10, r=10, t=30, b=10)
        )

        chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"chart_{uuid.uuid4()}")

        dominant_emotion = max(emotion_data, key=emotion_data.get)
        confidence = emotion_data[dominant_emotion]

        status = (
            f"Dominant emotion: **{dominant_emotion.title()}** ({confidence:.1%})"
            if confidence > 0.5 else
            f"Mixed emotions detected (highest: {dominant_emotion.title()} at {confidence:.1%})"
        )
        text_placeholder.markdown(status)

    
    def render_emotion_timeline(self, emotion_history: List[Dict]):
        """Render emotion timeline over the session"""
        
        if not emotion_history:
            st.info("No emotion history available")
            return
        
        st.subheader("📈 Emotion Timeline")
        
        # Prepare data for timeline
        timestamps = []
        emotions_data = {emotion: [] for emotion in self.emotion_colors.keys()}
        
        for entry in emotion_history[-20:]:  # Last 20 entries
            timestamps.append(entry['timestamp'])
            
            # Get dominant emotion for this timestamp
            facial_emotion = entry.get('facial_emotion', {})
            if facial_emotion:
                dominant = max(facial_emotion, key=facial_emotion.get)
                for emotion in emotions_data:
                    emotions_data[emotion].append(
                        facial_emotion.get(emotion, 0) if emotion == dominant else 0
                    )
            else:
                for emotion in emotions_data:
                    emotions_data[emotion].append(0)
        
        # Create timeline chart
        fig = go.Figure()
        
        for emotion, values in emotions_data.items():
            if any(v > 0 for v in values):  # Only show emotions that were detected
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=values,
                    mode='lines+markers',
                    name=emotion.title(),
                    line=dict(color=self.emotion_colors[emotion], width=2),
                    marker=dict(size=6)
                ))
        
        fig.update_layout(
            title="Emotion Changes Over Time",
            xaxis_title="Time",
            yaxis_title="Confidence",
            height=400,
            hovermode='x unified',
            margin=dict(l=10, r=10, t=30, b=10)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_live_emotion_placeholder(self):
        """Create a persistent emotion chart area and return update handle"""
        return st.empty()

    def render_aggregated_emotion_bars(self, frame_history: List[Dict[str, float]], chunk_size: int = 40):
        """Render bar chart showing dominant emotion over time (averaged chunks)"""

        if not frame_history or len(frame_history) < chunk_size:
            st.info("Not enough data to display emotion summary yet.")
            return

        chunked_results = []
        num_chunks = len(frame_history) // chunk_size

        for i in range(num_chunks):
            chunk = frame_history[i * chunk_size : (i + 1) * chunk_size]
            averaged = self.compute_average_emotion(chunk)
            if averaged:
                top_emotion = max(averaged, key=averaged.get)
                confidence = averaged[top_emotion]
                chunked_results.append({
                    'index': i,
                    'dominant_emotion': top_emotion,
                    'confidence': confidence,
                    'color': self.emotion_colors.get(top_emotion, '#888888')
                })

        if not chunked_results:
            st.info("No dominant emotion data available.")
            return

        fig = go.Figure(data=[
            go.Bar(
                x=[r['index'] for r in chunked_results],
                y=[r['confidence'] for r in chunked_results],
                marker_color=[r['color'] for r in chunked_results],
                text=[r['dominant_emotion'].title() for r in chunked_results],
                textposition='outside',
                hovertext=[
                    f"{r['dominant_emotion'].title()} ({r['confidence']:.2f})"
                    for r in chunked_results
                ],
                hoverinfo="text"
            )
        ])

        fig.update_layout(
            title="🧠 Dominant Emotions (Live)",
            xaxis_title="Chunk Index",
            yaxis_title="Confidence",
            height=250, # 👈 smaller height for sidebar
            margin=dict(l=5, r=5, t=30, b=5),
            font=dict(size=10)  # Optional: smaller font
        )

        st.plotly_chart(fig, use_container_width=True, key=f"emotion_bar_{uuid.uuid4()}")



    @staticmethod
    def compute_average_emotion(history: List[Dict[str, float]]) -> Dict[str, float]:
        if not history:
            logger.warning("[AVERAGING] No history available")
            return {}

        try:
            avg_scores = {k: 0.0 for k in history[0]}
            for frame in history:
                for k, v in frame.items():
                    avg_scores[k] += v

            count = len(history)
            result = {k: v / count for k, v in avg_scores.items()}

            top = max(result, key=result.get)
            #logger.info(f"[AVERAGING] Using {count} frames. Dominant: {top} ({result[top]:.2f})")
            return result
        except Exception as e:
            logger.error(f"[AVERAGING ERROR] Failed to average emotions: {e}")
            return {}
        

    # def render_voice_emotion(self, voice_emotion_data: Dict[str, float]):
    #     """Render voice emotion analysis"""
        
    #     if not voice_emotion_data:
    #         st.info("No voice emotion data available")
    #         return
        
    #     st.subheader("🎤 Voice Emotion Analysis")
        
    #     # Voice emotion metrics
    #     col1, col2, col3 = st.columns(3)
        
    #     with col1:
    #         valence = voice_emotion_data.get('valence', 0.5)
    #         st.metric(
    #             "Valence",
    #             f"{valence:.2f}",
    #             delta=f"{valence - 0.5:.2f}" if valence != 0.5 else None,
    #             help="Positive (happy) vs Negative (sad) emotion"
    #         )
        
    #     with col2:
    #         arousal = voice_emotion_data.get('arousal', 0.5)
    #         st.metric(
    #             "Arousal",
    #             f"{arousal:.2f}",
    #             delta=f"{arousal - 0.5:.2f}" if arousal != 0.5 else None,
    #             help="High (excited) vs Low (calm) energy"
    #         )
        
    #     with col3:
    #         dominance = voice_emotion_data.get('dominance', 0.5)
    #         st.metric(
    #             "Dominance",
    #             f"{dominance:.2f}",
    #             delta=f"{dominance - 0.5:.2f}" if dominance != 0.5 else None,
    #             help="Strong (confident) vs Weak (submissive) emotion"
    #         )
        
    #     # Voice emotion radar chart
    #     if len(voice_emotion_data) > 3:
    #         categories = list(voice_emotion_data.keys())
    #         values = list(voice_emotion_data.values())
            
    #         fig = go.Figure()
            
    #         fig.add_trace(go.Scatterpolar(
    #             r=values,
    #             theta=categories,
    #             fill='toself',
    #             name='Voice Emotion',
    #             line_color='#FF6B6B'
    #         ))
            
    #         fig.update_layout(
    #             polar=dict(
    #                 radialaxis=dict(
    #                     visible=True,
    #                     range=[0, 1]
    #                 )
    #             ),
    #             showlegend=False,
    #             height=300,
    #             margin=dict(l=10, r=10, t=30, b=10)
    #         )
            
    #         st.plotly_chart(fig, use_container_width=True)

    
    # def init_emotion_bar_widget(self, emotions: List[str]) -> go.FigureWidget:
    #     """Initialize persistent emotion bar chart with FigureWidget (for smooth updates)"""
    #     initial_values = [0.0] * len(emotions)

    #     fig = go.FigureWidget(
    #         data=[go.Bar(
    #             x=initial_values,
    #             y=emotions,
    #             orientation='h',
    #             marker_color=[self.emotion_colors.get(e, '#808080') for e in emotions],
    #             text=[f'{v:.2f}' for v in initial_values],
    #             textposition='auto'
    #         )]
    #     )

    #     fig.update_layout(
    #         title="😊 Real-Time Facial Emotion",
    #         xaxis_title="Confidence",
    #         yaxis_title="Emotions",
    #         height=300,
    #         margin=dict(l=10, r=10, t=30, b=10),
    #         showlegend=False
    #     )

    #     return fig
    
    # def update_emotion_bar_widget(self, fig_widget: go.FigureWidget, emotion_data: Dict[str, float]):
    #     """Efficiently update an existing Plotly FigureWidget (no flicker)"""
    #     if not fig_widget or not emotion_data:
    #         return

    #     # Use bar's existing y-values to keep order consistent
    #     emotions = fig_widget.data[0].y
    #     values = [emotion_data.get(emotion, 0.0) for emotion in emotions]

    #     fig_widget.data[0].x = values
    #     fig_widget.data[0].text = [f'{v:.2f}' for v in values]

    # def render_dominant_emotion_text(self, text_placeholder, emotion_data: Dict[str, float]):
    #     """Render dominant emotion summary separately"""
    #     if not emotion_data:
    #         text_placeholder.info("No emotion detected.")
    #         return

    #     dominant = max(emotion_data, key=emotion_data.get)
    #     confidence = emotion_data[dominant]

    #     if confidence > 0.5:
    #         text_placeholder.success(f"Dominant emotion: **{dominant.title()}** ({confidence:.1%})")
    #     else:
    #         text_placeholder.info(f"Mixed emotions detected (highest: {dominant.title()} at {confidence:.1%})")


class GROWPhaseTracker:
    """Component for tracking GROW model phases"""

    def __init__(self):
        self.phase_colors = {
            GROWPhase.GOAL: '#FF6B6B',
            GROWPhase.REALITY: '#4ECDC4',
            GROWPhase.OPTIONS: '#45B7D1',
            GROWPhase.WILL: '#96CEB4'
        }

        self.phase_icons = {
            GROWPhase.GOAL: '🎯',
            GROWPhase.REALITY: '🔍',
            GROWPhase.OPTIONS: '💡',
            GROWPhase.WILL: '✅'
        }

        self.phase_descriptions = {
            GROWPhase.GOAL: "What do you want to achieve?",
            GROWPhase.REALITY: "What is your current situation?",
            GROWPhase.OPTIONS: "What options do you have?",
            GROWPhase.WILL: "What will you do?"
        }

    def render(self, current_phase: GROWPhase, phase_history: List[Dict]):
        """Render basic GROW phase tracker with progress bar and icons"""

        phases = [GROWPhase.GOAL, GROWPhase.REALITY, GROWPhase.OPTIONS, GROWPhase.WILL]
        current_index = phases.index(current_phase)

        # Phase progress bar
        st.progress((current_index + 1) / len(phases))

        # Current phase display
        st.markdown(f"""
        **Current Phase:** {self.phase_icons[current_phase]} {current_phase.value.title()}
        
        *{self.phase_descriptions[current_phase]}*
        """)

        # Completed Phases
        if phase_history:
            completed = [p for p in phase_history if p.get("completed")]
            if completed:
                st.write("**Completed Phases:**")
                for phase_info in completed:
                    phase = phase_info.get('phase')
                    if isinstance(phase, GROWPhase):
                        icon = self.phase_icons.get(phase, "✅")
                        st.write(f"{icon} {phase.value.title()} ✓")

    def render_detailed_tracker(self, current_phase: GROWPhase, phase_history: List[Dict]):
        """Render advanced tracker with visual timeline and metadata"""

        st.subheader("🎯 GROW Model Progress")

        phases = [GROWPhase.GOAL, GROWPhase.REALITY, GROWPhase.OPTIONS, GROWPhase.WILL]
        fig = go.Figure()

        # Completed phases
        for i, phase_info in enumerate(phase_history):
            phase = phase_info.get('phase')
            completed_at = phase_info.get('completed_at')
            turns_spent = phase_info.get('turns_spent')

            if isinstance(phase, GROWPhase) and completed_at:
                color = self.phase_colors.get(phase, '#888')
                fig.add_trace(go.Scatter(
                    x=[i],
                    y=[phases.index(phase)],
                    mode='markers',
                    marker=dict(size=20, color=color, symbol='circle'),
                    name=f"{phase.value.title()} (Completed)",
                    hovertemplate=f"<b>{phase.value.title()}</b><br>" +
                                  f"Completed: {completed_at.strftime('%H:%M')}<br>" +
                                  f"Turns: {turns_spent}<extra></extra>"
                ))

        # Current phase marker
        current_index = phases.index(current_phase)
        fig.add_trace(go.Scatter(
            x=[len(phase_history)],
            y=[current_index],
            mode='markers',
            marker=dict(
                size=25,
                color=self.phase_colors.get(current_phase, '#999'),
                symbol='diamond',
                line=dict(width=3, color='white')
            ),
            name=f"{current_phase.value.title()} (Current)"
        ))

        fig.update_layout(
            title="GROW Phase Progression",
            xaxis_title="Session Progress",
            yaxis=dict(
                tickmode='array',
                tickvals=list(range(len(phases))),
                ticktext=[f"{self.phase_icons[ph]} {ph.value.title()}" for ph in phases]
            ),
            height=300,
            showlegend=False,
            margin=dict(l=10, r=10, t=30, b=10)
        )

        st.plotly_chart(fig, use_container_width=True)



class SessionMetricsComponent:
    """Component for displaying session metrics and analytics"""
    
    def render_mini_metrics(self, metrics: Dict[str, Any]):
        """Render condensed metrics for sidebar"""
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Turns", metrics.get('total_turns', 0))
            
        with col2:
            goal_clarity = metrics.get('goal_clarity', 0)
            st.metric("Goal Clarity", f"{goal_clarity:.1%}")
        
        # Current phase
        current_phase = metrics.get('current_phase', 'goal')
        st.write(f"**Phase:** {current_phase.title()}")
    
    def render_detailed_metrics(self, metrics: Dict[str, Any]):
        """Render detailed session metrics"""
        
        st.subheader("📊 Session Metrics")
        
        # Key metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            duration = metrics.get('session_duration', 0)
            st.metric("Duration", f"{duration} min")
            
        with col2:
            total_turns = metrics.get('total_turns', 0)
            st.metric("Total Turns", total_turns)
            
        with col3:
            engagement = metrics.get('engagement_score', 0)
            st.metric("Engagement", f"{engagement:.1%}")
            
        with col4:
            phase_info = metrics.get('phase_progression', {})
            progress = phase_info.get('progress_percentage', 0)
            st.metric("Progress", f"{progress:.0f}%")
        
        
        st.divider()
        
        # Phase completion analysis
        if 'phase_progression' in metrics:
            self.render_phase_completion_chart(metrics['phase_progression'])
        
        # Engagement trend
        if 'engagement_history' in metrics:
            self.render_engagement_trend(metrics['engagement_history'])
    
    def render_phase_completion_chart(self, phase_data: Dict):
        """Render phase completion visualization"""
        
        st.subheader("📈 Phase Completion")
        
        phases = ['Goal', 'Reality', 'Options', 'Will']
        completed = phase_data.get('completed_phases', 0)
        current_phase = phase_data.get('current_phase', 'goal')
        
        # Create completion status
        status = []
        colors = []
        
        for i, phase in enumerate(phases):
            if i < completed:
                status.append('Completed')
                colors.append('#28a745')  # Green
            elif phase.lower() == current_phase:
                status.append('In Progress')
                colors.append('#ffc107')  # Yellow
            else:
                status.append('Pending')
                colors.append('#e9ecef')  # Gray
        
        fig = go.Figure(data=[
            go.Bar(
                x=phases,
                y=[1] * len(phases),
                marker_color=colors,
                text=status,
                textposition='inside',
                hovertemplate='<b>%{x}</b><br>Status: %{text}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title="GROW Phase Status",
            xaxis_title="Phases",
            yaxis=dict(visible=False),
            height=200,
            margin=dict(l=10, r=10, t=30, b=10)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_engagement_trend(self, engagement_history: List[float]):
        """Render engagement level trend"""
        
        if not engagement_history:
            return
            
        st.subheader("📊 Engagement Trend")
        
        fig = px.line(
            y=engagement_history,
            title="Interest Level Over Time",
            labels={'y': 'Engagement Level', 'index': 'Interaction'}
        )
        
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", 
                     annotation_text="Baseline")
        
        fig.update_layout(
            height=250,
            margin=dict(l=10, r=10, t=30, b=10)
        )
        
        st.plotly_chart(fig, use_container_width=True)


class VARKStyleAnalyzer:
    """Component for VARK learning style analysis"""
    
    def __init__(self):
        self.vark_colors = {
            'visual': '#FF6B6B',
            'auditory': '#4ECDC4',
            'reading': '#45B7D1',
            'kinesthetic': '#96CEB4'
        }
        
        self.vark_descriptions = {
            'visual': "Learns best through images, diagrams, and visual aids",
            'auditory': "Learns best through discussion and verbal explanation",
            'reading': "Learns best through written materials and text",
            'kinesthetic': "Learns best through hands-on activities and movement"
        }
        
        self.vark_icons = {
            'visual': '👁️',
            'auditory': '👂',
            'reading': '📖',
            'kinesthetic': '🤲'
        }
    
    def render_mini_profile(self, vark_profile: Dict[str, float]):
        """Render compact VARK profile for sidebar"""
        
        dominant_style = max(vark_profile, key=vark_profile.get)
        confidence = vark_profile[dominant_style]
        
        st.write("**Learning Style:**")
        st.write(f"{self.vark_icons[dominant_style]} {dominant_style.title()}")
        st.write(f"Confidence: {confidence:.1%}")
        
        # Mini bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=list(vark_profile.values()),
                y=list(vark_profile.keys()),
                orientation='h',
                marker_color=[self.vark_colors[style] for style in vark_profile.keys()],
                text=[f'{value:.1%}' for value in vark_profile.values()],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            height=200,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_detailed_analysis(self, vark_profile: Dict[str, float]):
        """Render detailed VARK analysis"""
        
        #st.subheader("🎨 Learning Style Analysis")
        
        # VARK radar chart
        categories = list(vark_profile.keys())
        values = list(vark_profile.values())
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=[style.title() for style in categories],
            fill='toself',
            name='VARK Profile',
            line_color='#FF6B6B',
            fillcolor='rgba(255, 107, 107, 0.3)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=False,
            height=400,
            margin=dict(l=10, r=10, t=30, b=10)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Dominant style information
        dominant_style = max(vark_profile, key=vark_profile.get)
        confidence = vark_profile[dominant_style]
        
        st.info(f"""
        **Dominant Learning Style:** {self.vark_icons[dominant_style]} {dominant_style.title()} ({confidence:.1%})
        
        {self.vark_descriptions[dominant_style]}
        """)
        
        # Style breakdown
        st.subheader("Style Breakdown")
        
        for style, score in sorted(vark_profile.items(), key=lambda x: x[1], reverse=True):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"{self.vark_icons[style]} **{style.title()}**")
                st.progress(score)
                
            with col2:
                st.write(f"{score:.1%}")


class ChatInterface:
    """Component for chat interface"""
    
    def __init__(self):
        self.role_icons = {
            'user': '👤',
            'assistant': '🤖',
            'system': '⚙️'
        }
        
        self.phase_colors = {
            'goal': '#FF6B6B',
            'reality': '#4ECDC4',
            'options': '#45B7D1',
            'will': '#96CEB4'
        }

        self._sarcasm_component = None
    
    def render(self, conversation_history: List[Dict], on_message_submit: Callable[[str], None]):
        """Render chat interface"""
        
        # Chat history
        self.render_chat_history(conversation_history)
        
        # Message input
        self.render_message_input(on_message_submit)
    
    def render_chat_history(self, conversation_history: List[Dict]):
        """Render chat message history"""

        print("DEBUG: conversation_history =", conversation_history)

        # Chat container
        chat_container = st.container()
        
        with chat_container:
            if not conversation_history:
                st.info("Start the conversation by typing a message below!")
                return
            
            for message in conversation_history:
                role = message['role']
                content = message['content']
                timestamp = message.get('timestamp', datetime.now())
                phase = message.get('phase', 'goal')
                
                # Message container
                with st.container():
                    col1, col2 = st.columns([1, 10])
                    
                    with col1:
                        st.write(self.role_icons.get(role, '💬'))
                    
                    with col2:
                        # Message header
                        st.caption(f"{role.title()} • {timestamp.strftime('%H:%M')} • {phase.title()}")
                        
                        # Message content
                        if role == 'system':
                            st.info(content)
                        elif role == 'user':
                            st.write(f"**{content}**")
                        else:  # assistant
                            st.write(content)
                            
                            # **ADD SARCASM DETECTION HERE**
                            if 'context_data' in message:
                                context_data = message['context_data']
                                sarcasm_confidence = context_data.get('sarcasm_confidence', 0.0)
                                sarcasm_detected = context_data.get('sarcasm_detected', False)
                                
                                # Show sarcasm indicator if confidence > 0.3
                                if sarcasm_confidence >= 0.3:
                                    # Initialize component if needed
                                    if self._sarcasm_component is None:
                                        from app.ui.components import SarcasmDetectionComponent
                                        self._sarcasm_component = SarcasmDetectionComponent()
                                    
                                    # Compact inline indicator
                                    icon = "😏" if sarcasm_detected else "🤔"
                                    status = "Sarcastic" if sarcasm_detected else "Uncertain tone"
                                    color = "#FF6B6B" if sarcasm_detected else "#FFD93D"
                                    
                                    st.markdown(f"""
                                    <div style="display: inline-block; padding: 4px 12px; margin: 4px 0; 
                                         border-radius: 12px; background-color: {color}20; border: 1px solid {color}; 
                                         font-size: 12px; color: {color};">
                                        {icon} {status} ({sarcasm_confidence:.0%})
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Expandable detailed view
                                    with st.expander("🎭 Sarcasm Analysis", expanded=False):
                                        self._sarcasm_component.render(
                                            is_sarcastic=sarcasm_detected,
                                            confidence=sarcasm_confidence,
                                            text=content
                                        )
                            
                            # Show context data if available
                            if 'context_data' in message:
                                with st.expander("Context Data", expanded=False):
                                    context_data = message['context_data']
                                    
                                    col1, col2, col3, col4 = st.columns(4)
                                    
                                    with col1:
                                        emotion = context_data.get('dominant_emotion', 'neutral')
                                        st.write(f"**Emotion:** {emotion}")

                                        st.markdown("<br>", unsafe_allow_html=True) # Adds a small vertical space
                                        digression_score = context_data.get('digression_score', 0)
                                        if digression_score > 0.0:
                                            st.write(f"**Digression:** {digression_score:.1%}")
                                    
                                    with col2:
                                        interest = context_data.get('interest_level', 0)
                                        st.write(f"**Interest:** {interest:.1%}")
                                    
                                    with col3:
                                        vark = context_data.get('vark_type', 'visual')
                                        st.write(f"**Style:** {vark}")
                                    with col4:
                                        sarcasm_conf = context_data.get('sarcasm_confidence', 0)
                                        if sarcasm_conf > 0.0:
                                            st.write(f"**Sarcasm:** {sarcasm_conf:.1%}")
                    
                st.divider()
    
    def render_message_input(self, on_message_submit: Callable[[str], None]):
        """Render message input interface"""
        
        # Message input form
        with st.form(key="message_form", clear_on_submit=True):
            col1, col2 = st.columns([4, 1])
            
            with col1:
                user_message = st.text_area(
                    "Your response:",
                    placeholder="Type your message here...",
                    height=100,
                    key="user_input"
                )
            
            with col2:
                st.write("")  # Spacing
                submit_button = st.form_submit_button(
                    "Send",
                    type="primary",
                    use_container_width=True
                )
            
            if submit_button and user_message.strip():
                on_message_submit(user_message.strip())


class FeedbackCollector:
    """Component for collecting user feedback"""
    
    def render_feedback_form(self, context_id: str, on_feedback_submit: Callable[[Dict], None]):
        """Render feedback collection form"""
        
        st.subheader("📝 Feedback")
        
        with st.form(key=f"feedback_{context_id}"):
            # Effectiveness rating
            effectiveness = st.slider(
                "How helpful was this response?",
                min_value=1,
                max_value=5,
                value=3,
                help="1 = Not helpful, 5 = Very helpful"
            )
            
            # Specific feedback categories
            col1, col2 = st.columns(2)
            
            with col1:
                clarity = st.select_slider(
                    "Clarity",
                    options=["Poor", "Fair", "Good", "Excellent"],
                    value="Good"
                )
                
                relevance = st.select_slider(
                    "Relevance",
                    options=["Poor", "Fair", "Good", "Excellent"],
                    value="Good"
                )
            
            with col2:
                empathy = st.select_slider(
                    "Empathy",
                    options=["Poor", "Fair", "Good", "Excellent"],
                    value="Good"
                )
                
                actionability = st.select_slider(
                    "Actionability",
                    options=["Poor", "Fair", "Good", "Excellent"],
                    value="Good"
                )
            
            # Free text feedback
            feedback_text = st.text_area(
                "Additional comments (optional):",
                placeholder="What would make this response better?",
                height=80
            )
            
            # Submit feedback
            if st.form_submit_button("Submit Feedback"):
                feedback_data = {
                    'context_id': context_id,
                    'effectiveness_score': effectiveness / 5.0,  # Normalize to 0-1
                    'clarity': clarity,
                    'relevance': relevance,
                    'empathy': empathy,
                    'actionability': actionability,
                    'feedback_text': feedback_text,
                    'timestamp': datetime.now()
                }
                
                on_feedback_submit(feedback_data)
                st.success("Thank you for your feedback!")
    
    def render_quick_feedback(self, context_id: str, on_feedback_submit: Callable[[Dict], None]):
        """Render quick feedback buttons"""
        
        st.write("**Quick feedback:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("👍 Helpful", key=f"helpful_{context_id}"):
                feedback_data = {
                    'context_id': context_id,
                    'effectiveness_score': 0.8,
                    'feedback_type': 'helpful',
                    'timestamp': datetime.now()
                }
                on_feedback_submit(feedback_data)
                st.success("Thanks!")
        
        with col2:
            if st.button("👎 Not helpful", key=f"not_helpful_{context_id}"):
                feedback_data = {
                    'context_id': context_id,
                    'effectiveness_score': 0.2,
                    'feedback_type': 'not_helpful',
                    'timestamp': datetime.now()
                }
                on_feedback_submit(feedback_data)
                st.info("We'll improve!")
        
        with col3:
            if st.button("💡 Suggestion", key=f"suggestion_{context_id}"):
                # Show text input for suggestion
                suggestion = st.text_input(
                    "Your suggestion:",
                    key=f"suggestion_text_{context_id}"
                )
                
                if suggestion:
                    feedback_data = {
                        'context_id': context_id,
                        'effectiveness_score': 0.5,
                        'feedback_type': 'suggestion',
                        'feedback_text': suggestion,
                        'timestamp': datetime.now()
                    }
                    on_feedback_submit(feedback_data)
                    st.success("Suggestion noted!")

class SarcasmDetectionComponent:
    """Component for displaying sarcasm detection results"""

    def __init__(self):
        self.sarcasm_colors = {
            'detected': '#FF6B6B',      # Red for sarcasm
            'not_detected': '#96CEB4',   # Green for normal
            'uncertain': '#FFD93D'       # Yellow for uncertain
        }
        
        self.sarcasm_icons = {
            'detected': '😏',
            'not_detected': '😊', 
            'uncertain': '🤔'
        }

    def render(self, is_sarcastic: bool, confidence: float, text: str = ""):
        """Render basic sarcasm detection display"""
        
        # Determine status
        if confidence > 0.7:
            status = 'detected' if is_sarcastic else 'not_detected'
        else:
            status = 'uncertain'
        
        icon = self.sarcasm_icons[status]
        color = self.sarcasm_colors[status]
        
        # Main display
        st.markdown(f"""
        <div style="padding: 10px; border-left: 4px solid {color}; background-color: rgba({self._hex_to_rgb(color)}, 0.1);">
            <h4>{icon} Sarcasm Detection</h4>
            <p><strong>Status:</strong> {'Sarcasm Detected' if is_sarcastic else 'Normal Tone'}</p>
            <p><strong>Confidence:</strong> {confidence:.1%}</p>
        </div>
        """, unsafe_allow_html=True)

    def render_detailed(self, is_sarcastic: bool, confidence: float, text: str = "", history: List[Dict] = None):
        """Render detailed sarcasm analysis with confidence meter and history"""
        
        st.subheader("😏 Sarcasm Analysis")
        
        # Confidence gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = confidence * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Sarcasm Confidence"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': self.sarcasm_colors['detected'] if is_sarcastic else self.sarcasm_colors['not_detected']},
                'steps': [
                    {'range': [0, 30], 'color': self.sarcasm_colors['not_detected']},
                    {'range': [30, 70], 'color': self.sarcasm_colors['uncertain']},
                    {'range': [70, 100], 'color': self.sarcasm_colors['detected']}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
        # Current analysis
        status_text = "🎭 **Sarcasm Detected**" if is_sarcastic else "💬 **Normal Tone**"
        if confidence < 0.5:
            status_text = "🤷 **Uncertain**"
            
        st.markdown(f"""
        **Current Analysis:**
        - {status_text}
        - Confidence: {confidence:.1%}
        - Threshold: 70%
        """)
        
        if text:
            st.markdown(f"**Text analyzed:** *\"{text[:100]}{'...' if len(text) > 100 else ''}\"*")
        
        # Sarcasm history trend
        if history and len(history) > 1:
            self._render_sarcasm_trend(history)

    def render_live_indicator(self, is_sarcastic: bool, confidence: float):
        """Render a simple live indicator for real-time detection"""
        
        icon = self.sarcasm_icons['detected'] if is_sarcastic else self.sarcasm_icons['not_detected']
        status = "SARCASM" if is_sarcastic else "NORMAL"
        color = self.sarcasm_colors['detected'] if is_sarcastic else self.sarcasm_colors['not_detected']
        
        st.markdown(f"""
        <div style="text-align: center; padding: 15px; border-radius: 10px; background-color: {color}; color: white;">
            <h2 style="margin: 0;">{icon} {status}</h2>
            <p style="margin: 5px 0 0 0; opacity: 0.9;">{confidence:.1%} confidence</p>
        </div>
        """, unsafe_allow_html=True)

    def render_coaching_insight(self, is_sarcastic: bool, confidence: float, coaching_tip: str = ""):
        """Render sarcasm detection with coaching insights"""
        
        if is_sarcastic and confidence > 0.7:
            st.info(f"🎭 **Sarcasm detected** ({confidence:.1%} confidence)")
            if coaching_tip:
                st.markdown(f"💡 **Coaching Insight:** {coaching_tip}")
            else:
                st.markdown("💡 **Coaching Insight:** I notice some skepticism in your tone. Let's explore what might be behind that feeling.")
        elif confidence < 0.5:
            st.warning(f"🤔 **Uncertain tone** ({confidence:.1%} confidence) - Monitoring for context...")

    def _render_sarcasm_trend(self, history: List[Dict]):
        """Render sarcasm detection trend over time"""
        
        st.markdown("**📈 Sarcasm Trend**")
        
        if len(history) < 2:
            st.write("Not enough data for trend analysis")
            return
        
        # Extract data for plotting
        timestamps = [entry.get('timestamp', i) for i, entry in enumerate(history)]
        confidences = [entry.get('confidence', 0) for entry in history]
        sarcasm_flags = [entry.get('is_sarcastic', False) for entry in history]
        
        fig = go.Figure()
        
        # Confidence line
        fig.add_trace(go.Scatter(
            x=list(range(len(timestamps))),
            y=[c * 100 for c in confidences],
            mode='lines+markers',
            name='Confidence',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ))
        
        # Sarcasm detection markers
        sarcasm_x = [i for i, flag in enumerate(sarcasm_flags) if flag]
        sarcasm_y = [confidences[i] * 100 for i in sarcasm_x]
        
        if sarcasm_x:
            fig.add_trace(go.Scatter(
                x=sarcasm_x,
                y=sarcasm_y,
                mode='markers',
                name='Sarcasm Detected',
                marker=dict(
                    size=12,
                    color=self.sarcasm_colors['detected'],
                    symbol='diamond'
                )
            ))
        
        # Threshold line
        fig.add_hline(y=70, line_dash="dash", line_color="red", 
                     annotation_text="Sarcasm Threshold (70%)")
        
        fig.update_layout(
            title="Sarcasm Detection Over Time",
            xaxis_title="Message Number",
            yaxis_title="Confidence (%)",
            height=300,
            margin=dict(l=10, r=10, t=30, b=10)
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def _hex_to_rgb(self, hex_color: str) -> str:
        """Convert hex color to RGB string for CSS"""
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return f"{rgb[0]}, {rgb[1]}, {rgb[2]}"
