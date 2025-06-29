import sys
from pathlib import Path

# ✅ Add project root to sys.path before any local imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# External packages
import streamlit as st
import asyncio
import json
import time
import numpy as np
import cv2
from datetime import datetime
import logging
from typing import Dict, Optional, List, Any
import io
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tensorflow.keras.models import load_model

# App config and components
from config.settings import get_settings, MODEL_PATHS
from core.coaching_rag_system import CoachingRAGSystem, MultimodalContext, GROWPhase, VARKType
from core.session_manager import SessionManager
from models.inference_pipeline import MultimodalInferencePipeline
from models.multimodal_processor import MultimodalContextProcessor
from ui.components import (
    EmotionAnalysisComponent,
    GROWPhaseTracker,
    SessionMetricsComponent,
    VARKStyleAnalyzer,
    ChatInterface,
    FeedbackCollector
)

# ✅ Initialize Streamlit session state components once
settings = get_settings()

if "coaching_system" not in st.session_state:
    st.session_state.coaching_system = CoachingRAGSystem(
        gemini_api_key=settings.GEMINI_API_KEY,
        chroma_persist_dir=settings.CHROMA_DB_PATH
    )

if "session_manager" not in st.session_state:
    st.session_state.session_manager = SessionManager(
        coaching_system=st.session_state.coaching_system,
        inference_pipeline=MultimodalInferencePipeline(settings.MODEL_PATHS),
        timeout_minutes=settings.SESSION_CONFIG["timeout_minutes"]
    )

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # 👈 Use DEBUG to include .debug(), .info(), etc.
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

# ✅ Create your app/module logger
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AI Coaching Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.session_state.multimodal_processor = MultimodalContextProcessor(
    model_paths=MODEL_PATHS
)

class StreamlitCoachingApp:
    """Main Streamlit application for multimodal AI coaching"""
    
    def __init__(self, coaching_system=None, session_manager=None):
        self.settings = get_settings()

        # ✅ Use session_state fallback if arguments are not passed
        self.coaching_system = coaching_system or st.session_state.get("coaching_system")
        self.session_manager = session_manager or st.session_state.get("session_manager")

        if self.coaching_system is None or self.session_manager is None:
            st.warning("⚠️ Coaching system or session manager not found in session_state.")
            logger.warning("StreamlitCoachingApp initialized without full context.")

        self.initialize_session_state()
        self.load_components()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state"""
        
        # Core session data
        if 'session_id' not in st.session_state:
            st.session_state.session_id = None
        if 'user_id' not in st.session_state:
            st.session_state.user_id = None
        if 'coaching_system' not in st.session_state:
            st.session_state.coaching_system = None
        if 'multimodal_processor' not in st.session_state:
            st.session_state.multimodal_processor = None
        
        # GROW Model state
        if 'current_phase' not in st.session_state:
            st.session_state.current_phase = GROWPhase.GOAL
        if 'phase_history' not in st.session_state:
            st.session_state.phase_history = []
        if 'goal_clarity_score' not in st.session_state:
            st.session_state.goal_clarity_score = 0.0
        
        # Conversation state
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        if 'conversation_turn' not in st.session_state:
            st.session_state.conversation_turn = 0
        
        # Multimodal data
        if 'emotion_history' not in st.session_state:
            st.session_state.emotion_history = []
        if 'vark_profile' not in st.session_state:
            st.session_state.vark_profile = {'visual': 0.25, 'auditory': 0.25, 'reading': 0.25, 'kinesthetic': 0.25}
        if 'interest_levels' not in st.session_state:
            st.session_state.interest_levels = []
        
        # UI state
        if 'camera_enabled' not in st.session_state:
            st.session_state.camera_enabled = False
        if 'audio_enabled' not in st.session_state:
            st.session_state.audio_enabled = False
        if 'show_analytics' not in st.session_state:
            st.session_state.show_analytics = False
    
    def load_components(self):
        """Load and initialize system components"""
        try:
            if st.session_state.coaching_system is None:
                # Initialize coaching system
                st.session_state.coaching_system = CoachingRAGSystem(
                    gemini_api_key=self.settings.GEMINI_API_KEY,
                    chroma_persist_dir=self.settings.CHROMA_DB_PATH
                )
            
            if st.session_state.multimodal_processor is None:
                # Initialize multimodal processor
                st.session_state.multimodal_processor = MultimodalContextProcessor(
                model_paths=self.settings.MODEL_PATHS  # ✅ plural
                )
                
        except Exception as e:
            st.error(f"Failed to initialize system components: {e}")
            logger.error(f"Component initialization error: {e}")
    
    def run(self):
        """Main application entry point"""
        
        # Header
        st.title("🧠 AI Coaching Assistant")
        st.markdown("*Personalized coaching powered by multimodal AI*")
        
        # Sidebar
        self.render_sidebar()
        
        # Main content area
        if st.session_state.session_id is None:
            self.render_session_setup()
        else:
            self.render_coaching_interface()
    
    def render_sidebar(self):
        """Render sidebar with controls and analytics"""
        
        with st.sidebar:
            st.header("🎛️ Controls")
            
            # Session controls
            if st.session_state.session_id:
                st.success(f"Session: {st.session_state.session_id[:8]}...")
                if st.button("End Session", type="secondary"):
                    self.end_session()
            
            st.divider()
            
            # Multimodal settings
            st.subheader("📹 Input Settings")
            
            st.session_state.camera_enabled = st.checkbox(
                "Enable Camera", 
                value=st.session_state.camera_enabled,
                help="Capture facial emotions"
            )
            
            st.session_state.audio_enabled = st.checkbox(
                "Enable Microphone", 
                value=st.session_state.audio_enabled,
                help="Analyze voice emotions"
            )
            
            st.divider()
            
            # GROW Phase Display
            st.subheader("🎯 GROW Model")
            GROWPhaseTracker().render(
                current_phase=st.session_state.current_phase,
                phase_history=st.session_state.phase_history
            )
            
            st.divider()
            
            # Analytics toggle
            st.session_state.show_analytics = st.checkbox(
                "Show Analytics", 
                value=st.session_state.show_analytics
            )
            
            if st.session_state.show_analytics and st.session_state.session_id:
                self.render_sidebar_analytics()
    
    def render_sidebar_analytics(self):
        """Render analytics in sidebar"""

        st.subheader("📊 Session Analytics")

        # --- VARK Profile ---
        vark_profile = st.session_state.get("vark_profile", {})
        if vark_profile and all(k in vark_profile for k in ["visual", "auditory", "reading", "kinesthetic"]):
            VARKStyleAnalyzer().render_mini_profile(vark_profile)

            # Show dominant learning style
            dominant = max(vark_profile, key=vark_profile.get)
            st.markdown(f"**Dominant Style:** `{dominant.capitalize()}`")
        else:
            st.info("VARK profile is not available yet.")

        # --- Interest Level Trend ---
        interest_data = st.session_state.get("interest_levels", [])
        if interest_data:
            fig = px.line(
                y=interest_data[-10:],  # Last 10 points
                title="Interest Level (Recent)",
                labels={"y": "Interest Level"},
                height=200
            )
            fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Interest level trend will appear after a few turns.")

        # --- Session Metrics ---
        SessionMetricsComponent().render_mini_metrics({
            "total_turns": st.session_state.get("conversation_turn", 0),
            "current_phase": st.session_state.get("current_phase", GROWPhase.GOAL).value,
            "goal_clarity": round(st.session_state.get("goal_clarity_score", 0.0), 2)
        })

    
    def render_session_setup(self):
        """Render session initialization interface"""
        
        st.header("🚀 Start Your Coaching Session")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            Welcome to your AI coaching session! This system uses the GROW model 
            (Goal, Reality, Options, Will) to help you achieve your objectives.
            
            **Features:**
            - 🎯 Structured GROW model coaching
            - 😊 Emotion-aware responses
            - 🎨 Adaptive to your learning style (VARK)
            - 📊 Real-time session analytics
            """)
            
            # User input
            user_name = st.text_input("Your Name (optional)", placeholder="Enter your name")
            user_goal = st.text_area(
                "What would you like to work on today?", 
                placeholder="Describe your goal or challenge...",
                height=100
            )
            
            # Learning style preference
            if st.button("Start Coaching Session", type="primary", disabled=not user_goal.strip()):
                self.start_session(user_name, user_goal)
        
        with col2:
            st.image("https://via.placeholder.com/300x400/1f77b4/white?text=AI+Coach", 
                    caption="Your AI Coaching Assistant")
    
    def render_coaching_interface(self):
        """Render main coaching interface"""
        
        # Main content columns
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Chat interface
            st.header("💬 Coaching Conversation")
            ChatInterface().render(
                conversation_history=st.session_state.conversation_history,
                on_message_submit=self.handle_user_message
            )
        
        with col2:
            # Real-time analytics
            if st.session_state.show_analytics:
                self.render_analytics_panel()
            else:
                self.render_help_panel()
        
        # Bottom section - multimodal inputs
        if st.session_state.camera_enabled or st.session_state.audio_enabled:
            st.divider()
            self.render_multimodal_inputs()
    
    def render_analytics_panel(self):
        """Render detailed analytics panel"""
        
        st.header("📊 Session Analytics")
        
        # Emotion analysis
        if "emotion_chart_placeholder" not in st.session_state:
            st.session_state.emotion_chart_placeholder = EmotionAnalysisComponent().render_live_emotion_placeholder()
        
        st.divider()
        
        # VARK analysis
        VARKStyleAnalyzer().render_detailed_analysis(
            st.session_state.vark_profile
        )
        
        st.divider()
        
        # Session metrics
        SessionMetricsComponent().render_detailed_metrics({
            'session_duration': self.get_session_duration(),
            'total_turns': st.session_state.conversation_turn,
            'phase_progression': self.get_phase_progression(),
            'engagement_score': self.calculate_engagement_score()
        })
    
    def render_help_panel(self):
        """Render help and guidance panel"""
        
        st.header("ℹ️ Coaching Guide")
        
        # Current phase guidance
        phase_guidance = {
            GROWPhase.GOAL: {
                "title": "🎯 Goal Setting",
                "description": "Define what you want to achieve",
                "tips": [
                    "Be specific about your desired outcome",
                    "Make your goal measurable",
                    "Set a realistic timeframe"
                ]
            },
            GROWPhase.REALITY: {
                "title": "🔍 Current Reality",
                "description": "Explore your current situation",
                "tips": [
                    "Be honest about where you are now",
                    "Identify available resources",
                    "Acknowledge obstacles"
                ]
            },
            GROWPhase.OPTIONS: {
                "title": "💡 Explore Options",
                "description": "Generate possible solutions",
                "tips": [
                    "Think creatively",
                    "Consider all possibilities",
                    "Don't judge ideas yet"
                ]
            },
            GROWPhase.WILL: {
                "title": "✅ Way Forward",
                "description": "Commit to action",
                "tips": [
                    "Choose specific actions",
                    "Set deadlines",
                    "Identify accountability measures"
                ]
            }
        }
        
        current_guidance = phase_guidance[st.session_state.current_phase]
        
        st.subheader(current_guidance["title"])
        st.write(current_guidance["description"])
        
        st.write("**Tips for this phase:**")
        for tip in current_guidance["tips"]:
            st.write(f"• {tip}")
        
        st.divider()
        
        # Progress indicator
        phases = list(GROWPhase)
        current_index = phases.index(st.session_state.current_phase)
        progress = (current_index + 1) / len(phases)
        
        st.subheader("Progress")
        st.progress(progress)
        st.write(f"Phase {current_index + 1} of {len(phases)}: {st.session_state.current_phase.value.title()}")
    
    def render_multimodal_inputs(self):
        """Render multimodal input interfaces"""
        
        st.header("🎥 Multimodal Input")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.session_state.camera_enabled:
                st.subheader("📹 Real-Time Emotion Tracking")

                def run_real_time_emotion_loop():
                    model = load_model(r"C:\Users\vedan\Desktop\coachingSystem\models\saved_models\emotion_model.h5")
                    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                    emotion_component = EmotionAnalysisComponent()

                    FRAME_WINDOW = st.image([])

                    cap = cv2.VideoCapture(0)

                    if not cap.isOpened():
                        st.error("Could not open webcam.")
                        return

                    st.info("Press **Stop Webcam** in the sidebar to stop.")

                    # Create separate placeholders for chart and text
                    if "emotion_chart_placeholder" not in st.session_state:
                        st.session_state.emotion_chart_placeholder = st.empty()
                    if "emotion_text_placeholder" not in st.session_state:
                        st.session_state.emotion_text_placeholder = st.empty()

                    chart = st.session_state.emotion_chart_placeholder
                    text = st.session_state.emotion_text_placeholder

                    while st.session_state.camera_enabled:
                        ret, frame = cap.read()
                        if not ret:
                            st.warning("Webcam feed not available.")
                            break

                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                        latest_scores = {label: 0.0 for label in emotion_labels}

                        for (x, y, w, h) in faces:
                            roi_gray = gray[y:y+h, x:x+w]
                            roi = cv2.resize(roi_gray, (48, 48))
                            roi = roi.astype('float32') / 255.0
                            roi = np.expand_dims(roi, axis=0)
                            roi = np.expand_dims(roi, axis=-1)

                            preds = model.predict(roi, verbose=0)[0]
                            for i, label in enumerate(emotion_labels):
                                latest_scores[label] = float(preds[i])

                            # Annotate the frame
                            label = emotion_labels[np.argmax(preds)]
                            confidence = np.max(preds)
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            cv2.putText(frame, f"{label} ({confidence*100:.1f}%)", (x, y-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                            break  # just process one face

                        # Show webcam frame
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        FRAME_WINDOW.image(frame_rgb)

                        # Update live chart and text
                        if latest_scores:
                            emotion_component.update_real_time_emotion_chart(
                                chart, text, {k.lower(): v for k, v in latest_scores.items()}
                            )

                        time.sleep(0.1)

                    cap.release()

            if st.session_state.camera_enabled:
                st.subheader("📹 Real-Time Emotion Tracking")
                run_real_time_emotion_loop()
                
        with col2:
            if st.session_state.audio_enabled:
                st.subheader("🎤 Audio Input")
                
                # Audio recorder
                audio_input = st.audio_input("Record your voice for emotion analysis")
                
                if audio_input is not None:
                    # Process voice emotion
                    voice_emotion_data = self.process_voice_emotion(audio_input)
                    if voice_emotion_data:
                        EmotionAnalysisComponent().render_voice_emotion(voice_emotion_data)
    
    def start_session(self, user_name: str, user_goal: str):
        """Start a new coaching session with auto-inferred VARK"""

        try:
            # Identifiers
            session_id = f"session_{int(time.time())}"
            user_id = user_name.lower().replace(" ", "_") if user_name else f"user_{int(time.time())}"

            # Core state
            st.session_state.session_id = session_id
            st.session_state.user_id = user_id
            st.session_state.current_phase = GROWPhase.GOAL
            st.session_state.conversation_turn = 0
            st.session_state.phase_history = []
            st.session_state.conversation_history = []

            # Initialize metrics
            st.session_state.interest_levels = []
            st.session_state.vark_profile = {
                'visual': 0.25, 'auditory': 0.25, 'reading': 0.25, 'kinesthetic': 0.25
            }

            # Evaluate initial goal clarity
            clarity = st.session_state.coaching_system.evaluate_goal_clarity(user_goal)
            st.session_state.goal_clarity_score = clarity
            logger.info(f"[INIT] Goal clarity from goal input: {clarity}")

            # Add initial emotion data
            st.session_state.emotion_history = [{
                'timestamp': datetime.now(),
                'facial_emotion': {},
                'voice_emotion': {},
                'dominant_emotion': 'neutral'
            }]

            # Save user goal
            st.session_state.conversation_history.append({
                'role': 'user',
                'content': user_goal,
                'timestamp': datetime.now(),
                'phase': GROWPhase.GOAL.value
            })

            # Trigger optional VARK update
            if hasattr(self, "update_learning_style_via_model"):
                self.update_learning_style_via_model(user_goal)

            logger.info(f"🚀 Started session: {session_id} for user: {user_id}")

            # Kick off session
            self.generate_coaching_response(user_goal, initial=True)
            st.success("🎯 Coaching session started successfully!")
            st.rerun()

        except Exception as e:
            st.error(f"Failed to start session: {e}")
            logger.error(f"Session start error: {e}")


    def handle_user_message(self, message: str):
        """Handle user message input"""

        if not message.strip():
            return

        try:
            # Add user message to conversation history
            st.session_state.conversation_history.append({
                'role': 'user',
                'content': message,
                'timestamp': datetime.now(),
                'phase': st.session_state.current_phase.value
            })

            # Increment turn count
            st.session_state.conversation_turn += 1

            # Update goal clarity score BEFORE checking phase completion
            if st.session_state.current_phase == GROWPhase.GOAL:
                goal_clarity_score = st.session_state.coaching_system.evaluate_goal_clarity(message)
                st.session_state.goal_clarity_score = goal_clarity_score
                logger.info(f"🎯 Updated Goal clarity score: {goal_clarity_score}")

            # Check if current phase should transition BEFORE generating response
            self.check_phase_completion(message)

            # Update VARK profile dynamically every N turns
            if st.session_state.conversation_turn % 5 == 0:
                self.update_learning_style_via_model(message)

            # Generate coaching response
            self.generate_coaching_response(message)

            logger.info(f"📈 Goal clarity score: {st.session_state.goal_clarity_score}")
            logger.info(f"🔄 Current phase: {st.session_state.current_phase.value}")
            logger.info(f"👥 Conversation turn: {st.session_state.conversation_turn}")

        except Exception as e:
            st.error(f"Error processing message: {e}")
            logger.error(f"Message processing error: {e}")

    

    def generate_coaching_response(self, user_input: str, initial: bool = False):
        """Generate coaching response using the RAG system"""

        try:
            # Create multimodal context
            context = self.create_multimodal_context(user_input)
            
            # Generate response
            coaching_response = st.session_state.coaching_system.generate_coaching_response(context)
            
            # Add to conversation history
            st.session_state.conversation_history.append({
                'role': 'assistant',
                'content': coaching_response,
                'timestamp': datetime.now(),
                'phase': st.session_state.current_phase.value,
                'context_data': {
                    'dominant_emotion': self.get_dominant_emotion_from_context(context),
                    'interest_level': context.interest_level,
                    'vark_type': context.vark_type.value
                }
            })
            
            # Update analytics
            self.update_analytics(context)
            
            if not initial:
                st.rerun()
                
        except Exception as e:
            st.error(f"Failed to generate response: {e}")
            logger.error(f"Response generation error: {e}")
    
    def create_multimodal_context(self, user_input: str) -> MultimodalContext:
        """Create multimodal context from current state"""

        # Get latest emotion data
        latest_facial_emotion = (
            st.session_state.emotion_history[-1].get('facial_emotion', {}) 
            if st.session_state.emotion_history else {}
        )
        latest_voice_emotion = (
            st.session_state.emotion_history[-1].get('voice_emotion', {}) 
            if st.session_state.emotion_history else {}
        )

        # Determine dominant VARK type
        dominant_vark = max(st.session_state.vark_profile, key=st.session_state.vark_profile.get)
        vark_confidence = st.session_state.vark_profile[dominant_vark]

        # Interest level
        interest_level = (
            np.mean(st.session_state.interest_levels[-5:]) 
            if st.session_state.interest_levels else 0.5
        )

        # Text features
        text_features = self.analyze_text_features(user_input)

        # Use existing goal clarity score (don't recalculate here)
        goal_clarity_score = st.session_state.goal_clarity_score

        logger.info(f"[CTX] Current Phase: {st.session_state.current_phase}, Input: {user_input}")
        logger.info(f"[CTX] Using existing clarity score: {goal_clarity_score}")

        return MultimodalContext(
            user_id=st.session_state.user_id,
            session_id=st.session_state.session_id,
            timestamp=datetime.now(),
            grow_phase=st.session_state.current_phase,
            utterance=user_input,

            facial_emotion=latest_facial_emotion,
            voice_emotion=latest_voice_emotion,
            text_sentiment=text_features.get('sentiment', {}),

            vark_type=VARKType(dominant_vark),
            vark_confidence=vark_confidence,

            sarcasm_detected=text_features.get('sarcasm_detected', False),
            sarcasm_confidence=text_features.get('sarcasm_confidence', 0.0),
            interest_level=interest_level,
            digression_detected=text_features.get('digression_detected', False),

            conversation_turn=st.session_state.conversation_turn,
            previous_phase_completion=len(st.session_state.phase_history) > 0,
            goal_clarity_score=goal_clarity_score,
            system_instruction=""
        )

    
    def analyze_text_features(self, text: str) -> Dict[str, Any]:
        """Analyze text for sarcasm, sentiment, etc."""
        if not hasattr(st.session_state, 'multimodal_processor'):
            return {'sarcasm_detected': False, 'sarcasm_confidence': 0.0, 'sentiment': {}}
        
        # Ensure text is a string, not a list
        if isinstance(text, list):
            text = ' '.join(text) if text else ""
        
        return st.session_state.multimodal_processor.analyze_text(str(text))
    
    def process_facial_emotion(self, image_data) -> Optional[Dict]:
        """Process facial emotion from camera input"""
        
        try:
            if st.session_state.multimodal_processor:
                # Convert image data to format expected by processor
                image_bytes = image_data.getvalue()
                image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                image_array = np.array(image_pil)
                return st.session_state.multimodal_processor.process_facial_emotion(image_array)
        except Exception as e:
            logger.error(f"Facial emotion processing error: {e}")
        
        return None
    
    def process_voice_emotion(self, audio_data) -> Optional[Dict]:
        """Process voice emotion from audio input"""
        
        try:
            if st.session_state.multimodal_processor:
                return st.session_state.multimodal_processor.process_voice_emotion(audio_data)
        except Exception as e:
            logger.error(f"Voice emotion processing error: {e}")
        
        return None
    
    def update_analytics(self, context: MultimodalContext):
        """Update analytics data"""
        
        # Update emotion history
        emotion_data = {
            'timestamp': context.timestamp,
            'facial_emotion': context.facial_emotion,
            'voice_emotion': context.voice_emotion,
            'dominant_emotion': self.get_dominant_emotion_from_context(context)
        }
        st.session_state.emotion_history.append(emotion_data)
        
        # Update interest levels
        st.session_state.interest_levels.append(context.interest_level)
        
        # Update VARK profile (adaptive learning)
        if context.vark_confidence > 0.7:
            # Strengthen the detected VARK type
            current_weight = st.session_state.vark_profile[context.vark_type.value]
            st.session_state.vark_profile[context.vark_type.value] = min(1.0, current_weight + 0.1)
            
            # Normalize weights
            total_weight = sum(st.session_state.vark_profile.values())
            for key in st.session_state.vark_profile:
                st.session_state.vark_profile[key] /= total_weight
    
    def check_phase_transition(self):
        """Check if phase transition should occur"""
        turn = st.session_state.conversation_turn
        phase = st.session_state.current_phase

        if turn < 2:
            return  # Don’t transition too early

        if phase == GROWPhase.GOAL:
            if st.session_state.goal_clarity_score >= 0.7:
                self.transition_to_phase(GROWPhase.REALITY)

        elif phase == GROWPhase.REALITY:
            if turn >= 6:  # You can add LLM check here too later
                self.transition_to_phase(GROWPhase.OPTIONS)

        elif phase == GROWPhase.OPTIONS:
            if turn >= 9:  # Add LLM check for decision confidence later
                self.transition_to_phase(GROWPhase.WILL)

    
    def transition_to_phase(self, new_phase: GROWPhase):
        """Transition to a new GROW phase"""
        current_phase = st.session_state.current_phase

        # Avoid transitioning to the same phase
        if new_phase == current_phase:
            logger.warning("⚠️ Attempted to transition to the same phase.")
            return

        # Check if phase already completed (avoid duplicates)
        for phase_entry in st.session_state.phase_history:
            if isinstance(phase_entry, dict) and phase_entry.get('phase') == current_phase:
                logger.warning(f"⚠️ Phase {current_phase.value} already marked complete.")
                return

        # Count user turns in current phase
        turns_in_current_phase = sum(
            1 for msg in st.session_state.conversation_history
            if msg['role'] == 'user' and msg.get('phase') == current_phase.value
        )

        # Append completed phase to history
        st.session_state.phase_history.append({
            'phase': current_phase,
            'completed_at': datetime.now(),
            'turns_spent': turns_in_current_phase,
            'completed': True
        })

        # Update phase state
        st.session_state.current_phase = new_phase

        # Generate phase transition message
        phase_messages = {
            GROWPhase.REALITY: "Great! Now let's explore your current situation. What's happening right now?",
            GROWPhase.OPTIONS: "Perfect! Now let's brainstorm some options. What are different ways you could approach this?",
            GROWPhase.WILL: "Excellent! Now let's create your action plan. What specific steps will you take?"
        }

        transition_message = phase_messages.get(new_phase, f"Moving to {new_phase.value.title()} phase")

        # Add transition message to conversation history
        st.session_state.conversation_history.append({
            'role': 'assistant',
            'content': f"🎯 **Phase Transition:** {transition_message}",
            'timestamp': datetime.now(),
            'phase': new_phase.value,
            'is_transition': True
        })

        logger.info(f"🚦 Successfully transitioned: {current_phase.value.upper()} → {new_phase.value.upper()}")
        st.success(f"✅ Moved to {new_phase.value.title()} phase!")
        
        # Force UI refresh
        st.rerun()

    
    def end_session(self):
        """End the current coaching session"""
        
        # Reset session state
        st.session_state.session_id = None
        st.session_state.user_id = None
        st.session_state.current_phase = GROWPhase.GOAL
        st.session_state.conversation_history = []
        st.session_state.conversation_turn = 0
        st.session_state.phase_history = []
        st.session_state.emotion_history = []
        st.session_state.interest_levels = []
        
        st.success("Session ended successfully!")
        st.rerun()
    
    def get_dominant_emotion_from_context(self, context: MultimodalContext) -> str:
        """Get dominant emotion from context"""
        return st.session_state.coaching_system._get_dominant_emotion(
            context.facial_emotion, context.voice_emotion
        )
    
    def get_session_duration(self) -> int:
        """Get session duration in minutes"""
        if st.session_state.conversation_history:
            start_time = st.session_state.conversation_history[0]['timestamp']
            duration = datetime.now() - start_time
            return int(duration.total_seconds() / 60)
        return 0
    
    def get_phase_progression(self) -> Dict:
        """Get phase progression data"""
        phases = [GROWPhase.GOAL, GROWPhase.REALITY, GROWPhase.OPTIONS, GROWPhase.WILL]
        current_index = phases.index(st.session_state.current_phase)
        
        return {
            'current_phase': st.session_state.current_phase.value,
            'progress_percentage': ((current_index + 1) / len(phases)) * 100,
            'completed_phases': len(st.session_state.phase_history),
            'total_phases': len(phases)
        }
    
    def calculate_engagement_score(self) -> float:
        """Calculate overall engagement score"""
        if not st.session_state.interest_levels:
            return 0.0
        
        return np.mean(st.session_state.interest_levels)
    
    def update_learning_style_via_model(self, user_text: str):
        """Use VAK model to infer and update VARK profile from user text only"""
        try:
            vak_model = st.session_state.multimodal_processor.pipeline.models.get("vak")
            if vak_model:
                vak_type, confidence = vak_model.predict(user_text)
                if confidence >= 0.6:  # only apply confident updates
                    logger.info(f"message: {user_text}")
                    logger.info(f"🔁 VARK update from USER message: {vak_type} ({confidence:.2f})")

                    # Blend into existing profile using weighted average
                    for key in st.session_state.vark_profile:
                        if key == vak_type:
                            st.session_state.vark_profile[key] = (
                                st.session_state.vark_profile[key] * 0.7 + confidence * 0.3
                            )
                        else:
                            st.session_state.vark_profile[key] *= 0.95  # slight decay

                    # Normalize
                    total = sum(st.session_state.vark_profile.values())
                    for key in st.session_state.vark_profile:
                        st.session_state.vark_profile[key] /= total
        except Exception as e:
            logger.warning(f"⚠️ VARK model update failed: {e}")

    def check_phase_completion(self, message: str):
            """Check whether the current GROW phase should transition."""
            current_phase = st.session_state.current_phase
            current_turn = st.session_state.conversation_turn
            
            logger.info(f"🔄 [PHASE CHECK] Current Phase: {current_phase.value}")
            logger.info(f"💬 [USER MESSAGE] {message}")
            logger.info(f"🎯 [CLARITY SCORE] {st.session_state.goal_clarity_score}")
            logger.info(f"📊 [TURNS] Conversation Turn: {current_turn}")

            # ✅ GOAL → REALITY
            if current_phase == GROWPhase.GOAL:
                clarity_score = st.session_state.goal_clarity_score
                logger.info(f"🧠 [GOAL] Current Goal Clarity Score: {clarity_score}")

                # Check if goal is clear enough and we have enough turns
                if clarity_score >= 0.75 and current_turn >= 2:
                    logger.info("✅ [GOAL PHASE COMPLETE] Transitioning to REALITY phase")
                    self.transition_to_phase(GROWPhase.REALITY)
                    return
                else:
                    logger.info(f"⏳ [GOAL PHASE NOT READY] Clarity: {clarity_score}/0.75, Turns: {current_turn}/2")

            # ✅ REALITY → OPTIONS
            elif current_phase == GROWPhase.REALITY:
                # Check if user has described their current situation
                if current_turn >= 4:  # Allow at least 2 turns in REALITY phase
                    prompt = f"""Analyze if the user has described their current situation, challenges, or obstacles.
                    
                    User message: "{message}"
                    
                    Respond with only 'yes' or 'no'."""
                    
                    result = self.query_llm_boolean(prompt)
                    logger.info(f"🤖 [REALITY CHECK] LLM Response: {result}")
                    
                    if "yes" in result.lower():
                        logger.info("✅ [REALITY PHASE COMPLETE] Transitioning to OPTIONS phase")
                        self.transition_to_phase(GROWPhase.OPTIONS)
                        return
                    else:
                        logger.info("⏳ [REALITY PHASE NOT READY] Current situation not fully explored")

            # ✅ OPTIONS → WILL
            elif current_phase == GROWPhase.OPTIONS:
                # Check if user has generated options
                if current_turn >= 6:  # Allow at least 2 turns in OPTIONS phase
                    prompt = f"""Analyze if the user has proposed any options, solutions, or ideas for moving forward.
                    
                    User message: "{message}"
                    
                    Respond with only 'yes' or 'no'."""
                    
                    result = self.query_llm_boolean(prompt)
                    logger.info(f"🧪 [OPTIONS CHECK] LLM Response: {result}")
                    
                    if "yes" in result.lower():
                        logger.info("✅ [OPTIONS PHASE COMPLETE] Transitioning to WILL phase")
                        self.transition_to_phase(GROWPhase.WILL)
                        return
                    else:
                        logger.info("⏳ [OPTIONS PHASE NOT READY] No clear options identified")

            # ✅ WILL → Completion
            elif current_phase == GROWPhase.WILL:
                # Check if user has committed to action
                if current_turn >= 8:  # Allow at least 2 turns in WILL phase
                    prompt = f"""Analyze if the user has committed to concrete action steps or made specific commitments.
                    
                    User message: "{message}"
                    
                    Respond with only 'yes' or 'no'."""
                    
                    result = self.query_llm_boolean(prompt)
                    logger.info(f"✅ [WILL CHECK] LLM Response: {result}")
                    
                    if "yes" in result.lower():
                        logger.info("🎉 [SESSION COMPLETE] Coaching session completed")
                        st.success("🎉 Coaching session completed! You've made great progress.")
                        # You could add session completion logic here
                        self.end_session()
                        return
                    else:
                        logger.info("⏳ [WILL PHASE NOT READY] No clear commitment detected")



    def query_llm_boolean(self, prompt: str) -> str:
        """Query LLM for boolean response"""
        try:
            # Use the existing coaching system's Gemini model
            response = st.session_state.coaching_system.gemini_model.generate_content(prompt)
            result = response.text.lower().strip()
            logger.info(f"🤖 LLM Boolean Response: {result}")
            return result
        except Exception as e:
            logger.error(f"LLM boolean check failed: {e}")
            return "no"



def main():
    """Main application entry point"""
    app = StreamlitCoachingApp()
    app.run()

if __name__ == "__main__":
    main()