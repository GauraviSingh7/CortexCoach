import streamlit as st
import numpy as np
import cv2
import time
import logging
from tensorflow.keras.models import load_model
from core.coaching_rag_system import GROWPhase
from app.ui.components import (
    EmotionAnalysisComponent,
    GROWPhaseTracker,
    SessionMetricsComponent,
    VARKStyleAnalyzer,
    ChatInterface,
    FeedbackCollector
)

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- NEW: Helper to compute average emotion across frames ---
def compute_average_emotion(history: list[dict[str, float]]) -> dict[str, float]:
    if not history:
        logger.warning("[AVERAGING] No emotion history available.")
        return {}

    try:
        avg_scores = {k: 0.0 for k in history[0]}
        for i, frame in enumerate(history):
            for emotion, score in frame.items():
                if isinstance(score, (int, float)):
                    avg_scores[emotion] += score
                else:
                    logger.warning(f"[AVERAGING] Skipping non-numeric score at frame {i}: {emotion}={score}")

        count = len(history)
        averaged = {k: v / count for k, v in avg_scores.items()}

        # Log the result
        top = max(averaged, key=averaged.get)
        logger.info(f"[AVERAGING] Averaged {count} frames. Dominant: {top} ({averaged[top]:.2f})")
        return averaged

    except Exception as e:
        logger.error(f"[AVERAGING ERROR] Failed to compute average emotion: {e}")
        return {}


# --- Optional: Reset emotion history when camera stops ---
def reset_emotion_history():
    if "emotion_history" in st.session_state:
        del st.session_state.emotion_history


def render_coaching_interface(app):
    """Render main coaching interface with analytics always visible"""
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("💬 Coaching Conversation")
        ChatInterface().render(
            conversation_history=st.session_state.conversation_history,
            on_message_submit=app.handle_user_message
        )

    with col2:
        # Always show analytics panel (removed conditional)
        app.render_analytics_panel()

    if st.session_state.camera_enabled or st.session_state.audio_enabled:
        st.divider()
        app.render_multimodal_inputs()


def render_analytics_panel(app):
    """Render detailed analytics panel"""
    st.header("📊 Session Analytics")

    if "emotion_chart_placeholder" not in st.session_state:
        st.session_state.emotion_chart_placeholder = EmotionAnalysisComponent().render_live_emotion_placeholder()

    st.divider()

    VARKStyleAnalyzer().render_detailed_analysis(
        st.session_state.vark_profile
    )

    st.divider()

    SessionMetricsComponent().render_detailed_metrics({
        'session_duration': app.get_session_duration(),
        'total_turns': st.session_state.conversation_turn,
        'phase_progression': app.get_phase_progression(),
        'engagement_score': app.calculate_engagement_score()
    })


def render_multimodal_inputs(app):
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

                if "emotion_chart_placeholder" not in st.session_state:
                    st.session_state.emotion_chart_placeholder = st.empty()
                if "emotion_text_placeholder" not in st.session_state:
                    st.session_state.emotion_text_placeholder = st.empty()
                if "emotion_history" not in st.session_state:
                    st.session_state.emotion_history = []

                chart = st.session_state.emotion_chart_placeholder
                text = st.session_state.emotion_text_placeholder
                frame_count = 0

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

                        label = emotion_labels[np.argmax(preds)]
                        confidence = np.max(preds)
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label} ({confidence*100:.1f}%)", (x, y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                        break  # Only process first detected face

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    FRAME_WINDOW.image(frame_rgb)

                    # Save emotion data every frame
                    if latest_scores and any(latest_scores.values()):
                        st.session_state.emotion_history.append(latest_scores.copy()) 
                        logger.debug(f"[EMOTION] Added frame {frame_count} to history. Total stored: {len(st.session_state.emotion_history)}")

                        if len(st.session_state.emotion_history) > 40:
                            st.session_state.emotion_history.pop(0)

                    # if frame_count % 10 == 0:
                    #     logger.info(f"[INFO] Frame {frame_count}: Emotion snapshot: {latest_scores}")

                    # Update chart every 5 frames
                    frame_count += 1
                    if frame_count % 5 == 0 and latest_scores:
                        emotion_component.update_real_time_emotion_chart(
                            chart, text, {k.lower(): v for k, v in latest_scores.items()}
                        )

                    time.sleep(0.1)

                cap.release()
                reset_emotion_history()  # Clean up session state after stopping camera

            run_real_time_emotion_loop()

    with col2:
        if st.session_state.audio_enabled:
            st.subheader("🎤 Audio Input")
            audio_input = st.audio_input("Record your voice for emotion analysis")

            if audio_input is not None:
                voice_emotion_data = app.process_voice_emotion(audio_input)
                if voice_emotion_data:
                    EmotionAnalysisComponent().render_voice_emotion(voice_emotion_data)
