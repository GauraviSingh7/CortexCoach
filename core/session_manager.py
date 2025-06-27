import uuid
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor
import weakref
from config.settings import GEMINI_API_KEY

# Import from your existing system
from .coaching_rag_system import (
    CoachingRAGSystem, 
    MultimodalContext, 
    GROWPhase, 
    VARKType
)

from models.inference_pipeline import MultimodalInferencePipeline

logger = logging.getLogger(__name__)

class SessionStatus(Enum):
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    TERMINATED = "terminated"

class PhaseTransitionReason(Enum):
    COMPLETION = "completion"
    USER_REQUEST = "user_request"
    DIGRESSION_RECOVERY = "digression_recovery"
    CLARIFICATION_NEEDED = "clarification_needed"

@dataclass
class SessionMetrics:
    """Tracks key metrics for a coaching session"""
    total_turns: int = 0
    avg_interest_level: float = 0.0
    avg_goal_clarity: float = 0.0
    emotion_distribution: Dict[str, int] = None
    sarcasm_incidents: int = 0
    digression_count: int = 0
    phase_durations: Dict[str, float] = None
    user_satisfaction: Optional[float] = None
    
    def __post_init__(self):
        if self.emotion_distribution is None:
            self.emotion_distribution = {}
        if self.phase_durations is None:
            self.phase_durations = {}

@dataclass
class SessionContext:
    """Complete session context and state"""
    session_id: str
    user_id: str
    status: SessionStatus
    created_at: datetime
    updated_at: datetime
    
    # GROW Model State
    current_phase: GROWPhase
    phase_history: List[Dict[str, Any]]
    dominant_vark_type: VARKType
    goal_statement: Optional[str] = None
    goal_clarity_score: float = 0.0
    
    # User Profile
    vark_confidence: float = 0.0
    personality_indicators: Dict[str, float] = None
    
    # Conversation State
    conversation_history: List[Dict[str, Any]] = None
    current_turn: int = 0
    last_response_id: Optional[str] = None
    
    # Behavioral Tracking
    behavioral_patterns: Dict[str, Any] = None
    engagement_trends: List[float] = None
    
    # Session Metrics
    metrics: SessionMetrics = None
    
    # Metadata
    session_config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.personality_indicators is None:
            self.personality_indicators = {}
        if self.conversation_history is None:
            self.conversation_history = []
        if self.behavioral_patterns is None:
            self.behavioral_patterns = {
                "avg_response_time": 0.0,
                "typical_message_length": 0,
                "common_emotions": [],
                "sarcasm_frequency": 0.0,
                "digression_tendency": 0.0
            }
        if self.engagement_trends is None:
            self.engagement_trends = []
        if self.metrics is None:
            self.metrics = SessionMetrics()
        if self.session_config is None:
            self.session_config = {}

class SessionManager:
    """Manages coaching session lifecycle and state"""
    
    def __init__(
    self,
    coaching_system: CoachingRAGSystem,
    inference_pipeline,
    max_sessions: int = 1000,
    timeout_minutes: int = 20,
    cleanup_interval_minutes: int = 30
):
        self.coaching_system = coaching_system
        self.inference_pipeline = inference_pipeline
        self.max_sessions = max_sessions
        self.timeout_minutes = timeout_minutes

        # Active sessions storage
        self.active_sessions: Dict[str, SessionContext] = {}
        self.session_locks: Dict[str, asyncio.Lock] = {}

        # Background task management
        self.cleanup_task: Optional[asyncio.Task] = None
        self.analytics_task: Optional[asyncio.Task] = None

        # Thread pool for CPU-bound tasks
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Timing configuration
        self.session_timeout = timedelta(minutes=self.timeout_minutes)
        self.cleanup_interval = timedelta(minutes=cleanup_interval_minutes)
        self.phase_completion_threshold = 0.7

        # Weak references for lightweight tracking
        self._session_refs = weakref.WeakSet()

        # Start background task loop
        self._start_background_tasks()

    
    async def create_session(self, user_id: str, config: Dict[str, Any] = None) -> str:
        """Create a new coaching session"""
        
        session_id = str(uuid.uuid4())
        now = datetime.now()
        
        # Initialize session context
        session_context = SessionContext(
            session_id=session_id,
            user_id=user_id,
            status=SessionStatus.INITIALIZING,
            created_at=now,
            updated_at=now,
            current_phase=GROWPhase.GOAL,
            phase_history=[{
                "phase": GROWPhase.GOAL.value,
                "started_at": now.isoformat(),
                "reason": "session_start"
            }],
            dominant_vark_type=VARKType.VISUAL,  # Default, will be updated
            session_config=config or {}
        )
        
        # Create lock for this session
        self.session_locks[session_id] = asyncio.Lock()
        
        # Store session
        self.active_sessions[session_id] = session_context
        self._session_refs.add(session_context)
        
        # Initialize with user's historical preferences if available
        await self._initialize_user_preferences(session_context)
        
        # Mark as active
        session_context.status = SessionStatus.ACTIVE
        session_context.updated_at = datetime.now()
        
        logger.info(f"Created session {session_id} for user {user_id}")
        return session_id
    
    async def process_multimodal_input(
        self, 
        session_id: str, 
        utterance: str,
        facial_emotion: Dict[str, float],
        voice_emotion: Dict[str, float],
        text_sentiment: Dict[str, float],
        additional_context: Dict[str, Any] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Process multimodal input and generate coaching response"""
        
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        async with self.session_locks[session_id]:
            session = self.active_sessions[session_id]
            
            if session.status != SessionStatus.ACTIVE:
                raise ValueError(f"Session {session_id} is not active (status: {session.status})")
            
            # Update session state
            session.current_turn += 1
            session.updated_at = datetime.now()
            
            # Detect VARK type and update user profile
            vark_type, vark_confidence = await self._detect_vark_type(utterance, session)
            await self._update_user_profile(session, vark_type, vark_confidence)
            
            # Detect behavioral patterns
            behavioral_indicators = await self._analyze_behavioral_patterns(
                session, utterance, facial_emotion, voice_emotion, text_sentiment
            )
            
            # Create multimodal context
            multimodal_context = MultimodalContext(
                user_id=session.user_id,
                session_id=session_id,
                timestamp=session.updated_at,
                grow_phase=session.current_phase,
                utterance=utterance,
                
                facial_emotion=facial_emotion,
                voice_emotion=voice_emotion,
                text_sentiment=text_sentiment,
                
                vark_type=session.dominant_vark_type,
                vark_confidence=session.vark_confidence,
                
                sarcasm_detected=behavioral_indicators.get("sarcasm_detected", False),
                sarcasm_confidence=behavioral_indicators.get("sarcasm_confidence", 0.0),
                interest_level=behavioral_indicators.get("interest_level", 0.5),
                digression_detected=behavioral_indicators.get("digression_detected", False),
                
                conversation_turn=session.current_turn,
                previous_phase_completion=await self._check_phase_completion(session),
                goal_clarity_score=session.goal_clarity_score
            )
            
            # Store context in ChromaDB
            context_id = self.coaching_system.store_context(multimodal_context)
            
            # Generate coaching response
            coaching_response = await self._generate_coaching_response(multimodal_context)
            
            # Update session with response
            await self._update_session_with_response(
                session, multimodal_context, coaching_response, context_id
            )
            
            # Check for phase transition
            transition_info = await self._check_phase_transition(session, multimodal_context)
            
            # Update metrics
            await self._update_session_metrics(session, multimodal_context)
            
            # Prepare response
            response_data = {
                "response": coaching_response,
                "session_info": {
                    "current_phase": session.current_phase.value,
                    "goal_clarity_score": session.goal_clarity_score,
                    "conversation_turn": session.current_turn,
                    "dominant_vark_type": session.dominant_vark_type.value,
                    "interest_level": behavioral_indicators.get("interest_level", 0.5)
                },
                "context_id": context_id,
                "behavioral_indicators": behavioral_indicators,
                "transition_info": transition_info
            }
            
            return coaching_response, response_data
    
    async def transition_phase(
        self, 
        session_id: str, 
        new_phase: GROWPhase, 
        reason: PhaseTransitionReason,
        user_initiated: bool = False
    ) -> bool:
        """Manually transition to a new GROW phase"""
        
        if session_id not in self.active_sessions:
            return False
        
        async with self.session_locks[session_id]:
            session = self.active_sessions[session_id]
            
            if session.current_phase == new_phase:
                return False
            
            old_phase = session.current_phase
            now = datetime.now()
            
            # Update phase history
            if session.phase_history:
                session.phase_history[-1]["ended_at"] = now.isoformat()
                session.phase_history[-1]["duration"] = (
                    now - datetime.fromisoformat(session.phase_history[-1]["started_at"])
                ).total_seconds()
            
            # Add new phase
            session.phase_history.append({
                "phase": new_phase.value,
                "started_at": now.isoformat(),
                "reason": reason.value,
                "user_initiated": user_initiated,
                "previous_phase": old_phase.value
            })
            
            session.current_phase = new_phase
            session.updated_at = now
            
            logger.info(f"Session {session_id}: Phase transition {old_phase.value} -> {new_phase.value} ({reason.value})")
            return True
    
    async def update_response_effectiveness(
        self, 
        session_id: str, 
        context_id: str, 
        effectiveness_score: float,
        feedback: str = ""
    ):
        """Update response effectiveness based on user feedback"""
        
        if session_id not in self.active_sessions:
            return
        
        async with self.session_locks[session_id]:
            session = self.active_sessions[session_id]
            
            # Update in ChromaDB
            response_id = f"response_{context_id}"
            self.coaching_system.update_response_effectiveness(
                response_id, effectiveness_score, feedback
            )
            
            # Update session metrics
            if session.metrics.user_satisfaction is None:
                session.metrics.user_satisfaction = effectiveness_score
            else:
                # Running average
                total_feedback = getattr(session.metrics, '_feedback_count', 0) + 1
                session.metrics.user_satisfaction = (
                    (session.metrics.user_satisfaction * (total_feedback - 1) + effectiveness_score) / total_feedback
                )
                session.metrics._feedback_count = total_feedback
            
            session.updated_at = datetime.now()
            
            logger.info(f"Updated response effectiveness for session {session_id}: {effectiveness_score}")
    
    async def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive session summary"""
        
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        
        return {
            "session_id": session_id,
            "user_id": session.user_id,
            "status": session.status.value,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "duration_minutes": (session.updated_at - session.created_at).total_seconds() / 60,
            
            "grow_progress": {
                "current_phase": session.current_phase.value,
                "phase_history": session.phase_history,
                "goal_statement": session.goal_statement,
                "goal_clarity_score": session.goal_clarity_score
            },
            
            "user_profile": {
                "dominant_vark_type": session.dominant_vark_type.value,
                "vark_confidence": session.vark_confidence,
                "personality_indicators": session.personality_indicators,
                "behavioral_patterns": session.behavioral_patterns
            },
            
            "conversation_stats": {
                "total_turns": session.current_turn,
                "avg_interest_level": session.metrics.avg_interest_level,
                "emotion_distribution": session.metrics.emotion_distribution,
                "sarcasm_incidents": session.metrics.sarcasm_incidents,
                "digression_count": session.metrics.digression_count
            },
            
            "effectiveness": {
                "user_satisfaction": session.metrics.user_satisfaction,
                "phase_durations": session.metrics.phase_durations
            }
        }
    
    async def pause_session(self, session_id: str) -> bool:
        """Pause an active session"""
        
        if session_id not in self.active_sessions:
            return False
        
        async with self.session_locks[session_id]:
            session = self.active_sessions[session_id]
            
            if session.status == SessionStatus.ACTIVE:
                session.status = SessionStatus.PAUSED
                session.updated_at = datetime.now()
                logger.info(f"Paused session {session_id}")
                return True
            
            return False
    
    async def resume_session(self, session_id: str) -> bool:
        """Resume a paused session"""
        
        if session_id not in self.active_sessions:
            return False
        
        async with self.session_locks[session_id]:
            session = self.active_sessions[session_id]
            
            if session.status == SessionStatus.PAUSED:
                session.status = SessionStatus.ACTIVE
                session.updated_at = datetime.now()
                logger.info(f"Resumed session {session_id}")
                return True
            
            return False
    
    async def end_session(self, session_id: str, reason: str = "user_request") -> Optional[Dict[str, Any]]:
        """End a session and return final summary"""
        
        if session_id not in self.active_sessions:
            return None
        
        async with self.session_locks[session_id]:
            session = self.active_sessions[session_id]
            
            # Update final state
            session.status = SessionStatus.COMPLETED
            session.updated_at = datetime.now()
            
            # Complete current phase
            if session.phase_history:
                session.phase_history[-1]["ended_at"] = session.updated_at.isoformat()
                session.phase_history[-1]["completion_reason"] = reason
            
            # Get final summary
            summary = await self.get_session_summary(session_id)
            
            # Export session data for training
            await self._export_session_data(session)
            
            # Clean up
            del self.active_sessions[session_id]
            if session_id in self.session_locks:
                del self.session_locks[session_id]
            
            logger.info(f"Ended session {session_id}, reason: {reason}")
            return summary
    
    async def _initialize_user_preferences(self, session: SessionContext):
        """Initialize session with user's historical preferences from ChromaDB"""
        try:
            results = self.coaching_system.context_collection.query(
                query_texts=["coaching session context"],
                n_results=10,
                where={"user_id": session.user_id}
            )

            vark_counts = {}
            total_interest = 0.0
            total_clarity = 0.0
            valid_contexts = 0

            for meta in results.get("metadatas", []):
                vark = meta.get("vark_type")
                interest = meta.get("interest_level", 0.0)
                clarity = meta.get("goal_clarity_score", 0.0)

                if vark:
                    vark_counts[vark] = vark_counts.get(vark, 0) + 1
                total_interest += interest
                total_clarity += clarity
                valid_contexts += 1

            if vark_counts:
                dominant_vark = max(vark_counts, key=vark_counts.get)
                session.dominant_vark_type = VARKType[dominant_vark.upper()]
                session.vark_confidence = vark_counts[dominant_vark] / sum(vark_counts.values())

            if valid_contexts:
                session.metrics.avg_interest_level = total_interest / valid_contexts
                session.metrics.avg_goal_clarity = total_clarity / valid_contexts

            logger.info(f"Initialized preferences for user {session.user_id}: VARK={session.dominant_vark_type}, confidence={session.vark_confidence:.2f}")
        
        except Exception as e:
            logger.warning(f"Could not load user preferences: {e}")

    
    async def _detect_vark_type(self, utterance: str, session: SessionContext) -> Tuple[VARKType, float]:
        """Use trained VARK model from shared inference pipeline"""
        try:
            vak_str, confidence = self.inference_pipeline.models['vak'].predict(utterance)
            vak_enum = VARKType[vak_str.upper()]
            logger.info(f"Detected VARK type: {vak_enum} with confidence {confidence:.2f} for user {session.user_id}")
            return vak_enum, confidence
        except Exception as e:
            logger.warning(f"VARK detection failed for user {session.user_id}: {e}")
            return session.dominant_vark_type, session.vark_confidence
    
    async def _analyze_behavioral_patterns(
        self, 
        session: SessionContext,
        utterance: str,
        facial_emotion: Dict[str, float],
        voice_emotion: Dict[str, float],
        text_sentiment: Dict[str, float]
    ) -> Dict[str, Any]:
        """Analyze behavioral patterns from multimodal input"""
        
        # Combine emotions to get dominant emotion
        all_emotions = {}
        
        # Weight facial emotions (40%)
        for emotion, score in facial_emotion.items():
            all_emotions[emotion] = all_emotions.get(emotion, 0) + score * 0.4
        
        # Weight voice emotions (40%)
        for emotion, score in voice_emotion.items():
            all_emotions[emotion] = all_emotions.get(emotion, 0) + score * 0.4
        
        # Weight text sentiment (20%)
        for sentiment, score in text_sentiment.items():
            all_emotions[sentiment] = all_emotions.get(sentiment, 0) + score * 0.2
        
        dominant_emotion = max(all_emotions, key=all_emotions.get) if all_emotions else "neutral"
        
        # Calculate interest level (simplified)
        positive_emotions = ["happy", "excited", "interested", "positive"]
        negative_emotions = ["sad", "angry", "frustrated", "bored", "negative"]
        
        positive_score = sum(all_emotions.get(emotion, 0) for emotion in positive_emotions)
        negative_score = sum(all_emotions.get(emotion, 0) for emotion in negative_emotions)
        
        interest_level = max(0.1, min(0.9, 0.5 + (positive_score - negative_score)))
        
        # Placeholder for sarcasm detection (you'd use your trained model)
        sarcasm_detected = False
        sarcasm_confidence = 0.0
        
        # Placeholder for digression detection
        digression_detected = False
        
        return {
            "dominant_emotion": dominant_emotion,
            "interest_level": interest_level,
            "sarcasm_detected": sarcasm_detected,
            "sarcasm_confidence": sarcasm_confidence,
            "digression_detected": digression_detected,
            "emotion_scores": all_emotions
        }
    
    async def _update_user_profile(self, session: SessionContext, vark_type: VARKType, confidence: float):
        """Update user profile with new information"""
        
        # Update VARK type with confidence weighting
        if confidence > session.vark_confidence:
            session.dominant_vark_type = vark_type
            session.vark_confidence = confidence
        
        session.updated_at = datetime.now()
    
    async def _generate_coaching_response(self, context: MultimodalContext) -> str:
        """Generate coaching response using the RAG system"""
        
        # Use thread pool for CPU-intensive operations
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            self.executor,
            self.coaching_system.generate_coaching_response,
            context
        )
        
        return response
    
    async def _update_session_with_response(
        self,
        session: SessionContext,
        context: MultimodalContext,
        response: str,
        context_id: str
    ):
        """Update session state with the generated response"""
        
        # Add to conversation history
        session.conversation_history.append({
            "turn": session.current_turn,
            "timestamp": context.timestamp.isoformat(),
            "user_input": context.utterance,
            "coach_response": response,
            "context_id": context_id,
            "grow_phase": context.grow_phase.value,
            "emotion": self._get_dominant_emotion(context.facial_emotion, context.voice_emotion),
            "interest_level": context.interest_level
        })
        
        # Update goal statement if in GOAL phase
        if context.grow_phase == GROWPhase.GOAL and context.goal_clarity_score > session.goal_clarity_score:
            session.goal_statement = context.utterance
            session.goal_clarity_score = context.goal_clarity_score
        
        session.last_response_id = context_id
        session.updated_at = datetime.now()
    
    async def _check_phase_completion(self, session: SessionContext) -> bool:
        """Check if current phase is sufficiently complete"""
        
        if not session.conversation_history:
            return False
        
        # Simple heuristic: phase is complete if we have enough high-quality interactions
        phase_turns = [
            turn for turn in session.conversation_history
            if turn.get("grow_phase") == session.current_phase.value
        ]
        
        if len(phase_turns) >= 3:  # Minimum turns per phase
            avg_interest = sum(turn.get("interest_level", 0.5) for turn in phase_turns[-3:]) / 3
            return avg_interest > self.phase_completion_threshold
        
        return False
    
    async def _check_phase_transition(
        self, 
        session: SessionContext, 
        context: MultimodalContext
    ) -> Optional[Dict[str, Any]]:
        """Check if phase transition is needed"""
        
        # Check for explicit user request to change phase
        transition_keywords = {
            "goal": ["goal", "objective", "aim", "want to achieve"],
            "reality": ["current", "now", "situation", "where I am"],
            "options": ["options", "alternatives", "could do", "possibilities"],
            "will": ["will do", "commit", "action", "next step"]
        }
        
        utterance_lower = context.utterance.lower()
        
        for phase_name, keywords in transition_keywords.items():
            if any(keyword in utterance_lower for keyword in keywords):
                target_phase = GROWPhase(phase_name)
                if target_phase != session.current_phase:
                    await self.transition_phase(
                        session.session_id,
                        target_phase,
                        PhaseTransitionReason.USER_REQUEST,
                        user_initiated=True
                    )
                    return {
                        "transitioned": True,
                        "from_phase": session.current_phase.value,
                        "to_phase": target_phase.value,
                        "reason": "user_request"
                    }
        
        # Check for natural progression
        if await self._check_phase_completion(session):
            phase_order = [GROWPhase.GOAL, GROWPhase.REALITY, GROWPhase.OPTIONS, GROWPhase.WILL]
            current_index = phase_order.index(session.current_phase)
            
            if current_index < len(phase_order) - 1:
                next_phase = phase_order[current_index + 1]
                await self.transition_phase(
                    session.session_id,
                    next_phase,
                    PhaseTransitionReason.COMPLETION
                )
                return {
                    "transitioned": True,
                    "from_phase": session.current_phase.value,
                    "to_phase": next_phase.value,
                    "reason": "natural_progression"
                }
        
        return {"transitioned": False}
    
    async def _update_session_metrics(self, session: SessionContext, context: MultimodalContext):
        """Update session metrics with new interaction"""
        
        metrics = session.metrics
        
        # Update turn count
        metrics.total_turns = session.current_turn
        
        # Update average interest level
        if metrics.avg_interest_level == 0:
            metrics.avg_interest_level = context.interest_level
        else:
            metrics.avg_interest_level = (
                (metrics.avg_interest_level * (session.current_turn - 1) + context.interest_level) 
                / session.current_turn
            )
        
        # Update average goal clarity
        if metrics.avg_goal_clarity == 0:
            metrics.avg_goal_clarity = context.goal_clarity_score
        else:
            metrics.avg_goal_clarity = (
                (metrics.avg_goal_clarity * (session.current_turn - 1) + context.goal_clarity_score)
                / session.current_turn
            )
        
        # Update emotion distribution
        dominant_emotion = self._get_dominant_emotion(context.facial_emotion, context.voice_emotion)
        metrics.emotion_distribution[dominant_emotion] = metrics.emotion_distribution.get(dominant_emotion, 0) + 1
        
        # Update sarcasm count
        if context.sarcasm_detected:
            metrics.sarcasm_incidents += 1
        
        # Update digression count
        if context.digression_detected:
            metrics.digression_count += 1
        
        # Update engagement trends
        session.engagement_trends.append(context.interest_level)
        if len(session.engagement_trends) > 20:  # Keep only recent trends
            session.engagement_trends = session.engagement_trends[-20:]
    
    def _get_dominant_emotion(self, facial_emotions: Dict, voice_emotions: Dict) -> str:
        """Get dominant emotion from multimodal inputs"""
        
        combined_emotions = {}
        
        # Weight facial emotions (60%)
        for emotion, score in facial_emotions.items():
            combined_emotions[emotion] = combined_emotions.get(emotion, 0) + score * 0.6
        
        # Weight voice emotions (40%)
        for emotion, score in voice_emotions.items():
            combined_emotions[emotion] = combined_emotions.get(emotion, 0) + score * 0.4
        
        return max(combined_emotions, key=combined_emotions.get) if combined_emotions else "neutral"
    
    async def _export_session_data(self, session: SessionContext):
        """Export session data for training purposes"""
        
        try:
            # Export to your training data format
            export_data = {
                "session_summary": await self.get_session_summary(session.session_id),
                "conversation_history": session.conversation_history,
                "behavioral_patterns": session.behavioral_patterns,
                "effectiveness_metrics": {
                    "user_satisfaction": session.metrics.user_satisfaction,
                    "goal_achievement": session.goal_clarity_score,
                    "engagement_score": session.metrics.avg_interest_level
                }
            }
            
            # You could save this to file or database
            logger.info(f"Exported training data for session {session.session_id}")
            
        except Exception as e:
            logger.error(f"Failed to export session data: {e}")
    
    def _start_background_tasks(self):
        """Start background maintenance tasks"""

        async def cleanup_expired_sessions():
            """Clean up expired sessions"""
            logger.info(
                f"⚙️ Cleanup running every {self.cleanup_interval}, session timeout: {self.session_timeout}"
            )

            while True:
                try:
                    await asyncio.sleep(self.cleanup_interval.total_seconds())

                    now = datetime.now()
                    expired_sessions = []

                    for session_id, session in self.active_sessions.items():
                        if (now - session.updated_at) > self.session_timeout:
                            expired_sessions.append(session_id)

                    for session_id in expired_sessions:
                        await self.end_session(session_id, "timeout")
                        logger.info(f"🗑️ Cleaned up expired session: {session_id}")

                except Exception as e:
                    logger.error(f"❌ Error in cleanup task: {e}")

        # Start cleanup task
        try:
            self.cleanup_task = asyncio.create_task(cleanup_expired_sessions())
        except RuntimeError as e:
            logger.warning(f"⚠️ Async loop not available — cleanup task not started: {e}")

    
    async def shutdown(self):
        """Gracefully shutdown the session manager"""
        
        # Cancel background tasks
        if self.cleanup_task:
            self.cleanup_task.cancel()
        if self.analytics_task:
            self.analytics_task.cancel()
        
        # End all active sessions
        session_ids = list(self.active_sessions.keys())
        for session_id in session_ids:
            await self.end_session(session_id, "system_shutdown")
        
        # Shutdown thread pool
        self.executor.shutdown(wait=True)
        
        logger.info("Session manager shutdown complete")

# Usage Example
async def main():
    """Example usage of the session manager"""
    
    from coaching_rag_system import CoachingRAGSystem
    
    # Initialize the coaching system
    coaching_system = CoachingRAGSystem(
        gemini_api_key=GEMINI_API_KEY,
        chroma_persist_dir="./coaching_chroma_db"
    )
    
    # Initialize session manager
    session_manager = SessionManager(coaching_system)
    
    try:
        # Create a session
        session_id = await session_manager.create_session(
            user_id="user_123",
            config={"max_duration_minutes": 60}
        )
        
        # Process multimodal input
        response, response_data = await session_manager.process_multimodal_input(
            session_id=session_id,
            utterance="I want to improve my presentation skills",
            facial_emotion={"nervous": 0.6, "hopeful": 0.4},
            voice_emotion={"uncertain": 0.5, "motivated": 0.5},
            text_sentiment={"positive": 0.6, "negative": 0.4}
        )
        
        print(f"Coach Response: {response}")
        print(f"Response Data: {response_data}")
        
        # Get session summary
        summary = await session_manager.get_session_summary(session_id)
        print(f"Session Summary: {summary}")
        
        # End session
        final_summary = await session_manager.end_session(session_id)
        print(f"Final Summary: {final_summary}")
    except Exception as e:
        logger.error(f"Error during session management: {e}")