import asyncio
import websockets
import json
from typing import Dict, Any, List
import numpy as np
from datetime import datetime
import threading
import queue
import time
from dataclasses import dataclass, asdict

# Assuming the main coaching system is imported
from coaching_rag_system import CoachingRAGSystem, MultimodalContext, GROWPhase, VARKType

class RealTimeCoachingServer:
    """Real-time coaching server that processes multimodal inputs"""
    
    def __init__(self, coaching_system: CoachingRAGSystem):
        self.coaching_system = coaching_system
        self.active_sessions = {}  # session_id -> session_data
        self.processing_queue = queue.Queue()
        self.response_cache = {}  # For handling API rate limits
        
    async def handle_client(self, websocket, path):
        """Handle WebSocket client connections"""
        
        session_id = None
        try:
            async for message in websocket:
                data = json.loads(message)
                
                if data["type"] == "init_session":
                    session_id = await self._init_session(data, websocket)
                    
                elif data["type"] == "multimodal_input":
                    response = await self._process_multimodal_input(data, session_id)
                    await websocket.send(json.dumps(response))
                    
                elif data["type"] == "feedback":
                    await self._process_feedback(data, session_id)
                    
        except websockets.exceptions.ConnectionClosed:
            if session_id:
                await self._cleanup_session(session_id)
    
    async def _init_session(self, data: Dict, websocket) -> str:
        """Initialize a new coaching session"""
        
        session_id = f"session_{datetime.now().timestamp()}"
        
        self.active_sessions[session_id] = {
            "user_id": data["user_id"],
            "websocket": websocket,
            "start_time": datetime.now(),
            "current_phase": GROWPhase.GOAL,
            "conversation_turn": 0,
            "vark_type": VARKType(data.get("vark_type", "visual")),
            "context_history": [],
            "goal_clarity_score": 0.0
        }
        
        # Send initial coaching prompt
        initial_response = {
            "type": "coaching_response",
            "session_id": session_id,
            "message": "Welcome! I'm here to help you achieve your goals using a structured coaching approach. What would you like to work on today?",
            "phase": "goal",
            "suggestions": [
                "Tell me about a specific goal you'd like to achieve",
                "Describe a challenge you're currently facing", 
                "Share what success would look like for you"
            ]
        }
        
        await websocket.send(json.dumps(initial_response))
        return session_id
    
    async def _process_multimodal_input(self, data: Dict, session_id: str) -> Dict:
        """Process multimodal input and generate coaching response"""
        
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        session = self.active_sessions[session_id]
        session["conversation_turn"] += 1
        
        # Parse multimodal inputs
        context = self._parse_multimodal_data(data, session)
        
        # Store context
        context_id = self.coaching_system.store_context(context)
        session["context_history"].append(context_id)
        
        # Generate response (with caching for rate limiting)
        cache_key = self._get_cache_key(context)
        if cache_key in self.response_cache:
            response_text = self.response_cache[cache_key]
        else:
            response_text = self.coaching_system.generate_coaching_response(context)
            self.response_cache[cache_key] = response_text
        
        # Update session state
        session["goal_clarity_score"] = self._update_goal_clarity(context, session["goal_clarity_score"])
        
        # Determine if phase transition is needed
        phase_transition = self._check_phase_transition(context, session)
        if phase_transition:
            session["current_phase"] = phase_transition
        
        return {
            "type": "coaching_response",
            "session_id": session_id,
            "message": response_text,
            "phase": session["current_phase"].value,
            "context_id": context_id,
            "multimodal_analysis": {
                "dominant_emotion": self._get_dominant_emotion(
                    context.facial_emotion, context.voice_emotion
                ),
                "engagement_level": context.interest_level,
                "sarcasm_detected": context.sarcasm_detected,
                "digression_risk": context.digression_detected
            },
            "phase_progress": {
                "current": session["current_phase"].value,
                "completion_score": self._get_phase_completion_score(context, session),
                "suggested_transition": phase_transition.value if phase_transition else None
            }
        }
    
    def _parse_multimodal_data(self, data: Dict, session: Dict) -> MultimodalContext:
        """Parse incoming multimodal data into structured context"""
        
        return MultimodalContext(
            user_id=session["user_id"],
            session_id=data.get("session_id", ""),
            timestamp=datetime.now(),
            grow_phase=session["current_phase"],
            utterance=data.get("utterance", ""),
            
            # Emotion/Sentiment data
            facial_emotion=data.get("facial_emotion", {}),
            voice_emotion=data.get("voice_emotion", {}),
            text_sentiment=data.get("text_sentiment", {}),
            
            # Learning style
            vark_type=session["vark_type"],
            vark_confidence=data.get("vark_confidence", 0.8),
            
            # Behavioral indicators
            sarcasm_detected=data.get("sarcasm_detected", False),
            sarcasm_confidence=data.get("sarcasm_confidence", 0.0),
            interest_level=data.get("interest_level", 0.5),
            digression_detected=data.get("digression_detected", False),
            
            # Context
            conversation_turn=session["conversation_turn"],
            previous_phase_completion=False,  # Calculate this
            goal_clarity_score=session["goal_clarity_score"]
        )
    
    def _get_cache_key(self, context: MultimodalContext) -> str:
        """Generate cache key for similar contexts"""
        
        # Create a simplified representation for caching
        key_components = [
            context.grow_phase.value,
            context.vark_type.value,
            self._get_dominant_emotion(context.facial_emotion, context.voice_emotion),
            str(round(context.interest_level, 1)),
            str(context.sarcasm_detected)
        ]
        
        return "_".join(key_components)
    
    def _update_goal_clarity(self, context: MultimodalContext, current_score: float) -> float:
        """Update goal clarity score based on context"""
        
        # Simple heuristic - you can make this more sophisticated
        if context.grow_phase == GROWPhase.GOAL:
            # Increase clarity if user is providing specific information
            if any(word in context.utterance.lower() for word in ["by", "when", "specific", "measure", "achieve"]):
                return min(current_score + 0.2, 1.0)
        
        return current_score
    
    def _check_phase_transition(self, context: MultimodalContext, session: Dict) -> Optional[GROWPhase]:
        """Determine if it's time to transition to next GROW phase"""
        
        current_phase = session["current_phase"]
        completion_score = self._get_phase_completion_score(context, session)
        
        # Transition if phase is sufficiently complete
        if completion_score > 0.7:
            phase_order = [GROWPhase.GOAL, GROWPhase.REALITY, GROWPhase.OPTIONS, GROWPhase.WILL]
            current_index = phase_order.index(current_phase)
            
            if current_index < len(phase_order) - 1:
                return phase_order[current_index + 1]
        
        return None
    
    def _get_phase_completion_score(self, context: MultimodalContext, session: Dict) -> float:
        """Calculate how complete the current phase is"""
        
        # This is a simplified version - you'd want more sophisticated logic
        phase = session["current_phase"]
        
        if phase == GROWPhase.GOAL:
            return session["goal_clarity_score"]
        elif phase == GROWPhase.REALITY:
            # Check if user has described current situation adequately
            return min(len(session["context_history"]) * 0.25, 1.0)
        elif phase == GROWPhase.OPTIONS:
            # Check if multiple options have been discussed
            return min(session["conversation_turn"] * 0.2, 1.0)
        else:  # WILL
            # Check for commitment indicators
            commitment_words = ["will", "going to", "commit", "plan", "schedule"]
            if any(word in context.utterance.lower() for word in commitment_words):
                return 0.8
            return 0.3
    
    def _get_dominant_emotion(self, facial_emotions: Dict, voice_emotions: Dict) -> str:
        """Get dominant emotion (same as in main system)"""
        combined_emotions = {}
        
        for emotion, score in facial_emotions.items():
            combined_emotions[emotion] = combined_emotions.get(emotion, 0) + score * 0.6
        
        for emotion, score in voice_emotions.items():
            combined_emotions[emotion] = combined_emotions.get(emotion, 0) + score * 0.4
        
        return max(combined_emotions, key=combined_emotions.get) if combined_emotions else "neutral"
    
    async def _process_feedback(self, data: Dict, session_id: str):
        """Process user feedback on coaching responses"""
        
        if session_id not in self.active_sessions:
            return
        
        context_id = data.get("context_id")
        effectiveness_score = data.get("effectiveness_score", 0.5)
        feedback_text = data.get("feedback", "")
        
        # Update response effectiveness in ChromaDB
        response_id = f"response_{context_id}"
        self.coaching_system.update_response_effectiveness(
            response_id, effectiveness_score, feedback_text
        )
        
        # Update session data
        session = self.active_sessions[session_id]
        if "feedback_history" not in session:
            session["feedback_history"] = []
        
        session["feedback_history"].append({
            "context_id": context_id,
            "score": effectiveness_score,
            "feedback": feedback_text,
            "timestamp": datetime.now().isoformat()
        })
    
    async def _cleanup_session(self, session_id: str):
        """Clean up session data"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]


class MultimodalInputProcessor:
    """Process various multimodal inputs"""
    
    def __init__(self):
        self.emotion_classifiers = {
            "facial": FacialEmotionClassifier(),
            "voice": VoiceEmotionClassifier(),
            "text": TextSentimentAnalyzer()
        }
        self.vark_classifier = VARKClassifier()
        self.sarcasm_detector = SarcasmDetector()
        self.interest_tracker = InterestTracker()
    
    def process_inputs(self, audio_data: bytes, video_frame: np.ndarray, text: str) -> Dict:
        """Process all multimodal inputs and return structured data"""
        
        results = {}
        
        # Process facial emotions from video
        if video_frame is not None:
            results["facial_emotion"] = self.emotion_classifiers["facial"].predict(video_frame)
        
        # Process voice emotions from audio
        if audio_data:
            results["voice_emotion"] = self.emotion_classifiers["voice"].predict(audio_data)
        
        # Process text sentiment and other text-based features
        if text:
            results["text_sentiment"] = self.emotion_classifiers["text"].predict(text)
            results["sarcasm_detected"], results["sarcasm_confidence"] = self.sarcasm_detector.predict(text)
            results["vark_type"], results["vark_confidence"] = self.vark_classifier.predict(text)
        
        # Calculate interest level from multimodal cues
        results["interest_level"] = self.interest_tracker.calculate_interest(
            facial_emotion=results.get("facial_emotion", {}),
            voice_emotion=results.get("voice_emotion", {}),
            text_features={"text": text} if text else {}
        )
        
        # Simple digression detection (you'd want more sophisticated logic)
        results["digression_detected"] = self._detect_digression(text)
        
        return results
    
    def _detect_digression(self, text: str) -> bool:
        """Simple digression detection"""
        
        # Keywords that might indicate topic drift
        digression_indicators = [
            "by the way", "speaking of", "that reminds me", "oh wait",
            "actually", "on second thought", "random question"
        ]
        
        return any(indicator in text.lower() for indicator in digression_indicators)


# Mock classifier classes (replace with your actual implementations)
class FacialEmotionClassifier:
    """Mock facial emotion classifier"""
    
    def predict(self, video_frame: np.ndarray) -> Dict[str, float]:
        # Replace with actual facial emotion recognition
        return {
            "happy": np.random.random(),
            "sad": np.random.random(),
            "angry": np.random.random(),
            "surprised": np.random.random(),
            "fearful": np.random.random(),
            "neutral": np.random.random()
        }


class VoiceEmotionClassifier:
    """Mock voice emotion classifier"""
    
    def predict(self, audio_data: bytes) -> Dict[str, float]:
        # Replace with actual voice emotion recognition
        return {
            "happy": np.random.random(),
            "sad": np.random.random(),
            "angry": np.random.random(),
            "excited": np.random.random(),
            "calm": np.random.random(),
            "stressed": np.random.random()
        }


class TextSentimentAnalyzer:
    """Mock text sentiment analyzer"""
    
    def predict(self, text: str) -> Dict[str, float]:
        # Replace with actual sentiment analysis
        return {
            "positive": np.random.random(),
            "negative": np.random.random(),
            "neutral": np.random.random()
        }


class VARKClassifier:
    """Mock VARK learning style classifier"""
    
    def predict(self, text: str) -> tuple:
        # Replace with actual VARK classification
        vark_types = ["visual", "auditory", "reading", "kinesthetic"]
        return np.random.choice(vark_types), np.random.random()


class SarcasmDetector:
    """Mock sarcasm detector"""
    
    def predict(self, text: str) -> tuple:
        # Replace with actual sarcasm detection
        # Simple heuristic: check for certain patterns
        sarcasm_indicators = ["oh great", "wonderful", "fantastic", "perfect"]
        detected = any(indicator in text.lower() for indicator in sarcasm_indicators)
        confidence = np.random.random() if detected else 0.1
        return detected, confidence


class InterestTracker:
    """Track user interest/engagement levels"""
    
    def calculate_interest(self, facial_emotion: Dict, voice_emotion: Dict, text_features: Dict) -> float:
        """Calculate interest level from multimodal inputs"""
        
        interest_score = 0.5  # Base interest
        
        # Facial cues
        if facial_emotion:
            # High engagement emotions
            interest_score += facial_emotion.get("happy", 0) * 0.3
            interest_score += facial_emotion.get("surprised", 0) * 0.2
            
            # Low engagement emotions
            interest_score -= facial_emotion.get("bored", 0) * 0.4
            interest_score -= facial_emotion.get("confused", 0) * 0.2
        
        # Voice cues
        if voice_emotion:
            interest_score += voice_emotion.get("excited", 0) * 0.3
            interest_score += voice_emotion.get("engaged", 0) * 0.2
            interest_score -= voice_emotion.get("monotone", 0) * 0.3
        
        # Text cues
        text = text_features.get("text", "")
        if text:
            # Length can indicate engagement
            if len(text.split()) > 10:
                interest_score += 0.1
            
            # Question words indicate curiosity
            question_words = ["what", "how", "why", "when", "where"]
            if any(word in text.lower() for word in question_words):
                interest_score += 0.2
        
        return max(0.0, min(1.0, interest_score))


class GeminiAPIManager:
    """Manage Gemini API calls with rate limiting and error handling"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.rate_limiter = RateLimiter(calls_per_minute=60)  # Free tier limit
        self.retry_count = 3
        self.fallback_responses = self._load_fallback_responses()
    
    async def generate_response(self, prompt: str, context: MultimodalContext) -> str:
        """Generate response with proper error handling"""
        
        for attempt in range(self.retry_count):
            try:
                # Wait for rate limit
                await self.rate_limiter.wait()
                
                # Make API call
                response = await self._call_gemini_api(prompt)
                return response
                
            except Exception as e:
                if attempt == self.retry_count - 1:
                    # Use fallback response
                    return self._get_fallback_response(context)
                
                # Wait before retry
                await asyncio.sleep(2 ** attempt)
        
        return self._get_fallback_response(context)
    
    async def _call_gemini_api(self, prompt: str) -> str:
        """Make actual API call to Gemini"""
        # Implementation depends on your Gemini API client
        # This is a placeholder
        pass
    
    def _load_fallback_responses(self) -> Dict:
        """Load fallback responses for when API fails"""
        return {
            "goal": [
                "What specific outcome are you hoping to achieve?",
                "Can you help me understand your goal more clearly?",
                "What would success look like for you?"
            ],
            "reality": [
                "Tell me about your current situation.",
                "What's happening right now with this challenge?",
                "Help me understand where you are today."
            ],
            "options": [
                "What are some ways you could approach this?",
                "What options do you see available to you?",
                "Let's explore some different possibilities."
            ],
            "will": [
                "What's your next step?",
                "What are you committed to doing?",
                "When will you take action on this?"
            ]
        }
    
    def _get_fallback_response(self, context: MultimodalContext) -> str:
        """Get appropriate fallback response"""
        phase_responses = self.fallback_responses.get(context.grow_phase.value, ["How can I help you?"])
        return np.random.choice(phase_responses)


class RateLimiter:
    """Simple rate limiter for API calls"""
    
    def __init__(self, calls_per_minute: int):
        self.calls_per_minute = calls_per_minute
        self.calls = []
    
    async def wait(self):
        """Wait if necessary to respect rate limits"""
        now = time.time()
        
        # Remove old calls
        self.calls = [call_time for call_time in self.calls if now - call_time < 60]
        
        # Check if we need to wait
        if len(self.calls) >= self.calls_per_minute:
            wait_time = 60 - (now - self.calls[0])
            if wait_time > 0:
                await asyncio.sleep(wait_time)
        
        # Record this call
        self.calls.append(now)


class DataExporter:
    """Export data for model training and analysis"""
    
    def __init__(self, coaching_system: CoachingRAGSystem):
        self.coaching_system = coaching_system
    
    def export_for_digression_training(self, output_path: str):
        """Export data specifically for training digression detection models"""
        
        # Get all contexts where digression was detected
        all_contexts = self.coaching_system.context_collection.get(
            where={"digression_detected": True},
            include=["documents", "metadatas"]
        )
        
        training_data = []
        for i, doc in enumerate(all_contexts["documents"]):
            metadata = all_contexts["metadatas"][i]
            context_data = json.loads(metadata["context_data"])
            
            training_data.append({
                "text": context_data["utterance"],
                "label": "digression",
                "features": {
                    "grow_phase": context_data["grow_phase"],
                    "interest_level": context_data["interest_level"],
                    "conversation_turn": context_data["conversation_turn"]
                }
            })
        
        # Also get non-digression examples
        non_digression_contexts = self.coaching_system.context_collection.get(
            where={"digression_detected": False},
            include=["documents", "metadatas"]
        )
        
        for i, doc in enumerate(non_digression_contexts["documents"]):
            metadata = non_digression_contexts["metadatas"][i]
            context_data = json.loads(metadata["context_data"])
            
            training_data.append({
                "text": context_data["utterance"],
                "label": "on_topic",
                "features": {
                    "grow_phase": context_data["grow_phase"],
                    "interest_level": context_data["interest_level"],
                    "conversation_turn": context_data["conversation_turn"]
                }
            })
        
        with open(output_path, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        return len(training_data)
    
    def export_effectiveness_analysis(self, output_path: str):
        """Export data for analyzing coaching effectiveness"""
        
        # Get all responses with effectiveness scores
        responses = self.coaching_system.response_collection.get(
            include=["documents", "metadatas"]
        )
        
        analysis_data = []
        for i, doc in enumerate(responses["documents"]):
            metadata = responses["metadatas"][i]
            
            if "effectiveness_score" in metadata:
                context_summary = json.loads(metadata.get("context_summary", "{}"))
                
                analysis_data.append({
                    "response": doc,
                    "effectiveness_score": metadata["effectiveness_score"],
                    "grow_phase": metadata["grow_phase"],
                    "user_emotion": context_summary.get("emotion", "neutral"),
                    "vark_type": context_summary.get("vark", "unknown"),
                    "interest_level": context_summary.get("interest", 0.5),
                    "feedback": metadata.get("feedback", "")
                })
        
        with open(output_path, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        return len(analysis_data)


# Complete usage example
async def run_coaching_server():
    """Run the complete coaching server"""
    
    # Initialize the coaching system
    coaching_system = CoachingRAGSystem(
        gemini_api_key="your_gemini_api_key_here",
        chroma_persist_dir="./coaching_chroma_db"
    )
    
    # Initialize the real-time server
    server = RealTimeCoachingServer(coaching_system)
    
    # Start WebSocket server
    start_server = websockets.serve(server.handle_client, "localhost", 8765)
    
    print("Coaching server started on ws://localhost:8765")
    await start_server


# Example client usage
async def example_client():
    """Example client that sends multimodal data"""
    
    uri = "ws://localhost:8765"
    
    async with websockets.connect(uri) as websocket:
        # Initialize session
        init_message = {
            "type": "init_session",
            "user_id": "user_123",
            "vark_type": "visual"
        }
        await websocket.send(json.dumps(init_message))
        response = await websocket.recv()
        print(f"Server: {json.loads(response)['message']}")
        
        # Send multimodal input
        input_message = {
            "type": "multimodal_input",
            "utterance": "I want to improve my presentation skills but I get really nervous",
            "facial_emotion": {"nervous": 0.8, "hopeful": 0.2},
            "voice_emotion": {"anxious": 0.7, "determined": 0.3},
            "text_sentiment": {"positive": 0.3, "negative": 0.7},
            "sarcasm_detected": False,
            "sarcasm_confidence": 0.1,
            "interest_level": 0.7,
            "digression_detected": False
        }
        await websocket.send(json.dumps(input_message))
        response = await websocket.recv()
        
        response_data = json.loads(response)
        print(f"Coach: {response_data['message']}")
        print(f"Analysis: {response_data['multimodal_analysis']}")
        
        # Send feedback
        feedback_message = {
            "type": "feedback",
            "context_id": response_data["context_id"],
            "effectiveness_score": 0.8,
            "feedback": "Very helpful response, made me feel understood"
        }
        await websocket.send(json.dumps(feedback_message))


if __name__ == "__main__":
    # Run the server
    asyncio.run(run_coaching_server())