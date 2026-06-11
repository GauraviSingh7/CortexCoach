"""
Core Orchestrator - WITH IMPROVED SARCASM & VAK DETECTION
All models working properly with fallbacks
FULLY CORRECTED VERSION - Ready to use
"""
import asyncio
import logging
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, Set, List
from collections import Counter
import json

from backend.models.audio_processor import AudioProcessor
from backend.models.inference_engine import ModelInferenceEngine
from backend.models.gemini_analyzer import GeminiAnalyzer
from backend.models.file_audio_processor import FileAudioProcessor
from backend.models.enhanced_local_analyzer import EnhancedLocalAnalyzer
from backend.models.contextual_suggestion_engine import ContextualSuggestionEngine
from backend.schemas.data_models import AudioChunk, RealTimeFeedback, GROWPhase, SessionReport
from backend.models.storage import ChromaDBStorage

logger = logging.getLogger(__name__)


class CoachingObserverSystem:
    """Main orchestrator for the AI Coaching Observer system"""

    def __init__(self, assemblyai_key: str, gemini_key: str):
        self.assemblyai_key = assemblyai_key
        self.gemini_key = gemini_key
        
        # Core components
        self.audio_processor: Optional[AudioProcessor] = None
        self.file_processor: Optional[FileAudioProcessor] = None
        self.inference_engine = ModelInferenceEngine()
        self.gemini_analyzer = GeminiAnalyzer(gemini_key) if gemini_key else None
        try:
            self.storage = ChromaDBStorage()
        except Exception as e:
            logger.warning(f"ChromaDB storage unavailable: {e}")
            self.storage = None
        
        # Enhanced analyzers
        self.local_analyzer = EnhancedLocalAnalyzer()
        self.suggestion_engine = ContextualSuggestionEngine()
        
        # Session state
        self.session_id: Optional[str] = None
        self.session_active = False
        self.session_data: Dict[str, Any] = {}
        # Last successfully generated report — returned by /session/stop
        # retries so the endpoint is idempotent.
        self.last_report: Optional[SessionReport] = None
        
        # Audio queue for chunks
        self.audio_queue: Optional[asyncio.Queue] = None
        
        # WebSocket clients for real-time updates
        self.websocket_clients: Set = set()
        
        # Background tasks
        self.processing_task: Optional[asyncio.Task] = None
        
        logger.info("✅ CoachingObserverSystem initialized with enhanced analyzers")

    async def start_session(
        self,
        session_type: str = "live",
        device_index: Optional[int] = None,
        file_path: Optional[str] = None,
        coach_speaker_id: Optional[str] = None
    ) -> str:
        """
        Start a new coaching session

        Args:
            session_type: "live" or "file"
            device_index: Microphone device index (live mode only)
            file_path: Path to audio file (file mode only)
            coach_speaker_id: For file mode, specify which speaker is coach ("A" or "B").
                            If None, uses heuristic detection.
        """
        if self.session_active:
            raise RuntimeError("A session is already active")
        
        self.session_id = str(uuid.uuid4())
        self.session_active = True
        # New session — drop the cached previous report.
        self.last_report = None

        self.session_data = {
            "session_id": self.session_id,
            "start_time": datetime.now(),
            "chunks": [],
            "feedback_history": [],
            "type": session_type,
            "file_path": file_path,
            "sarcasm_detections": [],
            "digression_scores": [],
            "vak_scores": []
        }
        
        logger.info(f"🚀 Starting session {self.session_id} (type: {session_type})")
        
        try:
            self.audio_queue = asyncio.Queue()
            if session_type == "file":
                if not file_path:
                    raise ValueError("file_path required for file mode")

                self.file_processor = FileAudioProcessor(self.assemblyai_key, coach_speaker_id)
                if coach_speaker_id:
                    logger.info(f"📁 File mode: {file_path} (Speaker {coach_speaker_id} = coach)")
                else:
                    logger.info(f"📁 File mode: {file_path} (using heuristic detection)")

                self.processing_task = asyncio.create_task(self._processing_pipeline())
                logger.info("✅ Processing pipeline started")

                asyncio.create_task(self.file_processor.process_file(file_path, self.audio_queue))

            else:
                self.audio_processor = AudioProcessor(
                    self.assemblyai_key,
                    coach_speaker_id=coach_speaker_id
                )

                await self.audio_processor.start_live_transcription(
                    self.audio_queue,
                    device_index=device_index
                )
                
                logger.info("✅ Audio processor initialized and connected to AssemblyAI")
                
                self.processing_task = asyncio.create_task(self._processing_pipeline())
                logger.info("✅ Processing pipeline started")
            
            return self.session_id
            
        except Exception as e:
            error_msg = f"Failed to start session: {str(e)}"
            logger.error(f"❌ {error_msg}", exc_info=True)
            self.session_active = False
            if self.audio_processor:
                try:
                    await self.audio_processor.stop_transcription()
                except:
                    pass
            raise RuntimeError(error_msg) from e

    async def _processing_pipeline(self):
        """Main processing pipeline for audio chunks"""
        logger.info("🔄 Pipeline started")
        
        while self.session_active:
            try:
                chunk = await asyncio.wait_for(
                    self.audio_queue.get(),
                    timeout=1.0
                )
                
                await self._process_chunk(chunk)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                if self.session_active:
                    logger.error(f"❌ Pipeline error: {e}", exc_info=True)
        
        logger.info("⏹️ Pipeline stopped")

    async def _process_chunk(self, chunk: AudioChunk):
        """Process a single audio chunk - WITH IMPROVED SARCASM & VAK DETECTION"""
        try:
            logger.info(f"🔄 Processing chunk from {chunk.speaker}: {chunk.transcript[:50]}...")

            # Partial turns: broadcast live text update only, skip expensive ML
            if not chunk.is_final:
                await self._broadcast_partial(chunk)
                return

            self.session_data["chunks"].append(chunk)
            conversation_history = self.session_data["chunks"]
            
            # Run ML inference
            inferences = await self.inference_engine.process_chunk(chunk)
            
            # Only invoke text-based fallback when the emotion model produced nothing.
            # Empty dict → "Not Available"; never synthesize a neutral default.
            if not inferences.emotion:
                text_emotions = self._analyze_emotion_from_text(chunk.transcript)
                if text_emotions:
                    inferences.emotion = text_emotions
                    logger.info(f"🎭 Text-based emotion: {max(text_emotions.items(), key=lambda x: x[1])}")
            
            # IMPROVED: SARCASM DETECTION
            sarcasm_result = self._detect_sarcasm_improved(chunk, conversation_history)
            inferences.sarcasm_score = sarcasm_result['score']
            
            # Log if sarcasm detected
            if sarcasm_result['is_sarcastic']:
                logger.warning(f"😏 Sarcasm detected ({sarcasm_result['type']}): {chunk.transcript[:80]}")
            
            # REAL DIGRESSION DETECTION
            digression_score = self._detect_digression(chunk, conversation_history)
            inferences.digression_score = digression_score
            
            # IMPROVED: VAK DETECTION (only from coachee)
            vak_result = self._detect_vak_improved(chunk, conversation_history)
            
            logger.info(f"📊 Digression: {digression_score:.2f} | Sarcasm: {sarcasm_result['score']:.2f} | VAK: {vak_result['dominant']}")
            
            grow_phase = await self._analyze_grow_phase(chunk, inferences)
            engagement_score = inferences.interest_level
            coaching_quality = await self._assess_coaching_quality(chunk, inferences, grow_phase)
            suggestions = await self._generate_suggestions(chunk, inferences, grow_phase, sarcasm_result)
            
            feedback = RealTimeFeedback(
                timestamp=chunk.timestamp,
                speaker=chunk.speaker,
                grow_phase=grow_phase,
                emotion_trend=inferences.emotion,
                engagement_score=engagement_score,
                coaching_quality=coaching_quality,
                suggestions=suggestions
            )
            
            self.session_data["feedback_history"].append(feedback)
            
            # Store sarcasm, digression, and VAK
            self.session_data["sarcasm_detections"].append({
                'timestamp': chunk.timestamp,
                'speaker': chunk.speaker,
                'score': sarcasm_result['score'],
                'type': sarcasm_result['type'],
                'text': chunk.transcript[:100]
            })
            self.session_data["digression_scores"].append(digression_score)
            self.session_data["vak_scores"].append(vak_result)
            
            await self._broadcast_feedback(feedback, digression_score, sarcasm_result, vak_result)
            
            logger.info(f"✅ Processed: {grow_phase.phase} | Engagement: {engagement_score:.2f} | Digression: {digression_score:.2f} | Sarcasm: {sarcasm_result['score']:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Chunk processing error: {e}", exc_info=True)

    # ========================================================================
    # IMPROVED SARCASM DETECTION
    # ========================================================================
    
    def _detect_sarcasm_improved(self, chunk: AudioChunk, conversation_history: List[AudioChunk]) -> Dict:
        """
        IMPROVED sarcasm detection with better patterns
        
        Returns:
            {
                'score': float,
                'explanation': str,
                'type': str,
                'is_sarcastic': bool
            }
        """
        try:
            text = chunk.transcript.lower()
            score = 0.0
            sarcasm_type = "none"
            explanation = "No sarcasm detected"
            
            # PATTERN 1: Exaggerated enthusiasm
            exaggeration_words = ['absolutely', 'totally', 'completely', 'definitely', 'obviously', 
                                 'clearly', 'sure', 'perfect', 'wonderful', 'fantastic', 'brilliant']
            if any(word in text for word in exaggeration_words):
                # Check if followed by negative context
                if any(neg in text for neg in ['not', 'never', "can't", "won't", "couldn't"]):
                    score += 0.4
                    sarcasm_type = "mock_enthusiasm"
                    explanation = "Exaggerated positive word with negative context"
                # Or check if too many exclamations
                elif text.count('!') >= 2:
                    score += 0.3
                    sarcasm_type = "mock_enthusiasm"
            
            # PATTERN 2: "Yeah right" / "Oh great" patterns
            sarcastic_phrases = [
                'yeah right', 'oh great', 'how wonderful', 'just perfect',
                'exactly what i needed', 'oh joy', 'fantastic news',
                'couldn\'t be better', 'just what i wanted'
            ]
            for phrase in sarcastic_phrases:
                if phrase in text:
                    score += 0.6
                    sarcasm_type = "sarcastic_phrase"
                    explanation = f"Common sarcastic phrase: '{phrase}'"
                    break
            
            # PATTERN 3: Rhetorical questions implying opposite
            rhetorical_sarcasm = [
                'you think', 'you really believe', 'you expect me to',
                'do you seriously', 'are you kidding', 'you must be joking'
            ]
            if '?' in chunk.transcript:
                for phrase in rhetorical_sarcasm:
                    if phrase in text:
                        score += 0.5
                        sarcasm_type = "disbelief"
                        explanation = "Rhetorical question expressing disbelief"
                        break
            
            # PATTERN 4: Contradictory statements
            positive_words = ['good', 'great', 'nice', 'fine', 'okay', 'happy', 'love']
            negative_indicators = ['but', 'except', 'however', 'unfortunately', 'sadly']
            
            has_positive = any(word in text for word in positive_words)
            has_negative = any(word in text for word in negative_indicators)
            
            if has_positive and has_negative:
                score += 0.3
                sarcasm_type = "contradiction"
                explanation = "Positive word followed by negative context"
            
            # PATTERN 5: Passive-aggressive politeness
            passive_aggressive = [
                'no offense but', 'with all due respect', 'i mean', 
                'not to be rude but', 'just saying', 'no disrespect'
            ]
            for phrase in passive_aggressive:
                if phrase in text:
                    score += 0.5
                    sarcasm_type = "passive_aggressive"
                    explanation = f"Passive-aggressive phrase: '{phrase}'"
                    break
            
            # PATTERN 6: Check conversation context
            if len(conversation_history) >= 2:
                previous_chunk = conversation_history[-2]
                # If previous was coach asking something, and coachee responds with short positive
                if previous_chunk.speaker == 'coach' and chunk.speaker == 'coachee':
                    if len(chunk.transcript.split()) <= 4:  # Short response
                        if any(word in text for word in ['sure', 'fine', 'whatever', 'okay']):
                            score += 0.3
                            sarcasm_type = "dismissive"
                            explanation = "Short dismissive response"
            
            # Cap score at 1.0
            score = min(score, 1.0)
            
            # Determine if sarcastic (threshold at 0.4)
            is_sarcastic = score > 0.4
            
            if is_sarcastic:
                logger.info(f"😏 Sarcasm: {score:.2f} - {explanation}")
            
            return {
                'score': score,
                'explanation': explanation,
                'type': sarcasm_type if is_sarcastic else 'none',
                'is_sarcastic': is_sarcastic
            }
            
        except Exception as e:
            logger.error(f"Sarcasm detection error: {e}")
            return {
                'score': 0.0,
                'explanation': 'detection failed',
                'type': 'none',
                'is_sarcastic': False
            }

    # ========================================================================
    # IMPROVED VAK LEARNING STYLE DETECTION
    # ========================================================================
    
    def _detect_vak_improved(self, chunk: AudioChunk, conversation_history: List[AudioChunk]) -> Dict:
        """
        IMPROVED VAK detection - only analyzes coachee speech patterns
        Returns current VAK style with confidence
        """
        try:
            # Only analyze coachee chunks
            coachee_chunks = [c for c in conversation_history if c.speaker == "coachee"]
            
            if len(coachee_chunks) < 2:
                return {
                    "visual": 0.33,
                    "auditory": 0.33,
                    "kinesthetic": 0.34,
                    "dominant": "Unknown",
                    "confidence": 0.0
                }
            
            # Analyze recent coachee chunks (last 10)
            recent_coachee = coachee_chunks[-10:]
            
            # Stronger keyword indicators
            visual_patterns = {
                'strong': ['see', 'look', 'picture', 'imagine', 'visualize', 'view', 'watch', 'show me'],
                'medium': ['appears', 'bright', 'clear', 'focus', 'perspective', 'illustrate'],
                'phrases': ['i can see', 'looks like', 'picture this', 'from my perspective', 
                           'the way i see it', 'let me show you']
            }
            
            auditory_patterns = {
                'strong': ['hear', 'listen', 'sound', 'tell', 'say', 'talk', 'discuss', 'mention'],
                'medium': ['voice', 'tone', 'loud', 'quiet', 'resonate', 'harmonize'],
                'phrases': ['sounds like', 'i hear you', 'listen to this', 'tell me about', 
                           'that rings a bell', 'word for word']
            }
            
            kinesthetic_patterns = {
                'strong': ['feel', 'touch', 'grasp', 'hold', 'sense', 'experience', 'do', 'handle'],
                'medium': ['move', 'concrete', 'solid', 'pressure', 'comfortable', 'flow'],
                'phrases': ['i feel like', 'get a grip', 'hands on', 'gut feeling', 
                           'my sense is', 'concrete example']
            }
            
            visual_score = 0.0
            auditory_score = 0.0
            kinesthetic_score = 0.0
            
            for chunk_item in recent_coachee:
                text_lower = chunk_item.transcript.lower()
                
                # Visual scoring
                for word in visual_patterns['strong']:
                    if word in text_lower:
                        visual_score += 3
                for word in visual_patterns['medium']:
                    if word in text_lower:
                        visual_score += 1
                for phrase in visual_patterns['phrases']:
                    if phrase in text_lower:
                        visual_score += 5
                
                # Auditory scoring
                for word in auditory_patterns['strong']:
                    if word in text_lower:
                        auditory_score += 3
                for word in auditory_patterns['medium']:
                    if word in text_lower:
                        auditory_score += 1
                for phrase in auditory_patterns['phrases']:
                    if phrase in text_lower:
                        auditory_score += 5
                
                # Kinesthetic scoring
                for word in kinesthetic_patterns['strong']:
                    if word in text_lower:
                        kinesthetic_score += 3
                for word in kinesthetic_patterns['medium']:
                    if word in text_lower:
                        kinesthetic_score += 1
                for phrase in kinesthetic_patterns['phrases']:
                    if phrase in text_lower:
                        kinesthetic_score += 5
            
            # Normalize scores
            total_score = visual_score + auditory_score + kinesthetic_score
            
            if total_score == 0:
                return {
                    "visual": 0.33,
                    "auditory": 0.33,
                    "kinesthetic": 0.34,
                    "dominant": "Balanced (Mixed)",
                    "confidence": 0.1
                }
            
            visual_pct = visual_score / total_score
            auditory_pct = auditory_score / total_score
            kinesthetic_pct = kinesthetic_score / total_score
            
            # Determine dominant style
            max_score = max(visual_pct, auditory_pct, kinesthetic_pct)
            
            if visual_pct == max_score:
                dominant_style = "Visual"
                confidence = visual_pct
            elif auditory_pct == max_score:
                dominant_style = "Auditory"
                confidence = auditory_pct
            else:
                dominant_style = "Kinesthetic"
                confidence = kinesthetic_pct
            
            # Format output
            if confidence < 0.4:
                dominant_label = "Balanced (Mixed)"
            else:
                dominant_label = f"{dominant_style} ({confidence:.0%})"
            
            logger.info(f"👁️👂✋ VAK: V={visual_pct:.2f}, A={auditory_pct:.2f}, K={kinesthetic_pct:.2f} → {dominant_label}")
            
            return {
                "visual": visual_pct,
                "auditory": auditory_pct,
                "kinesthetic": kinesthetic_pct,
                "dominant": dominant_label,
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"VAK detection error: {e}")
            return {
                "visual": 0.33,
                "auditory": 0.33,
                "kinesthetic": 0.34,
                "dominant": "Unknown",
                "confidence": 0.0
            }

    # ========================================================================
    # EMOTION DETECTION
    # ========================================================================
    
    def _analyze_emotion_from_text(self, text: str) -> Optional[Dict[str, float]]:
        """Text-based emotion detection fallback"""
        text_lower = text.lower()
        
        # Positive emotions
        if any(word in text_lower for word in ['happy', 'great', 'excellent', 'ecstatic', 'joy', 'wonderful', 'amazing', 'love', 'fantastic']):
            return {"happy": 0.75, "excited": 0.15, "neutral": 0.1}
        
        # Excited/energized
        if any(word in text_lower for word in ['excited', 'pumped', 'thrilled', 'eager', 'energized', 'enthusiastic']):
            return {"excited": 0.75, "happy": 0.15, "neutral": 0.1}
        
        # Sad/down
        if any(word in text_lower for word in ['sad', 'depressed', 'down', 'unhappy', 'boring', 'miserable', 'blue', 'disappointed']):
            return {"sad": 0.75, "calm": 0.15, "neutral": 0.1}
        
        # Angry/frustrated
        if any(word in text_lower for word in ['angry', 'mad', 'frustrated', 'irritated', 'furious', 'annoyed', 'hate']):
            return {"angry": 0.75, "fearful": 0.15, "neutral": 0.1}
        
        # Fearful/anxious
        if any(word in text_lower for word in ['worried', 'anxious', 'scared', 'afraid', 'nervous', 'fearful', 'terrified']):
            return {"fearful": 0.75, "sad": 0.15, "neutral": 0.1}
        
        # No strong emotion detected
        return None

    # ========================================================================
    # DIGRESSION DETECTION
    # ========================================================================
    
    def _detect_digression(self, chunk: AudioChunk, conversation_history: List[AudioChunk]) -> float:
        """
        Detect off-topic drift via Jaccard overlap of content words against
        a rolling topic window of recent turns. Explicit digression markers
        short-circuit to a high score. Short utterances (backchannels,
        coaching prompts like "What else?") and early-session turns return
        0.0 — they cannot be evaluated reliably and previously caused most
        of the false-positive firings.
        """
        # Need enough prior context to build a meaningful topic vocabulary.
        if len(conversation_history) < 6:
            return 0.0

        current_text = chunk.transcript.lower()

        # High-precision: explicit digression cues.
        digression_phrases = [
            'by the way', 'speaking of', 'that reminds me', 'off topic',
            'random thought', 'unrelated', 'change the subject', 'oh also',
            'on another note', 'totally different'
        ]
        if any(phrase in current_text for phrase in digression_phrases):
            return 0.7

        current_keywords = set(self._extract_topic_keywords([chunk]))
        # Too few content words to judge — treat as on-topic.
        if len(current_keywords) < 2:
            return 0.0

        # Topic vocabulary from the last 10 turns (excluding the current one).
        topic_window = conversation_history[-11:-1]
        topic_vocab = set(self._extract_topic_keywords(topic_window))
        if not topic_vocab:
            return 0.0

        union = current_keywords | topic_vocab
        jaccard = len(current_keywords & topic_vocab) / len(union) if union else 0.0

        # Thresholds tuned so normal topic shifts (e.g. Goal → Reality)
        # don't register as digressions.
        if jaccard < 0.05:
            return 0.6
        if jaccard < 0.15:
            return 0.3
        return 0.1
    
    def _extract_topic_keywords(self, chunks: List[AudioChunk]) -> List[str]:
        """Extract main topic keywords from chunks"""
        if not chunks:
            return []
        
        combined_text = ' '.join([c.transcript.lower() for c in chunks])
        
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'is', 'was', 'are', 'were', 'been', 'be', 'have', 'has',
            'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may',
            'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
            'she', 'it', 'we', 'they', 'my', 'your', 'what', 'how', 'why', 'when'
        }
        
        words = combined_text.split()
        keywords = []
        
        for word in words:
            clean_word = word.strip('.,!?;:"()[]')
            
            if (len(clean_word) > 4 and 
                clean_word not in stop_words and 
                not clean_word.isdigit()):
                keywords.append(clean_word)
        
        keyword_counts = Counter(keywords)
        return [kw for kw, count in keyword_counts.most_common(5)]

    # ========================================================================
    # GROW PHASE ANALYSIS
    # ========================================================================

    async def _analyze_grow_phase(self, chunk: AudioChunk, inferences) -> GROWPhase:
        """Analyze GROW model phase with enhanced detection"""
        try:
            transcript_lower = chunk.transcript.lower()
            
            goal_keywords = ["goal", "want", "achieve", "objective", "aim", "target", "hope to", "wish", "aspire", "become"]
            reality_keywords = ["currently", "now", "situation", "reality", "actually", "right now", "present", 
                              "today", "at the moment", "happening", "problem", "challenge", "issue", "struggle"]
            options_keywords = ["option", "could", "might", "alternative", "what if", "perhaps", "maybe", 
                               "possibility", "choice", "consider", "explore", "idea"]
            wayforward_keywords = ["will", "plan", "next", "action", "step", "commit", "going to", "shall", 
                                  "decide", "do", "start", "begin", "implement"]
            
            goal_score = sum(2 if kw in transcript_lower else 0 for kw in goal_keywords)
            reality_score = sum(2 if kw in transcript_lower else 0 for kw in reality_keywords)
            options_score = sum(2 if kw in transcript_lower else 0 for kw in options_keywords)
            way_score = sum(2 if kw in transcript_lower else 0 for kw in wayforward_keywords)
            
            if "?" in chunk.transcript:
                if any(word in transcript_lower for word in ["want", "goal", "achieve"]):
                    goal_score += 3
                elif any(word in transcript_lower for word in ["what if", "could you"]):
                    options_score += 3
            
            scores = {
                "Goal": goal_score,
                "Reality": reality_score,
                "Options": options_score,
                "Way Forward": way_score
            }
            
            max_phase = max(scores.items(), key=lambda x: x[1])
            phase = max_phase[0]
            raw_score = max_phase[1]
            
            if raw_score >= 6:
                confidence = 0.9
            elif raw_score >= 4:
                confidence = 0.75
            elif raw_score >= 2:
                confidence = 0.6
            else:
                # No evidence — do NOT default to Reality.
                phase = "Uncertain"
                confidence = 0.0

            reasoning = f"Detected {raw_score} phase indicators"

            return GROWPhase(
                phase=phase,
                confidence=confidence,
                reasoning=reasoning
            )
        except Exception as e:
            logger.error(f"GROW analysis error: {e}")
            return GROWPhase(phase="Uncertain", confidence=0.0, reasoning="Error in analysis")

    async def _assess_coaching_quality(self, chunk: AudioChunk, inferences, grow_phase: GROWPhase) -> Dict[str, float]:
        """Assess coaching quality metrics — evidence-only, coach turns only."""
        try:
            # Coaching quality is only meaningful for coach utterances.
            if chunk.speaker != "coach":
                return {}

            transcript_lower = chunk.transcript.lower()

            # Score only from observed evidence; no inflated baselines.
            questioning_score = 0.0
            if "?" in chunk.transcript:
                if any(word in transcript_lower for word in ["what", "how", "why", "tell me"]):
                    questioning_score = 0.9
                else:
                    questioning_score = 0.5  # closed question observed

            listening_score = 0.0
            if any(word in transcript_lower for word in ["understand", "hear", "sounds like", "i see", "got it", "tell me more", "go on"]):
                listening_score = 0.85

            components = [s for s in (questioning_score, listening_score, grow_phase.confidence) if s > 0]
            overall = sum(components) / len(components) if components else 0.0

            return {
                "overall": overall,
                "questioning": questioning_score,
                "listening": listening_score
            }
        except Exception as e:
            logger.error(f"Quality assessment error: {e}")
            return {}

    async def _generate_suggestions(self, chunk: AudioChunk, inferences, grow_phase: GROWPhase, sarcasm_result: Dict = None) -> list:
        """Generate coaching suggestions - WITH SARCASM AWARENESS"""
        try:
            conversation_history = self.session_data["chunks"][-10:]
            
            # Use contextual suggestion engine
            suggestions = self.suggestion_engine.generate_suggestions(
                chunk=chunk,
                inferences=inferences,
                grow_phase=grow_phase,
                conversation_history=conversation_history
            )
            
            # NEW: Add sarcasm-specific suggestions
            if sarcasm_result and sarcasm_result['is_sarcastic']:
                if chunk.speaker == 'coachee':
                    if sarcasm_result['type'] == 'passive_aggressive':
                        suggestions.insert(0, "🚨 Passive-aggressive language detected. Explore what's really bothering them: 'What's frustrating you about this?'")
                    elif sarcasm_result['type'] == 'disbelief':
                        suggestions.insert(0, "😏 Coachee expressing doubt/disbelief. Acknowledge and explore: 'I sense some skepticism. What concerns you?'")
                    elif sarcasm_result['type'] == 'mock_enthusiasm':
                        suggestions.insert(0, "⚠️ Sarcasm detected (mock enthusiasm). Address underlying frustration: 'You seem frustrated. What's not working?'")
                    elif sarcasm_result['type'] == 'dismissive':
                        suggestions.insert(0, "⚠️ Short dismissive response detected. Dig deeper: 'Can you tell me more about that?'")
                    else:
                        suggestions.insert(0, f"😏 Sarcasm detected. This may indicate resistance or frustration. Explore deeper: 'What's really going on here?'")
                
                elif chunk.speaker == 'coach':
                    suggestions.insert(0, "⚠️ Your tone may come across as sarcastic. Stay authentic and supportive.")
            
            # Optional: Gemini AI enhancement
            chunk_count = len(self.session_data["chunks"])
            if (chunk_count % 5 == 0) and chunk.speaker == "coach" and self.gemini_analyzer and self.gemini_analyzer.model:
                try:
                    context = "\n".join([f"{c.speaker}: {c.transcript}" for c in conversation_history[-5:]])
                    prompt = f"""You are an expert coaching advisor. Based on this conversation, give ONE brief, actionable suggestion for the coach.

Recent conversation:
{context}

Current GROW phase: {grow_phase.phase}
{"Sarcasm detected in coachee response - may indicate resistance" if sarcasm_result and sarcasm_result['is_sarcastic'] else ""}

Provide ONE specific, actionable suggestion (max 15 words)."""

                    response = await self.gemini_analyzer.model.generate_content_async(prompt)
                    ai_suggestion = response.text.strip().replace('\n', ' ')[:150]
                    suggestions.insert(0, f"🤖 {ai_suggestion}")
                    
                except Exception as e:
                    logger.warning(f"Gemini suggestion failed: {e}")
            
            return suggestions[:7]
            
        except Exception as e:
            logger.error(f"Suggestion generation error: {e}", exc_info=True)
            return [
                "Continue with active listening",
                "Ask 'What else?' to explore deeper",
                "Reflect back what you're hearing"
            ]

    async def _broadcast_partial(self, chunk: AudioChunk):
        """Broadcast a partial (in-progress) utterance — no ML data, just live text."""
        if not self.websocket_clients:
            return
        message = json.dumps({
            "type": "partial",
            "speaker": chunk.speaker,
            "speaker_id": chunk.speaker_id,
            "transcript": chunk.transcript,
            "timestamp": chunk.timestamp,
        })
        disconnected = set()
        for client in self.websocket_clients:
            try:
                await client.send_text(message)
            except Exception:
                disconnected.add(client)
        self.websocket_clients -= disconnected

    async def _broadcast_feedback(self, feedback: RealTimeFeedback, digression_score: float, sarcasm_result: Dict, vak_result: Dict):
        """Broadcast feedback - WITH SARCASM AND VAK DATA"""
        if not self.websocket_clients:
            logger.debug("No WebSocket clients connected")
            return
        
        try:
            latest_chunk = self.session_data["chunks"][-1] if self.session_data["chunks"] else None
            
            # Get VAK learning style
            learning_style = vak_result['dominant']
            
            feedback_dict = {
                "type": "final",
                "timestamp": feedback.timestamp,
                "speaker": feedback.speaker,
                "speaker_id": latest_chunk.speaker_id if latest_chunk else None,
                "transcript": latest_chunk.transcript if latest_chunk else "",
                "grow_phase": {
                    "phase": feedback.grow_phase.phase,
                    "confidence": feedback.grow_phase.confidence,
                    "reasoning": feedback.grow_phase.reasoning
                },
                "emotion_trend": feedback.emotion_trend,
                "engagement_score": feedback.engagement_score,
                "coaching_quality": feedback.coaching_quality,
                "suggestions": feedback.suggestions,
                "learning_style": learning_style,
                "digression_level": digression_score,
                # Sarcasm data
                "sarcasm_detected": sarcasm_result['is_sarcastic'],
                "sarcasm_score": sarcasm_result['score'],
                "sarcasm_type": sarcasm_result['type'],
                # VAK data
                "vak_visual": vak_result['visual'],
                "vak_auditory": vak_result['auditory'],
                "vak_kinesthetic": vak_result['kinesthetic'],
                "vak_confidence": vak_result['confidence']
            }
            
            message = json.dumps(feedback_dict)
            logger.info(f"📤 Broadcasting: VAK={learning_style}, Digression={digression_score:.2f}, Sarcasm={sarcasm_result['score']:.2f}")
            
            disconnected = set()
            for client in self.websocket_clients:
                try:
                    await client.send_text(message)
                except Exception as e:
                    logger.warning(f"Failed to send to client: {e}")
                    disconnected.add(client)
            
            self.websocket_clients -= disconnected
            
        except Exception as e:
            logger.error(f"Broadcast error: {e}", exc_info=True)

    async def stop_session(self):
        """Stop session and generate ENHANCED report"""
        if not self.session_active:
            raise RuntimeError("No active session to stop")
        
        logger.info(f"⏹️ Stopping session {self.session_id}")
        
        self.session_active = False
        await asyncio.sleep(0.5)
        
        if self.processing_task:
            try:
                await asyncio.wait_for(self.processing_task, timeout=3.0)
                logger.info("✅ Processing task completed")
            except asyncio.TimeoutError:
                logger.warning("Processing task timed out")
                self.processing_task.cancel()
        
        if self.audio_processor:
            try:
                await asyncio.wait_for(
                    self.audio_processor.stop_transcription(),
                    timeout=5.0
                )
                logger.info("✅ Audio processor stopped")
            except asyncio.TimeoutError:
                logger.warning("Audio processor stop timed out")
            except Exception as e:
                logger.error(f"Error stopping audio processor: {e}")
        
        try:
            session_data_for_report = {
                'session_id': self.session_id,
                'duration': (datetime.now() - self.session_data["start_time"]).total_seconds() / 60,
                'chunks': self.session_data["chunks"],
                'feedback_history': self.session_data.get("feedback_history", []),
                'vak_scores': self.session_data.get("vak_scores", []),
                'sarcasm_detections': self.session_data.get("sarcasm_detections", []),
                'digression_scores': self.session_data.get("digression_scores", [])
            }
            
            gemini_succeeded = False
            if self.gemini_analyzer:
                try:
                    logger.info("Attempting Gemini report generation...")
                    gemini_report_dict = await asyncio.wait_for(
                        self.gemini_analyzer.generate_session_report({
                            'session_id': self.session_id,
                            'duration': session_data_for_report['duration'],
                            'chunks': [
                                {
                                    'speaker': c.speaker,
                                    'transcript': c.transcript,
                                    'timestamp': c.timestamp
                                }
                                for c in self.session_data["chunks"]
                            ]
                        }),
                        timeout=10.0
                    )
                    
                    if isinstance(gemini_report_dict, dict):
                        report = SessionReport(**gemini_report_dict)
                    else:
                        report = gemini_report_dict

                    # Gemini doesn't see the wired-in per-chunk metrics; overlay
                    # them so the final report still reflects collected data.
                    report.sarcasm_summary = self.local_analyzer._summarize_sarcasm(
                        session_data_for_report['sarcasm_detections']
                    )
                    report.digression_summary = self.local_analyzer._summarize_digression(
                        session_data_for_report['digression_scores']
                    )
                    vak_avg = self.local_analyzer._analyze_learning_styles(
                        session_data_for_report['vak_scores']
                    )
                    if vak_avg:
                        report.learning_style_analysis = vak_avg

                    logger.info("✅ Gemini report generated successfully")
                    gemini_succeeded = True
                    
                except asyncio.TimeoutError:
                    logger.warning("⏱️ Gemini timeout, using enhanced local")
                except Exception as e:
                    logger.warning(f"⚠️ Gemini failed ({str(e)}), using enhanced local")
            
            if not gemini_succeeded:
                logger.info("📊 Generating enhanced local analysis...")
                
                local_report_dict = self.local_analyzer.generate_comprehensive_report(
                    session_data_for_report
                )
                
                report = SessionReport(**local_report_dict)
                logger.info("✅ Enhanced local report generated")
            
        except Exception as e:
            logger.error(f"❌ Error generating report: {e}", exc_info=True)
            report = self._generate_basic_report({
                'session_id': self.session_id, 
                'duration': 0, 
                'chunks': []
            })
        
        # Fire-and-forget persistence — first ChromaDB write downloads an
        # ONNX embedding model (~80 MB) which would otherwise block the HTTP
        # response for minutes. The task logs its own errors.
        asyncio.create_task(self._store_session_safe(report))

        # Remember the report so /session/stop retries are idempotent.
        self.last_report = report

        logger.info(f"✅ Session {self.session_id} completed")

        return report

    async def _store_session_safe(self, report):
        try:
            await self._store_session(report)
        except Exception as e:
            logger.warning(f"⚠️ Background storage failed: {e}")

    async def _store_session(self, report):
        """Persist final session report to ChromaDB."""
        if not self.storage:
            return
        await self.storage.store_session_report(report)
        logger.info(f"💾 Stored session {report.session_id} in ChromaDB")

    def _generate_basic_report(self, report_data: Dict[str, Any]) -> SessionReport:
        """Generate basic report fallback"""
        chunks = report_data.get('chunks', [])
        coach_chunks = [c for c in chunks if c.get('speaker') == 'coach']
        coachee_chunks = [c for c in chunks if c.get('speaker') == 'coachee']
        
        return SessionReport(
            session_id=report_data.get('session_id', 'unknown'),
            duration_minutes=report_data.get('duration', 0),
            participants={
                "coach": {"engagement_avg": 0.0, "total_turns": len(coach_chunks)},
                "coachee": {"engagement_avg": 0.0, "total_turns": len(coachee_chunks)}
            },
            grow_phases=[],
            emotional_journey={"coach": [], "coachee": []},
            learning_style_analysis={},
            key_insights=[f"Session completed with {len(chunks)} total interactions"],
            coaching_effectiveness={},
            recommendations=["Insufficient data for recommendations"],
            transcript_summary=f"Session with {len(coach_chunks)} coach turns and {len(coachee_chunks)} coachee turns"
        )

    def get_available_audio_devices(self):
        """Get list of available audio input devices"""
        try:
            import pyaudio
            p = pyaudio.PyAudio()
            devices = []
            
            for i in range(p.get_device_count()):
                info = p.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    devices.append({
                        'index': i,
                        'name': info['name'],
                        'channels': info['maxInputChannels'],
                        'sample_rate': int(info['defaultSampleRate'])
                    })
            
            p.terminate()
            return devices
        except Exception as e:
            logger.error(f"Error getting audio devices: {e}")
            return []
