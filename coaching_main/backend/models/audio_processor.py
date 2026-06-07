"""
Fixed Audio Processor - Uses AssemblyAI's MicrophoneStream correctly
"""
import asyncio
import logging
import threading
from datetime import datetime
import assemblyai as aai
from assemblyai.streaming.v3 import (
    StreamingClient,
    StreamingClientOptions,
    StreamingParameters,
    StreamingEvents,
    TurnEvent,
    StreamingError,
)

from backend.schemas.data_models import AudioChunk

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Handles real-time audio processing with AssemblyAI (streaming.v3 API)"""

    def __init__(self, api_key: str, default_coach_role: bool = True):
        self.api_key = api_key
        self.client: StreamingClient | None = None
        self.session_active = False
        self.audio_queue: asyncio.Queue | None = None
        self.device_index = None
        self.event_loop: asyncio.AbstractEventLoop | None = None
        self.stream_thread = None
        self.default_coach_role = default_coach_role
        # Maps AssemblyAI speaker_id (e.g. "SPEAKER_A") to role ("coach"/"coachee")
        # Locked on first classification so same voice keeps same label all session
        self._speaker_map: dict[str, str] = {}

    async def start_live_transcription(self, audio_queue: asyncio.Queue, device_index=None):
        """Start live transcription with AssemblyAI streaming API"""
        self.audio_queue = audio_queue
        self.device_index = device_index
        # Store the event loop for use in handlers running in other threads
        self.event_loop = asyncio.get_running_loop()

        # Create streaming client
        self.client = StreamingClient(
            StreamingClientOptions(api_key=self.api_key)
        )

        # Register event handlers
        self.client.on(StreamingEvents.Turn, self._handle_turn_wrapper)
        self.client.on(StreamingEvents.Error, self._handle_error_wrapper)
        self.client.on(StreamingEvents.Begin, self._handle_begin)
        self.client.on(StreamingEvents.Termination, self._handle_termination)

        # Connect with desired parameters
        try:
            logger.info("Connecting to AssemblyAI streaming API...")
            self.client.connect(
                StreamingParameters(
                    sample_rate=16000,
                    format_turns=True,
                    speaker_labels=True,
                    speech_model="universal-streaming-english"
                )
            )
            logger.info("✅ Successfully connected to AssemblyAI streaming API")
            self.session_active = True
            
            # Start streaming in a separate thread (CRITICAL FIX)
            logger.info("🎤 Starting microphone stream...")
            self.stream_thread = threading.Thread(
                target=self._run_stream_blocking,
                daemon=True
            )
            self.stream_thread.start()
            logger.info("✅ Microphone stream thread started")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to AssemblyAI: {e}", exc_info=True)
            self.session_active = False
            raise RuntimeError(f"Failed to connect to AssemblyAI streaming API: {str(e)}") from e

    def _run_stream_blocking(self):
        """Run the blocking stream() call in a separate thread"""
        try:
            # Use AssemblyAI's built-in MicrophoneStream
            logger.info("🎤 Creating MicrophoneStream...")
            microphone_stream = aai.extras.MicrophoneStream(
                sample_rate=16000,
                device_index=self.device_index
            )
            
            logger.info("📡 Starting client.stream() - this will block until stopped...")
            # This is the blocking call that streams microphone audio to AssemblyAI
            self.client.stream(microphone_stream)
            logger.info("🔴 client.stream() completed")
            
        except Exception as e:
            if self.session_active:
                logger.error(f"❌ Error in stream thread: {e}", exc_info=True)
            else:
                logger.info("Stream thread stopped (expected)")

    def _handle_turn_wrapper(self, client: StreamingClient, event: TurnEvent):
        """Wrapper to handle turn events - schedules coroutine in event loop"""
        if not self.audio_queue:
            return
        
        if not self.event_loop:
            logger.error("No event loop stored - cannot schedule turn handler")
            return
            
        try:
            asyncio.run_coroutine_threadsafe(self._handle_turn(event), self.event_loop)
        except Exception as e:
            logger.error(f"Error scheduling turn handler: {e}", exc_info=True)

    async def _handle_turn(self, event: TurnEvent):
        """Handle a transcription turn event"""
        if not event:
            logger.warning("Received empty TurnEvent")
            return
            
        transcript_text = getattr(event, "transcript", None) or getattr(event, "text", None)
        if not transcript_text:
            logger.debug("TurnEvent has no transcript text, skipping")
            return
            
        if not self.audio_queue:
            logger.warning("Audio queue is None, cannot process transcription")
            return

        # Use AssemblyAI's speaker_id when available to ensure consistency,
        # falling back to heuristic-only when diarization produces no id
        speaker_id = getattr(event, "speaker_id", None)
        is_final   = bool(getattr(event, "end_of_turn", True))
        duration   = getattr(event, "audio_duration_seconds", 2.0)

        if speaker_id and speaker_id in self._speaker_map:
            speaker_label = self._speaker_map[speaker_id]
        else:
            speaker_label = self._detect_speaker(transcript_text)
            # Only lock the mapping once we have a final utterance — partials
            # may have too little text for a reliable heuristic classification.
            if speaker_id and is_final:
                self._speaker_map[speaker_id] = speaker_label
                logger.info(f"🔒 Locked speaker mapping: {speaker_id} → {speaker_label}")

        chunk = AudioChunk(
            timestamp=datetime.now().timestamp(),
            duration=duration,
            speaker=speaker_label,
            speaker_id=speaker_id,
            transcript=transcript_text,
            is_final=is_final,
        )
        
        try:
            await self.audio_queue.put(chunk)
            logger.info(f"📝 Transcription received: [{speaker_label}] {transcript_text[:50]}...")
        except Exception as e:
            logger.error(f"Error putting chunk in queue: {e}", exc_info=True)

    def _detect_speaker(self, transcript: str) -> str:
        """
        Detect speaker based on transcript content.
        Coach: asks questions, uses coaching language
        Coachee: shares problems, uncertainties, goals
        """
        transcript_lower = transcript.lower()
        
        # Strong coachee indicators (person seeking help)
        coachee_phrases = [
            "i don't know", "i'm not sure", "i worry", "i'm worried",
            "i feel", "i think", "my problem", "i want to", "i need",
            "i'm confused", "i'm stuck", "help me", "what should i",
            "i can't decide", "i'm struggling", "i don't understand",
            "my goal", "my issue", "my challenge"
        ]
        
        # Strong coach indicators (person helping)
        coach_phrases = [
            "what would you", "how do you feel", "tell me about",
            "what's stopping you", "what if you", "have you considered",
            "what are your options", "what's your goal", "let's explore",
            "how can i help", "what do you think", "describe",
            "what's important", "on a scale of"
        ]
        
        # Count matches
        coachee_score = sum(1 for phrase in coachee_phrases if phrase in transcript_lower)
        coach_score = sum(1 for phrase in coach_phrases if phrase in transcript_lower)
        
        # Questions are usually from coach
        if "?" in transcript:
            coach_score += 1
        
        # First person statements often from coachee
        if any(word in transcript_lower.split() for word in ["i", "my", "me"]):
            coachee_score += 0.5
        
        # Decide
        if coachee_score > coach_score:
            return "coachee"
        elif coach_score > coachee_score:
            return "coach"
        else:
            # Default: if uncertain and contains "I feel/think/want", it's coachee
            if any(phrase in transcript_lower for phrase in ["i feel", "i think", "i want", "i don't"]):
                return "coachee"
            return "coach" if self.default_coach_role else "coachee"

    def _handle_error_wrapper(self, client: StreamingClient, error: StreamingError):
        """Wrapper for error handler"""
        self._handle_error(error)

    def _handle_error(self, error: StreamingError):
        """Handle streaming errors"""
        error_msg = str(error)
        error_type = type(error).__name__
        logger.error(f"❌ AssemblyAI streaming error [{error_type}]: {error_msg}")
        
        if "authentication" in error_msg.lower() or "api key" in error_msg.lower():
            logger.error("Authentication error - check your ASSEMBLYAI_API_KEY")
            self.session_active = False

    def _handle_begin(self, client: StreamingClient, event):
        """Handle session begin event"""
        session_id = getattr(event, 'id', None) or getattr(event, 'session_id', None)
        logger.info(f"✅ AssemblyAI session began: {session_id or 'unknown'}")

    def _handle_termination(self, client: StreamingClient, event):
        """Handle session termination event"""
        duration = getattr(event, "audio_duration_seconds", 0)
        reason = getattr(event, "reason", None)
        logger.info(f"Session terminated: {duration}s audio processed. Reason: {reason}")

    async def stop_transcription(self):
        """Stop live transcription"""
        if not self.client:
            logger.info("No active client to stop")
            return

        self.session_active = False
        
        try:
            logger.info("Stopping transcription...")
            # Disconnect the client (this will stop the stream)
            await asyncio.wait_for(
                self._disconnect_client(), 
                timeout=5.0
            )
            
            # Wait for stream thread to finish
            if self.stream_thread and self.stream_thread.is_alive():
                logger.info("Waiting for stream thread to finish...")
                self.stream_thread.join(timeout=3.0)
            
            logger.info("✅ Live transcription stopped successfully")
            
        except asyncio.TimeoutError:
            logger.warning("Client disconnect timed out - forcing cleanup")
        except Exception as e:
            logger.error(f"Error during transcription stop: {e}")
        finally:
            self.session_active = False
            self.client = None
            self.audio_queue = None
            self.stream_thread = None

    async def _disconnect_client(self):
        """Disconnect client in executor to avoid blocking"""
        if self.client:
            loop = asyncio.get_event_loop()
            try:
                await loop.run_in_executor(
                    None, 
                    lambda: self.client.disconnect(terminate=True)
                )
            except Exception as e:
                if "ConnectionClosed" not in str(type(e).__name__):
                    raise
                logger.debug(f"Connection already closed: {e}")