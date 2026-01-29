"""
File Audio Processor - Process WAV files through AssemblyAI
"""
import asyncio
import logging
from pathlib import Path
import assemblyai as aai
from datetime import datetime

from backend.schemas.data_models import AudioChunk

logger = logging.getLogger(__name__)


class FileAudioProcessor:
    """Process audio from WAV files using AssemblyAI"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        aai.settings.api_key = api_key
        self.transcriber = aai.Transcriber()
        self.audio_queue = None
    
    async def process_file(self, file_path: str, audio_queue: asyncio.Queue):
        """
        Process a WAV file and send chunks to the queue
        
        Args:
            file_path: Path to the WAV file
            audio_queue: Queue to send AudioChunk objects to
        """
        self.audio_queue = audio_queue
        
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Audio file not found: {file_path}")
            
            logger.info(f"🎵 Processing audio file: {file_path.name}")
            
            # Configure transcription with speaker diarization
            config = aai.TranscriptionConfig(
                speaker_labels=True,
                speakers_expected=2
            )
            
            # Transcribe the file
            logger.info("📤 Uploading file to AssemblyAI...")
            transcript = self.transcriber.transcribe(str(file_path), config=config)
            
            if transcript.status == aai.TranscriptStatus.error:
                raise RuntimeError(f"Transcription failed: {transcript.error}")
            
            logger.info(f"✅ Transcription completed: {len(transcript.utterances)} utterances")
            
            # Process utterances and send as chunks
            await self._process_utterances(transcript.utterances)
            
            logger.info("✅ File processing completed")
            
        except Exception as e:
            logger.error(f"Error processing file: {e}", exc_info=True)
            raise
    
    async def _process_utterances(self, utterances):
        """Process transcript utterances and convert to AudioChunks"""
        
        for i, utterance in enumerate(utterances):
            try:
                # Map speaker labels
                speaker_label = self._map_speaker(utterance.speaker, utterance.text)
                
                # Create AudioChunk
                chunk = AudioChunk(
                    timestamp=datetime.now().timestamp(),
                    duration=(utterance.end - utterance.start) / 1000.0,
                    speaker=speaker_label,
                    transcript=utterance.text,
                    audio_data=None
                )
                
                # Send to queue
                await self.audio_queue.put(chunk)
                logger.info(f"📝 Utterance {i+1}/{len(utterances)}: [{speaker_label}] {utterance.text[:50]}...")
                
                # Small delay
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error processing utterance {i}: {e}")
                continue
    
    def _map_speaker(self, speaker_id: str, text: str) -> str:
        """Map AssemblyAI speaker labels to coach/coachee"""
        text_lower = text.lower()
        
        coachee_phrases = [
            "i don't know", "i'm not sure", "i worry", "i feel",
            "i think", "my problem", "i want to", "i need"
        ]
        
        coach_phrases = [
            "what would you", "how do you feel", "tell me about",
            "what's stopping you", "what if you", "have you considered"
        ]
        
        coachee_score = sum(1 for phrase in coachee_phrases if phrase in text_lower)
        coach_score = sum(1 for phrase in coach_phrases if phrase in text_lower)
        
        if "?" in text:
            coach_score += 1
        
        if coachee_score > coach_score:
            return "coachee"
        elif coach_score > coachee_score:
            return "coach"
        else:
            return "coach" if speaker_id == "A" else "coachee"