"""
Audio Capture System for PyAudio microphone input
Injects audio into AudioProcessor for AssemblyAI streaming.v3 API
"""

import asyncio
import logging
import pyaudio
from typing import Optional, Any

logger = logging.getLogger(__name__)


class AudioCaptureSystem:
    """
    Real-time audio capture using PyAudio
    Sends raw PCM audio bytes to AudioProcessor for AssemblyAI streaming
    """

    def __init__(self):
        # Audio configuration (must match AssemblyAI requirements)
        self.FRAMES_PER_BUFFER = 3200  # ~200ms at 16kHz
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        
        self.audio = pyaudio.PyAudio()
        self.stream: Optional[pyaudio.Stream] = None
        self.is_capturing = False
        self.audio_processor: Optional[Any] = None
        
    def get_available_devices(self):
        """Get list of available audio input devices"""
        devices = []
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                devices.append({
                    'index': i,
                    'name': info['name'],
                    'channels': info['maxInputChannels'],
                    'sample_rate': int(info['defaultSampleRate'])
                })
        return devices

    async def start_capture(
        self, 
        audio_processor: Any,
        device_index: Optional[int] = None
    ):
        """
        Start real-time audio capture and inject into AudioProcessor
        
        Args:
            audio_processor: AudioProcessor instance to send audio bytes to
            device_index: Optional audio device index
        """
        self.is_capturing = True
        self.audio_processor = audio_processor
        
        try:
            # Validate audio processor is ready
            if not audio_processor or not audio_processor.session_active:
                raise RuntimeError("AudioProcessor is not initialized or not active. Call start_live_transcription() first.")
            
            # Check if audio system is available
            device_count = self.audio.get_device_count()
            if device_count == 0:
                raise RuntimeError("No audio devices found on the system")
            
            # If device_index specified, validate it exists
            if device_index is not None:
                try:
                    device_info = self.audio.get_device_info_by_index(device_index)
                    if device_info['maxInputChannels'] == 0:
                        raise RuntimeError(f"Device {device_index} ({device_info.get('name', 'unknown')}) does not support input")
                except OSError:
                    raise RuntimeError(f"Invalid audio device index: {device_index}")
            
            # Open audio stream
            logger.info(f"Opening audio stream (device: {device_index or 'default'})")
            try:
                self.stream = self.audio.open(
                    format=self.FORMAT,
                    channels=self.CHANNELS,
                    rate=self.RATE,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=self.FRAMES_PER_BUFFER
                )
                logger.info("✅ Audio stream opened successfully")
            except OSError as e:
                error_msg = f"Failed to open audio stream: {str(e)}"
                if "Invalid sample rate" in str(e):
                    error_msg += f". Try using a device that supports {self.RATE}Hz sample rate."
                elif "No default input device" in str(e) or "Default input device not available" in str(e):
                    error_msg += ". Please check your microphone permissions and ensure a default input device is set."
                raise RuntimeError(error_msg)
            except Exception as e:
                raise RuntimeError(f"Unexpected error opening audio stream: {str(e)}")
            
            # Start sending audio to processor
            await self._send_audio_loop()
                
        except Exception as e:
            error_message = str(e)
            logger.error(f"❌ Capture error: {error_message}", exc_info=True)
            self.is_capturing = False
            raise RuntimeError(f"Audio capture failed: {error_message}") from e
        finally:
            self.is_capturing = False
            if self.stream:
                try:
                    if self.stream.is_active():
                        self.stream.stop_stream()
                    self.stream.close()
                except Exception as e:
                    logger.warning(f"Error closing stream: {e}")

    async def _send_audio_loop(self):
        """Read audio from microphone and send to AudioProcessor"""
        logger.info("🎤 Starting audio capture and injection loop")
        
        chunks_sent = 0
        last_log_time = asyncio.get_event_loop().time()
        
        while self.is_capturing and self.audio_processor:
            try:
                # Check if stream is still active
                if not self.stream or not self.stream.is_active():
                    logger.warning("Audio stream is not active, stopping capture")
                    break
                
                # Read audio from microphone (blocking call)
                # Use run_in_executor to avoid blocking the event loop
                loop = asyncio.get_event_loop()
                data = await loop.run_in_executor(
                    None,
                    lambda: self.stream.read(self.FRAMES_PER_BUFFER, exception_on_overflow=False)
                )
                
                # Check if we got data
                if not data or len(data) == 0:
                    logger.warning("No audio data read from stream")
                    continue
                
                # Send raw PCM audio bytes to AudioProcessor
                if self.audio_processor and self.audio_processor.session_active:
                    try:
                        await self.audio_processor.send_audio(data)
                        chunks_sent += 1
                        
                        # Log every 5 seconds to show it's working
                        current_time = asyncio.get_event_loop().time()
                        if current_time - last_log_time >= 5.0:
                            logger.info(f"📊 Audio capture active: {chunks_sent} chunks sent ({len(data)} bytes each)")
                            last_log_time = current_time
                    except Exception as send_error:
                        logger.error(f"Error sending audio to processor: {send_error}", exc_info=True)
                        # Don't break - continue trying
                else:
                    logger.warning("AudioProcessor not active or not set, skipping audio chunk")
                    if not self.audio_processor:
                        logger.error("AudioProcessor is None!")
                    elif not self.audio_processor.session_active:
                        logger.error("AudioProcessor.session_active is False!")
                
            except Exception as e:
                if self.is_capturing:  # Only log if we're supposed to be running
                    logger.error(f"Error capturing/sending audio: {e}", exc_info=True)
                    # Don't break immediately - might be temporary
                    await asyncio.sleep(0.1)
                else:
                    break
                
        logger.info(f"🔴 Audio capture loop stopped. Total chunks sent: {chunks_sent}")

    async def stop_capture(self):
        """Stop audio capture"""
        logger.info("⏹️ Stopping audio capture...")
        self.is_capturing = False
        
        # Give loops time to finish
        await asyncio.sleep(0.5)
        
        if self.stream and self.stream.is_active():
            self.stream.stop_stream()
            self.stream.close()
            logger.info("✅ Audio stream closed")

    def __del__(self):
        """Cleanup"""
        try:
            if self.stream:
                self.stream.close()
            self.audio.terminate()
        except:
            pass
