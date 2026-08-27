"""
Fixed Audio Processor - Uses AssemblyAI's MicrophoneStream correctly
"""
import asyncio
import logging
import queue
import threading
import time
from datetime import datetime
from typing import Optional

import numpy as np
import pyaudio
from assemblyai.streaming.v3 import (
    StreamingClient,
    StreamingClientOptions,
    StreamingParameters,
    StreamingEvents,
    TurnEvent,
    StreamingError,
)

from backend.models.speaker_router import SpeakerRouter
from backend.schemas.data_models import AudioChunk

logger = logging.getLogger(__name__)

# AssemblyAI streaming expects 16 kHz mono signed 16-bit PCM.
_SAMPLE_RATE = 16000
_CHANNELS = 1
_FORMAT = pyaudio.paInt16
#: ~200 ms of audio per read.
_FRAMES_PER_BUFFER = 3200

#: Host APIs to prefer when no device is pinned, best first. MME is last on
#: purpose: on Intel Smart Sound machines its default endpoint opens happily
#: and then returns digital silence forever, which looks exactly like a
#: working capture that nobody is talking into.
_HOST_API_PREFERENCE = (pyaudio.paWASAPI, pyaudio.paDirectSound, pyaudio.paMME)

#: RMS below this over the probe window is treated as "no signal at all".
_SILENCE_RMS = 3.0

#: Ceiling on buffered frames (~20s). Bounded so a stalled consumer drops
#: audio instead of growing the queue without limit.
_QUEUE_MAX_FRAMES = 100


class AudioProcessor:
    """Handles real-time audio processing with AssemblyAI (streaming.v3 API)"""

    def __init__(self, api_key: str, default_coach_role: bool = True, coach_speaker_id: Optional[str] = None):
        self.api_key = api_key
        self.client: StreamingClient | None = None
        self.session_active = False
        self.audio_queue: asyncio.Queue | None = None
        self.device_index = None
        self.event_loop: asyncio.AbstractEventLoop | None = None
        self.stream_thread = None
        self.default_coach_role = default_coach_role
        self._pyaudio: Optional[pyaudio.PyAudio] = None
        self._mic_stream = None
        self._frame_queue: "queue.Queue[bytes]" = queue.Queue(maxsize=_QUEUE_MAX_FRAMES)
        self._frames_dropped = 0
        #: Set when the streaming thread dies unexpectedly, so a silently
        #: dead microphone is visible rather than looking like a live session.
        self.stream_error: Optional[str] = None
        #: Set when the device opens but delivers silence - not fatal (the
        #: room may genuinely be quiet) but almost always the real problem.
        self.capture_warning: Optional[str] = None
        # Role assignment lives in SpeakerRouter: it accumulates evidence per
        # diarized speaker id and locks each role once, so a voice cannot
        # flip roles mid-session. Passing coach_speaker_id pins the mapping.
        self.router = SpeakerRouter(coach_speaker_id)

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

            # Open the microphone here rather than inside the worker thread:
            # a missing, busy or unsupported device has to fail the request,
            # not disappear into a log line while the session reports itself
            # as healthy.
            self._open_microphone(device_index)
            self.session_active = True
            self.stream_error = None

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

    def _candidate_devices(self, device_index) -> list:
        """Input devices to try, best first.

        A pinned device is honoured as-is. Otherwise each host API's default
        input is a candidate: they differ in both capability and behaviour,
        and no single one is reliable. WASAPI here refuses 16 kHz outright;
        MME opens happily and then returns nothing but zeroes forever, which
        is far worse because it looks like a working capture.
        """
        if device_index is not None:
            return [device_index]

        candidates, seen = [], set()
        for api_type in _HOST_API_PREFERENCE:
            try:
                api = self._pyaudio.get_host_api_info_by_type(api_type)
            except Exception:
                continue
            index = api.get("defaultInputDevice", -1)
            if index is not None and index >= 0 and index not in seen:
                seen.add(index)
                candidates.append(index)
        return candidates or [None]

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PortAudio hands us a frame; queue it for the streaming thread."""
        try:
            self._frame_queue.put_nowait(in_data)
        except queue.Full:
            self._frames_dropped += 1
        return (None, pyaudio.paContinue)

    def _read_frame(self, timeout: float = 0.5) -> bytes:
        """Take the next captured frame, or b'' if none arrived in time.

        Capture is callback-driven rather than read-driven because neither
        polling mechanism paces reliably across host APIs: MME blocks for
        the frame duration as expected, but the DirectSound endpoints return
        from ``read()`` instantly *and* report ``get_read_available() == 0``
        forever. Reading those in a loop either spins a core and grows the
        send queue without bound (~24 MB/s observed) or captures nothing at
        all. PortAudio's own callback paces correctly on every device here.
        """
        try:
            return self._frame_queue.get(timeout=timeout)
        except queue.Empty:
            return b""

    def _input_peak_rms(self, seconds: float = 1.0) -> float:
        """Peak RMS over a wall-clock window. Zero means pure silence."""
        peak = 0.0
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            raw = self._read_frame()
            if not raw:
                continue
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            if samples.size:
                peak = max(peak, float(np.sqrt(np.mean(samples ** 2))))
        return peak

    def _describe(self, index) -> str:
        if index is None:
            return "default"
        try:
            info = self._pyaudio.get_device_info_by_index(index)
            api = self._pyaudio.get_host_api_info_by_index(info["hostApi"])["name"]
            return f"[{index}] {info['name'][:40]} ({api})"
        except Exception:
            return f"[{index}]"

    def _open_microphone(self, device_index) -> None:
        """Open an input device that can actually be opened and heard.

        Each candidate is opened at 16 kHz and listened to briefly. One that
        refuses the rate, or that answers with silence, is discarded in
        favour of the next - a silent device is kept only as a last resort,
        flagged, because a genuinely quiet room sounds the same.
        """
        self._pyaudio = pyaudio.PyAudio()
        pinned = device_index is not None
        candidates = self._candidate_devices(device_index)
        errors, silent = [], None

        for index in candidates:
            try:
                with self._frame_queue.mutex:
                    self._frame_queue.queue.clear()
                stream = self._pyaudio.open(
                    format=_FORMAT,
                    channels=_CHANNELS,
                    rate=_SAMPLE_RATE,
                    input=True,
                    input_device_index=index,
                    frames_per_buffer=_FRAMES_PER_BUFFER,
                    stream_callback=self._audio_callback,
                )
                stream.start_stream()
            except Exception as exc:
                errors.append(f"{self._describe(index)}: {exc}")
                logger.info("Input %s unusable: %s", self._describe(index), exc)
                continue

            self._mic_stream = stream
            try:
                peak = self._input_peak_rms()
            except Exception as exc:
                logger.debug("Level probe failed on %s: %s", self._describe(index), exc)
                peak = None

            if peak is not None and peak <= _SILENCE_RMS and not pinned:
                # Hold on to it, but keep looking for one that can hear.
                if silent is None:
                    silent = (index, stream)
                    self._mic_stream = None
                    logger.info("Input %s is silent; trying the next one",
                                self._describe(index))
                    continue
                stream.close()
                continue

            self.capture_warning = None
            if peak is not None and peak <= _SILENCE_RMS:
                self.capture_warning = (
                    f"input device {self._describe(index)} is returning silence. "
                    "Pick a different microphone in the sidebar, or check that "
                    "it is not muted."
                )
                logger.warning("🔇 %s", self.capture_warning)
            logger.info("🎤 Microphone open %s at %d Hz mono (peak RMS %s)",
                        self._describe(index), _SAMPLE_RATE,
                        f"{peak:.0f}" if peak is not None else "unknown")
            if silent:
                silent[1].close()
            return

        # Nothing could hear anything - fall back to a silent-but-openable one.
        if silent is not None:
            index, stream = silent
            self._mic_stream = stream
            self.capture_warning = (
                f"every input device returned silence (using {self._describe(index)}). "
                "Check Windows sound settings, or pick a device in the sidebar."
            )
            logger.warning("🔇 %s", self.capture_warning)
            return

        self._close_microphone()
        raise RuntimeError(
            "Could not open any microphone. Tried: "
            + ("; ".join(errors) if errors else "no candidates")
        )

    def _microphone_frames(self):
        """Yield raw PCM frames until the session stops.

        Replaces ``aai.extras.MicrophoneStream``, which the assemblyai 1.0
        SDK removed. ``StreamingClient.stream()`` accepts any iterable of
        bytes, so the microphone is ours to supply - and doing it here
        keeps the code working on both SDK lines.
        """
        frames = 0
        while self.session_active and self._mic_stream is not None:
            try:
                data = self._read_frame()
            except Exception as exc:
                if self.session_active:
                    logger.error("Microphone read failed: %s", exc, exc_info=True)
                    self.stream_error = f"microphone read failed: {exc}"
                return
            if not data:
                continue
            if self.capture_warning is not None:
                # The open-time probe can land in a genuinely quiet moment.
                # Once real audio arrives the warning is simply wrong.
                samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                if samples.size and float(np.sqrt(np.mean(samples ** 2))) > _SILENCE_RMS:
                    logger.info("🎤 Audio detected; clearing the silence warning")
                    self.capture_warning = None
            frames += 1
            if frames % 25 == 0:  # roughly every 5 seconds
                logger.debug("🎤 %d audio frames sent", frames)
            yield data
        logger.info("🎤 Microphone loop ended after %d frames", frames)

    def _close_microphone(self) -> None:
        """Release the input device. Safe to call more than once."""
        if self._mic_stream is not None:
            try:
                if self._mic_stream.is_active():
                    self._mic_stream.stop_stream()
                self._mic_stream.close()
            except Exception as exc:
                logger.debug("Error closing microphone stream: %s", exc)
            self._mic_stream = None
        if self._pyaudio is not None:
            try:
                self._pyaudio.terminate()
            except Exception as exc:
                logger.debug("Error terminating PyAudio: %s", exc)
            self._pyaudio = None

    def _run_stream_blocking(self):
        """Run the blocking stream() call in a separate thread"""
        try:
            logger.info("📡 Starting client.stream() - this will block until stopped...")
            self.client.stream(self._microphone_frames())
            logger.info("🔴 client.stream() completed")

        except Exception as e:
            if self.session_active:
                # Record it: without this the thread dies and the session
                # still reports itself as active with nothing being captured.
                self.stream_error = str(e)
                logger.error(f"❌ Error in stream thread: {e}", exc_info=True)
            else:
                logger.info("Stream thread stopped (expected)")
        finally:
            self._close_microphone()

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

        speaker_id = getattr(event, "speaker_id", None)
        is_final = bool(getattr(event, "end_of_turn", True))
        duration = getattr(event, "audio_duration_seconds", 2.0)

        # Diarization decides *who* is speaking; the router decides which
        # role that speaker holds, once, from accumulated evidence. Only
        # final turns contribute evidence - partials are fragments.
        if is_final:
            self.router.observe(speaker_id, transcript_text)
        speaker_label = self.router.role_for(speaker_id, transcript_text)

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
            self._close_microphone()
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