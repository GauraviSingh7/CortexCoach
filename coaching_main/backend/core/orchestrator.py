"""
Session orchestration.

Owns the session lifecycle and wires the pieces together. All per-turn
analysis lives in :mod:`backend.core.turn_processor`, all report assembly
in :mod:`backend.reporting`, and all fan-out in
:mod:`backend.core.broadcaster` - this module coordinates, it does not
compute.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Dict, List, Optional

from backend.core.broadcaster import FeedbackBroadcaster
from backend.core.session_state import SessionState
from backend.core.suggestions import SuggestionBuilder
from backend.core.turn_processor import TurnProcessor, TurnResult
from backend.models.audio_processor import AudioProcessor
from backend.models.contextual_suggestion_engine import ContextualSuggestionEngine
from backend.models.file_audio_processor import FileAudioProcessor
from backend.models.gemini_analyzer import GeminiAnalyzer
from backend.models.inference_engine import ModelInferenceEngine
from backend.models.replay_processor import ReplayProcessor
from backend.models.storage import ChromaDBStorage
from backend.reporting import LocalAnalyzer
from backend.schemas.data_models import AudioChunk, RealTimeFeedback, SessionReport

logger = logging.getLogger(__name__)

#: How long to wait for Gemini before falling back to the local report.
GEMINI_TIMEOUT_SECONDS = 20.0


class CoachingObserverSystem:
    """Top-level coordinator for the AI Coaching Observer."""

    def __init__(self, assemblyai_key: str, gemini_key: str):
        self.assemblyai_key = assemblyai_key
        self.gemini_key = gemini_key

        self.inference_engine = ModelInferenceEngine()
        self.gemini_analyzer = GeminiAnalyzer(gemini_key) if gemini_key else None
        self.local_analyzer = LocalAnalyzer()
        self.suggestion_engine = ContextualSuggestionEngine()
        self.broadcaster = FeedbackBroadcaster()

        try:
            self.storage: Optional[ChromaDBStorage] = ChromaDBStorage()
        except Exception as exc:
            logger.warning("ChromaDB storage unavailable: %s", exc)
            self.storage = None

        self.audio_processor: Optional[AudioProcessor] = None
        self.file_processor = None  # FileAudioProcessor or ReplayProcessor
        self.audio_queue: Optional[asyncio.Queue] = None
        self.processing_task: Optional[asyncio.Task] = None
        self.source_task: Optional[asyncio.Task] = None
        self._processing_chunk = False

        self.state: Optional[SessionState] = None
        self.session_active = False
        self.last_report: Optional[SessionReport] = None

        self._turn_processor: Optional[TurnProcessor] = None
        self._suggestions: Optional[SuggestionBuilder] = None

        logger.info("CoachingObserverSystem initialised")

    # -- compatibility accessors ------------------------------------------

    @property
    def session_id(self) -> Optional[str]:
        return self.state.session_id if self.state else None

    @property
    def websocket_clients(self):
        """Exposed for the FastAPI websocket route."""
        return self.broadcaster.clients

    @property
    def session_data(self) -> Dict[str, Any]:
        """Legacy view of session contents, used by the status endpoint."""
        if not self.state:
            return {"chunks": [], "feedback_history": []}
        return {
            "session_id": self.state.session_id,
            "chunks": self.state.chunks,
            "feedback_history": self.state.feedback_history,
        }

    def get_available_audio_devices(self) -> List[Dict[str, Any]]:
        """Input devices for live mode. Empty list if PyAudio is missing."""
        try:
            from backend.models.audio_capture import AudioCaptureSystem

            return AudioCaptureSystem().get_available_devices()
        except Exception as exc:
            logger.warning("Could not enumerate audio devices: %s", exc)
            return []

    # -- lifecycle ---------------------------------------------------------

    async def start_session(
        self,
        session_type: str = "live",
        device_index: Optional[int] = None,
        file_path: Optional[str] = None,
        coach_speaker_id: Optional[str] = None,
    ) -> str:
        if self.session_active:
            raise RuntimeError("A session is already active")

        session_id = str(uuid.uuid4())
        self.state = SessionState(
            session_id=session_id, session_type=session_type, file_path=file_path
        )
        self.session_active = True
        self.last_report = None
        self.audio_queue = asyncio.Queue()
        self.source_task = None
        self._processing_chunk = False
        self._turn_processor = TurnProcessor(self.inference_engine)
        self._suggestions = SuggestionBuilder(
            self.suggestion_engine, self.gemini_analyzer
        )

        logger.info("Starting session %s (type=%s)", session_id, session_type)
        self._log_degraded_models()

        try:
            if session_type == "replay":
                # Runs the real pipeline over a stored transcript, with no
                # AssemblyAI or Gemini credentials. See ReplayProcessor.
                if not file_path:
                    raise ValueError("file_path (transcript) required for replay mode")
                self.file_processor = ReplayProcessor(file_path, coach_speaker_id)
                self.processing_task = asyncio.create_task(self._pipeline())
                self.source_task = asyncio.create_task(self._drain_source(file_path))
            elif session_type == "file":
                if not file_path:
                    raise ValueError("file_path required for file mode")
                self.file_processor = FileAudioProcessor(
                    self.assemblyai_key, coach_speaker_id
                )
                self.processing_task = asyncio.create_task(self._pipeline())
                self.source_task = asyncio.create_task(self._drain_source(file_path))
            else:
                self.audio_processor = AudioProcessor(
                    self.assemblyai_key, coach_speaker_id=coach_speaker_id
                )
                await self.audio_processor.start_live_transcription(
                    self.audio_queue, device_index=device_index
                )
                self.processing_task = asyncio.create_task(self._pipeline())

            return session_id

        except Exception as exc:
            self.session_active = False
            if self.audio_processor:
                try:
                    await self.audio_processor.stop_transcription()
                except Exception:
                    logger.debug("Audio processor cleanup failed", exc_info=True)
            raise RuntimeError(f"Failed to start session: {exc}") from exc

    async def stop_session(self) -> SessionReport:
        if not self.session_active or not self.state:
            raise RuntimeError("No active session to stop")

        logger.info("Stopping session %s", self.state.session_id)
        self.session_active = False

        # A source stopped mid-playback would otherwise keep pushing turns
        # into a queue nobody drains.
        if self.source_task and not self.source_task.done():
            self.source_task.cancel()

        await asyncio.sleep(0.5)

        if self.processing_task:
            try:
                await asyncio.wait_for(self.processing_task, timeout=3.0)
            except asyncio.TimeoutError:
                logger.warning("Processing task timed out; cancelling")
                self.processing_task.cancel()

        if self.audio_processor:
            try:
                await asyncio.wait_for(
                    self.audio_processor.stop_transcription(), timeout=5.0
                )
            except Exception as exc:
                logger.warning("Error stopping audio processor: %s", exc)

        report = await self._build_report()
        self.last_report = report

        asyncio.create_task(self._store_safely(report))
        logger.info("Session %s completed", self.state.session_id)
        return report

    # -- pipeline ----------------------------------------------------------

    async def _drain_source(self, file_path: str) -> None:
        """Feed a finite source, then announce that playback is over.

        Replay and file sessions end; live ones do not. Without this the
        dashboard keeps polling a session whose input dried up minutes
        ago, with nothing to mark the last turn as the last. The session
        deliberately stays active - stopping it is what builds the report
        - so this only sets :attr:`SessionState.source_finished` and tells
        the clients.
        """
        try:
            await self.file_processor.process_file(file_path, self.audio_queue)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("Source failed: %s", exc, exc_info=True)
            if self.state:
                self.state.source_finished = True
                await self.broadcaster.send_playback_complete(
                    turns=self.state.turn_count, error=str(exc)
                )
            return

        # The last turn is queued, not analysed. Wait for the pipeline to
        # catch up so "complete" means the dashboard has everything.
        while self.session_active and (
            not self.audio_queue.empty() or self._processing_chunk
        ):
            await asyncio.sleep(0.2)

        if not self.session_active or not self.state:
            return

        self.state.source_finished = True
        logger.info(
            "Playback finished after %d turns; session idle until stopped",
            self.state.turn_count,
        )
        await self.broadcaster.send_playback_complete(turns=self.state.turn_count)

    async def _pipeline(self) -> None:
        logger.info("Processing pipeline started")
        while self.session_active:
            try:
                chunk = await asyncio.wait_for(self.audio_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except Exception as exc:
                logger.error("Pipeline receive error: %s", exc, exc_info=True)
                continue

            try:
                self._processing_chunk = True
                await self._process_chunk(chunk)
            except Exception as exc:
                logger.error("Chunk processing error: %s", exc, exc_info=True)
            finally:
                self._processing_chunk = False
        logger.info("Processing pipeline stopped")

    async def _process_chunk(self, chunk: AudioChunk) -> None:
        if not chunk.is_final:
            await self.broadcaster.send_partial(chunk)
            return

        state = self.state
        state.chunks.append(chunk)

        result: TurnResult = await self._turn_processor.process(chunk, state.chunks)

        suggestions = await self._suggestions.build(
            chunk=chunk,
            inferences=result,
            grow_phase=result.grow_phase,
            history=state.recent(10),
            sarcasm=result.sarcasm,
            digression=result.digression,
            turn_index=state.turn_count,
        )

        feedback = RealTimeFeedback(
            timestamp=chunk.timestamp,
            speaker=chunk.speaker,
            grow_phase=result.grow_phase,
            emotion_trend=result.emotion,
            engagement_score=result.engagement,
            coaching_quality=result.quality,
            suggestions=suggestions,
            emotion_source=result.sources.get("emotion"),
        )
        state.feedback_history.append(feedback)

        state.sarcasm_detections.append({
            **result.sarcasm,
            "timestamp": chunk.timestamp,
            "speaker": chunk.speaker,
            "text": chunk.transcript[:160],
        })
        state.digression_records.append({
            **result.digression,
            "timestamp": chunk.timestamp,
            "speaker": chunk.speaker,
            "text": chunk.transcript[:160],
        })
        state.vak_scores.append(result.vak)

        await self.broadcaster.send_turn(chunk, feedback, result)

        logger.info(
            "Turn %d [%s] phase=%s engagement=%.2f sarcasm=%.2f digression=%.2f",
            state.turn_count, chunk.speaker, result.grow_phase.phase,
            result.engagement, result.sarcasm.get("score", 0.0),
            result.digression.get("score", 0.0),
        )

    # -- reporting ---------------------------------------------------------

    async def _build_report(self) -> SessionReport:
        model_status = self.inference_engine.get_model_status()
        sources = self._turn_processor.sources if self._turn_processor else {}
        report_input = self.state.to_report_input(model_status, sources)

        local_payload = self.local_analyzer.generate_comprehensive_report(report_input)

        if self.gemini_analyzer and getattr(self.gemini_analyzer, "model", None):
            try:
                narrative = await asyncio.wait_for(
                    self.gemini_analyzer.generate_session_report(
                        report_input, computed=local_payload
                    ),
                    timeout=GEMINI_TIMEOUT_SECONDS,
                )
                if isinstance(narrative, dict):
                    # Gemini narrates; computed metrics stay authoritative.
                    local_payload = self._merge_narrative(local_payload, narrative)
                    logger.info("Gemini narrative merged into report")
            except asyncio.TimeoutError:
                logger.warning("Gemini timed out; using local report only")
            except Exception as exc:
                logger.warning("Gemini failed (%s); using local report only", exc)

        try:
            return SessionReport(**local_payload)
        except Exception as exc:
            logger.error("Report validation failed: %s", exc, exc_info=True)
            return SessionReport(
                **self.local_analyzer.empty_report(report_input)
            )

    @staticmethod
    def _merge_narrative(
        computed: Dict[str, Any], narrative: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Let Gemini rewrite prose only.

        Metrics (GROW distribution, engagement, learning style, sarcasm,
        digression, participants) are computed locally from per-turn data
        and are never replaced by the model's own estimates.
        """
        merged = dict(computed)
        for field in ("key_insights", "recommendations", "transcript_summary"):
            value = narrative.get(field)
            if value:
                merged[field] = value
        return merged

    async def _store_safely(self, report: SessionReport) -> None:
        if not self.storage:
            return
        try:
            await self.storage.store_session_report(report)
            logger.info("Stored session %s in ChromaDB", report.session_id)
        except Exception as exc:
            logger.warning("Background storage failed: %s", exc)

    # -- diagnostics -------------------------------------------------------

    def _log_degraded_models(self) -> None:
        status = self.inference_engine.get_model_status()
        if not status.get("all_trained"):
            logger.warning(
                "Session starting with %d/%d models on trained weights. "
                "Degraded: %s. Affected metrics are computed by documented "
                "heuristics and are labelled as such in the report.",
                status.get("trained_count", 0), status.get("total_count", 0),
                ", ".join(status.get("degraded", [])) or "none",
            )
