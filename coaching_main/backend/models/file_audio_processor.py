"""
File-based audio processing via AssemblyAI batch transcription.

Role assignment is a two-pass operation here: file mode has the entire
transcript before any chunk is emitted, so speaker roles are decided once
from whole-session evidence rather than guessed per utterance. See
:mod:`backend.models.speaker_router` for why that matters.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional, Sequence

import assemblyai as aai

from backend.models.speaker_router import SpeakerRouter
from backend.schemas.data_models import AudioChunk

logger = logging.getLogger(__name__)

#: Delay between emitted chunks so the dashboard animates like a live
#: session. Set to 0 for batch processing with no playback pacing.
DEFAULT_PLAYBACK_DELAY = 0.5


class FileAudioProcessor:
    """Transcribes a media file and replays it as :class:`AudioChunk` items."""

    def __init__(
        self,
        api_key: str,
        coach_speaker_id: Optional[str] = None,
        playback_delay: float = DEFAULT_PLAYBACK_DELAY,
    ):
        self.api_key = api_key
        aai.settings.api_key = api_key
        self.transcriber = aai.Transcriber()
        self.audio_queue: Optional[asyncio.Queue] = None
        self.playback_delay = playback_delay
        self.router = SpeakerRouter(coach_speaker_id)

    async def process_file(self, file_path: str, audio_queue: asyncio.Queue) -> None:
        """Transcribe ``file_path`` and push every utterance onto the queue."""
        self.audio_queue = audio_queue
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        logger.info("Processing audio file: %s", path.name)

        config = aai.TranscriptionConfig(speaker_labels=True, speakers_expected=2)
        transcript = await asyncio.to_thread(
            self.transcriber.transcribe, str(path), config
        )

        if transcript.status == aai.TranscriptStatus.error:
            raise RuntimeError(f"Transcription failed: {transcript.error}")

        utterances = list(transcript.utterances or [])
        logger.info("Transcription completed: %d utterances", len(utterances))
        self._warn_on_suspicious_diarization(utterances)

        # Pass 1: decide roles from the whole session.
        roles = self.router.assign_batch(utterances)
        logger.info("Diarized speakers -> roles: %s", roles)

        # Pass 2: emit chunks with stable roles.
        await self._emit(utterances)
        logger.info("File processing completed")

    async def _emit(self, utterances: Sequence) -> None:
        total = len(utterances)
        for index, utterance in enumerate(utterances, start=1):
            try:
                role = self.router.role_for(utterance.speaker, utterance.text)
                chunk = AudioChunk(
                    timestamp=utterance.start / 1000.0,
                    duration=(utterance.end - utterance.start) / 1000.0,
                    speaker=role,
                    transcript=utterance.text,
                    audio_data=None,
                    speaker_id=str(utterance.speaker),
                    is_final=True,
                )
                await self.audio_queue.put(chunk)
                logger.info(
                    "Utterance %d/%d [%s/%s]: %.60s",
                    index, total, utterance.speaker, role, utterance.text,
                )
                if self.playback_delay:
                    await asyncio.sleep(self.playback_delay)
            except Exception as exc:
                logger.error("Error emitting utterance %d: %s", index, exc)
                continue

    @staticmethod
    def _warn_on_suspicious_diarization(utterances: Sequence) -> None:
        """Flag transcripts where diarization probably merged speakers.

        A back-and-forth coaching session should alternate. Long runs of
        consecutive utterances from one speaker, or only one speaker
        overall, usually mean the diarizer could not separate the voices -
        which silently halves the turn count and skews every per-speaker
        metric downstream. Worth saying out loud before anyone reads the
        numbers.
        """
        if not utterances:
            logger.warning("Transcription returned no utterances")
            return

        speakers = {str(u.speaker) for u in utterances}
        if len(speakers) < 2:
            logger.warning(
                "Diarization found only one speaker (%s) across %d utterances - "
                "coach/coachee metrics will not be meaningful.",
                speakers, len(utterances),
            )
            return

        longest_run = run = 1
        for previous, current in zip(utterances, utterances[1:]):
            run = run + 1 if current.speaker == previous.speaker else 1
            longest_run = max(longest_run, run)

        if longest_run >= 4:
            logger.warning(
                "Diarization produced a run of %d consecutive utterances from one "
                "speaker; adjacent turns may have been merged (%d utterances total).",
                longest_run, len(utterances),
            )
