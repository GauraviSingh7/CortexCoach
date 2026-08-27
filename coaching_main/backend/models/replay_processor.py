"""
Replay a stored transcript through the real analysis pipeline.

This exists so the whole system - backend, WebSocket, dashboard - can be
run and demonstrated without AssemblyAI or Gemini credentials, and so the
analysis layer can be exercised against a known conversation.

It deliberately mirrors :class:`FileAudioProcessor`: same constructor
shape, same ``process_file`` signature, same two-pass speaker assignment.
Everything downstream of the queue is the production code path.

Transcript format - a JSON list of ``{"speaker": ..., "text": ...}``::

    [
      {"speaker": "A", "text": "Hi Priya, good to see you..."},
      {"speaker": "B", "text": "Hi Marcus. Honestly, I want to..."}
    ]

``speaker`` is a diarization label ("A"/"B"), not a role - role assignment
is exactly what SpeakerRouter is being exercised on.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

from backend.models.speaker_router import SpeakerRouter
from backend.schemas.data_models import AudioChunk

logger = logging.getLogger(__name__)

DEFAULT_PLAYBACK_DELAY = 0.4

#: Nominal seconds attributed to each replayed turn, so timestamps advance.
_SECONDS_PER_TURN = 8.0


class ReplayProcessor:
    """Feeds a stored transcript into the pipeline as if it were live."""

    def __init__(
        self,
        transcript_path: str,
        coach_speaker_id: Optional[str] = None,
        playback_delay: float = DEFAULT_PLAYBACK_DELAY,
    ):
        self.transcript_path = Path(transcript_path)
        self.playback_delay = playback_delay
        self.router = SpeakerRouter(coach_speaker_id)
        self.audio_queue: Optional[asyncio.Queue] = None

    async def process_file(self, _file_path, audio_queue: asyncio.Queue) -> None:
        """Signature matches FileAudioProcessor so the orchestrator is shared."""
        self.audio_queue = audio_queue
        turns = self._load()
        logger.info(
            "Replaying %d turns from %s", len(turns), self.transcript_path.name
        )

        roles = self.router.assign_batch(turns)
        logger.info("Diarized speakers -> roles: %s", roles)

        for index, turn in enumerate(turns):
            chunk = AudioChunk(
                timestamp=index * _SECONDS_PER_TURN,
                duration=_SECONDS_PER_TURN,
                speaker=self.router.role_for(turn.speaker, turn.text),
                transcript=turn.text,
                audio_data=None,
                speaker_id=str(turn.speaker),
                is_final=True,
            )
            await audio_queue.put(chunk)
            if self.playback_delay:
                await asyncio.sleep(self.playback_delay)

        logger.info("Replay finished")

    def _load(self) -> List[SimpleNamespace]:
        if not self.transcript_path.exists():
            raise FileNotFoundError(f"Transcript not found: {self.transcript_path}")

        raw = json.loads(self.transcript_path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("Transcript must be a JSON list of turns")

        turns: List[SimpleNamespace] = []
        for entry in raw:
            text = entry.get("text") or entry.get("transcript") or ""
            speaker = entry.get("speaker")
            if not text or speaker is None:
                continue
            turns.append(SimpleNamespace(speaker=str(speaker), text=text))

        if not turns:
            raise ValueError("Transcript contained no usable turns")
        return turns
