"""WebSocket fan-out to connected dashboards."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Set

from backend.schemas.data_models import AudioChunk, RealTimeFeedback

logger = logging.getLogger(__name__)


class FeedbackBroadcaster:
    """Sends per-turn payloads to every connected client.

    Clients that fail to receive are dropped rather than retried, so one
    dead socket cannot stall the processing pipeline.
    """

    def __init__(self) -> None:
        self.clients: Set[Any] = set()

    def add(self, client: Any) -> None:
        self.clients.add(client)

    def discard(self, client: Any) -> None:
        self.clients.discard(client)

    @property
    def client_count(self) -> int:
        return len(self.clients)

    async def send(self, payload: Dict[str, Any]) -> None:
        if not self.clients:
            return
        message = json.dumps(payload, default=str)
        dead = set()
        for client in self.clients:
            try:
                await client.send_text(message)
            except Exception as exc:
                logger.debug("Dropping unreachable websocket client: %s", exc)
                dead.add(client)
        self.clients -= dead

    async def send_partial(self, chunk: AudioChunk) -> None:
        """Live text for an utterance still in progress - no analysis."""
        await self.send({
            "type": "partial",
            "speaker": chunk.speaker,
            "speaker_id": chunk.speaker_id,
            "transcript": chunk.transcript,
            "timestamp": chunk.timestamp,
        })

    async def send_turn(
        self,
        chunk: AudioChunk,
        feedback: RealTimeFeedback,
        result: Any,
    ) -> None:
        """Full analysis payload for a completed turn."""
        sarcasm = result.sarcasm or {}
        vak = result.vak or {}
        digression = result.digression or {}

        await self.send({
            "type": "final",
            "timestamp": feedback.timestamp,
            "speaker": feedback.speaker,
            "speaker_id": chunk.speaker_id,
            "transcript": chunk.transcript,
            "grow_phase": {
                "phase": feedback.grow_phase.phase,
                "confidence": feedback.grow_phase.confidence,
                "reasoning": feedback.grow_phase.reasoning,
                "inherited": feedback.grow_phase.inherited,
            },
            "emotion_trend": feedback.emotion_trend,
            "engagement_score": feedback.engagement_score,
            "coaching_quality": feedback.coaching_quality,
            "suggestions": feedback.suggestions,
            "learning_style": vak.get("dominant", "Insufficient Data"),
            "vak_visual": vak.get("visual", 0.0),
            "vak_auditory": vak.get("auditory", 0.0),
            "vak_kinesthetic": vak.get("kinesthetic", 0.0),
            "vak_confidence": vak.get("confidence", 0.0),
            "digression_level": digression.get("score", 0.0),
            "digression_detected": digression.get("is_digression", False),
            "sarcasm_detected": sarcasm.get("is_sarcastic", False),
            "sarcasm_score": sarcasm.get("score", 0.0),
            "sarcasm_type": sarcasm.get("type", "none"),
            # Provenance so the dashboard can label heuristic values.
            "sources": result.sources,
        })
