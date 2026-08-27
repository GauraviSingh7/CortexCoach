"""
Per-turn analysis pipeline.

This is where trained-model output and heuristic fallbacks are reconciled.
The rule is: **use the model when it produced something, fall back to the
documented heuristic otherwise, and always record which one was used.**

Previously the orchestrator called the inference engine and then
immediately overwrote its sarcasm, VAK and digression results with inline
rule-based functions - so even a correctly loaded model would have had its
output discarded. Provenance is now explicit and travels all the way into
the report and the dashboard.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Sequence

from backend.analysis.digression import DigressionAnalyzer
from backend.analysis.emotion import analyze_text_emotion
from backend.analysis.grow import GROWClassifier
from backend.analysis.quality import analyze_turn
from backend.analysis.sarcasm import SarcasmAnalyzer, SarcasmResult
from backend.analysis.vak import VAKAnalyzer, VAKResult
from backend.schemas.data_models import AudioChunk, GROWPhase, ModelInferences

logger = logging.getLogger(__name__)

MODEL = "model"
HEURISTIC = "heuristic"
UNAVAILABLE = "unavailable"

#: Engagement value used when nothing can score a turn. Reported with an
#: "unavailable" source so it is never mistaken for a measurement.
_NEUTRAL_ENGAGEMENT = 0.5


@dataclass
class TurnResult:
    """Everything computed for a single turn."""

    grow_phase: GROWPhase
    emotion: Dict[str, float] = field(default_factory=dict)
    engagement: float = _NEUTRAL_ENGAGEMENT
    quality: Dict[str, float] = field(default_factory=dict)
    sarcasm: Dict[str, Any] = field(default_factory=dict)
    digression: Dict[str, Any] = field(default_factory=dict)
    vak: Dict[str, Any] = field(default_factory=dict)
    sources: Dict[str, str] = field(default_factory=dict)

    @property
    def interest_level(self) -> float:
        """Alias so ContextualSuggestionEngine can consume a TurnResult."""
        return self.engagement


class TurnProcessor:
    """Session-scoped analysis pipeline for one conversation."""

    def __init__(self, inference_engine):
        self.engine = inference_engine
        self.grow = GROWClassifier()
        self.sarcasm = SarcasmAnalyzer()
        self.vak = VAKAnalyzer()
        self.digression = DigressionAnalyzer()
        self._sources: Dict[str, str] = {}

    @property
    def sources(self) -> Dict[str, str]:
        """Provenance of each signal, accumulated over the session."""
        return dict(self._sources)

    async def process(
        self, chunk: AudioChunk, history: Sequence[AudioChunk]
    ) -> TurnResult:
        inferences = await self._infer(chunk)

        emotion, emotion_source = self._resolve_emotion(chunk, inferences)
        engagement, engagement_source = self._resolve_engagement(inferences)
        sarcasm, sarcasm_source = self._resolve_sarcasm(chunk, inferences, history)
        vak, vak_source = self._resolve_vak(inferences, history)
        digression = self.digression.analyze(chunk.transcript, history)

        phase = self.grow.classify(chunk.transcript, chunk.speaker)
        grow_phase = GROWPhase(
            phase=phase.phase,
            confidence=phase.confidence,
            reasoning=phase.reasoning,
            inherited=phase.inherited,
        )
        quality = analyze_turn(chunk.transcript, chunk.speaker, phase.confidence)

        self._sources.update({
            "emotion": emotion_source,
            "engagement": engagement_source,
            "sarcasm": sarcasm_source,
            "vak": vak_source,
            "digression": HEURISTIC,
            "grow_phase": HEURISTIC,
            "coaching_quality": HEURISTIC,
        })

        return TurnResult(
            grow_phase=grow_phase,
            emotion=emotion,
            engagement=engagement,
            quality=quality.to_dict(),
            sarcasm=sarcasm.to_dict() if isinstance(sarcasm, SarcasmResult) else sarcasm,
            digression=digression.to_dict(),
            vak=vak.to_dict() if isinstance(vak, VAKResult) else vak,
            sources=self.sources,
        )

    # -- model / heuristic reconciliation ----------------------------------

    async def _infer(self, chunk: AudioChunk) -> ModelInferences:
        try:
            return await self.engine.process_chunk(chunk)
        except Exception as exc:
            logger.error("Model inference failed for chunk: %s", exc, exc_info=True)
            return ModelInferences()

    def _is_trained(self, model_name: str) -> bool:
        status = self.engine.status_of(model_name) if self.engine else None
        return bool(status and status.is_trained)

    def _resolve_emotion(self, chunk: AudioChunk, inferences: ModelInferences):
        if inferences.emotion:
            return inferences.emotion, MODEL
        heuristic = analyze_text_emotion(chunk.transcript)
        if heuristic:
            return heuristic, HEURISTIC
        # No signal at all. Return {} - never a manufactured neutral.
        return {}, UNAVAILABLE

    def _resolve_engagement(self, inferences: ModelInferences):
        if inferences.interest_level is None:
            return _NEUTRAL_ENGAGEMENT, UNAVAILABLE
        source = MODEL if self._is_trained("interest_detection") else HEURISTIC
        return float(inferences.interest_level), source

    def _resolve_sarcasm(
        self, chunk: AudioChunk, inferences: ModelInferences, history
    ):
        if inferences.sarcasm_score is not None:
            score = float(inferences.sarcasm_score)
            return (
                SarcasmResult(
                    score=score,
                    is_sarcastic=score >= 0.5,
                    type="model" if score >= 0.5 else "none",
                    explanation="trained sarcasm classifier",
                    source=MODEL,
                ),
                MODEL,
            )
        return self.sarcasm.analyze(chunk.transcript, chunk.speaker, history), HEURISTIC

    def _resolve_vak(self, inferences: ModelInferences, history):
        if inferences.vak_style:
            styles = inferences.vak_style
            dominant = max(styles, key=styles.get)
            return (
                VAKResult(
                    visual=styles.get("visual", 0.0),
                    auditory=styles.get("auditory", 0.0),
                    kinesthetic=styles.get("kinesthetic", 0.0),
                    dominant=f"{dominant.capitalize()} ({styles[dominant]:.0%})",
                    confidence=styles[dominant],
                    source=MODEL,
                ),
                MODEL,
            )
        return self.vak.analyze(history), HEURISTIC
