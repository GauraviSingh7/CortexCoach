"""
Off-topic drift detection.

Explicit discourse markers ("by the way", "random, but", "did you watch...")
are the authoritative signal and are the only thing that flags a turn as an
off-topic moment.

Lexical overlap against the recent topic window is kept as an *advisory*
score only, deliberately capped below the reporting threshold. Natural
coaching dialogue has high vocabulary turnover - a perfectly on-topic turn
routinely shares almost no content words with the preceding ones - so
overlap alone cannot separate genuine drift from ordinary variety, and
using it as a flag produced false positives on two thirds of the session.
Detecting semantic drift properly needs sentence embeddings; see the note
in ``docs/known-gaps.md``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from backend.analysis.text import content_words, has_any, overlap_ratio, top_keywords

logger = logging.getLogger(__name__)

#: Score above which a turn counts as an off-topic moment.
DIGRESSION_THRESHOLD = 0.5

# Explicit topic-change markers. High precision, so they short-circuit.
_MARKERS = (
    "by the way", "speaking of", "that reminds me", "off topic",
    "off-topic", "random thought", "random but", "unrelated",
    "change the subject", "oh also", "on another note", "totally different",
    "side note", "tangent", "did you watch", "did you see",
)

_MARKER_SCORE = 0.7

#: Turns of history needed before overlap can be measured meaningfully.
_MIN_HISTORY = 4

#: How many prior turns form the topic vocabulary.
_WINDOW = 10

#: Fewer content words than this and the turn is unjudgeable.
_MIN_CONTENT_WORDS = 2

#: Ceiling for overlap-derived scores. Sits below DIGRESSION_THRESHOLD on
#: purpose: weak lexical evidence must never flag a turn on its own.
_ADVISORY_CEILING = 0.35


@dataclass
class DigressionResult:
    score: float = 0.0
    is_digression: bool = False
    reason: str = "on topic"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "is_digression": self.is_digression,
            "reason": self.reason,
        }


class DigressionAnalyzer:
    """Stateless; the caller supplies the conversation history."""

    def analyze(
        self, transcript: str, history: Optional[Sequence[Any]] = None
    ) -> DigressionResult:
        try:
            return self._analyze(transcript, history or ())
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Digression analysis failed: %s", exc, exc_info=True)
            return DigressionResult()

    def _analyze(self, transcript: str, history: Sequence[Any]) -> DigressionResult:
        # An explicit marker is decisive regardless of how much history exists.
        if has_any(transcript, _MARKERS):
            return DigressionResult(
                score=_MARKER_SCORE, is_digression=True,
                reason="explicit topic-change marker",
            )

        if len(history) < _MIN_HISTORY:
            return DigressionResult(reason="insufficient context")

        current = set(content_words(transcript))
        if len(current) < _MIN_CONTENT_WORDS:
            return DigressionResult(reason="too few content words to judge")

        # Topic vocabulary from prior turns, excluding the current one.
        window = [
            getattr(c, "transcript", "") or "" for c in history[-(_WINDOW + 1):-1]
        ]
        vocabulary = set(top_keywords(window, limit=80))
        if not vocabulary:
            return DigressionResult(reason="no topic vocabulary yet")

        # Containment: what share of THIS turn's content is already in play.
        # Advisory only - never enough on its own to flag a digression.
        overlap = overlap_ratio(current, vocabulary)
        if overlap < 0.10:
            return DigressionResult(
                _ADVISORY_CEILING, False, f"low lexical overlap ({overlap:.2f})"
            )
        if overlap < 0.25:
            return DigressionResult(0.2, False, f"partial topic shift ({overlap:.2f})")
        return DigressionResult(0.1, False, f"on topic ({overlap:.2f})")


def summarize(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Roll per-turn digression scores up into a report section."""
    if not results:
        return {}
    scores = [float(r.get("score", 0.0)) for r in results]
    flagged = [r for r in results if r.get("is_digression")]
    return {
        "average_score": sum(scores) / len(scores),
        "max_score": max(scores),
        "off_topic_moments": len(flagged),
        "total_evaluated": len(results),
        "moments": [
            {
                "speaker": r.get("speaker"),
                "text": r.get("text", "")[:160],
                "score": r.get("score"),
                "reason": r.get("reason"),
            }
            for r in flagged[:10]
        ],
    }
