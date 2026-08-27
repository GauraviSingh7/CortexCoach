"""
VAK (Visual / Auditory / Kinesthetic) learning-style inference.

Heuristic only. The trained BERT classifier under ``models/vak_inference/``
ships its config, tokenizer and label encoder but **not** its weights, so it
cannot be loaded at all - see :mod:`backend.models.model_status`.

Only coachee speech is scored: the learning style being reported is the
coachee's, and mixing the coach's language into it was inflating whichever
style the coach happened to favour.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Sequence

from backend.analysis.text import count_terms, normalize

logger = logging.getLogger(__name__)

VISUAL = "visual"
AUDITORY = "auditory"
KINESTHETIC = "kinesthetic"
STYLES = (VISUAL, AUDITORY, KINESTHETIC)

#: Below this share, no single style is dominant enough to name.
_DOMINANCE_FLOOR = 0.40

#: Minimum coachee turns before a style is reported at all.
_MIN_TURNS = 2

_PHRASES: Dict[str, Sequence[str]] = {
    VISUAL: (
        "i can see", "i see it", "looks like", "picture this", "picture it",
        "from my perspective", "the way i see it", "let me show you",
        "laid out", "clear picture", "in focus", "looking at it",
    ),
    AUDITORY: (
        "sounds like", "i hear you", "listen to this", "tell me about",
        "rings a bell", "word for word", "talk it through", "say out loud",
        "out loud", "ask my manager", "ask him",
    ),
    KINESTHETIC: (
        "i feel like", "get a grip", "hands on", "gut feeling",
        "my sense is", "concrete example", "walk through", "hold on to",
        "feels like", "get a handle",
    ),
}

_STRONG: Dict[str, Sequence[str]] = {
    VISUAL: (
        "see", "look", "picture", "imagine", "visualize", "view", "watch",
        "show", "visible", "invisible", "blur", "clear",
    ),
    AUDITORY: (
        "hear", "listen", "sound", "tell", "say", "talk", "discuss",
        "mention", "ask", "conversation", "announce", "quiet", "loud",
    ),
    KINESTHETIC: (
        "feel", "touch", "grasp", "hold", "sense", "experience", "handle",
        "concrete", "solid", "heavy", "grip", "stuck", "move",
    ),
}

_MEDIUM: Dict[str, Sequence[str]] = {
    VISUAL: ("appears", "bright", "focus", "perspective", "illustrate", "vague"),
    AUDITORY: ("voice", "tone", "resonate", "verbal", "dialogue", "meetings"),
    KINESTHETIC: ("pressure", "comfortable", "flow", "lighter", "exposed"),
}

_PHRASE_WEIGHT = 5
_STRONG_WEIGHT = 3
_MEDIUM_WEIGHT = 1

#: Number of most recent coachee turns considered.
_WINDOW = 10


@dataclass
class VAKResult:
    """Learning-style estimate over the coachee's recent speech."""

    visual: float = 0.0
    auditory: float = 0.0
    kinesthetic: float = 0.0
    dominant: str = "Insufficient Data"
    confidence: float = 0.0
    source: str = "heuristic"

    @property
    def has_signal(self) -> bool:
        return self.confidence > 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "visual": self.visual,
            "auditory": self.auditory,
            "kinesthetic": self.kinesthetic,
            "dominant": self.dominant,
            "confidence": self.confidence,
            "source": self.source,
        }


class VAKAnalyzer:
    """Scores the coachee's sensory-predicate usage over a rolling window."""

    def analyze(self, history: Sequence[Any]) -> VAKResult:
        try:
            return self._analyze(history)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("VAK analysis failed: %s", exc, exc_info=True)
            return VAKResult()

    def _analyze(self, history: Sequence[Any]) -> VAKResult:
        coachee = [c for c in history if getattr(c, "speaker", None) == "coachee"]
        if len(coachee) < _MIN_TURNS:
            return VAKResult()

        raw = {style: 0.0 for style in STYLES}
        for chunk in coachee[-_WINDOW:]:
            text = getattr(chunk, "transcript", "") or ""
            for style in STYLES:
                raw[style] += (
                    count_terms(text, _PHRASES[style]) * _PHRASE_WEIGHT
                    + count_terms(text, _STRONG[style]) * _STRONG_WEIGHT
                    + count_terms(text, _MEDIUM[style]) * _MEDIUM_WEIGHT
                )

        shares = normalize(raw)
        if not shares:
            # No sensory language at all - do not invent 0.33/0.33/0.34.
            return VAKResult(dominant="Insufficient Data")

        dominant_style = max(shares, key=shares.get)
        confidence = shares[dominant_style]
        label = (
            f"{dominant_style.capitalize()} ({confidence:.0%})"
            if confidence >= _DOMINANCE_FLOOR else "Balanced (Mixed)"
        )

        return VAKResult(
            visual=shares[VISUAL],
            auditory=shares[AUDITORY],
            kinesthetic=shares[KINESTHETIC],
            dominant=label,
            confidence=confidence,
        )


def average_styles(vak_scores: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Average per-turn VAK estimates for the session report.

    Returns ``{}`` when nothing scored, so the UI can say "Insufficient
    Data" instead of showing a fabricated even split.
    """
    usable = [
        v for v in (vak_scores or [])
        if isinstance(v, dict) and float(v.get("confidence", 0.0)) > 0.0
    ]
    if not usable:
        return {}
    n = len(usable)
    return {
        style: sum(float(v.get(style, 0.0)) for v in usable) / n
        for style in STYLES
    }
