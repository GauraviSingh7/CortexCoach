"""
Sarcasm detection heuristics.

This is explicitly a *heuristic* detector. The trained Keras LSTM shipped in
``models/sarcasm_detection/model_lstm.pkl`` cannot be used because the Keras
tokenizer / word-index from training was never shipped alongside it, so there
is no way to map text onto its 30k-token embedding vocabulary. See
:mod:`backend.models.model_status` for how that is surfaced to the API.

The previous implementation missed the two most common forms of verbal irony
in coaching conversations:

* ironic commentary appended to a complaint
  ("... which is hilarious, because I stayed until midnight")
* epistemic overstatement attached to a positive outcome word
  ("Clearly that worked out great for me.")

Both are now first-class patterns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from backend.analysis.text import has_any, matched_terms, word_count

logger = logging.getLogger(__name__)

#: Score at or above which an utterance is reported as sarcastic.
SARCASM_THRESHOLD = 0.45

# --------------------------------------------------------------------------
# Pattern inventory
# --------------------------------------------------------------------------

# Ironic commentary a speaker appends to something that annoyed them.
_IRONIC_COMMENTARY = (
    "which is hilarious", "which is funny", "which is great", "which is rich",
    "which is perfect", "which is wonderful", "how nice", "how lovely",
    "how wonderful", "oh joy", "oh good", "yeah right", "sure sure",
    "big surprise", "what a shock", "go figure", "shocking",
    "who would have thought", "thanks for nothing", "just my luck",
)

# Overstated certainty. Sarcastic when paired with a positive outcome phrase.
_EPISTEMIC_OVERSTATEMENT = (
    "clearly", "obviously", "apparently", "evidently", "naturally",
    "of course", "sure enough", "needless to say",
)

# Positive-outcome language. Sincere on its own; ironic in a complaint frame.
_POSITIVE_OUTCOME = (
    "worked out great", "worked out well", "worked great", "worked well",
    "went great", "went well", "turned out great", "turned out well",
    "a great help", "so helpful", "real helpful", "really helpful",
    "just great", "just perfect", "just wonderful", "great for me",
    "perfect timing", "best thing ever",
)

_POSITIVE_WORDS = (
    "great", "wonderful", "perfect", "amazing", "fantastic", "excellent",
    "brilliant", "awesome", "lovely", "terrific", "delightful", "charming",
)

# Frames that make surrounding positive language read as ironic.
_COMPLAINT_FRAME = (
    "nobody", "no one", "never", "not", "midnight", "again", "still",
    "supposed to", "should have", "passed over", "ignored", "unnoticed",
    "nothing", "instead", "wasted", "pointless", "useless",
)

_PASSIVE_AGGRESSIVE = (
    "no offense but", "with all due respect", "not to be rude but",
    "no disrespect but", "if you say so", "whatever you say",
)

_DISMISSIVE = ("sure", "fine", "whatever", "okay", "great")


@dataclass
class SarcasmResult:
    """Outcome of scoring one utterance."""

    score: float = 0.0
    is_sarcastic: bool = False
    type: str = "none"
    explanation: str = "No sarcasm detected"
    source: str = "heuristic"
    signals: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Wire format consumed by the orchestrator, WebSocket and reports."""
        return {
            "score": self.score,
            "is_sarcastic": self.is_sarcastic,
            "type": self.type,
            "explanation": self.explanation,
            "source": self.source,
        }


class SarcasmAnalyzer:
    """Stateless scorer; safe to share across sessions."""

    def analyze(
        self,
        transcript: str,
        speaker: str = "coachee",
        history: Optional[Sequence[Any]] = None,
    ) -> SarcasmResult:
        try:
            return self._analyze(transcript, speaker, history or ())
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Sarcasm analysis failed: %s", exc, exc_info=True)
            return SarcasmResult(explanation="detection failed")

    # -- internals ---------------------------------------------------------

    def _analyze(
        self, transcript: str, speaker: str, history: Sequence[Any]
    ) -> SarcasmResult:
        text = transcript.strip()
        if not text:
            return SarcasmResult()

        score = 0.0
        signals: List[str] = []
        kind = "none"

        # 1. Ironic commentary markers - the strongest single signal.
        hits = matched_terms(text, _IRONIC_COMMENTARY)
        if hits:
            score += 0.65
            kind = "ironic_commentary"
            signals.extend(hits)

        # 2. Overstated certainty attached to a positive outcome.
        overstated = matched_terms(text, _EPISTEMIC_OVERSTATEMENT)
        outcome = matched_terms(text, _POSITIVE_OUTCOME)
        if overstated and outcome:
            score += 0.6
            kind = "mock_enthusiasm" if kind == "none" else kind
            signals.extend(overstated + outcome)
        elif outcome and has_any(text, _COMPLAINT_FRAME):
            # Positive outcome language inside a complaint frame.
            score += 0.45
            kind = "mock_enthusiasm" if kind == "none" else kind
            signals.extend(outcome)

        # 3. Positive sentiment word sitting in an explicit complaint frame.
        positives = matched_terms(text, _POSITIVE_WORDS)
        if positives and has_any(text, _COMPLAINT_FRAME) and not outcome:
            score += 0.3
            kind = "contradiction" if kind == "none" else kind
            signals.extend(positives)

        # 4. Passive-aggressive politeness formulas.
        pa = matched_terms(text, _PASSIVE_AGGRESSIVE)
        if pa:
            score += 0.5
            kind = "passive_aggressive" if kind == "none" else kind
            signals.extend(pa)

        # 5. Curt dismissal immediately after a coach question.
        if self._is_curt_dismissal(text, speaker, history):
            score += 0.35
            kind = "dismissive" if kind == "none" else kind
            signals.append("short dismissive reply")

        score = min(score, 1.0)
        is_sarcastic = score >= SARCASM_THRESHOLD

        return SarcasmResult(
            score=score,
            is_sarcastic=is_sarcastic,
            type=kind if is_sarcastic else "none",
            explanation=(
                f"Signals: {', '.join(dict.fromkeys(signals))}" if is_sarcastic
                else "No sarcasm detected"
            ),
            signals=list(dict.fromkeys(signals)),
        )

    @staticmethod
    def _is_curt_dismissal(
        text: str, speaker: str, history: Sequence[Any]
    ) -> bool:
        if speaker != "coachee" or word_count(text) > 4:
            return False
        if not history:
            return False
        previous = history[-1]
        prior_speaker = getattr(previous, "speaker", None)
        return prior_speaker == "coach" and has_any(text, _DISMISSIVE)


def summarize(detections: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Roll per-turn sarcasm results up into a report section."""
    if not detections:
        return {}
    scores = [float(d.get("score", 0.0)) for d in detections]
    flagged = [d for d in detections if d.get("is_sarcastic")]
    by_type: Dict[str, int] = {}
    for d in flagged:
        by_type[d.get("type", "none")] = by_type.get(d.get("type", "none"), 0) + 1
    return {
        "count_detected": len(flagged),
        "total_evaluated": len(detections),
        "average_score": sum(scores) / len(scores),
        "max_score": max(scores),
        "by_type": by_type,
        "moments": [
            {
                "speaker": d.get("speaker"),
                "text": d.get("text", "")[:160],
                "score": d.get("score"),
                "type": d.get("type"),
            }
            for d in flagged[:10]
        ],
        "source": detections[0].get("source", "heuristic"),
    }
