"""
GROW model phase classification and distribution.

Two behavioural changes over the previous inline implementation:

1. Keyword matching is word-boundary based (see :mod:`backend.analysis.text`).
   Substring matching previously let "now" fire inside "know" and "do" fire
   inside "don't", which made Reality swallow most of the session.

2. Phases are *sticky*. GROW phases are stretches of a conversation, not a
   property of an individual sentence, and the coach is the one who steers
   them. A coach turn with evidence opens a phase; turns without evidence
   continue the phase already in progress instead of being dumped into
   "Uncertain". Only turns before the first evidenced coach turn are
   genuinely unclassified.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from backend.analysis.text import count_terms, matched_terms

logger = logging.getLogger(__name__)

GOAL = "Goal"
REALITY = "Reality"
OPTIONS = "Options"
WAY_FORWARD = "Way Forward"
UNCERTAIN = "Uncertain"

#: The four real phases, in canonical GROW order.
CANONICAL_PHASES = (GOAL, REALITY, OPTIONS, WAY_FORWARD)

# Multi-word phrases are strong evidence; single keywords are weak evidence.
_PHRASES: Dict[str, Sequence[str]] = {
    GOAL: (
        "get out of our time", "walk away with", "what would you like",
        "look like for you", "your goal", "the goal for today", "ideal outcome",
        "what do you want", "hope to", "aiming for", "success look like",
    ),
    REALITY: (
        "right now", "at the moment", "current situation", "what is happening",
        "tell me about the current", "so far", "the gap", "what is actually",
        "where are you now", "say more about",
    ),
    OPTIONS: (
        "what are some options", "some options", "what else", "what if",
        "keep going", "other ways", "could you", "possibilities",
        "brainstorm", "what other",
    ),
    WAY_FORWARD: (
        "first step", "next step", "first move", "when will you", "by when",
        "how will you know", "on a scale", "recap", "going to",
        "what would make it", "committed to", "action plan",
    ),
}

_KEYWORDS: Dict[str, Sequence[str]] = {
    GOAL: ("goal", "objective", "aim", "target", "aspire", "achieve", "outcome"),
    REALITY: (
        "currently", "situation", "reality", "happening", "problem",
        "challenge", "issue", "struggle", "gap", "pattern",
    ),
    OPTIONS: (
        "option", "options", "alternative", "alternatives", "possibility",
        "choice", "choices", "consider", "explore", "idea", "ideas", "might",
    ),
    WAY_FORWARD: (
        "plan", "commit", "commitment", "decide", "implement",
        "deadline", "friday", "tomorrow", "reminder",
    ),
}

_PHRASE_WEIGHT = 3
_KEYWORD_WEIGHT = 2

#: Minimum score for a coach turn to open (or switch to) a phase.
_EVIDENCE_THRESHOLD = 3


@dataclass
class PhaseResult:
    """Classification of a single turn."""

    phase: str
    confidence: float
    reasoning: str
    inherited: bool = False

    @property
    def is_classified(self) -> bool:
        return self.phase in CANONICAL_PHASES


def score_phases(transcript: str) -> Dict[str, int]:
    """Evidence score per phase for one utterance."""
    return {
        phase: (
            count_terms(transcript, _PHRASES[phase]) * _PHRASE_WEIGHT
            + count_terms(transcript, _KEYWORDS[phase]) * _KEYWORD_WEIGHT
        )
        for phase in CANONICAL_PHASES
    }


class GROWClassifier:
    """Session-scoped, stateful GROW phase tracker.

    One instance per coaching session; :meth:`classify` is called once per
    final turn, in order.
    """

    def __init__(self) -> None:
        self._current: Optional[str] = None

    @property
    def current_phase(self) -> Optional[str]:
        return self._current

    def classify(self, transcript: str, speaker: str) -> PhaseResult:
        # The coach steers the session, so only coach turns may open or
        # switch a phase. Coachee turns sit inside whatever phase is open.
        if speaker != "coach":
            return self._inherit("coachee turn within the current phase")

        scores = score_phases(transcript)
        phase, raw = max(scores.items(), key=lambda kv: kv[1])

        if raw < _EVIDENCE_THRESHOLD:
            return self._inherit("no new phase evidence; continuing")

        self._current = phase
        evidence = matched_terms(
            transcript, list(_PHRASES[phase]) + list(_KEYWORDS[phase])
        )
        confidence = 0.9 if raw >= 6 else 0.75 if raw >= 4 else 0.6
        reasoning = (
            f"{raw} indicators ({', '.join(evidence[:3])})" if evidence
            else f"{raw} indicators"
        )
        return PhaseResult(phase=phase, confidence=confidence, reasoning=reasoning)

    def _inherit(self, why: str) -> PhaseResult:
        if self._current is None:
            # Nothing has opened yet - genuinely unclassified.
            return PhaseResult(UNCERTAIN, 0.0, "no phase established yet")
        return PhaseResult(self._current, 0.4, f"{why} ({self._current})", inherited=True)


def phase_distribution(feedback_history: Sequence[Any]) -> List[Dict[str, Any]]:
    """Percentage of the session spent in each GROW phase.

    Only the four canonical phases are returned, and each percentage is a
    share of the *classified* turns, so the values always sum to ~100%.

    The previous implementation excluded "Uncertain" from the denominator
    but still emitted it as a row, so an "Uncertain" row could read 120%
    on its own and the column could total well over 100%. Unclassified
    turns are now reported separately via :func:`phase_coverage`.
    """
    counts: Dict[str, int] = {p: 0 for p in CANONICAL_PHASES}
    confidences: Dict[str, List[float]] = {p: [] for p in CANONICAL_PHASES}

    for feedback in feedback_history:
        phase = feedback.grow_phase.phase
        if phase in counts:
            counts[phase] += 1
            confidences[phase].append(feedback.grow_phase.confidence)

    classified = sum(counts.values())
    if not classified:
        return []

    rows = [
        {
            "phase": phase,
            "turns": counts[phase],
            "percentage": counts[phase] / classified * 100.0,
            "avg_confidence": (
                sum(confidences[phase]) / len(confidences[phase])
                if confidences[phase] else 0.0
            ),
        }
        for phase in CANONICAL_PHASES
        if counts[phase] > 0
    ]
    rows.sort(key=lambda r: r["percentage"], reverse=True)
    return rows


def phase_coverage(feedback_history: Sequence[Any]) -> Dict[str, Any]:
    """How much of the session could be attributed to a GROW phase at all."""
    total = len(feedback_history)
    if not total:
        return {}
    classified = sum(
        1 for f in feedback_history if f.grow_phase.phase in CANONICAL_PHASES
    )
    observed = [
        p for p in CANONICAL_PHASES
        if any(f.grow_phase.phase == p for f in feedback_history)
    ]
    return {
        "total_turns": total,
        "classified_turns": classified,
        "unclassified_turns": total - classified,
        "coverage_pct": classified / total * 100.0,
        "phases_observed": observed,
        "phases_missing": [p for p in CANONICAL_PHASES if p not in observed],
    }


def phase_sequence(feedback_history: Sequence[Any]) -> List[str]:
    """Collapsed ordering of phases as the session moved through them."""
    seq: List[str] = []
    for feedback in feedback_history:
        phase = feedback.grow_phase.phase
        if phase in CANONICAL_PHASES and (not seq or seq[-1] != phase):
            seq.append(phase)
    return seq
