"""
Coaching quality metrics: questioning and active listening.

The previous listening detector recognised five literal phrases
(``i hear``, ``sounds like``, ``so you're saying``, ``if i understand``,
``let me check``). Genuine reflective moves - acknowledging what was said,
checking whether a reframe landed, inviting more, summarising at the close -
all scored zero, so a session full of good listening reported 0.0.

Listening is now scored across five recognised families of reflective move.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

from backend.analysis.text import has_any, is_question

# -- listening families -----------------------------------------------------

#: Explicitly naming what was heard.
_REFLECTING = (
    "i hear", "i'm hearing", "what i'm hearing", "sounds like", "it sounds",
    "so you're saying", "what you're saying", "what you just said",
    "you just said", "if i understand", "let me check", "you mentioned",
    "you said", "i noticed", "i'm noticing",
)

#: Naming the emotion or state behind the words.
_EMOTION_REFLECTING = (
    "i hear the", "you seem", "you sound", "that sounds", "how does that feel",
    "how do you feel", "what comes up", "feel to say", "sense some",
)

#: Confirming a reframe landed with the coachee.
_CHECKING = (
    "does that land", "does that resonate", "did i get that", "is that right",
    "does that make sense", "am i hearing", "does that fit", "land for you",
)

#: Explicit acknowledgement of what the coachee offered.
_ACKNOWLEDGING = (
    "that's a really", "that's a lot of", "that's a strong", "that's a good",
    "that's the shift", "i appreciate", "well done", "good catch",
    "interesting that", "thank you for", "that's honest",
)

#: Inviting the coachee to continue.
_INVITING = (
    "say more", "tell me more", "keep going", "go on", "what else",
    "and what else", "anything else", "what comes up",
)

#: Pulling the thread together.
_SUMMARISING = (
    "let's recap", "to recap", "so of everything", "to summarise",
    "to summarize", "what i've heard", "pulling that together",
    "so in the room", "let's hold that",
)

_LISTENING_FAMILIES = {
    "reflecting": (_REFLECTING, 0.9),
    "emotion_reflecting": (_EMOTION_REFLECTING, 0.85),
    "checking": (_CHECKING, 0.9),
    "summarising": (_SUMMARISING, 0.9),
    "acknowledging": (_ACKNOWLEDGING, 0.7),
    "inviting": (_INVITING, 0.6),
}

# -- questioning ------------------------------------------------------------

_OPEN_QUESTION_STEMS = ("what", "how", "why", "tell me", "describe", "walk me")

_POWERFUL_QUESTIONS = (
    "what else", "how do you feel", "what do you want", "what if",
    "what's stopping", "what would make", "what's underneath",
    "what comes up", "tell me more", "help me understand",
    "how will you know", "what would success", "on a scale",
    "what's really", "where else",
)


@dataclass
class TurnQuality:
    """Quality signals for a single coach turn."""

    questioning: float = 0.0
    listening: float = 0.0
    listening_moves: List[str] = field(default_factory=list)
    overall: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        if not self.overall and not self.questioning and not self.listening:
            return {}
        return {
            "overall": self.overall,
            "questioning": self.questioning,
            "listening": self.listening,
            "listening_moves": list(self.listening_moves),
        }


def score_listening(transcript: str) -> Tuple[float, List[str]]:
    """Best listening score for a turn, plus the families that matched."""
    best = 0.0
    moves: List[str] = []
    for name, (terms, weight) in _LISTENING_FAMILIES.items():
        if has_any(transcript, terms):
            moves.append(name)
            best = max(best, weight)
    return best, moves


def score_questioning(transcript: str) -> float:
    if not is_question(transcript):
        return 0.0
    if has_any(transcript, _POWERFUL_QUESTIONS):
        return 1.0
    if has_any(transcript, _OPEN_QUESTION_STEMS):
        return 0.85
    return 0.5  # closed question


def analyze_turn(transcript: str, speaker: str, phase_confidence: float) -> TurnQuality:
    """Quality of a single turn. Only coach turns carry coaching quality."""
    if speaker != "coach":
        return TurnQuality()

    questioning = score_questioning(transcript)
    listening, moves = score_listening(transcript)

    components = [s for s in (questioning, listening, phase_confidence) if s > 0]
    overall = sum(components) / len(components) if components else 0.0

    return TurnQuality(
        questioning=questioning,
        listening=listening,
        listening_moves=moves,
        overall=overall,
    )


def analyze_questions(coach_turns: Sequence[Any]) -> Dict[str, Any]:
    """Session-level questioning breakdown."""
    total = open_q = closed = powerful = 0
    for turn in coach_turns:
        text = getattr(turn, "transcript", "") or ""
        if not is_question(text):
            continue
        total += 1
        if has_any(text, _OPEN_QUESTION_STEMS):
            open_q += 1
        else:
            closed += 1
        if has_any(text, _POWERFUL_QUESTIONS):
            powerful += 1
    return {
        "total": total,
        "open": open_q,
        "closed": closed,
        "powerful": powerful,
        "ratio": open_q / total if total else 0.0,
    }


def analyze_listening(coach_turns: Sequence[Any]) -> Dict[str, Any]:
    """Session-level listening breakdown."""
    turns_with_listening = 0
    reflections = 0
    family_counts: Dict[str, int] = {}

    for turn in coach_turns:
        text = getattr(turn, "transcript", "") or ""
        score, moves = score_listening(text)
        if score > 0:
            turns_with_listening += 1
        for move in moves:
            family_counts[move] = family_counts.get(move, 0) + 1
        if has_any(text, _REFLECTING) or has_any(text, _SUMMARISING):
            reflections += 1

    total = len(coach_turns)
    return {
        "listening_indicators": turns_with_listening,
        "reflections": reflections,
        "by_move": family_counts,
        "frequency": turns_with_listening / total if total else 0.0,
    }


def effectiveness(questions: Dict[str, Any], listening: Dict[str, Any],
                  engagement_avg: float) -> Dict[str, float]:
    """Overall coaching effectiveness from the session-level breakdowns."""
    total_q = questions.get("total", 0)
    questioning_score = min(
        1.0,
        questions.get("ratio", 0.0)
        + (questions.get("powerful", 0) / total_q * 0.3 if total_q else 0.0),
    )
    listening_score = min(1.0, listening.get("frequency", 0.0) * 1.5)
    overall = (
        questioning_score * 0.4 + listening_score * 0.4 + engagement_avg * 0.2
    )
    return {
        "overall": overall,
        "questioning": questioning_score,
        "listening": listening_score,
        "engagement_management": engagement_avg,
    }
