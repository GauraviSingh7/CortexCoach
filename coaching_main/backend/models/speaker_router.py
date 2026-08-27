"""
Coach / coachee role assignment.

The previous implementation discarded the diarizer's speaker label and
re-decided the role for every utterance independently, using a short
keyword list plus ``if "?" in text: coach_score += 1`` and a tie-break of
``"coach" if speaker_id == "A" else "coachee"``.

Two consequences, both observed on a 40-turn test transcript:

* the same physical speaker flipped roles mid-session; and
* because which label the diarizer assigns to whom is arbitrary, whenever
  the coachee happened to be labelled "A" nearly every tie resolved to
  coach - producing a 33/7 split on a conversation that was 20/20.

This module instead treats the diarization label as ground truth for
*who is speaking*, aggregates role evidence per speaker across the whole
session, and assigns each role exactly once.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from backend.analysis.text import has_any, is_question, matched_terms, word_count

logger = logging.getLogger(__name__)

COACH = "coach"
COACHEE = "coachee"

# Language characteristic of the person facilitating.
_COACH_MARKERS = (
    "what would you", "how do you feel", "tell me about", "what's stopping",
    "what if you", "have you considered", "what are your options",
    "what's your goal", "let's explore", "what do you think", "describe",
    "what's important", "on a scale", "say more", "what else",
    "keep going", "let's recap", "what comes up", "does that land",
    "i hear", "sounds like", "let's hold", "let's park", "good to see you",
)

# Language characteristic of the person being coached.
_COACHEE_MARKERS = (
    "i don't know", "i'm not sure", "i worry", "i'm worried", "i feel",
    "i guess", "my problem", "i want to", "i need", "i'm confused",
    "i'm stuck", "help me", "what should i", "i'm struggling",
    "my goal", "my issue", "my challenge", "my manager", "my team",
    "i could", "i'll", "i assumed", "i think i",
)

#: Final turns required from a speaker before its role is locked.
_MIN_TURNS_TO_LOCK = 3


@dataclass
class _SpeakerEvidence:
    """Running evidence for one diarized speaker."""

    turns: int = 0
    questions: int = 0
    coach_hits: int = 0
    coachee_hits: int = 0
    total_words: int = 0

    def observe(self, text: str) -> None:
        self.turns += 1
        self.total_words += word_count(text)
        if is_question(text):
            self.questions += 1
        self.coach_hits += len(matched_terms(text, _COACH_MARKERS))
        self.coachee_hits += len(matched_terms(text, _COACHEE_MARKERS))

    @property
    def coach_score(self) -> float:
        """Higher means more coach-like. Normalised per turn."""
        if not self.turns:
            return 0.0
        question_rate = self.questions / self.turns
        coach_rate = self.coach_hits / self.turns
        coachee_rate = self.coachee_hits / self.turns
        avg_words = self.total_words / self.turns
        # Coaches ask more and say less; coachees disclose more and say more.
        brevity = 1.0 if avg_words < 25 else 0.0
        return (
            question_rate * 2.0
            + coach_rate * 1.5
            - coachee_rate * 1.5
            + brevity * 0.5
        )


class SpeakerRouter:
    """Maps diarization speaker ids to coach/coachee roles, once."""

    def __init__(self, coach_speaker_id: Optional[str] = None):
        self.pinned = self._normalize(coach_speaker_id) if coach_speaker_id else None
        self._evidence: Dict[str, _SpeakerEvidence] = {}
        self._roles: Dict[str, str] = {}
        self._locked = False

    # -- batch (file mode) -------------------------------------------------

    def assign_batch(self, utterances: Sequence) -> Dict[str, str]:
        """Assign roles using every utterance at once.

        File mode has the whole transcript up front, so there is no reason
        to guess incrementally. Returns the speaker-id to role mapping.
        """
        for utterance in utterances:
            speaker_id = self._normalize(getattr(utterance, "speaker", None))
            text = getattr(utterance, "text", None) or getattr(utterance, "transcript", "")
            if speaker_id is None:
                continue
            self._evidence.setdefault(speaker_id, _SpeakerEvidence()).observe(text or "")

        self._decide()
        self._locked = True
        logger.info("Speaker roles assigned: %s", self._roles or "none")
        return dict(self._roles)

    # -- incremental (live mode) -------------------------------------------

    def observe(self, speaker_id: Optional[str], text: str) -> None:
        """Record evidence for a final turn."""
        speaker_id = self._normalize(speaker_id)
        if speaker_id is None:
            return
        self._evidence.setdefault(speaker_id, _SpeakerEvidence()).observe(text or "")
        if not self._locked:
            self._decide()
            if self._ready_to_lock():
                self._locked = True
                logger.info("Speaker roles locked: %s", self._roles)

    def role_for(self, speaker_id: Optional[str], text: str = "") -> str:
        """Role for a diarized speaker id.

        Once roles are decided the mapping is stable - the same speaker id
        always returns the same role, regardless of what this particular
        utterance happens to contain.
        """
        speaker_id = self._normalize(speaker_id)

        if self.pinned is not None and speaker_id is not None:
            return COACH if speaker_id == self.pinned else COACHEE

        if speaker_id is not None and speaker_id in self._roles:
            return self._roles[speaker_id]

        # No diarization id at all (some streaming events omit it): fall back
        # to per-utterance evidence, which is all that is available.
        return self._role_from_text(text)

    # -- internals ---------------------------------------------------------

    def _ready_to_lock(self) -> bool:
        seen = [e for e in self._evidence.values() if e.turns >= _MIN_TURNS_TO_LOCK]
        return len(self._evidence) >= 2 and len(seen) >= 2

    def _decide(self) -> None:
        if self.pinned is not None:
            self._roles = {
                sid: (COACH if sid == self.pinned else COACHEE)
                for sid in self._evidence
            }
            return

        if not self._evidence:
            return

        ranked: List[Tuple[str, float]] = sorted(
            ((sid, ev.coach_score) for sid, ev in self._evidence.items()),
            key=lambda kv: kv[1],
            reverse=True,
        )

        if len(ranked) == 1:
            # Only one speaker so far - decide from its own evidence rather
            # than defaulting to coach.
            sid, score = ranked[0]
            self._roles = {sid: COACH if score >= 0.5 else COACHEE}
            return

        # Highest coach-score speaker is the coach; everyone else coachee.
        self._roles = {ranked[0][0]: COACH}
        for sid, _ in ranked[1:]:
            self._roles[sid] = COACHEE

    @staticmethod
    def _role_from_text(text: str) -> str:
        if has_any(text, _COACH_MARKERS):
            return COACH
        if has_any(text, _COACHEE_MARKERS):
            return COACHEE
        return COACH if is_question(text) else COACHEE

    @staticmethod
    def _normalize(speaker_id) -> Optional[str]:
        """Reduce "A" / "SPEAKER_A" / "speaker_a" to a single form."""
        if speaker_id is None:
            return None
        text = str(speaker_id).strip().upper()
        if text.startswith("SPEAKER_"):
            text = text[len("SPEAKER_"):]
        return text or None

    @property
    def roles(self) -> Dict[str, str]:
        return dict(self._roles)

    @property
    def is_locked(self) -> bool:
        return self._locked
