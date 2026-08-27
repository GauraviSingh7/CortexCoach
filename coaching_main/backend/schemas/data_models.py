from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


@dataclass
class AudioChunk:
    """A processed audio chunk with its transcript and speaker attribution."""

    timestamp: float
    duration: float
    speaker: str  # "coach" or "coachee"
    transcript: str
    audio_data: Optional[bytes] = None
    speaker_id: Optional[str] = None  # Raw diarization id, e.g. "SPEAKER_A"
    is_final: bool = True             # False while the utterance is in progress


@dataclass
class ModelInferences:
    """Raw output of the trained models for one chunk.

    Every field is ``Optional`` and defaults to ``None``, which means "no
    trained model produced this value". That is deliberately distinct from
    a real prediction: the orchestrator substitutes a labelled heuristic
    and records which source was used. Defaulting these to neutral values
    is what previously made every turn report 100% neutral emotion and a
    constant 0.30 sarcasm score.
    """

    emotion: Optional[Dict[str, float]] = None
    interest_level: Optional[float] = None
    sarcasm_score: Optional[float] = None
    vak_style: Optional[Dict[str, float]] = None
    digression_score: Optional[float] = None

    # Provenance: "model", "heuristic" or None.
    emotion_source: Optional[str] = None
    sarcasm_source: Optional[str] = None
    vak_source: Optional[str] = None
    interest_source: Optional[str] = None


@dataclass
class GROWPhase:
    """GROW model phase classification for one turn."""

    phase: str  # "Goal", "Reality", "Options", "Way Forward" or "Uncertain"
    confidence: float
    reasoning: str
    inherited: bool = False  # True when continuing the phase already in progress


@dataclass
class RealTimeFeedback:
    """Per-turn feedback pushed to connected dashboards."""

    timestamp: float
    speaker: str
    grow_phase: GROWPhase
    emotion_trend: Dict[str, float]
    engagement_score: float
    coaching_quality: Dict[str, Any]  # scores plus listening_moves
    suggestions: List[str]
    emotion_source: Optional[str] = None


class SessionReport(BaseModel):
    """Final session assessment."""

    session_id: str
    duration_minutes: float
    participants: Dict[str, Dict[str, float]]
    grow_phases: List[Dict]
    emotional_journey: Dict[str, List]
    learning_style_analysis: Dict[str, float]
    key_insights: List[str]
    coaching_effectiveness: Dict[str, float]
    recommendations: List[str]
    transcript_summary: str
    sarcasm_summary: Dict[str, Any] = {}
    digression_summary: Dict[str, Any] = {}

    #: Turn-level GROW coverage: how much of the session could be attributed
    #: to a phase at all. Kept separate from ``grow_phases`` so percentages
    #: there always describe classified turns and sum to 100%.
    grow_coverage: Dict[str, Any] = {}

    #: Which models were trained vs. running on documented heuristics.
    model_status: Dict[str, Any] = {}

    #: Provenance per signal, e.g. {"emotion": "heuristic", ...}.
    analysis_sources: Dict[str, str] = {}
