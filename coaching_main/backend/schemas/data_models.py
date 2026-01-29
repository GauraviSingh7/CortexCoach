from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from pydantic import BaseModel


@dataclass
class AudioChunk:
    """Represents a processed audio chunk with metadata"""
    timestamp: float
    duration: float
    speaker: str  # "coach" or "coachee"
    transcript: str
    audio_data: Optional[bytes] = None


@dataclass
class ModelInferences:
    """Container for all AI model inference results"""
    emotion: Dict[str, float]  # {emotion: confidence}
    interest_level: float
    sarcasm_score: float
    vak_style: Dict[str, float]  # {visual: 0.3, auditory: 0.4, kinesthetic: 0.3}
    digression_score: float


@dataclass
class GROWPhase:
    """GROW model phase classification"""
    phase: str  # "Goal", "Reality", "Options", "Way Forward"
    confidence: float
    reasoning: str


@dataclass
class RealTimeFeedback:
    """Real-time feedback structure"""
    timestamp: float
    speaker: str
    grow_phase: GROWPhase
    emotion_trend: Dict[str, float]
    engagement_score: float
    coaching_quality: Dict[str, float]   # made tighter
    suggestions: List[str]


class SessionReport(BaseModel):
    """Final session assessment report"""
    session_id: str
    duration_minutes: float
    participants: Dict[str, Dict[str, float]]  # coach/coachee engagement metrics
    grow_phases: List[Dict]
    emotional_journey: Dict[str, List]
    learning_style_analysis: Dict[str, float]
    key_insights: List[str]
    coaching_effectiveness: Dict[str, float]
    recommendations: List[str]
    transcript_summary: str