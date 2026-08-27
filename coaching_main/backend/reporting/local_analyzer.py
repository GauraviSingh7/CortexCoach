"""
Local (no-API) session report generation.

Assembles the report from the per-turn data the pipeline collected. All
metric computation is delegated to :mod:`backend.analysis`, so the same
code path produces the numbers whether or not Gemini is available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from backend.analysis import digression as digression_analysis
from backend.analysis import emotion as emotion_analysis
from backend.analysis import grow as grow_analysis
from backend.analysis import quality as quality_analysis
from backend.analysis import sarcasm as sarcasm_analysis
from backend.analysis import themes as theme_analysis
from backend.analysis import vak as vak_analysis
from backend.reporting.insights import build_insights, build_recommendations
from backend.reporting.summaries import build_summary

logger = logging.getLogger(__name__)


@dataclass
class SessionAnalysis:
    """Intermediate analysis shared by insights, recommendations and summary."""

    questions: Dict[str, Any] = field(default_factory=dict)
    listening: Dict[str, Any] = field(default_factory=dict)
    themes: List[str] = field(default_factory=list)
    theme_breakdown: Dict[str, int] = field(default_factory=dict)
    engagement: Dict[str, Any] = field(default_factory=dict)
    coach_turns: int = 0
    coachee_turns: int = 0
    grow_sequence: List[str] = field(default_factory=list)
    grow_coverage: Dict[str, Any] = field(default_factory=dict)


class LocalAnalyzer:
    """Builds a full :class:`SessionReport` payload from session data."""

    def generate_comprehensive_report(
        self, session_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        chunks = session_data.get("chunks", []) or []
        feedback_history = session_data.get("feedback_history", []) or []

        if not chunks:
            return self.empty_report(session_data)

        analysis = self.analyze(chunks, feedback_history)
        sarcasm_records = session_data.get("sarcasm_detections", []) or []
        digression_records = session_data.get("digression_scores", []) or []
        vak_scores = session_data.get("vak_scores", []) or []

        return {
            "session_id": session_data.get("session_id", "unknown"),
            "duration_minutes": session_data.get("duration", 0.0),
            "participants": self.participants(chunks, feedback_history),
            "grow_phases": grow_analysis.phase_distribution(feedback_history),
            "grow_coverage": analysis.grow_coverage,
            "emotional_journey": emotion_analysis.emotional_journey(feedback_history),
            "learning_style_analysis": vak_analysis.average_styles(vak_scores),
            "key_insights": build_insights(analysis),
            "coaching_effectiveness": quality_analysis.effectiveness(
                analysis.questions, analysis.listening,
                analysis.engagement.get("average", 0.0),
            ),
            "recommendations": build_recommendations(analysis),
            "transcript_summary": build_summary(analysis, chunks),
            "sarcasm_summary": sarcasm_analysis.summarize(sarcasm_records),
            "digression_summary": digression_analysis.summarize(digression_records),
            "model_status": session_data.get("model_status", {}),
            "analysis_sources": session_data.get("analysis_sources", {}),
        }

    # -- analysis ----------------------------------------------------------

    def analyze(
        self, chunks: Sequence[Any], feedback_history: Sequence[Any]
    ) -> SessionAnalysis:
        coach_turns = [c for c in chunks if c.speaker == "coach"]
        coachee_turns = [c for c in chunks if c.speaker == "coachee"]

        return SessionAnalysis(
            questions=quality_analysis.analyze_questions(coach_turns),
            listening=quality_analysis.analyze_listening(coach_turns),
            themes=theme_analysis.extract_themes(coachee_turns),
            theme_breakdown=theme_analysis.theme_breakdown(coachee_turns),
            engagement=self.engagement_patterns(feedback_history),
            coach_turns=len(coach_turns),
            coachee_turns=len(coachee_turns),
            grow_sequence=grow_analysis.phase_sequence(feedback_history),
            grow_coverage=grow_analysis.phase_coverage(feedback_history),
        )

    @staticmethod
    def engagement_patterns(feedback_history: Sequence[Any]) -> Dict[str, Any]:
        """Average engagement, direction of travel and low points."""
        if not feedback_history:
            return {"average": 0.0, "trend": "unknown", "low_points": 0}

        scores = [f.engagement_score for f in feedback_history]
        midpoint = len(scores) // 2
        first = scores[:midpoint] or scores
        second = scores[midpoint:] or scores
        avg_first = sum(first) / len(first)
        avg_second = sum(second) / len(second)

        if avg_second > avg_first + 0.1:
            trend = "increasing"
        elif avg_second < avg_first - 0.1:
            trend = "decreasing"
        else:
            trend = "stable"

        return {
            "average": sum(scores) / len(scores),
            "trend": trend,
            "low_points": sum(1 for s in scores if s < 0.4),
        }

    @staticmethod
    def participants(
        chunks: Sequence[Any], feedback_history: Sequence[Any]
    ) -> Dict[str, Dict[str, float]]:
        """Per-role turn counts, engagement and verbosity."""
        result: Dict[str, Dict[str, float]] = {}
        for role in ("coach", "coachee"):
            turns = [c for c in chunks if c.speaker == role]
            engagement = [
                f.engagement_score for f in feedback_history if f.speaker == role
            ]
            result[role] = {
                "total_turns": len(turns),
                "engagement_avg": (
                    sum(engagement) / len(engagement) if engagement else 0.0
                ),
                "avg_words": (
                    sum(len(c.transcript.split()) for c in turns) / len(turns)
                    if turns else 0.0
                ),
            }
        return result

    # -- fallback ----------------------------------------------------------

    @staticmethod
    def empty_report(session_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "session_id": session_data.get("session_id", "unknown"),
            "duration_minutes": session_data.get("duration", 0.0),
            "participants": {},
            "grow_phases": [],
            "grow_coverage": {},
            "emotional_journey": {"coach": [], "coachee": []},
            "learning_style_analysis": {},
            "key_insights": ["No conversation data was captured for this session."],
            "coaching_effectiveness": {},
            "recommendations": ["Record a coaching session to receive insights."],
            "transcript_summary": "No data recorded.",
            "sarcasm_summary": {},
            "digression_summary": {},
            "model_status": session_data.get("model_status", {}),
            "analysis_sources": session_data.get("analysis_sources", {}),
        }
