"""Key insights and recommendations derived from a :class:`SessionAnalysis`."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:  # pragma: no cover
    from backend.reporting.local_analyzer import SessionAnalysis


def build_insights(analysis: "SessionAnalysis") -> List[str]:
    """Narrative observations about how the session went."""
    insights: List[str] = []

    questions = analysis.questions
    if questions.get("total"):
        insights.append(
            f"Coach asked {questions['total']} questions, "
            f"{questions['ratio']:.0%} open-ended, including "
            f"{questions['powerful']} powerful questions that deepened exploration."
        )

    if analysis.grow_sequence:
        insights.append(
            "Session moved through: " + " -> ".join(analysis.grow_sequence) + "."
        )

    missing = analysis.grow_coverage.get("phases_missing") or []
    if missing:
        insights.append(
            "GROW phases not reached: " + ", ".join(missing) + "."
        )

    if analysis.themes:
        insights.append("Main coaching themes: " + ", ".join(analysis.themes) + ".")

    engagement = analysis.engagement
    if engagement:
        insights.append(
            f"Coachee engagement {engagement.get('trend', 'unknown')} through the "
            f"session (avg {engagement.get('average', 0.0):.2f}); "
            f"{engagement.get('low_points', 0)} low-engagement moments."
        )

    listening = analysis.listening
    if listening.get("listening_indicators"):
        moves = ", ".join(sorted(listening.get("by_move", {}))) or "listening"
        insights.append(
            f"Coach used active listening on "
            f"{listening['listening_indicators']} of {analysis.coach_turns} turns "
            f"({listening['frequency']:.0%}), via {moves}."
        )

    return insights


def build_recommendations(analysis: "SessionAnalysis") -> List[str]:
    """Actionable suggestions, strengths first when they are earned."""
    recommendations: List[str] = []
    questions = analysis.questions
    listening = analysis.listening
    engagement = analysis.engagement

    if questions.get("ratio", 0.0) < 0.6:
        recommendations.append(
            "Increase open-ended questions. Try 'What else?', "
            "'How do you feel about that?', 'What would success look like?'"
        )

    if questions.get("powerful", 0) < 3:
        recommendations.append(
            "Use more powerful questions to deepen thinking: "
            "'What if that weren't true?', 'What's really important here?', "
            "'What are you not saying?'"
        )

    if listening.get("frequency", 0.0) < 0.3:
        recommendations.append(
            "Reflect back more of what you hear: 'So what I'm hearing is...', "
            "'It sounds like...', 'Let me check I understand...'"
        )

    if engagement.get("low_points", 0) > 2:
        recommendations.append(
            f"Engagement dropped {engagement['low_points']} times. When energy "
            "dips, check in: 'What's coming up for you right now?'"
        )

    missing = analysis.grow_coverage.get("phases_missing") or []
    if missing:
        recommendations.append(
            "Session did not reach " + ", ".join(missing) +
            ". Leave time to convert insight into concrete commitments."
        )

    if "challenges" in analysis.themes:
        recommendations.append(
            "When exploring challenges, balance problem analysis with "
            "solution focus - move from 'What's wrong?' to 'What's possible?'"
        )

    if questions.get("powerful", 0) >= 3:
        recommendations.insert(
            0,
            f"Strong use of powerful questions ({questions['powerful']} detected) - "
            "keep building on this.",
        )

    if listening.get("frequency", 0.0) >= 0.5:
        recommendations.insert(
            0,
            f"Active listening was consistent ({listening['frequency']:.0%} of "
            "coach turns) - a clear strength in this session.",
        )

    return recommendations[:5]
