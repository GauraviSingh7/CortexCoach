"""Narrative session summary: what was discussed and how it was handled."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Sequence

from backend.analysis.text import has_any

if TYPE_CHECKING:  # pragma: no cover
    from backend.reporting.local_analyzer import SessionAnalysis

_ISSUE_MARKERS: Dict[str, Sequence[str]] = {
    "goal": ("want to", "goal", "achieve", "become", "aspire", "figure out"),
    "emotion": ("feel", "sad", "happy", "worried", "anxious", "excited",
                "resentful", "relief", "heavy", "exposed"),
    "challenge": ("difficult", "hard", "struggling", "problem", "stuck",
                  "passed over", "blur"),
    "obstacle": ("can't", "don't know", "unable", "impossible", "guesswork"),
}

_TECHNIQUES: Dict[str, Sequence[str]] = {
    "reframing": ("what if", "another way", "there may be a difference",
                  "that's the shift"),
    "acknowledgment": ("i hear", "i understand", "i appreciate",
                       "that's a really", "that's a lot of"),
    "challenging": ("what stops", "what if you", "don't filter",
                    "what's underneath"),
    "goal clarification": ("what do you want", "your goal", "look like for you",
                           "walk away with"),
    "summarising": ("let's recap", "to recap", "so of everything"),
}


def build_summary(analysis: "SessionAnalysis", chunks: Sequence[Any]) -> str:
    """One-paragraph account of the session."""
    parts: List[str] = []

    themes = ", ".join(analysis.themes) if analysis.themes else "various topics"
    parts.append(
        f"The coach explored {themes} with the coachee across "
        f"{analysis.coach_turns} coach turns and "
        f"{analysis.coachee_turns} coachee responses."
    )

    if analysis.grow_sequence:
        parts.append("The session moved " + " -> ".join(analysis.grow_sequence) + ".")

    issues = _describe_issues(chunks)
    if issues:
        parts.append(issues)

    approach = _describe_approach(chunks, analysis)
    if approach:
        parts.append(approach)

    questions = analysis.questions
    if questions.get("total"):
        parts.append(
            f"Questioning was {questions['ratio']:.0%} open-ended with "
            f"{questions['powerful']} powerful questions."
        )

    engagement = analysis.engagement
    if engagement:
        parts.append(
            f"Coachee engagement {engagement.get('trend', 'unknown')} throughout, "
            f"averaging {engagement.get('average', 0.0):.2f} with "
            f"{engagement.get('low_points', 0)} moments needing re-engagement."
        )

    return " ".join(parts)


def _describe_issues(chunks: Sequence[Any]) -> str:
    """Quote the coachee's own framing of goal, feeling and obstacle."""
    coachee = [c for c in chunks if c.speaker == "coachee"]
    found: Dict[str, str] = {}

    for chunk in coachee:
        for kind, markers in _ISSUE_MARKERS.items():
            if kind in found:
                continue
            for sentence in chunk.transcript.split("."):
                if sentence.strip() and has_any(sentence, markers):
                    found[kind] = sentence.strip()[:90]
                    break

    described = [
        f"expressed goal: '{found['goal']}'" if "goal" in found else "",
        f"shared feeling: '{found['emotion']}'" if "emotion" in found else "",
        f"named challenge: '{found['challenge']}'" if "challenge" in found else "",
    ]
    described = [d for d in described if d]
    return "Coachee " + "; ".join(described[:2]) + "." if described else ""


def _describe_approach(chunks: Sequence[Any], analysis: "SessionAnalysis") -> str:
    """Name the coaching techniques actually observed."""
    coach = [c for c in chunks if c.speaker == "coach"]
    approaches: List[str] = []

    if analysis.questions.get("powerful", 0) >= 2:
        approaches.append("used powerful questions")
    if analysis.listening.get("listening_indicators", 0) >= 2:
        approaches.append("demonstrated active listening")

    observed = {
        name for name, markers in _TECHNIQUES.items()
        if any(has_any(c.transcript, markers) for c in coach)
    }
    if observed:
        approaches.append("employed " + ", ".join(sorted(observed)[:3]))

    return "Coach " + "; ".join(approaches[:3]) + "." if approaches else ""
