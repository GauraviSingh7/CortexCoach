"""
Theme extraction from coachee dialogue.

The previous keyword sets keyed "relationships" on ``team``, ``manager`` and
``people``, so a session entirely about career visibility reported
"Relationships" as a main theme - the coachee said "team-lead" (a job title)
and "my manager" (the person granting the promotion) throughout.

Two changes fix that class of false positive:

* relational themes now require genuinely relational vocabulary; an
  authority figure mentioned in a career context is career, not relationship;
* a theme must clear a minimum share of coachee turns before it is reported,
  so a single incidental mention no longer becomes a headline theme.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from backend.analysis.text import has_any

_THEMES: Dict[str, Sequence[str]] = {
    "career": (
        "job", "career", "promotion", "promoted", "team lead", "team-lead",
        "role", "position", "company", "review", "raise", "cycle",
        "roadmap", "manager", "leadership", "analytics pod",
    ),
    "visibility": (
        "visible", "invisible", "visibility", "notice", "noticed", "seen",
        "recognition", "credit", "announce", "showcase", "present",
        "showing", "show up", "hang back",
    ),
    "goals": (
        "goal", "achieve", "aspire", "target", "outcome", "want to",
        "walk away with", "aiming",
    ),
    "challenges": (
        "problem", "difficult", "struggle", "struggling", "challenge",
        "issue", "stuck", "blocked", "hard", "blur",
    ),
    "growth": (
        "learn", "grow", "develop", "improve", "skill", "mentor",
        "feedback", "self-awareness", "practice",
    ),
    "relationships": (
        # Genuinely relational language only.
        "colleague", "colleagues", "coworker", "teammate", "teammates",
        "relationship", "relationships", "conflict", "peers", "get along",
        "rapport", "trust", "fell out", "tension with",
    ),
    "confidence": (
        "confident", "confidence", "doubt", "afraid", "nervous", "unsure",
        "imposter", "hesitant", "quiet", "not certain", "exposed",
        "hundred percent sure",
    ),
    "decision": (
        "decide", "decision", "decisions", "choice", "choices", "choose",
        "option", "options",
    ),
    "emotions": (
        "feel", "feeling", "resentful", "relief", "heavy", "hurts",
        "stings", "frustrated", "excited",
    ),
}

#: A theme must appear in at least this share of coachee turns...
_MIN_SHARE = 0.15

#: ...and in at least this many turns, whichever is greater.
_MIN_TURNS = 2


def extract_themes(coachee_turns: Sequence[Any], limit: int = 3) -> List[str]:
    """Main themes in the coachee's dialogue, most prominent first."""
    if not coachee_turns:
        return []

    counts: Dict[str, int] = {theme: 0 for theme in _THEMES}
    for turn in coachee_turns:
        text = getattr(turn, "transcript", "") or ""
        for theme, keywords in _THEMES.items():
            if has_any(text, keywords):
                counts[theme] += 1

    floor = max(_MIN_TURNS, int(len(coachee_turns) * _MIN_SHARE))
    ranked = sorted(
        ((theme, n) for theme, n in counts.items() if n >= floor),
        key=lambda kv: kv[1],
        reverse=True,
    )
    return [theme for theme, _ in ranked[:limit]]


def theme_breakdown(coachee_turns: Sequence[Any]) -> Dict[str, int]:
    """Raw per-theme turn counts, for diagnostics and report detail."""
    counts: Dict[str, int] = {}
    for turn in coachee_turns:
        text = getattr(turn, "transcript", "") or ""
        for theme, keywords in _THEMES.items():
            if has_any(text, keywords):
                counts[theme] = counts.get(theme, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))
