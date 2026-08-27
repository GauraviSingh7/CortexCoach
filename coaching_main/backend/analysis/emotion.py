"""
Text-based emotion estimation.

This is a lexicon heuristic and is labelled as such everywhere it surfaces.
The shipped ``models/emotion_recognition/model_weight.pth`` is a graph
convolution network over 40-dimensional *audio* features with 6 output
classes; it needs torch-geometric plus a real audio feature pipeline, and
cannot score text at all. See :mod:`backend.models.model_status`.

The function returns ``{}`` when it finds no emotional signal. Callers must
render that as "not available" rather than substituting neutral - reporting
a confident ``{"neutral": 1.0}`` for every turn was the single most
misleading output in the previous build.
"""

from __future__ import annotations

import re
from typing import Dict, Sequence

from backend.analysis.text import normalize

#: Emotion vocabulary, weighted by how strongly each term implies the label.
_LEXICON: Dict[str, Dict[str, float]] = {
    "happy": {
        "happy": 2.0, "glad": 1.5, "pleased": 1.5, "joy": 2.0, "delighted": 2.0,
        "wonderful": 1.5, "great": 1.0, "love": 1.5, "enjoy": 1.5, "grateful": 1.5,
        "thank you": 1.0, "lighter": 1.5,
    },
    "excited": {
        "excited": 2.0, "excites": 2.0, "exciting": 2.0, "thrilled": 2.0,
        "eager": 1.5, "energised": 1.5, "energized": 1.5, "pumped": 1.5,
        "can't wait": 2.0, "genuinely": 0.5, "unreal": 1.0,
    },
    "hopeful": {
        "hope": 1.5, "hopeful": 2.0, "optimistic": 2.0, "looking forward": 1.5,
        "workable": 1.5, "possible": 1.0, "i'll do it": 1.0, "committed": 1.0,
    },
    "relieved": {
        "relief": 2.5, "relieved": 2.5, "lighter": 2.0, "weight off": 2.0,
        "finally": 1.0, "at last": 1.0, "makes sense now": 1.5,
    },
    "sad": {
        "sad": 2.0, "hurt": 1.8, "hurts": 1.8, "stings": 1.8, "down": 1.0,
        "disappointed": 2.0, "unhappy": 2.0, "miserable": 2.0, "heavy": 1.5,
        "lonely": 2.0, "defeated": 2.0,
    },
    "frustrated": {
        "frustrated": 2.5, "frustration": 2.5, "annoyed": 2.0, "annoying": 2.0,
        "angry": 2.0, "mad": 1.5, "irritated": 2.0, "resentful": 2.5,
        "fed up": 2.0, "sick of": 2.0, "passed over": 1.5, "nobody": 0.8,
        "hilarious": 0.8,
    },
    "anxious": {
        "worried": 2.0, "worry": 2.0, "anxious": 2.5, "nervous": 2.0,
        "scared": 2.0, "afraid": 2.0, "uneasy": 1.5, "exposed": 1.8,
        "vulnerable": 1.8, "lose my nerve": 2.0, "not sure": 1.0,
        "stuck": 1.5, "guessing": 1.0,
    },
    "conflicted": {
        "complicated": 1.5, "torn": 2.0, "mixed": 1.5, "but also": 1.2,
        "weirdly": 1.2, "kind of": 0.6, "pretending": 1.5,
    },
}

# Clause splitters - negation is scoped to the clause it appears in.
_CLAUSE_SPLIT = re.compile(r"[.!?;,]| but | and | because | although | though ")
_NEGATORS = ("not", "never", "no", "dont", "don't", "didnt", "didn't",
             "isnt", "isn't", "wasnt", "wasn't", "cant", "can't", "wouldn't")

#: Minimum total weight before we claim any emotional reading at all.
_MIN_EVIDENCE = 1.5


def _clause_is_negated(clause: str) -> bool:
    tokens = re.findall(r"[a-z']+", clause.lower())
    return any(tok in _NEGATORS for tok in tokens)


def analyze_text_emotion(text: str) -> Dict[str, float]:
    """Estimate an emotion distribution for one utterance.

    Returns a normalised distribution, or ``{}`` when there is no signal.
    Terms inside a negated clause are skipped rather than inverted - "I'm
    not happy" should not read as happiness, and guessing the opposite
    label would be worse than abstaining.
    """
    if not text or not text.strip():
        return {}

    lowered = text.lower()
    scores: Dict[str, float] = {label: 0.0 for label in _LEXICON}

    for clause in _CLAUSE_SPLIT.split(lowered):
        clause = clause.strip()
        if not clause:
            continue
        negated = _clause_is_negated(clause)
        for label, terms in _LEXICON.items():
            for term, weight in terms.items():
                if term in clause:
                    if negated:
                        continue
                    scores[label] += weight

    if sum(scores.values()) < _MIN_EVIDENCE:
        return {}

    distribution = normalize({k: v for k, v in scores.items() if v > 0})
    # Keep the report readable: drop negligible tails.
    trimmed = {k: v for k, v in distribution.items() if v >= 0.05}
    return normalize(trimmed) if trimmed else distribution


def dominant_emotion(distribution: Dict[str, float]) -> str:
    """Label of the highest-scoring emotion, or "unknown" for no signal."""
    if not distribution:
        return "unknown"
    return max(distribution.items(), key=lambda kv: kv[1])[0]


def emotional_journey(feedback_history: Sequence) -> Dict[str, list]:
    """Per-speaker timeline of dominant emotions.

    Turns with no emotional signal are skipped entirely rather than being
    plotted as neutral.
    """
    journey: Dict[str, list] = {"coach": [], "coachee": []}
    for feedback in feedback_history:
        trend = getattr(feedback, "emotion_trend", None)
        if not trend:
            continue
        label, confidence = max(trend.items(), key=lambda kv: kv[1])
        bucket = journey.get(feedback.speaker)
        if bucket is None:
            continue
        bucket.append({
            "timestamp": feedback.timestamp,
            "emotion": label,
            "confidence": confidence,
        })
    return journey
