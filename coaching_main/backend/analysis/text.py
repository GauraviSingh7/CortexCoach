"""
Shared text utilities for the analysis layer.

Everything here is pure: no I/O, no model loading, no logging side effects.
The single most important export is :func:`has_term`, which does
word-boundary matching. The previous code used plain ``substring in text``
checks, which produced silent false positives that skewed every downstream
metric ("now" matching inside "know", "do" inside "don't", "team" inside
"team-lead").
"""

from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
from typing import Iterable, List, Sequence

# Words with no topical content. Used for keyword extraction and the
# digression overlap measure.
STOP_WORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "is", "was", "are", "were", "been", "be", "have", "has",
    "had", "do", "does", "did", "will", "would", "should", "could", "may",
    "might", "can", "this", "that", "these", "those", "i", "you", "he",
    "she", "it", "we", "they", "my", "your", "what", "how", "why", "when",
    "there", "their", "them", "then", "than", "about", "just", "like",
    "really", "actually", "yeah", "okay", "right", "well", "some", "which",
    "because", "from", "into", "over", "such", "very", "more", "most",
    "into", "also", "here", "were", "with",
})

_WORD_RE = re.compile(r"[a-z0-9']+")


@lru_cache(maxsize=4096)
def _term_pattern(term: str) -> re.Pattern:
    """Compile a word-boundary pattern for a single- or multi-word term.

    Internal whitespace also matches a hyphen, so "team lead" matches
    "team-lead".

    Apostrophes are optional. Speech-to-text output routinely drops them,
    so "dont", "lets" and "whats" must still match terms written as
    "don't", "let's" and "what's" - otherwise every apostrophised phrase
    in the keyword lists silently fails on real transcripts.
    """
    parts = [re.escape(p) for p in term.lower().split()]
    body = r"[\s\-]+".join(parts)
    # re.escape leaves "'" alone on current Python; handle the escaped form
    # too so this keeps working if that ever changes.
    body = body.replace("\\'", "'").replace("'", "'?")
    return re.compile(rf"(?<!\w){body}(?!\w)")


def has_term(text: str, term: str) -> bool:
    """True when ``term`` appears in ``text`` as a whole word or phrase."""
    return bool(_term_pattern(term).search(text.lower()))


def has_any(text: str, terms: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(_term_pattern(t).search(lowered) for t in terms)


def matched_terms(text: str, terms: Iterable[str]) -> List[str]:
    """Return the subset of ``terms`` present in ``text`` (for explanations)."""
    lowered = text.lower()
    return [t for t in terms if _term_pattern(t).search(lowered)]


def count_terms(text: str, terms: Iterable[str]) -> int:
    """Total number of occurrences of all ``terms`` in ``text``."""
    lowered = text.lower()
    return sum(len(_term_pattern(t).findall(lowered)) for t in terms)


def words(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def content_words(text: str, min_length: int = 3) -> List[str]:
    """Topical words: long enough, not a stop word, not a bare number.

    The threshold is deliberately low (4+ characters). At 5+ the most
    topical nouns in a coaching session - work, role, team, goal - were
    all discarded, which wrecked any topic-overlap measure built on top.
    """
    return [
        w for w in words(text)
        if len(w) > min_length and w not in STOP_WORDS and not w.isdigit()
    ]


def top_keywords(texts: Sequence[str], limit: int = 5) -> List[str]:
    counts = Counter()
    for t in texts:
        counts.update(content_words(t))
    return [kw for kw, _ in counts.most_common(limit)]


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    """Symmetric set similarity. Only meaningful for similarly sized sets."""
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0.0


def overlap_ratio(subject: Iterable[str], reference: Iterable[str]) -> float:
    """Share of ``subject`` that also appears in ``reference``.

    Containment, not Jaccard. When one set is much larger than the other
    - as when comparing a single utterance against a whole topic window -
    Jaccard is dominated by the larger set's size and collapses toward
    zero for on-topic text, which made every turn look like a digression.
    """
    ss, sr = set(subject), set(reference)
    if not ss:
        return 0.0
    return len(ss & sr) / len(ss)


def is_question(text: str) -> bool:
    return "?" in text


def word_count(text: str) -> int:
    return len(text.split())


def normalize(scores: dict) -> dict:
    """Scale a dict of non-negative scores so the values sum to 1.0.

    Returns ``{}`` for an all-zero or empty input rather than inventing a
    uniform distribution — callers must be able to tell "no signal" apart
    from "genuinely balanced".
    """
    total = sum(scores.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in scores.items()}
