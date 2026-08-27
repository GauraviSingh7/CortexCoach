"""
Regression tests for the issues raised in the AI Coach Observer review.

Each test names the finding it guards. Run with:

    python -m pytest tests/ -v
"""

from types import SimpleNamespace

import pytest

from backend.analysis import emotion, quality, themes
from backend.analysis.digression import DigressionAnalyzer
from backend.analysis.grow import (
    CANONICAL_PHASES, GROWClassifier, phase_coverage, phase_distribution,
)
from backend.analysis.sarcasm import SarcasmAnalyzer
from backend.analysis.text import has_term
from backend.analysis.vak import VAKAnalyzer
from backend.models.speaker_router import SpeakerRouter
from tests.fixtures import DIGRESSION_TURN, SARCASTIC_TURNS, TURNS, chunks, utterances


# -- word-boundary matching (root cause of several findings) ---------------

def test_substring_matches_do_not_fire():
    """"now" must not match inside "know", "do" not inside "don't"."""
    assert not has_term("I know the answer", "now")
    assert not has_term("I don't announce it", "do")
    assert not has_term("my team-lead role", "relationship")
    assert has_term("right now", "now")
    assert has_term("my team-lead role", "team lead")


# -- Finding 4: speaker identification --------------------------------------

@pytest.mark.parametrize("coach_label", ["A", "B"])
def test_speaker_split_is_correct_regardless_of_diarization_label(coach_label):
    """Was 33/7 when the coachee happened to be labelled "A"."""
    router = SpeakerRouter()
    router.assign_batch(utterances(coach_label))
    roles = [router.role_for(u.speaker, u.text) for u in utterances(coach_label)]
    assert roles.count("coach") == 20
    assert roles.count("coachee") == 20
    for (expected, _), actual in zip(TURNS, roles):
        assert expected == actual


def test_pinned_coach_speaker_is_honoured():
    router = SpeakerRouter(coach_speaker_id="B")
    router.assign_batch(utterances("B"))
    assert router.role_for("B") == "coach"
    assert router.role_for("A") == "coachee"


def test_role_is_stable_for_a_speaker():
    """The same speaker id must never flip role between utterances."""
    router = SpeakerRouter()
    router.assign_batch(utterances("A"))
    first = router.role_for("A", "What would you like to focus on?")
    second = router.role_for("A", "I feel stuck and I don't know why.")
    assert first == second


# -- Findings 5 & 6: GROW phases -------------------------------------------

def _feedback_history():
    classifier = GROWClassifier()
    history = []
    for speaker, text in TURNS:
        result = classifier.classify(text, speaker)
        history.append(SimpleNamespace(
            grow_phase=SimpleNamespace(
                phase=result.phase, confidence=result.confidence
            ),
            speaker=speaker,
        ))
    return history


def test_grow_percentages_sum_to_100():
    """Was 220% total with a single row reading 120%."""
    rows = phase_distribution(_feedback_history())
    assert rows
    total = sum(row["percentage"] for row in rows)
    assert total == pytest.approx(100.0, abs=0.01)


def test_grow_distribution_never_emits_uncertain_as_a_phase():
    rows = phase_distribution(_feedback_history())
    assert all(row["phase"] in CANONICAL_PHASES for row in rows)


def test_all_four_grow_phases_are_detected():
    """Options and Way Forward were previously never reported."""
    coverage = phase_coverage(_feedback_history())
    assert coverage["phases_missing"] == []
    assert coverage["coverage_pct"] == pytest.approx(100.0)


def test_grow_progression_is_in_canonical_order():
    classifier = GROWClassifier()
    opened = [
        (index, classifier.classify(text, speaker).phase)
        for index, (speaker, text) in enumerate(TURNS, start=1)
    ]
    ordering = []
    for _, phase in opened:
        if not ordering or ordering[-1] != phase:
            ordering.append(phase)
    assert ordering == ["Goal", "Reality", "Options", "Way Forward"]


# -- Findings 1 & 13: emotion ----------------------------------------------

def test_emotion_is_not_always_neutral():
    """Every turn previously reported 100% neutral."""
    readings = [emotion.analyze_text_emotion(text) for _, text in TURNS]
    labels = {emotion.dominant_emotion(r) for r in readings if r}
    assert len(labels) >= 4
    assert "neutral" not in labels


def test_emotion_returns_empty_when_there_is_no_signal():
    """No signal must be {}, never a fabricated neutral."""
    assert emotion.analyze_text_emotion("The meeting is at three.") == {}
    assert emotion.analyze_text_emotion("") == {}


def test_emotion_respects_negation():
    assert "happy" not in emotion.analyze_text_emotion("I am not happy about it")


# -- Finding 3: sarcasm -----------------------------------------------------

def test_obvious_sarcasm_is_detected():
    """Both flagged lines previously scored below threshold."""
    analyzer = SarcasmAnalyzer()
    history = []
    detected = []
    for index, (speaker, text) in enumerate(TURNS, start=1):
        result = analyzer.analyze(text, speaker, history)
        history.append(SimpleNamespace(speaker=speaker, transcript=text))
        if result.is_sarcastic:
            detected.append(index)
    for turn in SARCASTIC_TURNS:
        assert turn in detected, f"missed sarcasm on turn {turn}"


def test_sarcasm_has_no_false_positives_on_this_session():
    analyzer = SarcasmAnalyzer()
    history = []
    detected = []
    for index, (speaker, text) in enumerate(TURNS, start=1):
        if analyzer.analyze(text, speaker, history).is_sarcastic:
            detected.append(index)
        history.append(SimpleNamespace(speaker=speaker, transcript=text))
    assert detected == list(SARCASTIC_TURNS)


def test_sarcasm_score_is_not_a_constant():
    """The old fallback returned a flat 0.30 for every utterance."""
    analyzer = SarcasmAnalyzer()
    scores = {analyzer.analyze(text, speaker).score for speaker, text in TURNS}
    assert len(scores) > 1


# -- Finding 7: listening ---------------------------------------------------

def test_listening_recognises_real_reflective_moves():
    """Scored 1/20 coach turns (0.04 average) before."""
    coach_turns = [c for c in chunks() if c.speaker == "coach"]
    result = quality.analyze_listening(coach_turns)
    assert result["listening_indicators"] >= 10
    assert result["frequency"] >= 0.5


def test_effectiveness_reflects_observed_listening():
    coach_turns = [c for c in chunks() if c.speaker == "coach"]
    scores = quality.effectiveness(
        quality.analyze_questions(coach_turns),
        quality.analyze_listening(coach_turns),
        0.6,
    )
    assert scores["listening"] > 0.5
    assert scores["overall"] > 0.5


# -- Finding 10: digression -------------------------------------------------

def test_explicit_digression_is_detected():
    analyzer = DigressionAnalyzer()
    history = []
    flagged = []
    for index, chunk in enumerate(chunks(), start=1):
        if analyzer.analyze(chunk.transcript, history).is_digression:
            flagged.append(index)
        history.append(chunk)
    assert flagged == [DIGRESSION_TURN]


def test_on_topic_turns_are_not_flagged():
    """Lexical overlap alone must never flag a turn."""
    analyzer = DigressionAnalyzer()
    history = list(chunks()[:10])
    result = analyzer.analyze(
        "I want to talk about the promotion and my manager again.", history
    )
    assert not result.is_digression


# -- Finding 11: themes -----------------------------------------------------

def test_relationships_is_not_a_false_positive():
    """"team-lead" and "my manager" are career signals, not relational ones."""
    coachee = [c for c in chunks() if c.speaker == "coachee"]
    assert "relationships" not in themes.extract_themes(coachee)
    assert "career" in themes.extract_themes(coachee)


# -- Finding 9: VAK ---------------------------------------------------------

def test_vak_reports_insufficient_data_rather_than_a_fake_split():
    result = VAKAnalyzer().analyze([])
    assert result.dominant == "Insufficient Data"
    assert result.confidence == 0.0
    assert result.to_dict()["visual"] == 0.0


def test_vak_scores_only_coachee_speech():
    coach_only = [c for c in chunks() if c.speaker == "coach"]
    assert VAKAnalyzer().analyze(coach_only).confidence == 0.0
