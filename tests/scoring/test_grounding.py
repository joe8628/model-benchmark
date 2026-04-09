from benchmark.scoring.grounding import (
    atomize, score_grounded_mode, score_withheld_mode, score_grounding,
)
from benchmark.schema import BenchmarkItem, ItemAxis, GroundingMode


def _grounded_item(context: str, correct: str = "") -> BenchmarkItem:
    return BenchmarkItem(
        id="g001", axis=ItemAxis.GROUNDING,
        prompt="Answer based on the context.",
        correct_answer=correct,
        context=context,
        grounding_mode=GroundingMode.GROUNDED,
    )


def _withheld_item(context: str, withheld: list[str]) -> BenchmarkItem:
    return BenchmarkItem(
        id="g002", axis=ItemAxis.GROUNDING,
        prompt="Describe the subject.",
        correct_answer="",
        context=context,
        withheld_facts=withheld,
        grounding_mode=GroundingMode.WITHHELD,
    )


def test_atomize_splits_two_sentences():
    claims = atomize("Water boils at 100°C. Ice melts at 0°C.")
    assert len(claims) == 2

def test_atomize_single_sentence():
    assert len(atomize("Just one claim.")) == 1

def test_atomize_empty_returns_empty():
    assert atomize("") == []

def test_atomize_strips_whitespace():
    claims = atomize("  First.  Second.  ")
    assert all(c == c.strip() for c in claims)

def test_score_grounded_mode_fully_supported():
    context = "Water is a liquid that boils at 100 degrees Celsius at sea level."
    claims = ["Water boils at 100 degrees.", "Water is a liquid."]
    assert score_grounded_mode(claims, context) == 1.0

def test_score_grounded_mode_unsupported_claim():
    context = "Water is a liquid."
    claims = ["Jupiter is the largest planet in the solar system."]
    score = score_grounded_mode(claims, context)
    assert score < 0.5

def test_score_grounded_mode_empty_claims():
    assert score_grounded_mode([], "some context") == 1.0

def test_score_withheld_mode_no_hallucination():
    claims = ["Water is a liquid.", "It is clear and odorless."]
    withheld = ["Water boils at 100 degrees Celsius"]
    assert score_withheld_mode(claims, withheld) == 1.0

def test_score_withheld_mode_hallucination_detected():
    claims = ["Water boils at 100 degrees Celsius at sea level."]
    withheld = ["Water boils at 100 degrees Celsius"]
    score = score_withheld_mode(claims, withheld)
    assert score < 1.0

def test_score_withheld_mode_empty_withheld():
    assert score_withheld_mode(["any claim"], []) == 1.0

def test_score_grounding_dispatches_grounded():
    item = _grounded_item("Water boils at 100 degrees Celsius at sea level.")
    score = score_grounding(item, "Water boils at 100 degrees.")
    assert 0.0 <= score <= 1.0

def test_score_grounding_dispatches_withheld():
    item = _withheld_item(
        context="Water is a clear liquid.",
        withheld=["Water boils at 100 degrees Celsius"],
    )
    score = score_grounding(item, "Water is a clear liquid.")
    assert score == 1.0

def test_score_grounding_no_mode_returns_zero():
    item = BenchmarkItem(
        id="g001", axis=ItemAxis.GROUNDING,
        prompt="Q", correct_answer="", context="ctx",
    )
    assert score_grounding(item, "response") == 0.0
