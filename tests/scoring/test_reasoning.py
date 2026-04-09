from benchmark.scoring.reasoning import parse_multiple_choice, score_reasoning
from benchmark.schema import BenchmarkItem, ItemAxis

CHOICES = ["A. Photosynthesis", "B. Respiration", "C. Fermentation", "D. Digestion"]

def _item(correct: str) -> BenchmarkItem:
    return BenchmarkItem(
        id="r001", axis=ItemAxis.REASONING,
        prompt="Which process converts CO2 to glucose?",
        correct_answer=correct, choices=CHOICES,
    )

def test_parse_answer_is_pattern():
    assert parse_multiple_choice("The answer is A.", CHOICES) == "A"

def test_parse_answer_colon():
    assert parse_multiple_choice("Answer: B", CHOICES) == "B"

def test_parse_cot_trailing_answer():
    response = "I considered all options carefully. The answer is C."
    assert parse_multiple_choice(response, CHOICES) == "C"

def test_parse_standalone_letter_last_line():
    response = "After reasoning through the options:\n\nD"
    assert parse_multiple_choice(response, CHOICES) == "D"

def test_parse_parenthesized_letter():
    assert parse_multiple_choice("The correct choice is (B).", CHOICES) == "B"

def test_parse_returns_none_on_ambiguous():
    assert parse_multiple_choice("It could be any of them.", CHOICES) is None

def test_parse_case_insensitive():
    assert parse_multiple_choice("the answer is a.", CHOICES) == "A"

def test_score_correct():
    assert score_reasoning(_item("A"), "The answer is A.") == 1.0

def test_score_incorrect():
    assert score_reasoning(_item("A"), "The answer is B.") == 0.0

def test_score_unparseable():
    assert score_reasoning(_item("A"), "I have no idea.") == 0.0

def test_score_no_choices_returns_zero():
    item = BenchmarkItem(id="r001", axis=ItemAxis.REASONING, prompt="Q", correct_answer="A")
    assert score_reasoning(item, "The answer is A.") == 0.0
