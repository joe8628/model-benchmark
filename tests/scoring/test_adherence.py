import pytest
from benchmark.scoring.adherence import (
    MinWordCount, MaxWordCount, MustContainKeyword, MustNotContainKeyword,
    MustStartWith, MustEndWith, MustUseBulletFormat, MustUseNumberedFormat,
    build_constraint, score_adherence,
)
from benchmark.schema import BenchmarkItem, ItemAxis


def _item(constraints: list[dict]) -> BenchmarkItem:
    return BenchmarkItem(
        id="a001", axis=ItemAxis.ADHERENCE,
        prompt="Describe the topic.", correct_answer="",
        constraints=constraints,
    )


def test_min_word_count_pass():
    assert MinWordCount(3).check("one two three") is True

def test_min_word_count_fail():
    assert MinWordCount(5).check("one two three") is False

def test_max_word_count_pass():
    assert MaxWordCount(10).check("one two three") is True

def test_max_word_count_fail():
    assert MaxWordCount(2).check("one two three") is False

def test_must_contain_keyword_present():
    assert MustContainKeyword("water").check("Water is a liquid.") is True

def test_must_contain_keyword_absent():
    assert MustContainKeyword("fire").check("Water is a liquid.") is False

def test_must_contain_keyword_min_count():
    assert MustContainKeyword("the", min_count=2).check("the cat and the dog") is True
    assert MustContainKeyword("the", min_count=3).check("the cat and the dog") is False

def test_must_not_contain_keyword_absent():
    assert MustNotContainKeyword("fire").check("Water is a liquid.") is True

def test_must_not_contain_keyword_present():
    assert MustNotContainKeyword("water").check("Water is a liquid.") is False

def test_must_start_with_pass():
    assert MustStartWith("The").check("The quick brown fox") is True

def test_must_start_with_fail():
    assert MustStartWith("A").check("The quick brown fox") is False

def test_must_end_with_pass():
    assert MustEndWith("fox").check("The quick brown fox") is True

def test_must_end_with_fail():
    assert MustEndWith("dog").check("The quick brown fox") is False

def test_bullet_format_pass():
    response = "- Item one\n- Item two\n- Item three"
    assert MustUseBulletFormat().check(response) is True

def test_bullet_format_fail():
    assert MustUseBulletFormat().check("Just a sentence.") is False

def test_numbered_format_pass():
    response = "1. First\n2. Second\n3. Third"
    assert MustUseNumberedFormat().check(response) is True

def test_numbered_format_fail():
    assert MustUseNumberedFormat().check("No numbers here.") is False

def test_build_constraint_min_words():
    c = build_constraint({"type": "min_words", "n": 5})
    assert c.check("a b c d e") is True

def test_build_constraint_unknown_raises():
    with pytest.raises(ValueError, match="Unknown constraint type"):
        build_constraint({"type": "unknown_type"})

def test_score_adherence_all_pass():
    item = _item([{"type": "min_words", "n": 3}, {"type": "format_bullet"}])
    assert score_adherence(item, "- Apple\n- Banana\n- Cherry") == 1.0

def test_score_adherence_half_pass():
    item = _item([{"type": "min_words", "n": 500}, {"type": "format_bullet"}])
    assert score_adherence(item, "- Apple\n- Banana\n- Cherry") == 0.5

def test_score_adherence_none_pass():
    item = _item([{"type": "min_words", "n": 500}, {"type": "must_contain", "keyword": "zebra"}])
    assert score_adherence(item, "Short response.") == 0.0

def test_score_adherence_no_constraints_returns_one():
    item = BenchmarkItem(id="a001", axis=ItemAxis.ADHERENCE, prompt="Q", correct_answer="")
    assert score_adherence(item, "any response") == 1.0
