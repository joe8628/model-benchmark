import math
from benchmark.tokens import count_tokens, ockscore

def test_count_tokens_empty():
    assert count_tokens("") == 0

def test_count_tokens_nonempty():
    assert count_tokens("hello world") > 0

def test_count_tokens_consistent():
    text = "The quick brown fox"
    assert count_tokens(text) == count_tokens(text)

def test_ockscore_zero_accuracy_returns_zero():
    assert ockscore(0.0, 100, 2048) == 0.0

def test_ockscore_full_accuracy_at_budget():
    # budget/actual = 1 → log(2) ≈ 0.693
    score = ockscore(1.0, 2048, 2048)
    assert abs(score - math.log(2)) < 1e-9

def test_ockscore_penalizes_verbosity():
    concise = ockscore(1.0, 200, 2048)
    verbose = ockscore(1.0, 1800, 2048)
    assert concise > verbose

def test_ockscore_zero_tokens_returns_zero():
    assert ockscore(1.0, 0, 2048) == 0.0

def test_ockscore_partial_accuracy_scales():
    half = ockscore(0.5, 200, 2048)
    full = ockscore(1.0, 200, 2048)
    assert abs(half - full / 2) < 1e-9
