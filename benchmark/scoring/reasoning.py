import re
from benchmark.schema import BenchmarkItem


def parse_multiple_choice(response: str, choices: list[str]) -> str | None:
    """Extract the chosen answer letter (A–D) from a chain-of-thought response.

    Recognises:
      - "The answer is A" / "Answer: B"
      - "(C)" at end of sentence
      - Standalone letter on the last non-empty line
    Returns uppercase letter or None if unparseable.
    """
    text = response.strip()

    # "the answer is X" or "answer: X"
    match = re.search(r'\bthe answer is\s*[:\-]?\s*\(?([A-Da-d])\)?', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    match = re.search(r'\banswer\s*[:\-]\s*\(?([A-Da-d])\)?', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Parenthesised letter in any sentence: "correct choice is (B)"
    match = re.search(r'correct\s+\w+\s+is\s+\(([A-Da-d])\)', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Standalone letter on last non-empty line
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        match = re.match(r'^\(?([A-Da-d])\)?\.?$', lines[-1])
        if match:
            return match.group(1).upper()

    return None


def score_reasoning(item: BenchmarkItem, response: str) -> float:
    """Returns 1.0 for correct answer, 0.0 for incorrect or unparseable."""
    if not item.choices:
        return 0.0
    chosen = parse_multiple_choice(response, item.choices)
    if chosen is None:
        return 0.0
    return 1.0 if chosen == item.correct_answer.upper() else 0.0
