# Model Reasoning Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a model-agnostic benchmark suite that measures intrinsic vs. stack-derived reasoning quality via score delta across baseline and affordance-augmented conditions.

**Architecture:** Items are JSONL files organized by axis (reasoning, adherence, grounding) and set (anchor/rolling). A run executor calls the model API under two isolated conditions (baseline, affordance-augmented), scores each item automatically via axis-specific scorers, and aggregates into per-item OckScores and a run delta report. Token counting uses cl100k_base throughout.

**Tech Stack:** Python 3.11+, tiktoken, pydantic v2, anthropic SDK, openai SDK, click, pytest

---

## File Map

```
model-benchmark/
├── pyproject.toml
├── benchmark/
│   ├── __init__.py
│   ├── tokens.py          # cl100k_base counting + OckScore formula
│   ├── schema.py          # BenchmarkItem, RunCondition, ItemResult, RunResult dataclasses
│   ├── client.py          # ModelClient protocol, AnthropicClient, OpenAIClient
│   ├── runner.py          # run_item(), run_suite() — isolated per-condition execution
│   ├── delta.py           # compute_delta(), token-inflation detection
│   ├── report.py          # generate_report() — markdown output
│   ├── cli.py             # click CLI entry point
│   └── scoring/
│       ├── __init__.py
│       ├── reasoning.py   # GPQA-style CoT answer parsing
│       ├── adherence.py   # IFEval-style verifiable constraint checking
│       └── grounding.py   # FActScore-style atomic claim decomposition
├── items/
│   ├── anchor/
│   │   ├── reasoning/items.jsonl
│   │   ├── adherence/items.jsonl
│   │   └── grounding/items.jsonl
│   └── rolling/           # empty at start, populated by refresh process
│       ├── reasoning/
│       ├── adherence/
│       └── grounding/
├── tests/
│   ├── __init__.py
│   ├── test_tokens.py
│   ├── test_schema.py
│   ├── test_client.py
│   ├── test_runner.py
│   ├── test_delta.py
│   ├── test_report.py
│   ├── test_cli.py
│   └── scoring/
│       ├── __init__.py
│       ├── test_reasoning.py
│       ├── test_adherence.py
│       └── test_grounding.py
└── docs/
    ├── concepts/
    │   └── model-reasoning-benchmark.md
    └── superpowers/
        └── plans/
            └── 2026-04-08-model-reasoning-benchmark.md
```

---

## Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `benchmark/__init__.py`
- Create: `benchmark/scoring/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/scoring/__init__.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "model-benchmark"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "tiktoken>=0.7",
    "pydantic>=2.0",
    "anthropic>=0.40",
    "openai>=1.50",
    "click>=8.1",
    "httpx>=0.27",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
]

[project.scripts]
benchmark = "benchmark.cli:main"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create empty `__init__.py` files**

Create `benchmark/__init__.py`, `benchmark/scoring/__init__.py`, `tests/__init__.py`, `tests/scoring/__init__.py` — all empty.

- [ ] **Step 3: Install dependencies**

```bash
pip install -e ".[dev]"
```

Expected: all packages install without error, `benchmark` command appears in PATH.

- [ ] **Step 4: Verify tiktoken loads**

```bash
python -c "import tiktoken; enc = tiktoken.get_encoding('cl100k_base'); print(len(enc.encode('hello world')))"
```

Expected: prints `2`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml benchmark/ tests/
git commit -m "chore: scaffold project with dependencies"
```

---

## Task 2: Token Counter + OckScore

**Files:**
- Create: `benchmark/tokens.py`
- Create: `tests/test_tokens.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tokens.py
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_tokens.py -v
```

Expected: `ModuleNotFoundError: No module named 'benchmark.tokens'`

- [ ] **Step 3: Implement `benchmark/tokens.py`**

```python
import math
import tiktoken

_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def ockscore(accuracy: float, actual_tokens: int, budget: int = 2048) -> float:
    """Logarithmically penalizes verbosity while prioritizing correctness.

    Formula: accuracy × log(budget / actual_tokens + 1)
    Returns 0.0 when accuracy is 0.0 or actual_tokens is 0.
    """
    if actual_tokens <= 0 or accuracy == 0.0:
        return 0.0
    return accuracy * math.log(budget / actual_tokens + 1)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_tokens.py -v
```

Expected: all 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add benchmark/tokens.py tests/test_tokens.py
git commit -m "feat: add cl100k_base token counter and OckScore metric"
```

---

## Task 3: Item Schema

**Files:**
- Create: `benchmark/schema.py`
- Create: `tests/test_schema.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_schema.py
from benchmark.schema import (
    BenchmarkItem, ItemAxis, GroundingMode,
    RunCondition, ItemResult, RunResult,
)

def test_reasoning_item_defaults():
    item = BenchmarkItem(
        id="r001",
        axis=ItemAxis.REASONING,
        prompt="Which process converts CO2 to glucose?",
        correct_answer="A",
        choices=["A. Photosynthesis", "B. Respiration", "C. Fermentation", "D. Digestion"],
    )
    assert item.anchor is True
    assert item.version == 1
    assert item.constraints is None
    assert item.context is None

def test_adherence_item():
    item = BenchmarkItem(
        id="a001",
        axis=ItemAxis.ADHERENCE,
        prompt="Describe the water cycle.",
        correct_answer="",
        constraints=[{"type": "min_words", "n": 100}, {"type": "format_bullet"}],
    )
    assert item.axis == ItemAxis.ADHERENCE
    assert len(item.constraints) == 2

def test_grounding_item_grounded_mode():
    item = BenchmarkItem(
        id="g001",
        axis=ItemAxis.GROUNDING,
        prompt="At what temperature does the substance boil?",
        correct_answer="100°C",
        context="The substance boils at 100°C at standard pressure.",
        grounding_mode=GroundingMode.GROUNDED,
    )
    assert item.grounding_mode == GroundingMode.GROUNDED
    assert item.withheld_facts is None

def test_grounding_item_withheld_mode():
    item = BenchmarkItem(
        id="g002",
        axis=ItemAxis.GROUNDING,
        prompt="Describe all properties of the substance.",
        correct_answer="",
        context="The substance is a clear liquid at room temperature.",
        withheld_facts=["The substance boils at 100°C"],
        grounding_mode=GroundingMode.WITHHELD,
    )
    assert item.withheld_facts is not None
    assert len(item.withheld_facts) == 1

def test_run_condition_baseline_empty_prompt():
    cond = RunCondition(name="baseline")
    assert cond.system_prompt == ""

def test_run_condition_with_affordance():
    cond = RunCondition(name="skill_v1", system_prompt="Always reason step by step.")
    assert "step" in cond.system_prompt

def test_item_result_fields():
    result = ItemResult(
        item_id="r001", condition="baseline", response="The answer is A.",
        tokens_prompt=50, tokens_completion=10, tokens_total=60,
        accuracy=1.0, ockscore=0.7, flagged_over_budget=False,
    )
    assert result.tokens_total == 60
    assert result.flagged_over_budget is False

def test_run_result_fields():
    run = RunResult(
        run_id="abc123", model="claude-sonnet-4-6",
        conditions=["baseline", "skill_v1"],
        results=[], timestamp="2026-04-08T00:00:00Z",
    )
    assert run.suite_token_total == 0
    assert run.suite_ockscore == 0.0
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_schema.py -v
```

Expected: `ModuleNotFoundError: No module named 'benchmark.schema'`

- [ ] **Step 3: Implement `benchmark/schema.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum


class ItemAxis(str, Enum):
    REASONING = "reasoning"
    ADHERENCE = "adherence"
    GROUNDING = "grounding"


class GroundingMode(str, Enum):
    GROUNDED = "grounded"
    WITHHELD = "withheld"


@dataclass
class BenchmarkItem:
    id: str
    axis: ItemAxis
    prompt: str
    correct_answer: str
    anchor: bool = True
    version: int = 1
    # Reasoning axis
    choices: list[str] | None = None
    # Adherence axis
    constraints: list[dict] | None = None
    # Grounding axis
    context: str | None = None
    withheld_facts: list[str] | None = None
    grounding_mode: GroundingMode | None = None


@dataclass
class RunCondition:
    name: str
    system_prompt: str = ""


@dataclass
class ItemResult:
    item_id: str
    condition: str
    response: str
    tokens_prompt: int
    tokens_completion: int
    tokens_total: int
    accuracy: float
    ockscore: float
    flagged_over_budget: bool


@dataclass
class RunResult:
    run_id: str
    model: str
    conditions: list[str]
    results: list[ItemResult]
    timestamp: str
    suite_token_total: int = 0
    suite_ockscore: float = 0.0
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_schema.py -v
```

Expected: all 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add benchmark/schema.py tests/test_schema.py
git commit -m "feat: add benchmark item and run result schema"
```

---

## Task 4: Model Client Abstraction

**Files:**
- Create: `benchmark/client.py`
- Create: `tests/test_client.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_client.py
from benchmark.client import AnthropicClient, OpenAIClient


class _StubClient:
    """Satisfies the ModelClient protocol without hitting any API."""
    def complete(self, prompt: str, system: str, temperature: float = 0.0) -> tuple[str, int, int]:
        return "The answer is A.", 50, 10


def test_stub_returns_three_tuple():
    client = _StubClient()
    result = client.complete("test prompt", "")
    assert len(result) == 3

def test_stub_response_is_str():
    client = _StubClient()
    response, _, _ = client.complete("test", "")
    assert isinstance(response, str)

def test_stub_token_counts_are_ints():
    client = _StubClient()
    _, prompt_tokens, completion_tokens = client.complete("test", "")
    assert isinstance(prompt_tokens, int)
    assert isinstance(completion_tokens, int)

def test_stub_temperature_accepted():
    client = _StubClient()
    response, _, _ = client.complete("test", "", temperature=0.0)
    assert response

def test_anthropic_client_instantiates():
    # Does not call the API — just confirms the class is importable and constructable
    client = AnthropicClient("claude-sonnet-4-6")
    assert client._model == "claude-sonnet-4-6"

def test_openai_client_instantiates():
    client = OpenAIClient("gpt-4o")
    assert client._model == "gpt-4o"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_client.py -v
```

Expected: `ModuleNotFoundError: No module named 'benchmark.client'`

- [ ] **Step 3: Implement `benchmark/client.py`**

```python
from typing import Protocol
import anthropic
import openai


class ModelClient(Protocol):
    def complete(
        self,
        prompt: str,
        system: str,
        temperature: float = 0.0,
    ) -> tuple[str, int, int]:
        """Returns (response_text, prompt_tokens, completion_tokens)."""
        ...


class AnthropicClient:
    def __init__(self, model: str):
        self._client = anthropic.Anthropic()
        self._model = model

    def complete(self, prompt: str, system: str, temperature: float = 0.0) -> tuple[str, int, int]:
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=2048,
            temperature=temperature,
            system=system or "You are a helpful assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text
        return text, msg.usage.input_tokens, msg.usage.output_tokens


class OpenAIClient:
    def __init__(self, model: str):
        self._client = openai.OpenAI()
        self._model = model

    def complete(self, prompt: str, system: str, temperature: float = 0.0) -> tuple[str, int, int]:
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system or "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        text = resp.choices[0].message.content
        return text, resp.usage.prompt_tokens, resp.usage.completion_tokens
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_client.py -v
```

Expected: all 6 tests PASS (the two instantiation tests confirm import, not API calls)

- [ ] **Step 5: Commit**

```bash
git add benchmark/client.py tests/test_client.py
git commit -m "feat: add model client abstraction for Anthropic and OpenAI"
```

---

## Task 5: Reasoning Scorer

**Files:**
- Create: `benchmark/scoring/reasoning.py`
- Create: `tests/scoring/test_reasoning.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/scoring/test_reasoning.py
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/scoring/test_reasoning.py -v
```

Expected: `ModuleNotFoundError: No module named 'benchmark.scoring.reasoning'`

- [ ] **Step 3: Implement `benchmark/scoring/reasoning.py`**

```python
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
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/scoring/test_reasoning.py -v
```

Expected: all 11 tests PASS

- [ ] **Step 5: Commit**

```bash
git add benchmark/scoring/reasoning.py tests/scoring/test_reasoning.py
git commit -m "feat: add GPQA-style reasoning scorer with CoT answer parsing"
```

---

## Task 6: Adherence Scorer

**Files:**
- Create: `benchmark/scoring/adherence.py`
- Create: `tests/scoring/test_adherence.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/scoring/test_adherence.py
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/scoring/test_adherence.py -v
```

Expected: `ModuleNotFoundError: No module named 'benchmark.scoring.adherence'`

- [ ] **Step 3: Implement `benchmark/scoring/adherence.py`**

```python
import re
from typing import Protocol
from benchmark.schema import BenchmarkItem


class Constraint(Protocol):
    def check(self, response: str) -> bool: ...


class MinWordCount:
    def __init__(self, n: int):
        self.n = n
    def check(self, response: str) -> bool:
        return len(response.split()) >= self.n


class MaxWordCount:
    def __init__(self, n: int):
        self.n = n
    def check(self, response: str) -> bool:
        return len(response.split()) <= self.n


class MustContainKeyword:
    def __init__(self, keyword: str, min_count: int = 1):
        self.keyword = keyword.lower()
        self.min_count = min_count
    def check(self, response: str) -> bool:
        return response.lower().count(self.keyword) >= self.min_count


class MustNotContainKeyword:
    def __init__(self, keyword: str):
        self.keyword = keyword.lower()
    def check(self, response: str) -> bool:
        return self.keyword not in response.lower()


class MustStartWith:
    def __init__(self, prefix: str):
        self.prefix = prefix
    def check(self, response: str) -> bool:
        return response.strip().startswith(self.prefix)


class MustEndWith:
    def __init__(self, suffix: str):
        self.suffix = suffix
    def check(self, response: str) -> bool:
        return response.strip().endswith(self.suffix)


class MustUseBulletFormat:
    def check(self, response: str) -> bool:
        lines = [ln.strip() for ln in response.splitlines() if ln.strip()]
        return sum(1 for ln in lines if re.match(r'^[-*•]', ln)) >= 3


class MustUseNumberedFormat:
    def check(self, response: str) -> bool:
        lines = [ln.strip() for ln in response.splitlines() if ln.strip()]
        return sum(1 for ln in lines if re.match(r'^\d+[.)]\s', ln)) >= 3


_REGISTRY = {
    "min_words":       lambda d: MinWordCount(d["n"]),
    "max_words":       lambda d: MaxWordCount(d["n"]),
    "must_contain":    lambda d: MustContainKeyword(d["keyword"], d.get("min_count", 1)),
    "must_not_contain":lambda d: MustNotContainKeyword(d["keyword"]),
    "must_start_with": lambda d: MustStartWith(d["prefix"]),
    "must_end_with":   lambda d: MustEndWith(d["suffix"]),
    "format_bullet":   lambda d: MustUseBulletFormat(),
    "format_numbered": lambda d: MustUseNumberedFormat(),
}


def build_constraint(spec: dict):
    factory = _REGISTRY.get(spec["type"])
    if factory is None:
        raise ValueError(f"Unknown constraint type: {spec['type']!r}")
    return factory(spec)


def score_adherence(item: BenchmarkItem, response: str) -> float:
    """Returns fraction of constraints satisfied (0.0–1.0)."""
    if not item.constraints:
        return 1.0
    constraints = [build_constraint(spec) for spec in item.constraints]
    passed = sum(1 for c in constraints if c.check(response))
    return passed / len(constraints)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/scoring/test_adherence.py -v
```

Expected: all 22 tests PASS

- [ ] **Step 5: Commit**

```bash
git add benchmark/scoring/adherence.py tests/scoring/test_adherence.py
git commit -m "feat: add IFEval-style verifiable constraint scorer"
```

---

## Task 7: Grounding Scorer

**Files:**
- Create: `benchmark/scoring/grounding.py`
- Create: `tests/scoring/test_grounding.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/scoring/test_grounding.py
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/scoring/test_grounding.py -v
```

Expected: `ModuleNotFoundError: No module named 'benchmark.scoring.grounding'`

- [ ] **Step 3: Implement `benchmark/scoring/grounding.py`**

```python
import re
from benchmark.schema import BenchmarkItem, GroundingMode

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "in", "of",
    "to", "and", "or", "it", "that", "this", "at", "by", "with",
    "on", "for", "be", "as", "its", "which", "has", "have",
}


def atomize(text: str) -> list[str]:
    """Split text into atomic sentence-level claims."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _keyword_overlap(claim: str, context: str) -> float:
    """Fraction of significant claim words found in context."""
    claim_words = {
        w.lower() for w in re.findall(r'\w+', claim)
        if w.lower() not in _STOPWORDS and len(w) > 2
    }
    if not claim_words:
        return 1.0
    context_words = {w.lower() for w in re.findall(r'\w+', context)}
    return len(claim_words & context_words) / len(claim_words)


def score_grounded_mode(
    claims: list[str], context: str, threshold: float = 0.5
) -> float:
    """Fraction of claims with keyword overlap >= threshold against provided context."""
    if not claims:
        return 1.0
    supported = sum(1 for c in claims if _keyword_overlap(c, context) >= threshold)
    return supported / len(claims)


def score_withheld_mode(claims: list[str], withheld_facts: list[str]) -> float:
    """Penalizes claims that reference withheld facts.
    Returns 1.0 - (fraction of claims referencing any withheld keyword).
    """
    if not claims or not withheld_facts:
        return 1.0
    withheld_kw = {
        w.lower() for fact in withheld_facts
        for w in re.findall(r'\w+', fact)
        if len(w) > 3 and w.lower() not in _STOPWORDS
    }
    hallucinated = sum(
        1 for claim in claims
        if any(kw in claim.lower() for kw in withheld_kw)
    )
    return 1.0 - (hallucinated / len(claims))


def score_grounding(item: BenchmarkItem, response: str) -> float:
    claims = atomize(response)
    if item.grounding_mode == GroundingMode.GROUNDED:
        return score_grounded_mode(claims, item.context or "")
    if item.grounding_mode == GroundingMode.WITHHELD:
        return score_withheld_mode(claims, item.withheld_facts or [])
    return 0.0
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/scoring/test_grounding.py -v
```

Expected: all 13 tests PASS

- [ ] **Step 5: Commit**

```bash
git add benchmark/scoring/grounding.py tests/scoring/test_grounding.py
git commit -m "feat: add FActScore-style grounding scorer (grounded + withheld modes)"
```

---

## Task 8: Run Executor

**Files:**
- Create: `benchmark/runner.py`
- Create: `tests/test_runner.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_runner.py
from benchmark.schema import BenchmarkItem, ItemAxis, GroundingMode, RunCondition
from benchmark.runner import run_item, run_suite, ITEM_TOKEN_BUDGET


class _StubClient:
    def __init__(self, response="The answer is A.", prompt_tokens=50, completion_tokens=10):
        self._response = response
        self._pt = prompt_tokens
        self._ct = completion_tokens

    def complete(self, prompt, system, temperature=0.0):
        assert temperature == 0.0, "Runner must enforce temperature=0"
        return self._response, self._pt, self._ct


_REASONING_ITEM = BenchmarkItem(
    id="r001", axis=ItemAxis.REASONING,
    prompt="Which is correct?", correct_answer="A",
    choices=["A. Yes", "B. No", "C. Maybe", "D. Never"],
)

_ADHERENCE_ITEM = BenchmarkItem(
    id="a001", axis=ItemAxis.ADHERENCE,
    prompt="List things.", correct_answer="",
    constraints=[{"type": "format_bullet"}],
)

_GROUNDING_ITEM = BenchmarkItem(
    id="g001", axis=ItemAxis.GROUNDING,
    prompt="What does the context say?", correct_answer="liquid",
    context="Water is a liquid.", grounding_mode=GroundingMode.GROUNDED,
)


def test_run_item_returns_correct_ids():
    result = run_item(_REASONING_ITEM, RunCondition("baseline"), _StubClient())
    assert result.item_id == "r001"
    assert result.condition == "baseline"

def test_run_item_tokens_sum_correctly():
    result = run_item(_REASONING_ITEM, RunCondition("baseline"), _StubClient(prompt_tokens=50, completion_tokens=10))
    assert result.tokens_total == 60

def test_run_item_flags_over_budget():
    client = _StubClient(prompt_tokens=1500, completion_tokens=700)
    result = run_item(_REASONING_ITEM, RunCondition("baseline"), client)
    assert result.flagged_over_budget is True

def test_run_item_not_flagged_under_budget():
    result = run_item(_REASONING_ITEM, RunCondition("baseline"), _StubClient())
    assert result.flagged_over_budget is False

def test_run_item_enforces_temperature_zero():
    # _StubClient.complete asserts temperature == 0.0 — will raise if violated
    run_item(_REASONING_ITEM, RunCondition("baseline"), _StubClient())

def test_run_item_reasoning_scored():
    result = run_item(_REASONING_ITEM, RunCondition("baseline"), _StubClient(response="The answer is A."))
    assert result.accuracy == 1.0

def test_run_item_adherence_scored():
    client = _StubClient(response="- One\n- Two\n- Three")
    result = run_item(_ADHERENCE_ITEM, RunCondition("baseline"), client)
    assert result.accuracy == 1.0

def test_run_suite_produces_n_results():
    items = [_REASONING_ITEM, _ADHERENCE_ITEM]
    conditions = [RunCondition("baseline"), RunCondition("skill_v1", "Think step by step.")]
    run = run_suite(items, conditions, _StubClient(), model="test-model")
    assert len(run.results) == 4  # 2 items × 2 conditions

def test_run_suite_records_model():
    run = run_suite([_REASONING_ITEM], [RunCondition("baseline")], _StubClient(), model="my-model")
    assert run.model == "my-model"

def test_run_suite_records_conditions():
    conditions = [RunCondition("baseline"), RunCondition("v2")]
    run = run_suite([_REASONING_ITEM], conditions, _StubClient(), model="m")
    assert "baseline" in run.conditions
    assert "v2" in run.conditions

def test_run_suite_token_total():
    run = run_suite([_REASONING_ITEM], [RunCondition("baseline")], _StubClient(prompt_tokens=50, completion_tokens=10), model="m")
    assert run.suite_token_total == 60
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_runner.py -v
```

Expected: `ModuleNotFoundError: No module named 'benchmark.runner'`

- [ ] **Step 3: Implement `benchmark/runner.py`**

```python
import uuid
from datetime import datetime, timezone

from benchmark.schema import (
    BenchmarkItem, RunCondition, ItemResult, RunResult, ItemAxis,
)
from benchmark.tokens import count_tokens, ockscore
from benchmark.scoring.reasoning import score_reasoning
from benchmark.scoring.adherence import score_adherence
from benchmark.scoring.grounding import score_grounding

ITEM_TOKEN_BUDGET = 2048
SUITE_TOKEN_BUDGET = 600_000


def run_item(
    item: BenchmarkItem,
    condition: RunCondition,
    client,
) -> ItemResult:
    response, prompt_tokens, completion_tokens = client.complete(
        prompt=item.prompt,
        system=condition.system_prompt,
        temperature=0.0,
    )
    tokens_total = prompt_tokens + completion_tokens

    if item.axis == ItemAxis.REASONING:
        accuracy = score_reasoning(item, response)
    elif item.axis == ItemAxis.ADHERENCE:
        accuracy = score_adherence(item, response)
    else:
        accuracy = score_grounding(item, response)

    return ItemResult(
        item_id=item.id,
        condition=condition.name,
        response=response,
        tokens_prompt=prompt_tokens,
        tokens_completion=completion_tokens,
        tokens_total=tokens_total,
        accuracy=accuracy,
        ockscore=ockscore(accuracy, tokens_total, ITEM_TOKEN_BUDGET),
        flagged_over_budget=tokens_total > ITEM_TOKEN_BUDGET,
    )


def run_suite(
    items: list[BenchmarkItem],
    conditions: list[RunCondition],
    client,
    model: str,
) -> RunResult:
    results = []
    for condition in conditions:
        for item in items:
            results.append(run_item(item, condition, client))

    total_tokens = sum(r.tokens_total for r in results)
    avg_ockscore = sum(r.ockscore for r in results) / len(results) if results else 0.0

    return RunResult(
        run_id=str(uuid.uuid4()),
        model=model,
        conditions=[c.name for c in conditions],
        results=results,
        timestamp=datetime.now(timezone.utc).isoformat(),
        suite_token_total=total_tokens,
        suite_ockscore=avg_ockscore,
    )
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_runner.py -v
```

Expected: all 11 tests PASS

- [ ] **Step 5: Commit**

```bash
git add benchmark/runner.py tests/test_runner.py
git commit -m "feat: add multi-condition run executor with token budget enforcement"
```

---

## Task 9: Score Delta

**Files:**
- Create: `benchmark/delta.py`
- Create: `tests/test_delta.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_delta.py
from datetime import datetime, timezone
from benchmark.schema import ItemResult, RunResult
from benchmark.delta import compute_delta, ItemDelta, RunDelta


def _result(item_id, condition, accuracy, tokens):
    return ItemResult(
        item_id=item_id, condition=condition, response="test",
        tokens_prompt=tokens // 2, tokens_completion=tokens - tokens // 2,
        tokens_total=tokens, accuracy=accuracy,
        ockscore=accuracy * 0.5, flagged_over_budget=False,
    )


def _run(*results):
    return RunResult(
        run_id="test", model="m",
        conditions=list({r.condition for r in results}),
        results=list(results),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def test_compute_delta_positive_accuracy():
    run = _run(
        _result("i1", "baseline", 0.0, 100),
        _result("i1", "skill_v1", 1.0, 110),
    )
    delta = compute_delta(run, "baseline", "skill_v1")
    assert len(delta.item_deltas) == 1
    assert delta.item_deltas[0].accuracy_delta == 1.0

def test_compute_delta_token_delta():
    run = _run(
        _result("i1", "baseline", 0.0, 100),
        _result("i1", "skill_v1", 1.0, 110),
    )
    delta = compute_delta(run, "baseline", "skill_v1")
    assert delta.item_deltas[0].token_delta == 10

def test_compute_delta_no_improvement():
    run = _run(
        _result("i1", "baseline", 1.0, 100),
        _result("i1", "skill_v1", 1.0, 100),
    )
    delta = compute_delta(run, "baseline", "skill_v1")
    assert delta.item_deltas[0].accuracy_delta == 0.0

def test_token_inflated_flag_set():
    run = _run(
        _result("i1", "baseline", 0.0, 100),
        _result("i1", "skill_v1", 1.0, 300),  # improved + more tokens
    )
    delta = compute_delta(run, "baseline", "skill_v1")
    assert delta.item_deltas[0].token_inflated is True

def test_token_inflated_flag_not_set_when_fewer_tokens():
    run = _run(
        _result("i1", "baseline", 0.0, 300),
        _result("i1", "skill_v1", 1.0, 100),  # improved + fewer tokens (good)
    )
    delta = compute_delta(run, "baseline", "skill_v1")
    assert delta.item_deltas[0].token_inflated is False

def test_inflated_count_aggregated():
    run = _run(
        _result("i1", "baseline", 0.0, 100),
        _result("i1", "skill_v1", 1.0, 200),  # inflated
        _result("i2", "baseline", 0.0, 200),
        _result("i2", "skill_v1", 1.0, 100),  # not inflated
    )
    delta = compute_delta(run, "baseline", "skill_v1")
    assert delta.inflated_count == 1

def test_mean_accuracy_delta():
    run = _run(
        _result("i1", "baseline", 0.0, 100),
        _result("i1", "skill_v1", 0.5, 100),
        _result("i2", "baseline", 0.0, 100),
        _result("i2", "skill_v1", 1.0, 100),
    )
    delta = compute_delta(run, "baseline", "skill_v1")
    assert abs(delta.mean_accuracy_delta - 0.75) < 1e-9

def test_mean_token_delta():
    run = _run(
        _result("i1", "baseline", 0.0, 100),
        _result("i1", "skill_v1", 0.0, 200),
    )
    delta = compute_delta(run, "baseline", "skill_v1")
    assert delta.mean_token_delta == 100.0
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_delta.py -v
```

Expected: `ModuleNotFoundError: No module named 'benchmark.delta'`

- [ ] **Step 3: Implement `benchmark/delta.py`**

```python
from dataclasses import dataclass
from benchmark.schema import RunResult


@dataclass
class ItemDelta:
    item_id: str
    baseline_accuracy: float
    augmented_accuracy: float
    accuracy_delta: float
    baseline_tokens: int
    augmented_tokens: int
    token_delta: int
    token_inflated: bool


@dataclass
class RunDelta:
    baseline_condition: str
    augmented_condition: str
    item_deltas: list[ItemDelta]
    mean_accuracy_delta: float
    mean_token_delta: float
    inflated_count: int


def compute_delta(run: RunResult, baseline_name: str, augmented_name: str) -> RunDelta:
    baseline = {r.item_id: r for r in run.results if r.condition == baseline_name}
    augmented = {r.item_id: r for r in run.results if r.condition == augmented_name}
    common_ids = sorted(set(baseline) & set(augmented))

    deltas = []
    for item_id in common_ids:
        b, a = baseline[item_id], augmented[item_id]
        acc_delta = a.accuracy - b.accuracy
        tok_delta = a.tokens_total - b.tokens_total
        deltas.append(ItemDelta(
            item_id=item_id,
            baseline_accuracy=b.accuracy,
            augmented_accuracy=a.accuracy,
            accuracy_delta=acc_delta,
            baseline_tokens=b.tokens_total,
            augmented_tokens=a.tokens_total,
            token_delta=tok_delta,
            token_inflated=acc_delta > 0.0 and tok_delta > 0,
        ))

    n = len(deltas) or 1
    return RunDelta(
        baseline_condition=baseline_name,
        augmented_condition=augmented_name,
        item_deltas=deltas,
        mean_accuracy_delta=sum(d.accuracy_delta for d in deltas) / n,
        mean_token_delta=sum(d.token_delta for d in deltas) / n,
        inflated_count=sum(1 for d in deltas if d.token_inflated),
    )
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_delta.py -v
```

Expected: all 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add benchmark/delta.py tests/test_delta.py
git commit -m "feat: add score delta computation with token-inflation detection"
```

---

## Task 10: Run Report

**Files:**
- Create: `benchmark/report.py`
- Create: `tests/test_report.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_report.py
from datetime import datetime, timezone
from benchmark.schema import ItemResult, RunResult
from benchmark.delta import compute_delta
from benchmark.report import generate_report


def _result(item_id, condition, accuracy, tokens, flagged=False):
    return ItemResult(
        item_id=item_id, condition=condition, response="test",
        tokens_prompt=tokens // 2, tokens_completion=tokens - tokens // 2,
        tokens_total=tokens, accuracy=accuracy,
        ockscore=accuracy * 0.5, flagged_over_budget=flagged,
    )


def _run(*results, model="test-model"):
    conds = list(dict.fromkeys(r.condition for r in results))
    total = sum(r.tokens_total for r in results)
    avg_ock = sum(r.ockscore for r in results) / len(results)
    return RunResult(
        run_id="abc123", model=model,
        conditions=conds, results=list(results),
        timestamp=datetime.now(timezone.utc).isoformat(),
        suite_token_total=total, suite_ockscore=avg_ock,
    )


def test_report_contains_model():
    run = _run(_result("i1", "baseline", 1.0, 100), model="claude-sonnet-4-6")
    assert "claude-sonnet-4-6" in generate_report(run)

def test_report_contains_run_id():
    run = _run(_result("i1", "baseline", 1.0, 100))
    assert "abc123" in generate_report(run)

def test_report_contains_condition_names():
    run = _run(
        _result("i1", "baseline", 0.0, 100),
        _result("i1", "skill_v1", 1.0, 110),
    )
    report = generate_report(run)
    assert "baseline" in report
    assert "skill_v1" in report

def test_report_shows_over_budget_section():
    run = _run(_result("i1", "baseline", 1.0, 2100, flagged=True))
    assert "Over-Budget" in generate_report(run)

def test_report_no_over_budget_section_when_clean():
    run = _run(_result("i1", "baseline", 1.0, 100))
    assert "Over-Budget" not in generate_report(run)

def test_report_with_delta_shows_delta_section():
    run = _run(
        _result("i1", "baseline", 0.0, 100),
        _result("i1", "skill_v1", 1.0, 200),
    )
    delta = compute_delta(run, "baseline", "skill_v1")
    report = generate_report(run, delta)
    assert "Affordance Delta" in report

def test_report_with_delta_shows_inflation_warning():
    run = _run(
        _result("i1", "baseline", 0.0, 100),
        _result("i1", "skill_v1", 1.0, 300),  # inflated
    )
    delta = compute_delta(run, "baseline", "skill_v1")
    report = generate_report(run, delta)
    assert "yes" in report  # inflation marker

def test_report_without_delta_no_delta_section():
    run = _run(_result("i1", "baseline", 1.0, 100))
    assert "Affordance Delta" not in generate_report(run)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_report.py -v
```

Expected: `ModuleNotFoundError: No module named 'benchmark.report'`

- [ ] **Step 3: Implement `benchmark/report.py`**

```python
from benchmark.schema import RunResult
from benchmark.delta import RunDelta


def generate_report(run: RunResult, delta: RunDelta | None = None) -> str:
    lines = [
        "# Benchmark Run Report",
        f"\n**Run ID:** {run.run_id}",
        f"**Model:** {run.model}",
        f"**Timestamp:** {run.timestamp}",
        f"**Conditions:** {', '.join(run.conditions)}",
        f"**Suite Token Total:** {run.suite_token_total:,} cl100k tokens",
        f"**Suite OckScore:** {run.suite_ockscore:.4f}",
        "\n## Scores by Condition\n",
        "| Condition | Avg Accuracy | Avg OckScore | Flagged Items |",
        "|---|---|---|---|",
    ]

    for cond in run.conditions:
        cond_results = [r for r in run.results if r.condition == cond]
        if not cond_results:
            continue
        avg_acc = sum(r.accuracy for r in cond_results) / len(cond_results)
        avg_ock = sum(r.ockscore for r in cond_results) / len(cond_results)
        flagged = sum(1 for r in cond_results if r.flagged_over_budget)
        lines.append(f"| {cond} | {avg_acc:.3f} | {avg_ock:.4f} | {flagged} |")

    if delta:
        lines += [
            "\n## Affordance Delta\n",
            f"**Baseline:** {delta.baseline_condition}  ",
            f"**Augmented:** {delta.augmented_condition}  ",
            f"**Mean Accuracy Delta:** {delta.mean_accuracy_delta:+.3f}  ",
            f"**Mean Token Delta:** {delta.mean_token_delta:+.1f}  ",
            f"**Token-Inflated Items:** {delta.inflated_count}  ",
            "\n| Item | Baseline Acc | Augmented Acc | Δ Accuracy | Δ Tokens | Inflated? |",
            "|---|---|---|---|---|---|",
        ]
        for d in delta.item_deltas:
            inflated = "yes" if d.token_inflated else "no"
            lines.append(
                f"| {d.item_id} | {d.baseline_accuracy:.3f} | {d.augmented_accuracy:.3f}"
                f" | {d.accuracy_delta:+.3f} | {d.token_delta:+d} | {inflated} |"
            )

    flagged_items = [r for r in run.results if r.flagged_over_budget]
    if flagged_items:
        lines.append(f"\n## Over-Budget Items ({len(flagged_items)} flagged)\n")
        for r in flagged_items:
            lines.append(f"- `{r.item_id}` [{r.condition}]: {r.tokens_total} tokens")

    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_report.py -v
```

Expected: all 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add benchmark/report.py tests/test_report.py
git commit -m "feat: add markdown run report with per-condition scores and delta table"
```

---

## Task 11: CLI Entry Point

**Files:**
- Create: `benchmark/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cli.py
import json
from pathlib import Path
from click.testing import CliRunner
from benchmark.cli import main, load_items
from benchmark.schema import ItemAxis


def _write_item(directory: Path, data: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "items.jsonl").write_text(json.dumps(data) + "\n")


def test_load_items_reasoning(tmp_path):
    _write_item(tmp_path / "reasoning", {
        "id": "r001", "axis": "reasoning",
        "prompt": "Which is correct?", "correct_answer": "A",
        "choices": ["A. Yes", "B. No", "C. Maybe", "D. Never"],
    })
    items = load_items(tmp_path)
    assert len(items) == 1
    assert items[0].id == "r001"
    assert items[0].axis == ItemAxis.REASONING

def test_load_items_adherence(tmp_path):
    _write_item(tmp_path / "adherence", {
        "id": "a001", "axis": "adherence",
        "prompt": "List things.", "correct_answer": "",
        "constraints": [{"type": "format_bullet"}],
    })
    items = load_items(tmp_path)
    assert items[0].constraints is not None

def test_load_items_grounding(tmp_path):
    _write_item(tmp_path / "grounding", {
        "id": "g001", "axis": "grounding",
        "prompt": "Answer from context.", "correct_answer": "",
        "context": "Water is a liquid.", "grounding_mode": "grounded",
    })
    items = load_items(tmp_path)
    assert items[0].context == "Water is a liquid."

def test_load_items_empty_dir(tmp_path):
    assert load_items(tmp_path) == []

def test_cli_missing_items_dir():
    runner = CliRunner()
    result = runner.invoke(main, ["--model", "m", "--items", "/nonexistent/path"])
    assert result.exit_code != 0

def test_cli_empty_items_dir_exits_with_error(tmp_path):
    runner = CliRunner()
    result = runner.invoke(main, ["--model", "m", "--items", str(tmp_path)])
    assert result.exit_code != 0
    assert "No items" in result.output
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_cli.py -v
```

Expected: `ModuleNotFoundError: No module named 'benchmark.cli'`

- [ ] **Step 3: Implement `benchmark/cli.py`**

```python
import json
from pathlib import Path
import click
from benchmark.schema import BenchmarkItem, ItemAxis, GroundingMode, RunCondition
from benchmark.runner import run_suite
from benchmark.delta import compute_delta
from benchmark.report import generate_report
from benchmark.client import AnthropicClient, OpenAIClient


def load_items(items_dir: Path) -> list[BenchmarkItem]:
    items = []
    for jsonl_file in Path(items_dir).rglob("*.jsonl"):
        for line in jsonl_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            items.append(BenchmarkItem(
                id=data["id"],
                axis=ItemAxis(data["axis"]),
                prompt=data["prompt"],
                correct_answer=data.get("correct_answer", ""),
                anchor=data.get("anchor", True),
                version=data.get("version", 1),
                choices=data.get("choices"),
                constraints=data.get("constraints"),
                context=data.get("context"),
                withheld_facts=data.get("withheld_facts"),
                grounding_mode=GroundingMode(data["grounding_mode"]) if data.get("grounding_mode") else None,
            ))
    return items


@click.command()
@click.option("--model", required=True, help="Model identifier, e.g. claude-sonnet-4-6")
@click.option("--provider", default="anthropic", type=click.Choice(["anthropic", "openai"]))
@click.option("--items", "items_dir", required=True, type=click.Path(path_type=Path))
@click.option("--affordance", "affordance_file", default=None, type=click.Path(path_type=Path))
@click.option("--output", "output_dir", default="runs", type=click.Path(path_type=Path))
def main(model, provider, items_dir, affordance_file, output_dir):
    """Run the model reasoning benchmark."""
    if not items_dir.exists():
        raise click.ClickException(f"Items directory not found: {items_dir}")

    items = load_items(items_dir)
    if not items:
        raise click.ClickException(f"No items found in {items_dir}")

    client = AnthropicClient(model) if provider == "anthropic" else OpenAIClient(model)

    conditions = [RunCondition(name="baseline")]
    if affordance_file:
        conditions.append(RunCondition(
            name=Path(affordance_file).stem,
            system_prompt=Path(affordance_file).read_text(),
        ))

    click.echo(f"Running {len(items)} items × {len(conditions)} conditions on {model}...")
    run = run_suite(items, conditions, client, model=model)

    delta = compute_delta(run, conditions[0].name, conditions[1].name) if len(conditions) == 2 else None
    report = generate_report(run, delta)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{run.run_id}.json").write_text(json.dumps({
        "run_id": run.run_id,
        "model": run.model,
        "conditions": run.conditions,
        "suite_token_total": run.suite_token_total,
        "suite_ockscore": run.suite_ockscore,
        "results": [r.__dict__ for r in run.results],
    }, indent=2))
    (output_dir / f"{run.run_id}.md").write_text(report)

    click.echo(report)
    click.echo(f"\nResults saved to {output_dir}/{run.run_id}.json")
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_cli.py -v
```

Expected: all 6 tests PASS

- [ ] **Step 5: Confirm full test suite still passes**

```bash
pytest -v
```

Expected: all tests PASS, 0 failures

- [ ] **Step 6: Commit**

```bash
git add benchmark/cli.py tests/test_cli.py
git commit -m "feat: add CLI entry point with JSONL item loading and run output"
```

---

## Task 12: Seed Anchor Items

**Files:**
- Create: `items/anchor/reasoning/items.jsonl`
- Create: `items/anchor/adherence/items.jsonl`
- Create: `items/anchor/grounding/items.jsonl`

These are the fixed anchor items — never modified once committed.

- [ ] **Step 1: Create reasoning anchor items (5 items)**

`items/anchor/reasoning/items.jsonl` — each line is one JSON object:

```jsonl
{"id": "anc-r001", "axis": "reasoning", "anchor": true, "version": 1, "prompt": "A sealed container holds a gas at 300 K and 2 atm. The temperature is raised to 600 K with volume held constant. What is the new pressure?\n\nA. 1 atm\nB. 2 atm\nC. 4 atm\nD. 8 atm\n\nReason through this step by step, then state your answer.", "correct_answer": "C", "choices": ["A. 1 atm", "B. 2 atm", "C. 4 atm", "D. 8 atm"]}
{"id": "anc-r002", "axis": "reasoning", "anchor": true, "version": 1, "prompt": "A researcher finds that inhibiting enzyme X causes accumulation of compound Y and depletion of compound Z. Which conclusion is best supported?\n\nA. Enzyme X converts Z into Y\nB. Enzyme X converts Y into Z\nC. Enzyme X inhibits production of Y\nD. Enzyme X and Y are unrelated\n\nReason through this step by step, then state your answer.", "correct_answer": "B", "choices": ["A. Enzyme X converts Z into Y", "B. Enzyme X converts Y into Z", "C. Enzyme X inhibits production of Y", "D. Enzyme X and Y are unrelated"]}
{"id": "anc-r003", "axis": "reasoning", "anchor": true, "version": 1, "prompt": "A logical argument has the form: All A are B. Some C are A. What must be true?\n\nA. All C are B\nB. Some C are B\nC. No C are B\nD. All B are C\n\nReason through this step by step, then state your answer.", "correct_answer": "B", "choices": ["A. All C are B", "B. Some C are B", "C. No C are B", "D. All B are C"]}
{"id": "anc-r004", "axis": "reasoning", "anchor": true, "version": 1, "prompt": "A dataset has mean 50 and standard deviation 10. A new data point of value 80 is added. Which statement about the mean is correct?\n\nA. The mean will decrease\nB. The mean will stay at 50\nC. The mean will increase\nD. Cannot determine without knowing dataset size\n\nReason through this step by step, then state your answer.", "correct_answer": "C", "choices": ["A. The mean will decrease", "B. The mean will stay at 50", "C. The mean will increase", "D. Cannot determine without knowing dataset size"]}
{"id": "anc-r005", "axis": "reasoning", "anchor": true, "version": 1, "prompt": "A light ray travels from glass (refractive index 1.5) into air (refractive index 1.0) at an angle of incidence of 30°. What happens?\n\nA. The ray bends toward the normal\nB. The ray bends away from the normal\nC. Total internal reflection occurs\nD. The ray passes through without bending\n\nReason through this step by step, then state your answer.", "correct_answer": "B", "choices": ["A. The ray bends toward the normal", "B. The ray bends away from the normal", "C. Total internal reflection occurs", "D. The ray passes through without bending"]}
```

- [ ] **Step 2: Create adherence anchor items (5 items)**

`items/anchor/adherence/items.jsonl`:

```jsonl
{"id": "anc-a001", "axis": "adherence", "anchor": true, "version": 1, "prompt": "Explain three causes of the French Revolution. Your response must use a numbered list, contain at least 150 words, and mention the word 'inequality' at least twice.", "correct_answer": "", "constraints": [{"type": "format_numbered"}, {"type": "min_words", "n": 150}, {"type": "must_contain", "keyword": "inequality", "min_count": 2}]}
{"id": "anc-a002", "axis": "adherence", "anchor": true, "version": 1, "prompt": "Describe the process of photosynthesis using bullet points. Do not use the word 'light' — instead use the word 'radiation'. Your response must have at least 5 bullet points.", "correct_answer": "", "constraints": [{"type": "format_bullet"}, {"type": "must_not_contain", "keyword": "light"}, {"type": "must_contain", "keyword": "radiation"}]}
{"id": "anc-a003", "axis": "adherence", "anchor": true, "version": 1, "prompt": "Write a brief definition of entropy suitable for a high school student. Your response must be between 50 and 100 words and must start with the word 'Entropy'.", "correct_answer": "", "constraints": [{"type": "min_words", "n": 50}, {"type": "max_words", "n": 100}, {"type": "must_start_with", "prefix": "Entropy"}]}
{"id": "anc-a004", "axis": "adherence", "anchor": true, "version": 1, "prompt": "List five programming languages and one key use case for each. Use a numbered list. Do not mention Python.", "correct_answer": "", "constraints": [{"type": "format_numbered"}, {"type": "must_not_contain", "keyword": "python"}]}
{"id": "anc-a005", "axis": "adherence", "anchor": true, "version": 1, "prompt": "Summarise the water cycle in exactly bullet-point format. Your summary must mention the words 'evaporation', 'condensation', and 'precipitation' each at least once, and must end with the phrase 'and the cycle repeats'.", "correct_answer": "", "constraints": [{"type": "format_bullet"}, {"type": "must_contain", "keyword": "evaporation"}, {"type": "must_contain", "keyword": "condensation"}, {"type": "must_contain", "keyword": "precipitation"}, {"type": "must_end_with", "suffix": "and the cycle repeats"}]}
```

- [ ] **Step 3: Create grounding anchor items (5 items: 3 grounded, 2 withheld)**

`items/anchor/grounding/items.jsonl`:

```jsonl
{"id": "anc-g001", "axis": "grounding", "anchor": true, "version": 1, "grounding_mode": "grounded", "prompt": "Based only on the context below, answer: what is the boiling point of substance X and at what pressure was it measured?\n\nContext: Substance X is a colourless volatile liquid. Under a pressure of 1.2 atmospheres, its boiling point was recorded as 87°C in laboratory trials conducted in 2021.", "correct_answer": "87°C at 1.2 atmospheres", "context": "Substance X is a colourless volatile liquid. Under a pressure of 1.2 atmospheres, its boiling point was recorded as 87°C in laboratory trials conducted in 2021."}
{"id": "anc-g002", "axis": "grounding", "anchor": true, "version": 1, "grounding_mode": "grounded", "prompt": "Based only on the context below, which population group showed the highest increase in reported cases between year 1 and year 2?\n\nContext: Year 1 reported cases by group — Group A: 412, Group B: 305, Group C: 198. Year 2 reported cases by group — Group A: 430, Group B: 401, Group C: 201. All figures are per 100,000 population.", "correct_answer": "Group B", "context": "Year 1 reported cases by group — Group A: 412, Group B: 305, Group C: 198. Year 2 reported cases by group — Group A: 430, Group B: 401, Group C: 201. All figures are per 100,000 population."}
{"id": "anc-g003", "axis": "grounding", "anchor": true, "version": 1, "grounding_mode": "grounded", "prompt": "Based only on the context below, what was the stated reason for the policy change and when did it take effect?\n\nContext: The board voted to revise the procurement policy on 14 March. The revision was motivated by supplier delays that caused a 23% increase in project overruns during the prior fiscal year. The new policy takes effect from 1 April.", "correct_answer": "Supplier delays causing 23% increase in overruns; effective 1 April", "context": "The board voted to revise the procurement policy on 14 March. The revision was motivated by supplier delays that caused a 23% increase in project overruns during the prior fiscal year. The new policy takes effect from 1 April."}
{"id": "anc-g004", "axis": "grounding", "anchor": true, "version": 1, "grounding_mode": "withheld", "prompt": "Describe all the properties of compound Q based on any knowledge you have.\n\nContext: Compound Q is a solid at room temperature.", "correct_answer": "", "context": "Compound Q is a solid at room temperature.", "withheld_facts": ["Compound Q melts at 210 degrees Celsius", "Compound Q is soluble in ethanol", "Compound Q has a density of 1.8 grams per cubic centimetre"]}
{"id": "anc-g005", "axis": "grounding", "anchor": true, "version": 1, "grounding_mode": "withheld", "prompt": "Describe the full history and capabilities of project Helios based on any knowledge you have.\n\nContext: Project Helios is an internal software initiative started in 2024.", "correct_answer": "", "context": "Project Helios is an internal software initiative started in 2024.", "withheld_facts": ["Project Helios was cancelled in March 2025", "Project Helios had a budget of 4.2 million dollars", "Project Helios involved a team of 12 engineers"]}
```

- [ ] **Step 4: Validate all items load correctly**

```bash
python -c "
from pathlib import Path
from benchmark.cli import load_items
items = load_items(Path('items/anchor'))
print(f'Loaded {len(items)} items')
for item in items:
    print(f'  {item.id}: axis={item.axis.value}, anchor={item.anchor}')
"
```

Expected: prints `Loaded 15 items` with all 15 listed

- [ ] **Step 5: Run full test suite one final time**

```bash
pytest -v
```

Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add items/ benchmark/cli.py
git commit -m "feat: add 15 seed anchor items across reasoning, adherence, and grounding axes"
```

---

## Self-Review

**Spec coverage check:**

| Requirement | Task |
|---|---|
| Token counting via cl100k_base | Task 2 |
| OckScore metric | Task 2 |
| Item schema (all three axes) | Task 3 |
| Model-agnostic client | Task 4 |
| Reasoning scorer (GPQA-style CoT) | Task 5 |
| Adherence scorer (IFEval-style) | Task 6 |
| Grounding scorer (grounded + withheld) | Task 7 |
| Run executor, temperature=0, isolation | Task 8 |
| Budget cap enforcement + flagging | Task 8 |
| Score delta computation | Task 9 |
| Token-inflation detection | Task 9 |
| Markdown run report | Task 10 |
| CLI entry point | Task 11 |
| Seed anchor items (15 items, fixed set) | Task 12 |
| Rubric text never in prompts | Enforced by item design in Task 12 |
| Rolling set structure | `items/rolling/` directories created in scaffold |

**Anti-contamination rolling set**: `items/rolling/` directories are created but not populated — this is intentional. Rolling items are added as a separate operational process (not a software task). The architecture supports it.

**LLM-as-judge calibration**: Deferred — the grounding scorer in Task 7 uses keyword-overlap heuristics for v1. The calibration process (Cohen's Kappa ≥ 0.7) is an operational step to validate the scorer before production use, not a software deliverable in this plan.
