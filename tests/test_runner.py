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
