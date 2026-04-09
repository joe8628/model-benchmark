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
