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
