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
