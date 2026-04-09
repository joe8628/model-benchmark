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
