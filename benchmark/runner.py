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
