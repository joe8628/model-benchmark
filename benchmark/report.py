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
