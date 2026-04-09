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
