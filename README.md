# model-benchmark

A model-agnostic benchmark suite that measures how much of a model's reasoning quality is intrinsic versus contributed by the surrounding stack — prompt engineering, system prompts, injected skills, and other affordances.

---

## What this is

Most LLM benchmarks measure raw capability on fixed tasks, but they cannot tell you whether a model reasons well on its own or only when guided by a carefully engineered stack. This benchmark addresses that gap by running identical items against the same model under two conditions: a bare baseline (no system prompt, no affordances) and an affordance-augmented condition of your choosing. The primary output is a **score delta** — the difference in reasoning quality between conditions — which shows how much the stack is contributing versus the model itself.

Items are evaluated across three axes: **reasoning** (multi-step chain-of-thought problems), **rule adherence** (verifiable instruction-following), and **grounding** (fact-grounded answers vs. hallucination). Token efficiency is a first-class constraint; every run reports token totals and per-item budget flags using cl100k_base as a tokenizer-agnostic unit.

---

## How it works

Each run executes every benchmark item in fully isolated sessions — one per condition — so no state leaks between them. Temperature is fixed at 0 throughout to minimise score variance. Each item is scored automatically: reasoning items are parsed for a final answer letter, adherence items are checked programmatically against their constraints, and grounding items are decomposed into atomic claims and matched against the provided context (or, in withheld mode, penalised for hallucinating facts that were deliberately omitted).

After all conditions complete, the runner computes an **OckScore** per item — a logarithmic verbosity-penalised accuracy metric — and aggregates results into a run report. When two conditions are present, it also computes a delta table showing per-item accuracy change, token change, and whether any accuracy gain came at the cost of inflated token usage.

---

## Prerequisites

- Python 3.11 or higher
- An Anthropic or OpenAI API key set in your environment:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
# or
export OPENAI_API_KEY=sk-...
```

---

## Installation

```bash
git clone https://github.com/your-org/model-benchmark.git
cd model-benchmark
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

---

## Running a baseline-only benchmark

The simplest run evaluates a model against a set of items with no system prompt. This gives you the model's unassisted performance across all three axes.

```bash
benchmark \
  --model claude-sonnet-4-6 \
  --provider anthropic \
  --items items/anchor
```

The CLI loads every `.jsonl` file found recursively under `items/anchor`, runs each item once under a single `baseline` condition, scores results automatically, and writes a run report to `runs/`. You will see output like:

```
Running 15 items × 1 conditions on claude-sonnet-4-6...

# Benchmark Run Report

**Run ID:** run-20260409-143201
**Model:** claude-sonnet-4-6
**Conditions:** baseline
**Suite Token Total:** 18,432 cl100k tokens
**Suite OckScore:** 0.7841

## Scores by Condition

| Condition | Avg Accuracy | Avg OckScore | Flagged Items |
|---|---|---|---|
| baseline | 0.800 | 0.7841 | 0 |

Results saved to runs/run-20260409-143201.json
```

Two files are written to `runs/`: a `.json` with the full item-level data and a `.md` with the human-readable report.

---

## Running a two-condition delta run

To measure what your affordance stack actually contributes, pass an affordance file containing the system prompt you want to test. The benchmark runs every item twice — once baseline, once with your system prompt — then produces a delta table.

First, write your affordance system prompt to a file:

```bash
cat > affordances/my-stack.txt <<'EOF'
You are a careful, methodical reasoner. Before answering any question, restate the key facts, identify what is being asked, and reason step by step. Only then give your final answer.
EOF
```

Then run with both conditions:

```bash
benchmark \
  --model claude-sonnet-4-6 \
  --provider anthropic \
  --items items/anchor \
  --affordance affordances/my-stack.txt
```

The affordance file's stem (`my-stack`) becomes the condition name. The report will include an **Affordance Delta** section:

```
## Affordance Delta

**Baseline:** baseline
**Augmented:** my-stack
**Mean Accuracy Delta:** +0.133
**Mean Token Delta:** +312.4
**Token-Inflated Items:** 3

| Item       | Baseline Acc | Augmented Acc | Δ Accuracy | Δ Tokens | Inflated? |
|---|---|---|---|---|---|
| anc-r001   | 1.000        | 1.000         | +0.000     | +284     | no        |
| anc-r002   | 0.000        | 1.000         | +1.000     | +401     | yes       |
| ...
```

A **token-inflated** item is one where accuracy improved but token usage also increased — a signal that the affordance is trading verbosity for correctness rather than making the model intrinsically better.

---

## Output files

Each run writes two files to the `runs/` directory (created automatically):

- **`<run_id>.json`** — Full structured results: every item, every condition, token counts, accuracy, OckScore, and budget flags. Use this for programmatic analysis or longitudinal tracking.
- **`<run_id>.md`** — The human-readable report shown in the terminal. Includes the scores table, delta table (if two conditions), and a list of any over-budget items.

---

## Project layout

```
model-benchmark/
├── benchmark/
│   ├── cli.py          # Click entry point — parses args, orchestrates the run
│   ├── schema.py       # BenchmarkItem, RunCondition, ItemResult, RunResult dataclasses
│   ├── client.py       # ModelClient protocol + AnthropicClient / OpenAIClient
│   ├── runner.py       # run_item(), run_suite() — isolated per-condition execution
│   ├── delta.py        # compute_delta() — accuracy and token delta, inflation detection
│   ├── report.py       # generate_report() — markdown report generation
│   ├── tokens.py       # cl100k_base token counting + OckScore formula
│   └── scoring/
│       ├── reasoning.py   # CoT answer parsing (GPQA-style)
│       ├── adherence.py   # Constraint verification (IFEval-style)
│       └── grounding.py   # Atomic claim decomposition (FActScore-style)
├── items/
│   ├── anchor/         # Fixed items — never modified once committed
│   │   ├── reasoning/items.jsonl
│   │   ├── adherence/items.jsonl
│   │   └── grounding/items.jsonl
│   └── rolling/        # Refreshed periodically to prevent contamination
├── tests/              # pytest suite — 102 tests, all axes and components covered
└── pyproject.toml
```
