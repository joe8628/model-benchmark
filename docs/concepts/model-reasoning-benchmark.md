# Concept: Model Reasoning Benchmark with Affordance Delta Measurement

**Status**: READY
**Complexity**: COMPLEX
**Created**: 2026-04-08
**Updated**: 2026-04-08

---

## Summary

Most existing LLM benchmarks measure raw capability on fixed tasks, but cannot distinguish whether a model's reasoning quality is intrinsic or stack-derived — that is, whether a model reasons well on its own or only when guided by injected plugins, skills, or rules. This benchmark addresses that gap by running identical reasoning items across two conditions for any single model under test: a bare baseline condition (no affordances) and one or more affordance-augmented conditions. The primary output is a **score delta** — the difference in reasoning quality between conditions — which reveals how much of the model's apparent capability is contributed by the surrounding stack versus the model itself.

The benchmark evaluates three discrimination axes — genuine reasoning versus memorized templates, rule adherence versus rule-skipping, and grounded answers versus hallucination — using established scoring techniques (GPQA-style multiple-choice with chain-of-thought parsing, IFEval-style verifiable instruction checking, and FActScore-style atomic fact decomposition). Token efficiency is a first-class constraint enforced via an adapted OckScore metric using cl100k_base token counting as a tokenizer-agnostic unit. Anti-contamination is maintained through a rolling item refresh inspired by LiveBench, anchored to a fixed set for longitudinal comparison.

---

## Accepted

- Token budget is a first-class design constraint, not a secondary concern
- Token counting unit: **cl100k_base tokens** (tokenizer-agnostic proxy); provider-native counts logged alongside for reference
- Per-item ceiling: **2,048 cl100k tokens** (prompt + completion combined); starting point, subject to reduction
- Suite-level ceiling: **600K cl100k tokens** per full run
- Rubric and scoring text: **amortized across the suite**, never charged per item, never included in item prompts
- Token efficiency metric: **OckScore** (logarithmic verbosity penalty, prioritizes correctness) adapted to cl100k_base counts
- Benchmark is **model-agnostic**; one model under test per run
- Evaluation is **on-policy, multi-condition**: same items run against (a) bare model, (b) model + each affordance configuration
- Score delta (`Score(model+stack) − Score(baseline)`) is the primary signal; it is within-model, not cross-model
- Affordances are **experimental conditions**, not scoring dimensions; the benchmark does not reward affordance usage
- Scoring is **automatic-first**:
  - Reasoning axis: GPQA-style multiple-choice answer parsing with zero-shot chain-of-thought
  - Rule adherence axis: IFEval-style programmatic verification of verifiable instructions
  - Grounding axis: FActScore-style atomic fact decomposition, two modes (grounded-mode, withheld-mode)
- LLM-as-judge permitted **only as fallback** for grounding edge cases; MUST achieve Cohen's Kappa ≥ 0.7 against a human calibration set before use
- Human review capped at **<5% of items**; used only where auto-scoring is undefined
- Anti-contamination: **rolling item refresh** (new items sourced from post-training-cutoff material) plus a **fixed anchor set** never refreshed, enabling longitudinal comparison
- Rubric text MUST NEVER appear in item prompts; distractors MUST be included that match rubric surface form but lead to wrong conclusions, to detect keyword surfing
- Temperature MUST be fixed to 0 (or lowest deterministic setting) for all conditions to minimize score variance
- Each condition MUST run in a fully isolated session with no shared state

---

## Blocked

*(none — all open questions resolved)*

---

## Discarded

- **Scoring based on affordance usage count** — benchmark is agnostic to how many affordances are invoked
- **Offline / cached-completion design** — rejected in favor of on-policy multi-condition runs
- **TruthfulQA** — saturated, training-contaminated, deprecated for frontier use
- **MMLU / GSM8K / HellaSwag as reasoning components** — saturated at 95–99% for frontier models; not discriminative
- **Provider-native token counting** — cannot be the canonical unit in a model-agnostic benchmark
- **Inventing a new token efficiency metric** — OckScore covers this; adopt and cite (arXiv 2511.05722)
- **Cross-model comparison as primary design goal** — supported only as a derived use case

---

## Sub-concepts

### Scoring System
**Status**: READY

Three axes, each with a distinct automatic scoring method:

1. **Reasoning vs. template** — GPQA-style items: expert-authored multiple-choice questions requiring multi-step chain-of-thought. Scored by answer parsing (automatic). Items MUST expose failure on distractor options that match surface reasoning patterns without correct conclusions.
2. **Rule adherence vs. skipping** — IFEval-style items: prompts with one or more verifiable instructions (format, length, keyword constraints). Scored programmatically; no human judgment required.
3. **Grounding vs. hallucination** — FActScore-style atomic decomposition in two modes:
   - *Grounded-mode*: all relevant facts provided in context; atomic claims scored against provided context only
   - *Withheld-mode*: facts intentionally omitted; claims about withheld facts are penalized

LLM-as-judge fallback: documented CoT rubric, randomized response position, inter-judge Kappa threshold ≥ 0.7.

---

### Token Budget Framework
**Status**: READY

- Counting unit: cl100k_base tokens
- Per-item cap: 2,048 tokens (prompt + completion); items exceeding cap are truncated and flagged, not silently scored
- Suite cap: 600K tokens per full run
- Rubric overhead: logged once per run, not charged per item
- OckScore per item: `S = Accuracy × log(budget / actual_tokens + 1)` — prioritizes correctness, penalizes verbosity logarithmically
- Suite OckScore: mean of per-item scores

---

### Affordance Discrimination Layer
**Status**: READY

- Each benchmark run executes two passes on identical items: baseline (no affordances) and affordance-augmented (plugins/skills/rules injected via system prompt segments or tool definitions)
- Token delta between conditions is logged; score improvement MUST be normalized against token delta — score gains driven purely by longer prompts are flagged
- Score delta is the primary affordance signal; a positive delta with a non-positive token delta is the target signature of a genuinely helpful affordance

---

### Anti-Contamination
**Status**: READY

- Rolling set: new items added periodically, sourced from material published after the model's training cutoff
- Fixed anchor set: never modified; used for longitudinal score comparison across benchmark versions
- Watermarking: item reformulation with a marked LLM before release to detect training leakage
- Items MUST NOT reproduce verbatim text from any known training corpus

---

### Grounding (Provided / Withheld Facts)
**Status**: EXPLORING

This is the novel component not covered by existing benchmarks. Design is clear conceptually but item construction methodology needs validation:
- Grounded-mode items must be carefully authored so that the correct answer is fully derivable from provided context, with no reliance on world knowledge
- Withheld-mode items must be validated to confirm that the withheld fact is genuinely absent from the provided context and not inferable
- Item authoring protocol: expert author + independent validator confirms each mode's integrity before inclusion

---

## Open Questions

*(all resolved — none remaining)*

---

## Ready for Architecture

**Concept summary**: A model-agnostic, on-policy reasoning benchmark that runs identical items under two conditions — bare baseline and affordance-augmented — for any single model under test. The primary output is a within-model score delta revealing how much reasoning quality is stack-derived versus intrinsic. Three scoring axes (reasoning, rule adherence, grounding) use established automatic methods. Token efficiency is enforced via OckScore with cl100k_base counting. Anti-contamination uses rolling refresh plus a fixed anchor set.

**Key constraints**:
- Per-item token ceiling: 2,048 cl100k tokens (hard cap, flagged not silently truncated)
- Suite ceiling: 600K cl100k tokens per run
- Scoring MUST be automatic-first; human review <5%
- Temperature MUST be fixed to 0 for all conditions
- Rubric text MUST NEVER appear in item prompts
- LLM-as-judge fallback requires Cohen's Kappa ≥ 0.7 calibration before use

**Confirmed directions**:
- OckScore for token efficiency (adapt arXiv 2511.05722 to cl100k_base)
- IFEval-style verifiable instructions for rule adherence
- GPQA-style multiple-choice + CoT for reasoning
- FActScore atomic decomposition (grounded-mode + withheld-mode) for grounding
- LiveBench rolling refresh + watermarking for anti-contamination
- Score delta normalized against token delta to detect affordance gaming

**Discarded alternatives**: TruthfulQA, MMLU, GSM8K, offline evaluation, cross-model primary design, provider-native token counting, custom efficiency metrics

**Blocked / deferred**: Item authoring protocol for withheld-mode grounding items (needs validation pass before production)

**Open questions for the architect**:
- What is the minimum viable item count per axis for statistical power on the score delta signal?
- How are affordance configurations versioned and diffed between benchmark runs?
- What is the storage and execution model for isolated per-condition sessions at scale?
