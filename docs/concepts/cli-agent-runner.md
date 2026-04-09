# Concept: CLI Agent Runner

**Status**: READY
**Complexity**: COMPLEX
**Created**: 2026-04-09
**Updated**: 2026-04-09

---

## Summary

A benchmark execution layer that integrates with an agent CLI (such as Claude Code, but not exclusively) to run benchmark items through the CLI's own API call mechanism rather than making independent API calls. The runner operates within the CLI's chat session and executes two conditions: (1) a **baseline** sub-agent with a clean, stripped system prompt — no skills, MCP servers, plugins, or rules — and (2) an **affordance-augmented** sub-agent that loads the full current CLI configuration including active MCP servers, skills, and plugins. The runner introspects the loaded CLI context to identify which affordances are present and includes this metadata in the report. Benchmark validity is verified by running the tool across multiple machines and comparing scores for consistency.

---

## Accepted

- Benchmark items are dispatched via the CLI's native sub-agent mechanism, not via direct API calls from the runner
- The runner inherits the active model from the CLI session — the user selects the model in the CLI, not in the benchmark config
- **Model identity**: extracted from sub-agent response metadata (`model` field in provider API response) — authoritative, zero noise injected. Fallback: CLI config/env vars. Self-report via prompt is rejected.
- **Baseline floor**: CLI's unavoidable injections (e.g., safety preambles) are accepted as the baseline floor; they are consistent across machines and documented in the report. The delta is measured relative to this floor, not relative to a raw API call.
- **Baseline condition**: sub-agent spawned with a clean, stripped system prompt — no affordances of any kind
- **Affordance-augmented condition**: sub-agent spawned with the full current CLI configuration loaded
- The loaded context (MCP servers, skills, plugins, agent identity) is introspected and included in the run report as metadata
- Multiple models are benchmarked via separate runs; reports are generated separately per run/model
- **Cross-machine validity**: two-tier empirical approach — within-machine CoV <2% required before cross-machine test; cross-machine CoV <5% acceptable, 5–10% warning, >10% validity failure; N≥3 machines, 95% CI non-overlap = failure
- **CLI scope**: Claude Code only in v1; other CLIs deferred; no abstraction layer built now
- **Sub-agent prompt control**: full control assumed for Claude Code; CLIs without this are unsupported in v1

---

## Blocked

*(none yet)*

---

## Discarded

- **Option B (CLI-native floor as baseline)** — rejected; a truly stripped baseline is required for a clean delta signal
- **Option C (raw dispatch mode)** — not needed; sub-agent with stripped prompt achieves the same isolation

---

## Sub-concepts

### Condition Dispatch Layer
**Status**: READY
- Spawns two sub-agents per run: one with stripped prompt (baseline), one with full config (affordance-augmented)
- Full control over sub-agent system prompt is assumed for Claude Code
- Baseline floor (CLI's unavoidable injections) is consistent and documented, not eliminated

### Configuration Introspection
**Status**: READY
- At run start, enumerates what is active: MCP servers, skills, plugins, agent identity, model
- Model ID sourced from sub-agent response metadata (`model` field); fallback to CLI config
- Output is run report metadata only — not used to modify scoring

### Cross-Machine Validity Testing
**Status**: READY
- Within-machine: 5 runs, CoV <2% required to proceed
- Cross-machine: N≥3 machines, 95% CI comparison (Hoefler-Belli); CoV <5% = acceptable, 5–10% = warning, >10% = failure
- Temperature=0 means variance primarily reflects CLI environment drift, not model noise

### Report Generation
**Status**: READY
- One JSONL + markdown report per run, written to disk; summary surfaced in chat
- Tagged with: model identity, CLI type, active affordances, condition scores, score delta, OckScore
- Reports from multiple runs aggregatable for cross-machine validity analysis

---

## Open Questions

1. ~~**OQ1: CLI abstraction boundary**~~ — **RESOLVED (deferred).** Claude Code is the only supported CLI in v1. A named-integration model (one adapter per CLI) is the likely direction for future CLIs, but no abstraction layer is built now.
2. ~~**OQ2: Condition isolation**~~ — **RESOLVED.** Baseline = sub-agent with clean stripped system prompt. Affordance-augmented = sub-agent with full current config loaded.
3. ~~**OQ3: Model identity**~~ — **RESOLVED.** Primary: extract model ID from sub-agent response metadata (the `model` field in provider API responses — authoritative, zero noise). Fallback: read from CLI config/env vars if response metadata is not surfaced by the CLI's dispatch layer. Self-reporting via prompt (asking the model) is explicitly rejected — unreliable at version granularity and injects noise into the session context before benchmark items run.
4. ~~**OQ4: Affordance injection**~~ — **RESOLVED.** Baseline strips all affordances via sub-agent system prompt control. Augmented condition uses the CLI's loaded config as-is; introspection enumerates what's present.
5. ~~**OQ5: Result persistence**~~ — **RESOLVED.** Structured files written to disk (JSONL + markdown report, consistent with existing runner). Run summary also surfaced in the chat window. Both outputs are produced on every run.
6. ~~**OQ6: Cross-machine validity threshold**~~ — **RESOLVED.** Two-tier empirical approach (Hoefler-Belli methodology): (1) within-machine: run 5× at temperature=0, compute CoV — must be <2% before cross-machine testing; (2) cross-machine: N≥3 machines, compare 95% confidence intervals — CoV <5% = acceptable, 5–10% = warning flagged in report, >10% = validity failure. Non-overlapping 95% CIs between machines = failure. Thresholds are relative to empirical floor, not pre-declared fixed numbers. Temperature=0 mandate means most variance will reflect CLI environment drift, not model noise — itself a useful signal.
7. ~~**OQ7: Sub-agent prompt control**~~ — **RESOLVED.** Claude Code is the initial target; full sub-agent prompt control is assumed available. CLIs with insufficient control are deferred — no graceful degradation in v1.

---

## Ready for Architecture

**Concept summary**: A CLI-native benchmark runner for Claude Code (v1) that dispatches benchmark items through the CLI's own sub-agent mechanism rather than making independent API calls. Two conditions run per benchmark: (1) a baseline sub-agent with a stripped system prompt — only the CLI's unavoidable injections remain, accepted as the baseline floor — and (2) an affordance-augmented sub-agent with the full active CLI configuration (MCP servers, skills, plugins). Model identity is extracted from sub-agent response metadata, not self-reported. Configuration introspection enumerates active affordances as report metadata. Results are written to disk (JSONL + markdown) and summarized in chat. Cross-machine validity uses a two-tier Hoefler-Belli approach: within-machine CoV <2% gates cross-machine testing; cross-machine CoV <5% is acceptable, >10% is a validity failure.

**Key constraints**:
- Claude Code is the only supported CLI in v1 — no abstraction layer, no graceful degradation for other CLIs
- Model identity MUST come from sub-agent response metadata; self-report via prompt is prohibited
- Temperature=0 is mandatory for all conditions (inherited from parent benchmark concept)
- Within-machine CoV must be <2% before cross-machine validity testing is run
- Baseline floor = CLI's unavoidable injections; score delta is measured relative to this floor, not a raw API call

**Confirmed directions**:
- Sub-agent dispatch via Claude Code's native Agent tool mechanism
- Two-condition execution: stripped prompt baseline vs. full config affordance-augmented
- Hoefler-Belli two-tier CoV methodology for cross-machine validity (within-machine first, then cross-machine)
- JSONL + markdown report to disk, summary to chat
- Configuration introspection as metadata only (MCP servers, skills, plugins, model ID)

**Discarded alternatives**:
- Direct API calls from runner — rejected; CLI owns the transport
- Model self-report via prompt — rejected; unreliable at version granularity, injects noise
- CLI-native floor as the only baseline (no stripped condition) — rejected; clean delta signal requires stripped baseline
- Raw dispatch mode bypass — not needed; sub-agent prompt control achieves isolation
- Fixed declared variance thresholds — rejected in favor of empirical Hoefler-Belli approach

**Blocked (deferred)**:
- Multi-CLI support (other than Claude Code) — deferred to v2; named-integration model is the likely direction
- Graceful degradation for CLIs without sub-agent prompt control — deferred

**Open questions for the architect**:
- How does Claude Code's Agent tool expose (or not expose) the raw `model` field from the provider API response? If it doesn't surface it, what is the most reliable fallback?
- How should the configuration introspection enumerate MCP servers and skills — read from `~/.claude/settings.json` and skill directories, or query the live CLI session state?
- What is the invocation surface for the runner — a Claude Code slash command (`/benchmark`), a skill, or a CLI argument?
