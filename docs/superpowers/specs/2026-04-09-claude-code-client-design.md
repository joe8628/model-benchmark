# Design: Claude Code Client (API-Key-Free Provider)

**Status:** APPROVED  
**Created:** 2026-04-09  

---

## Problem

The benchmark CLI currently requires an Anthropic or OpenAI API key. Users running the benchmark from within a Claude Code session already have a live, authenticated model available via the `claude` CLI binary. There is no way to reuse that auth without a separate API key.

---

## Goal

Add a `--provider claude-code` option that runs benchmark items against the active Claude Code session's model using the `claude -p` subprocess interface — no API key required.

---

## Design

### New: `ClaudeCodeClient` in `benchmark/client.py`

A new class implementing the existing `ModelClient` protocol:

```python
class ClaudeCodeClient:
    def complete(self, prompt: str, system: str, temperature: float = 0.0) -> tuple[str, int, int]:
        ...
```

**Behaviour:**
- Calls `subprocess.run(["claude", "-p", prompt, "--system", system], capture_output=True, text=True, timeout=120)`
- If `system` is empty, omits `--system` flag entirely
- If the subprocess exits with a non-zero return code, raises `RuntimeError` with stderr content
- If `claude` is not found in PATH, the `FileNotFoundError` from subprocess propagates with no wrapping (the message is already clear)
- Returns `(stdout.strip(), estimated_prompt_tokens, estimated_completion_tokens)` where both token counts are estimated via `count_tokens()` from `benchmark/tokens.py` — the CLI does not return usage data
- `temperature` parameter is accepted but ignored (the `claude` CLI does not expose a temperature flag; benchmark always runs at 0 anyway)

### CLI changes in `benchmark/cli.py`

- `--provider` gains `claude-code` as a valid choice: `type=click.Choice(["anthropic", "openai", "claude-code"])`
- `--model` becomes optional (`required=False`, `default=None`)
- Validation: if `provider != "claude-code"` and `model is None`, raise `click.UsageError("--model is required for anthropic and openai providers")`
- Client selection adds a third branch:
  ```python
  if provider == "claude-code":
      client = ClaudeCodeClient()
  elif provider == "anthropic":
      client = AnthropicClient(model)
  else:
      client = OpenAIClient(model)
  ```
- `run.model` is set to `"claude-code"` when provider is `claude-code`

### Tests in `tests/test_client.py`

New test group for `ClaudeCodeClient` using `unittest.mock.patch("subprocess.run", ...)`:

- `test_claude_code_client_returns_response` — mock returns `CompletedProcess(stdout="answer", returncode=0)`; assert response text is `"answer"`
- `test_claude_code_client_token_counts_are_ints` — assert both token counts are integers
- `test_claude_code_client_raises_on_nonzero_exit` — mock returns `returncode=1, stderr="error"`; assert `RuntimeError` raised
- `test_claude_code_client_omits_system_when_empty` — assert `--system` not in subprocess args when `system=""`

---

## What does NOT change

- Scoring, delta, report, runner, schema, tokens — no changes
- Existing `AnthropicClient` and `OpenAIClient` — no changes
- Item JSONL format — no changes
- All 102 existing tests continue to pass

---

## Token count note

Because `claude -p` does not return token usage, both prompt and completion token counts are estimated via `count_tokens()` (cl100k_base). This means per-item token totals and OckScores are approximate when using the `claude-code` provider. The run report will reflect this accurately since it uses the same estimated counts throughout.
