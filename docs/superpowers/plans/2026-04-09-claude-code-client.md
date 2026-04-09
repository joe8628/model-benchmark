# Claude Code Client Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `ClaudeCodeClient` and `--provider claude-code` CLI option so the benchmark runs without an API key via the `claude -p` subprocess interface.

**Architecture:** New class in `benchmark/client.py` implementing the existing `ModelClient` protocol via `subprocess.run`. CLI gains `claude-code` as a valid provider choice and makes `--model` optional when that provider is selected.

**Tech Stack:** Python stdlib `subprocess`, `unittest.mock`, Click, existing `count_tokens()` from `benchmark/tokens.py`.

---

### Task 1: Add `ClaudeCodeClient` to `benchmark/client.py`

**Files:**
- Modify: `benchmark/client.py`

- [ ] **Step 1: Add the class**

Append after `OpenAIClient` in `benchmark/client.py`:

```python
import subprocess
from benchmark.tokens import count_tokens


class ClaudeCodeClient:
    def complete(self, prompt: str, system: str, temperature: float = 0.0) -> tuple[str, int, int]:
        cmd = ["claude", "-p", prompt]
        if system:
            cmd += ["--system", system]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(result.stderr)
        text = result.stdout.strip()
        return text, count_tokens(prompt), count_tokens(text)
```

Note: `import subprocess` and `from benchmark.tokens import count_tokens` go at the top of the file.

---

### Task 2: Write and run tests for `ClaudeCodeClient`

**Files:**
- Modify: `tests/test_client.py`

- [ ] **Step 1: Add the four tests**

Append to `tests/test_client.py`:

```python
from unittest.mock import patch
import subprocess
from benchmark.client import ClaudeCodeClient


def _ok_process(stdout="answer", stderr=""):
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr=stderr)

def _err_process(stderr="error msg"):
    return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr=stderr)


def test_claude_code_client_returns_response():
    with patch("subprocess.run", return_value=_ok_process("answer")) as mock_run:
        client = ClaudeCodeClient()
        text, _, _ = client.complete("q", "sys")
    assert text == "answer"

def test_claude_code_client_token_counts_are_ints():
    with patch("subprocess.run", return_value=_ok_process("answer")):
        client = ClaudeCodeClient()
        _, pt, ct = client.complete("q", "sys")
    assert isinstance(pt, int)
    assert isinstance(ct, int)

def test_claude_code_client_raises_on_nonzero_exit():
    with patch("subprocess.run", return_value=_err_process("bad")):
        client = ClaudeCodeClient()
        try:
            client.complete("q", "sys")
            assert False, "should have raised"
        except RuntimeError as e:
            assert "bad" in str(e)

def test_claude_code_client_omits_system_when_empty():
    with patch("subprocess.run", return_value=_ok_process()) as mock_run:
        client = ClaudeCodeClient()
        client.complete("q", "")
    args = mock_run.call_args[0][0]
    assert "--system" not in args
```

- [ ] **Step 2: Run tests**

```
python -m pytest tests/test_client.py -v
```

Expected: all tests pass including the 4 new ones.

---

### Task 3: Update `benchmark/cli.py`

**Files:**
- Modify: `benchmark/cli.py`

- [ ] **Step 1: Apply changes**

1. Add `ClaudeCodeClient` to the import line:
   ```python
   from benchmark.client import AnthropicClient, OpenAIClient, ClaudeCodeClient
   ```

2. Change `--model` to optional and add `claude-code` to `--provider`:
   ```python
   @click.option("--model", required=False, default=None, help="Model identifier, e.g. claude-sonnet-4-6")
   @click.option("--provider", default="anthropic", type=click.Choice(["anthropic", "openai", "claude-code"]))
   ```

3. Add validation at the top of `main()` body (before `items_dir.exists()` check):
   ```python
   if provider != "claude-code" and model is None:
       raise click.UsageError("--model is required for anthropic and openai providers")
   ```

4. Replace the client selection line:
   ```python
   # old:
   client = AnthropicClient(model) if provider == "anthropic" else OpenAIClient(model)
   # new:
   if provider == "claude-code":
       client = ClaudeCodeClient()
       model = "claude-code"
   elif provider == "anthropic":
       client = AnthropicClient(model)
   else:
       client = OpenAIClient(model)
   ```

---

### Task 4: Write and run CLI tests for `--provider claude-code`

**Files:**
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Add CLI tests**

Append to `tests/test_cli.py`:

```python
from unittest.mock import patch, MagicMock
import subprocess


def test_cli_model_required_for_anthropic():
    runner = CliRunner()
    result = runner.invoke(main, ["--provider", "anthropic", "--items", "/nonexistent"])
    assert result.exit_code != 0


def test_cli_model_required_for_openai():
    runner = CliRunner()
    result = runner.invoke(main, ["--provider", "openai", "--items", "/nonexistent"])
    assert result.exit_code != 0


def test_cli_claude_code_provider_no_model_needed(tmp_path):
    _write_item(tmp_path / "reasoning", {
        "id": "r001", "axis": "reasoning",
        "prompt": "Which?", "correct_answer": "A",
        "choices": ["A. Yes", "B. No", "C. Maybe", "D. Never"],
    })
    ok = subprocess.CompletedProcess(args=[], returncode=0, stdout="A", stderr="")
    with patch("subprocess.run", return_value=ok):
        runner = CliRunner()
        result = runner.invoke(main, [
            "--provider", "claude-code",
            "--items", str(tmp_path),
            "--output", str(tmp_path / "runs"),
        ])
    assert result.exit_code == 0, result.output
```

- [ ] **Step 2: Run full test suite**

```
python -m pytest -v
```

Expected: all tests pass, no regressions.
