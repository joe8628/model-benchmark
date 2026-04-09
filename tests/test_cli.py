import json
from pathlib import Path
from click.testing import CliRunner
from benchmark.cli import main, load_items
from benchmark.schema import ItemAxis


def _write_item(directory: Path, data: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "items.jsonl").write_text(json.dumps(data) + "\n")


def test_load_items_reasoning(tmp_path):
    _write_item(tmp_path / "reasoning", {
        "id": "r001", "axis": "reasoning",
        "prompt": "Which is correct?", "correct_answer": "A",
        "choices": ["A. Yes", "B. No", "C. Maybe", "D. Never"],
    })
    items = load_items(tmp_path)
    assert len(items) == 1
    assert items[0].id == "r001"
    assert items[0].axis == ItemAxis.REASONING

def test_load_items_adherence(tmp_path):
    _write_item(tmp_path / "adherence", {
        "id": "a001", "axis": "adherence",
        "prompt": "List things.", "correct_answer": "",
        "constraints": [{"type": "format_bullet"}],
    })
    items = load_items(tmp_path)
    assert items[0].constraints is not None

def test_load_items_grounding(tmp_path):
    _write_item(tmp_path / "grounding", {
        "id": "g001", "axis": "grounding",
        "prompt": "Answer from context.", "correct_answer": "",
        "context": "Water is a liquid.", "grounding_mode": "grounded",
    })
    items = load_items(tmp_path)
    assert items[0].context == "Water is a liquid."

def test_load_items_empty_dir(tmp_path):
    assert load_items(tmp_path) == []

def test_cli_missing_items_dir():
    runner = CliRunner()
    result = runner.invoke(main, ["--model", "m", "--items", "/nonexistent/path"])
    assert result.exit_code != 0

def test_cli_empty_items_dir_exits_with_error(tmp_path):
    runner = CliRunner()
    result = runner.invoke(main, ["--model", "m", "--items", str(tmp_path)])
    assert result.exit_code != 0
    assert "No items" in result.output
