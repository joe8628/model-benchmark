from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class ItemAxis(str, Enum):
    REASONING = "reasoning"
    ADHERENCE = "adherence"
    GROUNDING = "grounding"


class GroundingMode(str, Enum):
    GROUNDED = "grounded"
    WITHHELD = "withheld"


@dataclass
class BenchmarkItem:
    id: str
    axis: ItemAxis
    prompt: str
    correct_answer: str
    anchor: bool = True
    version: int = 1
    # Reasoning axis
    choices: list[str] | None = None
    # Adherence axis
    constraints: list[dict] | None = None
    # Grounding axis
    context: str | None = None
    withheld_facts: list[str] | None = None
    grounding_mode: GroundingMode | None = None


@dataclass
class RunCondition:
    name: str
    system_prompt: str = ""


@dataclass
class ItemResult:
    item_id: str
    condition: str
    response: str
    tokens_prompt: int
    tokens_completion: int
    tokens_total: int
    accuracy: float
    ockscore: float
    flagged_over_budget: bool


@dataclass
class RunResult:
    run_id: str
    model: str
    conditions: list[str]
    results: list[ItemResult]
    timestamp: str
    suite_token_total: int = 0
    suite_ockscore: float = 0.0
