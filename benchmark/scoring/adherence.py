import re
from typing import Protocol
from benchmark.schema import BenchmarkItem


class Constraint(Protocol):
    def check(self, response: str) -> bool: ...


class MinWordCount:
    def __init__(self, n: int):
        self.n = n
    def check(self, response: str) -> bool:
        return len(response.split()) >= self.n


class MaxWordCount:
    def __init__(self, n: int):
        self.n = n
    def check(self, response: str) -> bool:
        return len(response.split()) <= self.n


class MustContainKeyword:
    def __init__(self, keyword: str, min_count: int = 1):
        self.keyword = keyword.lower()
        self.min_count = min_count
    def check(self, response: str) -> bool:
        return response.lower().count(self.keyword) >= self.min_count


class MustNotContainKeyword:
    def __init__(self, keyword: str):
        self.keyword = keyword.lower()
    def check(self, response: str) -> bool:
        return self.keyword not in response.lower()


class MustStartWith:
    def __init__(self, prefix: str):
        self.prefix = prefix
    def check(self, response: str) -> bool:
        return response.strip().startswith(self.prefix)


class MustEndWith:
    def __init__(self, suffix: str):
        self.suffix = suffix
    def check(self, response: str) -> bool:
        return response.strip().endswith(self.suffix)


class MustUseBulletFormat:
    def check(self, response: str) -> bool:
        lines = [ln.strip() for ln in response.splitlines() if ln.strip()]
        return sum(1 for ln in lines if re.match(r'^[-*•]', ln)) >= 3


class MustUseNumberedFormat:
    def check(self, response: str) -> bool:
        lines = [ln.strip() for ln in response.splitlines() if ln.strip()]
        return sum(1 for ln in lines if re.match(r'^\d+[.)]\s', ln)) >= 3


_REGISTRY = {
    "min_words":        lambda d: MinWordCount(d["n"]),
    "max_words":        lambda d: MaxWordCount(d["n"]),
    "must_contain":     lambda d: MustContainKeyword(d["keyword"], d.get("min_count", 1)),
    "must_not_contain": lambda d: MustNotContainKeyword(d["keyword"]),
    "must_start_with":  lambda d: MustStartWith(d["prefix"]),
    "must_end_with":    lambda d: MustEndWith(d["suffix"]),
    "format_bullet":    lambda d: MustUseBulletFormat(),
    "format_numbered":  lambda d: MustUseNumberedFormat(),
}


def build_constraint(spec: dict):
    factory = _REGISTRY.get(spec["type"])
    if factory is None:
        raise ValueError(f"Unknown constraint type: {spec['type']!r}")
    return factory(spec)


def score_adherence(item: BenchmarkItem, response: str) -> float:
    """Returns fraction of constraints satisfied (0.0–1.0)."""
    if not item.constraints:
        return 1.0
    constraints = [build_constraint(spec) for spec in item.constraints]
    passed = sum(1 for c in constraints if c.check(response))
    return passed / len(constraints)
