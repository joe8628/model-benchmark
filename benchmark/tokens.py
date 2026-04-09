import math
import tiktoken

_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def ockscore(accuracy: float, actual_tokens: int, budget: int = 2048) -> float:
    """Logarithmically penalizes verbosity while prioritizing correctness.

    Formula: accuracy × log(budget / actual_tokens + 1)
    Returns 0.0 when accuracy is 0.0 or actual_tokens is 0.
    """
    if actual_tokens <= 0 or accuracy == 0.0:
        return 0.0
    return accuracy * math.log(budget / actual_tokens + 1)
