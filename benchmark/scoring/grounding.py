import re
from benchmark.schema import BenchmarkItem, GroundingMode

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "in", "of",
    "to", "and", "or", "it", "that", "this", "at", "by", "with",
    "on", "for", "be", "as", "its", "which", "has", "have",
}


def atomize(text: str) -> list[str]:
    """Split text into atomic sentence-level claims."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _keyword_overlap(claim: str, context: str) -> float:
    """Fraction of significant claim words found in context."""
    claim_words = {
        w.lower() for w in re.findall(r'\w+', claim)
        if w.lower() not in _STOPWORDS and len(w) > 2
    }
    if not claim_words:
        return 1.0
    context_words = {w.lower() for w in re.findall(r'\w+', context)}
    return len(claim_words & context_words) / len(claim_words)


def score_grounded_mode(
    claims: list[str], context: str, threshold: float = 0.5
) -> float:
    """Fraction of claims with keyword overlap >= threshold against provided context."""
    if not claims:
        return 1.0
    supported = sum(1 for c in claims if _keyword_overlap(c, context) >= threshold)
    return supported / len(claims)


def _claim_reveals_fact(claim: str, fact: str, min_matches: int = 2) -> bool:
    """True when a claim contains >= min_matches keywords from a withheld fact."""
    fact_kw = {
        w.lower() for w in re.findall(r'\w+', fact)
        if len(w) > 3 and w.lower() not in _STOPWORDS
    }
    if not fact_kw:
        return False
    claim_lower = claim.lower()
    return sum(1 for kw in fact_kw if kw in claim_lower) >= min_matches


def score_withheld_mode(claims: list[str], withheld_facts: list[str]) -> float:
    """Penalizes claims that reveal withheld facts (>= 2 keyword matches required).
    Returns 1.0 - (fraction of claims revealing any withheld fact).
    """
    if not claims or not withheld_facts:
        return 1.0
    hallucinated = sum(
        1 for claim in claims
        if any(_claim_reveals_fact(claim, fact) for fact in withheld_facts)
    )
    return 1.0 - (hallucinated / len(claims))


def score_grounding(item: BenchmarkItem, response: str) -> float:
    claims = atomize(response)
    if item.grounding_mode == GroundingMode.GROUNDED:
        return score_grounded_mode(claims, item.context or "")
    if item.grounding_mode == GroundingMode.WITHHELD:
        return score_withheld_mode(claims, item.withheld_facts or [])
    return 0.0
