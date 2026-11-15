"""Small deterministic metrics used by the local and CI evaluation runner."""

from collections.abc import Sequence

GoldLocations = tuple[str, int] | Sequence[tuple[str, int]]


def _location_set(gold: GoldLocations) -> set[tuple[str, int]]:
    if isinstance(gold, tuple) and len(gold) == 2 and isinstance(gold[0], str):
        return {gold}
    return set(gold)


def hit_rate_at_k(results: Sequence[Sequence[dict]], expected: Sequence[GoldLocations]) -> float:
    """Fraction of queries with the gold source/page among the first k chunks."""
    if not expected:
        return 0.0
    hits = sum(
        any(
            (item.get("source"), int(item.get("page", 0))) in _location_set(gold) for item in ranked
        )
        for ranked, gold in zip(results, expected)
    )
    return hits / len(expected)


def mean_reciprocal_rank(
    results: Sequence[Sequence[dict]], expected: Sequence[GoldLocations]
) -> float:
    """Mean reciprocal rank of the first matching source/page, zero on a miss."""
    if not expected:
        return 0.0
    reciprocal_ranks = []
    for ranked, gold in zip(results, expected):
        rank = next(
            (
                index
                for index, item in enumerate(ranked, start=1)
                if (item.get("source"), int(item.get("page", 0))) in _location_set(gold)
            ),
            0,
        )
        reciprocal_ranks.append(1 / rank if rank else 0.0)
    return sum(reciprocal_ranks) / len(expected)


def rate(numerator: int, denominator: int) -> float:
    """Return a bounded rate, using zero when the evaluated set is empty."""
    return numerator / denominator if denominator else 0.0
