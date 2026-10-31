"""Small-sample uncertainty helpers for Pramaan's benchmark leaderboard."""

from __future__ import annotations

from math import sqrt


def wilson_interval(successes: int, total: int, *, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return the two-sided Wilson score interval for a binomial proportion."""
    if isinstance(successes, bool) or isinstance(total, bool):
        raise ValueError("successes and total must be integers")
    if not isinstance(successes, int) or not isinstance(total, int):
        raise ValueError("successes and total must be integers")
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("require 0 <= successes <= total")
    if total == 0:
        return (0.0, 1.0)
    if not isinstance(z, (int, float)) or z <= 0:
        raise ValueError("z must be positive")
    p = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denominator
    half = z * sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total) / denominator
    return (max(0.0, center - half), min(1.0, center + half))


def intervals_overlap(first: tuple[float, float], second: tuple[float, float]) -> bool:
    """Whether two closed confidence intervals share any point."""
    first_lo, first_hi = first
    second_lo, second_hi = second
    if first_lo > first_hi or second_lo > second_hi:
        raise ValueError("interval lower bounds must not exceed upper bounds")
    return first_lo <= second_hi and second_lo <= first_hi


def tie_badge(first: tuple[float, float], second: tuple[float, float]) -> str | None:
    """Return the UI badge for overlapping intervals; never infer a strict rank."""
    return "Intervals overlap" if intervals_overlap(first, second) else None
