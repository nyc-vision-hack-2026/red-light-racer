"""Pure scoring functions for Red Light Racer."""

from __future__ import annotations


def clamp_elapsed_ms(elapsed_ms: int | float) -> int:
    """Clamp guess elapsed time to the allowed window without erroring."""
    return int(max(150, min(30000, elapsed_ms)))


def compute_points(*, correct: bool, elapsed_ms: int | float, streak: int) -> int:
    """
    Score a single guess.

    streak is consecutive correct guesses *before* this one.
    """
    elapsed = clamp_elapsed_ms(elapsed_ms)
    base = 100 if correct else 0
    speed_bonus = round(50 * max(0, 1 - elapsed / 10000)) if correct else 0
    multiplier = min(1 + 0.25 * max(0, streak), 3.0)
    return round((base + speed_bonus) * multiplier)
