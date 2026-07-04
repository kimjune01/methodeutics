"""Likelihood-ratio models: each returns a callable lr(x) = P_alt(x) / P_null(x).

Any callable with that signature works as a Bankroll model; these are the
three a first course needs.
"""

import math
from typing import Callable, Sequence


def normal(null_mean: float, alt_mean: float, sd: float = 1.0) -> Callable[[float], float]:
    """Gaussian mean test with known sd. One observation's likelihood ratio:
    exp(delta * (x - null_mean) / sd^2 - delta^2 / (2 sd^2)), delta = alt - null.
    """
    if sd <= 0:
        raise ValueError("sd must be positive")
    delta = alt_mean - null_mean

    def lr(x: float) -> float:
        z = x - null_mean
        return math.exp((delta * z - delta * delta / 2) / (sd * sd))

    return lr


def bernoulli(p0: float, p1: float) -> Callable[[int], float]:
    """Success-rate test. Observation is truthy (success) or falsy (failure)."""
    for name, p in (("p0", p0), ("p1", p1)):
        if not 0 < p < 1:
            raise ValueError(f"{name} must be strictly between 0 and 1, got {p}")

    def lr(x) -> float:
        return p1 / p0 if x else (1 - p1) / (1 - p0)

    return lr


def categorical(null: Sequence[float], alt: Sequence[float]) -> Callable[[int], float]:
    """K-outcome test (dice, funnels, multinomial checks). Observation is the
    outcome's index. Both distributions must sum to 1; null must never assign
    zero to a possible outcome (the bet's odds would be infinite).
    """
    if len(null) != len(alt):
        raise ValueError("null and alt must have the same length")
    for name, dist in (("null", null), ("alt", alt)):
        if abs(sum(dist) - 1.0) > 1e-9:
            raise ValueError(f"{name} probabilities sum to {sum(dist)}, not 1")
    if any(p <= 0 for p in null):
        raise ValueError("null must assign positive probability to every outcome")

    ratios = [a / n for n, a in zip(null, alt)]

    def lr(x: int) -> float:
        return ratios[x]

    return lr
