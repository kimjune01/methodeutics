"""The Bankroll: an e-value trajectory with a pre-committed threshold."""

import math
from typing import Callable, List, Optional


class Bankroll:
    """A running bet against the null hypothesis.

    Starts at one dollar. Each observation multiplies the bankroll by the
    model's likelihood ratio. The bankroll's value is the e-value: the
    factor by which the data has beaten the null's odds so far.

    alpha sets the pre-committed threshold 1/alpha. By Ville's inequality
    the bankroll crosses it with probability at most alpha if the null is
    true, no matter when or how often you look. A crossing is remembered
    (`decided`, `decided_at`) even if the trajectory later collapses;
    acting on the first crossing is what the guarantee prices.
    """

    def __init__(self, model: Callable[..., float], alpha: float = 0.05):
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be strictly between 0 and 1, got {alpha}")
        self.model = model
        self.alpha = alpha
        self._log_e = 0.0
        self._log_trajectory: List[float] = []
        self.decided_at: Optional[int] = None

    @property
    def e(self) -> float:
        """Current e-value: the bankroll, in dollars per starting dollar."""
        return math.exp(self._log_e)

    @property
    def n(self) -> int:
        """Observations seen."""
        return len(self._log_trajectory)

    @property
    def threshold(self) -> float:
        """The pre-committed decision line, 1/alpha."""
        return 1 / self.alpha

    @property
    def trajectory(self) -> List[float]:
        """E-value after each observation."""
        return [math.exp(v) for v in self._log_trajectory]

    @property
    def decided(self) -> bool:
        """True if the bankroll has ever crossed the threshold."""
        return self.decided_at is not None

    def update(self, x) -> float:
        """Feed one observation; returns the new e-value."""
        ratio = self.model(x)
        if ratio < 0:
            raise ValueError(f"model returned a negative likelihood ratio: {ratio}")
        self._log_e += math.log(ratio) if ratio > 0 else -math.inf
        self._log_trajectory.append(self._log_e)
        if self.decided_at is None and self._log_e >= math.log(self.threshold):
            self.decided_at = self.n - 1
        return self.e

    def extend(self, xs) -> "Bankroll":
        """Feed many observations; returns self for chaining."""
        for x in xs:
            self.update(x)
        return self

    def compose(self, other: "Bankroll") -> float:
        """Combined evidence from two independent bankrolls: the product.

        Two honest bets composed are still an honest bet; no correction
        for multiple testing is needed.
        """
        return math.exp(self._log_e + other._log_e)

    def plot(self, ax=None):
        """Plot the trajectory on a log scale with the break-even and
        threshold lines. Requires the [plot] extra."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as err:
            raise ImportError(
                "plotting requires matplotlib: pip install methodeutics[plot]"
            ) from err

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))
        ts = range(1, self.n + 1)
        ax.plot(ts, self.trajectory, linewidth=1.8)
        ax.axhline(1, linestyle="--", linewidth=1, color="gray", label="break even (e = 1)")
        ax.axhline(
            self.threshold, linestyle=":", linewidth=1, color="tab:purple",
            label=f"threshold (e = {self.threshold:g})",
        )
        if self.decided:
            ax.axvline(self.decided_at + 1, linestyle=":", linewidth=1, color="tab:red")
            ax.annotate(
                f"decided at t = {self.decided_at + 1}",
                (self.decided_at + 1, self.threshold),
                textcoords="offset points", xytext=(5, 5), fontsize=9, color="tab:red",
            )
        ax.set_yscale("log")
        ax.set_xlabel("observation t")
        ax.set_ylabel("e-value (bankroll)")
        ax.legend(loc="best", fontsize=9)
        return ax

    def __repr__(self) -> str:
        state = f"decided at t={self.decided_at + 1}" if self.decided else "undecided"
        return f"Bankroll(e={self.e:.4g}, n={self.n}, threshold={self.threshold:g}, {state})"
