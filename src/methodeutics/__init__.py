"""methodeutics: anytime-valid evidence as a running bet.

A Bankroll is an e-value trajectory: start with one dollar, bet the
likelihood ratio on every observation, let winnings ride. If the null
hypothesis is true, no strategy grows the dollar on average, and the
bankroll exceeds 1/alpha with probability at most alpha, no matter when
or how often you look. If the null is false, the bankroll compounds.

    from methodeutics import Bankroll, bernoulli

    b = Bankroll(bernoulli(p0=0.5, p1=0.6), alpha=0.05)
    for outcome in observations:
        b.update(outcome)
    b.plot()

Not the epidemiology E-value (VanderWeele & Ding's sensitivity-analysis
quantity); this is the sequential-testing e-value of Shafer, Ramdas, and
Grünwald. Companion package to the Methodeutics textbook, chapters 9-10:
https://june.kim/reading/methodeutics/
"""

from .bankroll import Bankroll
from .models import bernoulli, categorical, normal

__version__ = "0.1.0"
__all__ = ["Bankroll", "bernoulli", "categorical", "normal"]
