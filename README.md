# methodeutics

Anytime-valid evidence as a running bet. The e-value bankroll, in three lines:

```python
from methodeutics import Bankroll, bernoulli

b = Bankroll(bernoulli(p0=0.5, p1=0.6), alpha=0.05).extend(observations)
b.plot()  # pip install methodeutics[plot]
```

The core is one small file of plain Python with zero dependencies: read it
before you trust it. The tests check the package against the published
theorems (Ville's inequality, the supermartingale property) and against the
textbook's worked examples.

Start with one dollar. Each observation multiplies your bankroll by the
likelihood ratio the alternative assigns against the null. If the null is
true, no strategy grows the dollar on average, and the bankroll crosses
1/alpha with probability at most alpha, no matter when or how often you
look (Ville's inequality). If the null is false, the bankroll compounds.
Look every day. Stop whenever you want. The guarantee holds.

```python
b.e           # current e-value
b.trajectory  # e-value after each observation
b.threshold   # 1/alpha, pre-committed
b.decided     # crossed the threshold? (remembered even if it later falls)
b.compose(b2) # evidence from an independent bankroll: multiply, no correction
```

Models: `normal(null_mean, alt_mean, sd)`, `bernoulli(p0, p1)`,
`categorical(null, alt)`, or any callable `lr(x)` returning a likelihood ratio.

This is the companion package to chapters 9-10 of the
[Methodeutics textbook](https://june.kim/reading/methodeutics/), where the
bankroll is derived from scratch and the three guarantees (anytime validity,
optional stopping, composition) are stated with their lineage: Ville 1939,
Doob 1953, Shafer 2021, Ramdas 2023, Grünwald 2024.

**Not** the epidemiology E-value (VanderWeele & Ding's sensitivity-analysis
quantity). Same letter, unrelated concept.

See also: [confseq](https://github.com/gostevehoward/confseq) (confidence
sequences, the reference implementation), [expectation](https://github.com/jakorostami/expectation)
(a fuller e-process library), [savvi](https://github.com/assuncaolfi/savvi)
(anytime-valid inference for specific models).

AGPL-3.0-or-later license.
