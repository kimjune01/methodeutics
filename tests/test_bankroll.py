"""Tests written first. They define the contract of the package:

1. A Bankroll is a running product of likelihood ratios (an e-process).
2. Under the null, it is a supermartingale: E[e] <= 1, so wealth can't
   grow on average, and P(sup e >= 1/alpha) <= alpha (Ville).
3. Bankrolls compose by multiplication with no correction.
4. The pre-committed threshold is 1/alpha and crossing it is remembered
   (anytime validity: a crossing counts even if the trajectory later falls).
"""

import math
import random

import pytest

from methodeutics import Bankroll, bernoulli, categorical, normal


# --- The chapter-10 coin: H,H,T,H,H at 1.2x / 0.8x -----------------------

def test_coin_trajectory_matches_textbook_exercise():
    b = Bankroll(bernoulli(p0=0.5, p1=0.6), alpha=0.05)
    for outcome in [1, 1, 0, 1, 1]:
        b.update(outcome)
    expected = [1.2, 1.44, 1.152, 1.3824, 1.65888]
    assert b.trajectory == pytest.approx(expected)
    assert b.e == pytest.approx(1.65888)


def test_starts_with_one_dollar():
    b = Bankroll(bernoulli(p0=0.5, p1=0.6))
    assert b.e == 1.0
    assert b.trajectory == []
    assert b.n == 0


# --- Supermartingale property under the null ------------------------------

def test_null_cannot_grow_wealth_on_average():
    rng = random.Random(42)
    finals = []
    for _ in range(2000):
        b = Bankroll(bernoulli(p0=0.5, p1=0.7))
        for _ in range(30):
            b.update(1 if rng.random() < 0.5 else 0)  # null is true
        finals.append(b.e)
    assert sum(finals) / len(finals) <= 1.1  # E[e] <= 1 plus sampling slack


def test_ville_false_alarm_rate_bounded():
    rng = random.Random(7)
    alpha = 0.1
    alarms = 0
    trials = 2000
    for _ in range(trials):
        b = Bankroll(bernoulli(p0=0.5, p1=0.7), alpha=alpha)
        for _ in range(50):
            b.update(1 if rng.random() < 0.5 else 0)
        if b.decided:
            alarms += 1
    assert alarms / trials <= alpha + 0.02


def test_true_effect_compounds():
    rng = random.Random(3)
    b = Bankroll(normal(null_mean=0.0, alt_mean=0.5))
    for _ in range(200):
        b.update(rng.gauss(0.5, 1))  # alternative is true
    assert b.e > 20


# --- Threshold and decision ------------------------------------------------

def test_threshold_is_one_over_alpha():
    assert Bankroll(bernoulli(0.5, 0.6), alpha=0.05).threshold == 20
    assert Bankroll(bernoulli(0.5, 0.6), alpha=0.01).threshold == 100


def test_crossing_is_remembered_even_if_trajectory_falls():
    b = Bankroll(bernoulli(p0=0.5, p1=0.9), alpha=0.25)  # threshold = 4
    for outcome in [1, 1, 1, 1]:      # 1.8^3 = 5.832 crosses 4 at the third
        b.update(outcome)
    assert b.decided
    for outcome in [0, 0, 0, 0, 0]:   # 0.2x each: collapses far below 1
        b.update(outcome)
    assert b.e < 1
    assert b.decided                  # anytime validity: the crossing stands
    assert b.decided_at == 2          # 0-indexed observation of first crossing


# --- Composition ------------------------------------------------------------

def test_bankrolls_multiply_with_no_correction():
    a = Bankroll(bernoulli(0.5, 0.6))
    b = Bankroll(normal(0.0, 0.5))
    for outcome in [1, 1, 0]:
        a.update(outcome)
    for x in [0.4, 0.9]:
        b.update(x)
    assert a.compose(b) == pytest.approx(a.e * b.e)


# --- Models ------------------------------------------------------------------

def test_normal_likelihood_ratio_formula():
    lr = normal(null_mean=0.0, alt_mean=0.5)
    x = 1.3
    assert lr(x) == pytest.approx(math.exp(0.5 * x - 0.5**2 / 2))


def test_categorical_die():
    # Null: fair die. Alt: six comes up 30% of the time, rest uniform.
    null = [1 / 6] * 6
    alt = [0.14] * 5 + [0.30]
    lr = categorical(null, alt)
    assert lr(5) == pytest.approx(0.30 / (1 / 6))  # rolled a six (0-indexed)
    assert lr(0) == pytest.approx(0.14 / (1 / 6))


def test_model_rejects_invalid_probabilities():
    with pytest.raises(ValueError):
        bernoulli(p0=0.0, p1=0.5)
    with pytest.raises(ValueError):
        categorical([0.5, 0.5], [0.7, 0.4])  # alt doesn't sum to 1


# --- Custom model: any callable works ---------------------------------------

def test_any_callable_is_a_model():
    b = Bankroll(lambda x: 2.0 if x else 0.5)
    b.update(True)
    b.update(True)
    b.update(False)
    assert b.e == pytest.approx(2.0)


# --- Plot (smoke) ------------------------------------------------------------

def test_plot_smoke():
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    b = Bankroll(bernoulli(0.5, 0.6), alpha=0.05)
    rng = random.Random(1)
    for _ in range(40):
        b.update(1 if rng.random() < 0.6 else 0)
    ax = b.plot()
    assert ax is not None


def test_plot_without_matplotlib_raises_helpfully(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise ImportError("no matplotlib")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    b = Bankroll(bernoulli(0.5, 0.6))
    with pytest.raises(ImportError, match="pip install methodeutics\\[plot\\]"):
        b.plot()
