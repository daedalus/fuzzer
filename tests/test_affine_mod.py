"""Affine-modular compares: recover (a, b, m) from probes and solve (core/affine_mod.py).

Each fake target is the model ``v = (a*x + b) % m``; expected values come from
the model, never from the solver.
"""

import random

import pytest

from fuzzer_tool.core.affine_mod import (
    MAX_FIT_EXECS,
    WINDOW,
    AffineMod,
    Outcome,
    Ramp,
    fit,
    solve,
)

SEED = 20261012


def _target(a: int, b: int, m: int):
    return lambda x: (a * x + b) % m


# ── fit ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("a", "b", "m", "width", "x0"),
    [
        (37, 11, 65521, 16, 1000),  # steps wrap often
        (1103515245, 12345, 2**31 - 1, 32, 7),  # LCG-sized constants
        (7, 3, 251, 8, 100),  # byte field
        (1, 0, 1000, 32, 123456),  # x % m: no wrap in the window, gallop
        (3, 5, 99991, 32, 2**32 - 2),  # window abuts the top: gallop downward
    ],
)
def test_fit_recovers_the_map(a, b, m, width, x0):
    truth = _target(a, b, m)

    result = fit(truth, x0, width)

    assert result.outcome is Outcome.FITTED
    model = result.model
    assert isinstance(model, AffineMod)
    assert model.m == m
    rng = random.Random(SEED)
    for _ in range(200):
        x = rng.randrange(1 << width)
        assert model(x) == truth(x)


def test_fit_stays_inside_the_budget():
    calls = []

    def truth(x):
        calls.append(x)
        return (1 * x + 0) % 65521

    result = fit(truth, 5, 32)

    assert result.execs == len(calls) <= MAX_FIT_EXECS
    assert result.outcome is Outcome.FITTED


def test_unwrapped_field_is_a_ramp():
    """m above the field's range: v is a plain ramp, solved by division."""
    result = fit(_target(5, 3, 1 << 40), 10, 16)

    assert result.outcome is Outcome.RAMP
    assert isinstance(result.model, Ramp)
    assert solve(result.model, 5 * 400 + 3, 16, near=10) == [400]


def test_flat_operand_is_reported():
    """Falsification: a field that never moves v has nothing to fit."""
    assert fit(lambda _x: 42, 0, 16).outcome is Outcome.FLAT


def test_unreached_compare_is_blind():
    assert fit(lambda _x: None, 0, 16).outcome is Outcome.BLIND


def test_budget_stops_the_fit():
    result = fit(_target(1, 0, 65521), 5, 32, max_execs=WINDOW + 2)

    assert result.outcome is Outcome.BUDGET
    assert result.model is None


# ── adversarial: not affine mod m ────────────────────────────────────


@pytest.mark.parametrize(
    "name",
    ["square", "xor", "overflow32", "floor_div", "cube", "hash"],
)
def test_non_affine_maps_fail_closed(name):
    """A wrong model would make solve() write garbage; none may be returned."""
    rng = random.Random(SEED)
    maps = {
        "square": lambda x, m: (x * x) % m,
        "xor": lambda x, m: (x ^ 0x5A5A) % m,
        "overflow32": lambda x, m: ((x * 2654435761 + 7) & 0xFFFFFFFF) % m,
        "floor_div": lambda x, m: (x // 3) % m,
        "cube": lambda x, m: (x**3 + 1) % m,
        "hash": lambda x, m: random.Random(x).randrange(m),
    }
    for _ in range(100):
        m = rng.randrange(3, 70000)
        x0 = rng.randrange(1 << 32)

        result = fit(lambda x, m=m: maps[name](x, m), x0, 32)

        assert result.model is None, (name, m, x0)
        assert result.outcome in (Outcome.BAD_FIT, Outcome.FLAT)


def test_random_models_never_fit_wrong():
    """Soundness sweep: every FITTED model equals the truth on the whole field sample."""
    rng = random.Random(SEED)
    for _ in range(1500):
        width = rng.choice([8, 16, 32])
        top = (1 << width) - 1
        m = rng.randrange(2, 1 << min(width + 1, 32))
        a, b = rng.randrange(1, top + 1), rng.randrange(top + 1)
        truth = _target(a, b, m)

        result = fit(truth, rng.choice([0, top, rng.randrange(top + 1)]), width)

        if result.outcome is Outcome.FITTED:
            assert all(
                result.model(x) == truth(x) for x in (rng.randrange(top + 1) for _ in range(50))
            )


# ── solve ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("a", "b", "m", "t"),
    [
        (37, 11, 65521, 12345),
        (6, 4, 18, 10),  # gcd(a, m) = 6: a residue class mod 3
        (1, 0, 7, 3),
    ],
)
def test_solve_inverts_the_map(a, b, m, t):
    model = AffineMod(a, b, m)

    xs = solve(model, t, 16, near=500)

    assert xs
    assert all(0 <= x < 1 << 16 and (a * x + b) % m == t for x in xs)


def test_solve_returns_the_nearest_class_member_first():
    model = AffineMod(1, 0, 1000)

    assert solve(model, 7, 16, near=30007)[0] == 30007


def test_solve_is_empty_when_gcd_does_not_divide():
    """Falsification: 6x + 4 mod 18 is always even, so t = 9 has no preimage."""
    assert solve(AffineMod(6, 4, 18), 9, 16) == []


def test_solve_rejects_a_target_at_or_above_m():
    assert solve(AffineMod(3, 1, 100), 100, 16) == []


def test_solve_is_empty_when_the_class_misses_the_field():
    """Adversarial: m far above 2**8, class member beyond the byte."""
    assert solve(AffineMod(1, 0, 100000), 5000, 8) == []


def test_ramp_solution_outside_its_probed_range_is_dropped():
    """The ramp is only known to be wrap-free where the gallop walked."""
    ramp = Ramp(x_ref=100, v_ref=50, step=1, lo=100, hi=200)

    assert solve(ramp, 40, 16) == []
    assert solve(ramp, 60, 16) == [110]
