"""Chaotic inertia schedule (core/chaos.py) and its MOpt / firefly wiring."""

import pytest

from fuzzer_tool.core.chaos import (
    LOGISTIC_EPS,
    W_MAX,
    W_MIN,
    InertiaMode,
    LogisticMap,
)
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.op_firefly import OpFireflyScheduler
from fuzzer_tool.core.schedulers.op_mopt import MOptScheduler
from tests.support.scripted_rng import ScriptedRng

# --- LogisticMap -----------------------------------------------------------


def test_step_follows_r4_logistic():
    # 4 * 0.2 * 0.8 = 0.64; 4 * 0.64 * 0.36 = 0.9216
    m = LogisticMap(ScriptedRng(), z0=0.2)
    assert m.step() == pytest.approx(0.64)
    assert m.step() == pytest.approx(0.9216)


@pytest.mark.parametrize("z0", [0.5, 0.25, 0.75])
def test_regression_float_collapse_reseeds(z0):
    # 0.5 -> 1.0 -> 0 forever; 0.25 -> 0.75 (fixed point). Both must redraw.
    m = LogisticMap(ScriptedRng(randoms=[0.3]), z0=z0)
    expected = LOGISTIC_EPS + (1.0 - 2.0 * LOGISTIC_EPS) * 0.3
    assert m.step() == pytest.approx(expected)


def test_seeded_start_draws_from_rng():
    m = LogisticMap(ScriptedRng(randoms=[0.6]))
    assert m.z == pytest.approx(LOGISTIC_EPS + (1.0 - 2.0 * LOGISTIC_EPS) * 0.6)


def test_long_orbit_stays_open_and_aperiodic():
    """Falsification: a broken guard collapses to 0/1 or a short cycle."""
    m = LogisticMap(RandPool(seed=7))
    tail = [m.step() for _ in range(50_000)][-1000:]

    assert all(LOGISTIC_EPS < z < 1.0 - LOGISTIC_EPS for z in tail)
    assert len(set(tail)) >= 990


def test_orbit_is_arcsine_not_uniform():
    """Arcsine density: mass in [0.4, 0.6] is ~0.13, uniform would give 0.2."""
    m = LogisticMap(RandPool(seed=3))
    zs = [m.step() for _ in range(20_000)]
    mid = sum(0.4 <= z <= 0.6 for z in zs) / len(zs)
    assert mid < 0.16


# --- MOpt ------------------------------------------------------------------


def _mopt(mode):
    s = MOptScheduler(
        n_particles=1,
        window_size=8,
        c1=0.0,
        c2=0.0,
        c3=0.0,
        max_vel=10.0,
        rng=RandPool(seed=1),
        inertia=mode,
    )
    s.init_arm("a")
    s.init_arm("b")
    s.particles[0].vel = [0.1, -0.1]
    return s


def test_mopt_constant_inertia_unchanged():
    s = _mopt(InertiaMode.CONSTANT)
    s._pso_update()
    assert s.particles[0].vel == pytest.approx([0.07, -0.07])


def test_mopt_chaotic_inertia_scales_velocity():
    s = _mopt(InertiaMode.CHAOTIC)
    s._chaos = LogisticMap(ScriptedRng(), z0=0.2)
    s._pso_update()

    w = W_MIN + (W_MAX - W_MIN) * 0.64
    assert s.particles[0].vel == pytest.approx([0.1 * w, -0.1 * w])


def test_mopt_chaotic_inertia_varies_per_window():
    """Adversarial: a map that never advances gives a constant w."""
    s = _mopt(InertiaMode.CHAOTIC)
    s._chaos = LogisticMap(ScriptedRng(), z0=0.2)
    s._pso_update()
    v1 = s.particles[0].vel[0]
    s._pso_update()
    v2 = s.particles[0].vel[0]

    w2 = W_MIN + (W_MAX - W_MIN) * 0.9216
    assert v2 == pytest.approx(v1 * w2)


def test_mopt_constant_mode_draws_no_chaos():
    assert _mopt(InertiaMode.CONSTANT)._chaos is None


# --- Firefly ---------------------------------------------------------------


def _firefly(mode):
    s = OpFireflyScheduler(n_fireflies=1, window_size=8, rng=RandPool(seed=1), inertia=mode)
    s.init_arm("a")
    s.init_arm("b")
    s.fireflies[0].pos = [0.5, 0.5]
    s._rng = ScriptedRng(randoms=[1.0, 0.0])
    return s


def test_firefly_constant_alpha_unchanged():
    s = _firefly(InertiaMode.CONSTANT)
    s._firefly_update()
    assert s.fireflies[0].pos == pytest.approx([0.7, 0.3])


def test_firefly_chaotic_alpha_scales_step():
    s = _firefly(InertiaMode.CHAOTIC)
    s._chaos = LogisticMap(ScriptedRng(), z0=0.2)
    s._firefly_update()

    a = 0.2 * 2.0 * 0.64
    assert s.fireflies[0].pos == pytest.approx([0.5 + a, 0.5 - a])


def test_firefly_chaotic_keeps_alpha_decay():
    s = _firefly(InertiaMode.CHAOTIC)
    s._chaos = LogisticMap(ScriptedRng(), z0=0.2)
    s._firefly_update()
    assert s.alpha == pytest.approx(0.2 * 0.97)


# --- Wiring ----------------------------------------------------------------


def test_fuzzer_default_is_constant():
    import inspect

    from fuzzer_tool.services.fuzzer import Fuzzer

    default = inspect.signature(Fuzzer.__init__).parameters["swarm_inertia"].default
    assert default is InertiaMode.CONSTANT


def test_cli_passes_swarm_inertia():
    import ast
    import inspect

    from fuzzer_tool.cli import commands
    from tests.test_regression_cli_fuzzer_kwargs import _fuzz_parser_dests

    tree = ast.parse(inspect.getsource(commands))
    assert "swarm_inertia" in _fuzz_parser_dests(tree)
