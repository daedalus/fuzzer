"""Vectorized swarm updates and byte-redundancy table vs scalar oracles.

Each oracle is the pre-vectorization scalar code. Pools are seeded
identically, so a faithful port matches draw-for-draw.
"""

import math
import os

import numpy as np
import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.op_firefly import OpFireflyScheduler
from fuzzer_tool.core.schedulers.op_mopt import MOptScheduler
from fuzzer_tool.core.simplex import project_rows
from fuzzer_tool.services.operators import OperatorEngine

ATOL = 1e-12
N_OPS = 12
OPS = [f"op{i}" for i in range(N_OPS)]
SEEDS = (1, 7, 42)


# --- scalar oracles ---------------------------------------------------------


def _simplex_scalar(pos, frac):
    n = len(pos)
    clipped = [x if x > 0.0 else 0.0 for x in pos]
    total = sum(clipped)
    out = [x / total for x in clipped] if total > 0.0 else [1.0 / n] * n
    floor = frac / n
    if floor > 0.0:
        out = [max(x, floor) for x in out]
        total = sum(out)
        out = [x / total for x in out]
    return out


def _firefly_scalar(s, *, ignore_brightness=False):
    n = len(s.operators)
    old_pos = [list(f.pos) for f in s.fireflies]
    old_fit = [f.fitness for f in s.fireflies]
    for i, fly in enumerate(s.fireflies):
        new_pos = list(old_pos[i])
        for j in range(len(s.fireflies)):
            if j == i or (not ignore_brightness and old_fit[j] <= old_fit[i]):
                continue
            r2 = sum((old_pos[j][k] - old_pos[i][k]) ** 2 for k in range(n))
            beta = s.beta0 * math.exp(-s.gamma * r2)
            for k in range(n):
                new_pos[k] += beta * (old_pos[j][k] - old_pos[i][k])
        for k in range(n):
            new_pos[k] += s.alpha * (s._rng.random() * 2.0 - 1.0)
        fly.pos = _simplex_scalar(new_pos, s.min_prob_frac)


def _pso_scalar(s, eff, *, operator_major=False):
    n = len(s.operators)
    draws = {}
    if operator_major:
        for i in range(n):
            for pi in range(len(s.particles)):
                draws[pi, i] = (s._rng.random(), s._rng.random(), s._rng.random())
    for pi, p in enumerate(s.particles):
        for i in range(n):
            r1, r2, r3 = draws[pi, i] if operator_major else (s._rng.random() for _ in range(3))
            v = (
                s.w * p.vel[i]
                + s.c1 * r1 * (p.pbest_pos[i] - p.pos[i])
                + s.c2 * r2 * (s.global_best_pos[i] - p.pos[i])
                + s.c3 * r3 * (eff[i] - p.pos[i])
            )
            p.vel[i] = max(-s.max_vel, min(s.max_vel, v))
        for i in range(n):
            p.pos[i] += p.vel[i]
        p.pos = _simplex_scalar(p.pos, s.min_prob_frac)


def _redundant_scalar(original, candidate):
    from fuzzer_tool.core.mutations.generic import (
        could_be_arith,
        could_be_bitflip,
        could_be_interest,
    )

    if len(original) != len(candidate):
        return False
    for a, b in zip(original, candidate, strict=True):
        if a == b:
            continue
        if could_be_bitflip(a ^ b) or could_be_arith(a, b, 1) or could_be_interest(a, b, 1):
            continue
        return False
    return True


# --- builders ---------------------------------------------------------------


def _fireflies(seed, n_ops=N_OPS, **kw):
    s = OpFireflyScheduler(n_fireflies=5, window_size=1, rng=RandPool(seed), **kw)
    for op in OPS[:n_ops]:
        s.init_arm(op)
    for i, fly in enumerate(s.fireflies):
        fly.fitness = (i * 0.37) % 1.0
    return s


def _swarm(seed, n_ops=N_OPS, **kw):
    s = MOptScheduler(n_particles=5, rng=RandPool(seed), **kw)
    for op in OPS[:n_ops]:
        s.init_arm(op)
    # Pin pbest/gbest so the pre-velocity bookkeeping leaves them untouched.
    for p in s.particles:
        p.pbest_fitness = 1e9
        p.pbest_pos = list(np.random.default_rng(3).dirichlet(np.ones(n_ops)))
        p.vel = list(np.random.default_rng(4).uniform(-0.1, 0.1, n_ops))
    s.global_best_fitness = 1e9
    s.global_best_pos = list(np.random.default_rng(5).dirichlet(np.ones(n_ops)))
    return s


def _pos(s):
    return np.array([p.pos for p in (getattr(s, "fireflies", None) or s.particles)])


# --- project_rows -----------------------------------------------------------


@pytest.mark.parametrize("frac", [0.0, 0.1, 1.0])
def test_project_rows_matches_scalar(frac):
    rng = np.random.default_rng(0)
    rows = rng.normal(0.0, 1.0, (8, 13))
    got = project_rows(rows, frac)
    want = np.array([_simplex_scalar(list(r), frac) for r in rows])
    np.testing.assert_allclose(got, want, atol=ATOL)


def test_project_rows_adversarial_degenerate_rows():
    rows = np.array([[-1.0, -2.0, 0.0], [0.0, 0.0, 0.0], [1e300, 0.0, 0.0], [5.0, 5.0, 5.0]])
    got = project_rows(rows, 0.1)
    np.testing.assert_allclose(got.sum(axis=1), 1.0, atol=ATOL)
    assert (got >= 0.1 / 3 / 1.1).all()
    np.testing.assert_allclose(got[0], got[1])  # all-nonpositive -> uniform
    assert project_rows(np.empty((2, 0)), 0.1).shape == (2, 0)


# --- firefly ----------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_firefly_matches_scalar(seed):
    new, old = _fireflies(seed), _fireflies(seed)
    new._firefly_update()
    _firefly_scalar(old)
    np.testing.assert_allclose(_pos(new), _pos(old), atol=ATOL)


def test_firefly_falsification_wrong_oracle_differs():
    new, wrong = _fireflies(1), _fireflies(1)
    new._firefly_update()
    _firefly_scalar(wrong, ignore_brightness=True)
    assert not np.allclose(_pos(new), _pos(wrong), atol=1e-6)


@pytest.mark.parametrize(
    "case",
    ["one_op", "equal_fitness", "far_apart", "no_fireflies"],
)
def test_firefly_adversarial(case):
    seed = 9
    n_ops = 1 if case == "one_op" else len(OPS)
    n_fl = 0 if case == "no_fireflies" else 5
    new = OpFireflyScheduler(n_fireflies=n_fl, window_size=1, rng=RandPool(seed))
    old = OpFireflyScheduler(n_fireflies=n_fl, window_size=1, rng=RandPool(seed))
    for s in (new, old):
        for op in OPS[:n_ops]:
            s.init_arm(op)
        for i, fly in enumerate(s.fireflies):
            fly.fitness = 0.5 if case == "equal_fitness" else i * 0.2
            if case == "far_apart":
                fly.pos = [1e3 * (i + 1) * (-1) ** k for k in range(n_ops)]
    new._firefly_update()
    _firefly_scalar(old)

    if n_fl:
        np.testing.assert_allclose(_pos(new), _pos(old), atol=ATOL)
        np.testing.assert_allclose(_pos(new).sum(axis=1), 1.0, atol=ATOL)
    assert new.alpha == old.alpha * old.alpha_decay or case == "no_fireflies"


def test_firefly_window_state_reset():
    s = _fireflies(1)
    for fly in s.fireflies:
        fly.execs_in_window = 7
        fly.discoveries.append(1.0)
    s._firefly_update()
    assert all(f.execs_in_window == 0 and not f.discoveries for f in s.fireflies)
    assert all(isinstance(x, float) for x in s.fireflies[0].pos)


# --- PSO --------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_pso_matches_scalar(seed):
    new, old = _swarm(seed), _swarm(seed)
    eff = old._efficiency_distribution(len(OPS))
    new._pso_update()
    _pso_scalar(old, eff)
    np.testing.assert_allclose(_pos(new), _pos(old), atol=ATOL)
    np.testing.assert_allclose(
        [p.vel for p in new.particles], [p.vel for p in old.particles], atol=ATOL
    )


def test_pso_falsification_wrong_draw_order_differs():
    new, wrong = _swarm(1), _swarm(1)
    eff = wrong._efficiency_distribution(len(OPS))
    new._pso_update()
    _pso_scalar(wrong, eff, operator_major=True)
    assert not np.allclose(_pos(new), _pos(wrong), atol=1e-6)


@pytest.mark.parametrize("case", ["one_op", "clamped", "no_particles"])
def test_pso_adversarial(case):
    seed = 11
    n_ops = 1 if case == "one_op" else len(OPS)
    n_p = 0 if case == "no_particles" else 5
    new = _swarm(seed, n_ops=n_ops)
    old = _swarm(seed, n_ops=n_ops)
    if case == "no_particles":
        new.particles.clear()
        old.particles.clear()
    if case == "clamped":
        for s in (new, old):
            for p in s.particles:
                p.pos = [-50.0 + k for k in range(n_ops)]
                p.vel = [10.0] * n_ops
    eff = old._efficiency_distribution(n_ops)
    new._pso_update()
    _pso_scalar(old, eff)

    if n_p:
        np.testing.assert_allclose(_pos(new), _pos(old), atol=ATOL)
        np.testing.assert_allclose(_pos(new).sum(axis=1), 1.0, atol=ATOL)
        assert np.abs([p.vel for p in new.particles]).max() <= new.max_vel + ATOL
    assert not new._op_execs


# --- byte redundancy --------------------------------------------------------


def _mutate(buf, kind):
    out = bytearray(buf)
    n = len(out)
    if kind == "bitflip":
        for i in range(0, n, max(1, n // 5)):
            out[i] ^= 1 << (i % 8)
    if kind == "random":
        out[:] = os.urandom(n)
    if kind == "last_byte":
        out[-1] ^= 0xA5
    return bytes(out)


@pytest.mark.parametrize("n", [1, 255, 256, 257, 1024, 4096])
@pytest.mark.parametrize("kind", ["same", "bitflip", "random", "last_byte"])
def test_redundant_matches_scalar(n, kind):
    original = os.urandom(n)
    candidate = _mutate(original, kind)
    got = OperatorEngine._is_deterministically_redundant(original, candidate)
    assert got == _redundant_scalar(original, candidate)


def test_redundant_falsification_non_redundant_detected():
    original = bytes(512)
    candidate = bytearray(original)
    candidate[300] = 0x5B  # 0 -> 0x5B: no bitflip, arith, or interesting value
    assert not OperatorEngine._is_deterministically_redundant(original, bytes(candidate))


def test_redundant_adversarial_edges():
    check = OperatorEngine._is_deterministically_redundant
    assert check(b"", b"")
    assert not check(bytes(300), bytes(301))
    assert not check(bytes(10), bytes(11))
    # Every (old, new) pair, placed at the end of a LUT-sized buffer.
    pad = bytes(255)
    for old in range(0, 256, 5):
        for new in range(256):
            a, b = pad + bytes([old]), pad + bytes([new])
            assert check(a, b) == _redundant_scalar(a, b)
