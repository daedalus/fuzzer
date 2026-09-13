"""Regression tests for the fluctuation work functional's probability source.

Three defects are pinned here, all from the thermo handover audit (P2-T4):

1. ``self._operators._available`` was consulted through a ``hasattr`` guard
   and is not an attribute of the operators service. The guard never raised;
   it always fell through to the trajectory itself, so every step got
   probability ``1/L`` and the recorded work was exactly ``L*log(L)`` -- a
   function of the mutation-stack depth and nothing else. A live run produced
   5,697 samples with 7 distinct values, all ``n*log(n)``.

2. ``WorkFunctional`` pooled those values anyway, so ``jarzynski_estimator``
   returned a number with the shape of an entropy and none of its meaning.
   Its documented identity -- the Renyi entropy of order ``1+beta`` of the
   operator-path distribution -- holds only when the trajectory was drawn
   from the recorded probabilities.

3. ``state_key`` hashed the operator tuple with the builtin ``hash()``, which
   is salted per process, so the same trajectory keyed differently on every
   run and any state restored from disk was orphaned under a key the new
   process could not reproduce.
"""

from __future__ import annotations

import math
import subprocess
import sys

import pytest

from fuzzer_tool.core.fluctuation import TrajectoryRecord, WorkFunctional
from fuzzer_tool.services import fuzzer as fuzzer_mod

# --- Defect 1: the phantom attribute, and the guard against the next one ----


def test_selection_prob_sources_name_real_attributes() -> None:
    """Every name in _SELECTION_PROB_SOURCES must be a real Fuzzer attribute.

    This is the guard that would have caught the original bug. A name that no
    instance carries disables the feature silently instead of failing.
    """
    assert fuzzer_mod._SELECTION_PROB_SOURCES, "source list must not be empty"
    src = fuzzer_mod.Fuzzer.__init__.__code__.co_names + tuple(
        getattr(fuzzer_mod.Fuzzer, "__slots__", ()) or ()
    )
    text = __import__("inspect").getsource(fuzzer_mod.Fuzzer)
    for attr in fuzzer_mod._SELECTION_PROB_SOURCES:
        assert f"self.{attr}" in text, (
            f"_SELECTION_PROB_SOURCES names {attr!r}, which Fuzzer never assigns; "
            "a hasattr/getattr on it would fall through forever"
        )
        assert attr in src or f"self.{attr}" in text


def test_operators_service_has_no_available_attribute() -> None:
    """Pin the fact that motivated the fix, so a re-introduction is loud."""
    from fuzzer_tool.services import operators as ops_mod

    assert not hasattr(ops_mod, "_available")
    # Comments are stripped first: the fix's own explanatory comment names the
    # dead attribute on purpose, and an unstripped scan matches that.
    src = __import__("inspect").getsource(fuzzer_mod.Fuzzer)
    code = "\n".join(
        line for line in src.splitlines() if not line.lstrip().startswith("#")
    )
    assert "self._operators._available" not in code, (
        "the phantom attribute is back; the work functional degenerates to L*log(L)"
    )


# --- Defect 2: provenance gating -------------------------------------------


def test_untrue_probs_are_counted_but_not_pooled() -> None:
    wf = WorkFunctional(beta=1.0, window=1000)
    for length in (2, 3, 4, 6, 8, 12):
        ops = tuple(f"op{i}" for i in range(length))
        wf.observe(
            TrajectoryRecord(
                ops=ops,
                probs=tuple(1.0 / length for _ in ops),
                outcome="boring",
                state_key="S",
                probs_are_true=False,
            )
        )
    assert wf.jarzynski_estimator("S") is None
    assert wf.stats("S")["unpooled"] == 6


def test_degenerate_work_is_l_log_l_and_stays_out() -> None:
    """The exact signature of the shipped defect: W = L*log(L)."""
    wf = WorkFunctional(beta=1.0, window=1000)
    seen = []
    for length in (1, 2, 3, 4, 6, 8, 12):
        ops = tuple(f"op{i}" for i in range(length))
        w = wf.observe(
            TrajectoryRecord(
                ops=ops,
                probs=tuple(1.0 / length for _ in ops),
                outcome="boring",
                state_key="S",
                probs_are_true=False,
            )
        )
        seen.append((length, w))
    for length, w in seen:
        assert w == pytest.approx(length * math.log(length), abs=1e-9)
    # ... and none of it reaches the estimator.
    assert wf.jarzynski_estimator("S") is None


def test_true_probs_are_pooled_and_recover_renyi_entropy() -> None:
    """The documented identity, checked against ground truth.

    ``-log(E[e^{-beta W}])/beta`` is the Renyi entropy of order ``1+beta`` of
    the trajectory distribution when the trajectory is drawn from the recorded
    probabilities. Two operators, length-2 paths, so the exact value is
    computable in closed form.
    """
    import itertools
    import random

    p = {"a": 0.7, "b": 0.3}
    law = {
        path: p[path[0]] * p[path[1]]
        for path in itertools.product(("a", "b"), repeat=2)
    }
    beta = 1.0
    truth = -math.log(sum(v ** (1.0 + beta) for v in law.values())) / beta

    rng = random.Random(1234)
    wf = WorkFunctional(beta=beta, window=10**7)
    keys, weights = zip(*law.items(), strict=True)
    for _ in range(200000):
        path = rng.choices(keys, weights=weights, k=1)[0]
        wf.observe(
            TrajectoryRecord(
                ops=tuple(path),
                probs=tuple(p[o] for o in path),
                outcome="boring",
                state_key="S",
                probs_are_true=True,
            )
        )
    est = wf.jarzynski_estimator("S")
    assert est is not None
    assert est == pytest.approx(truth, abs=0.02)


def test_probs_are_true_defaults_to_false() -> None:
    """A caller has to assert the property, not remember to deny it."""
    rec = TrajectoryRecord(ops=("a",), probs=(0.5,), outcome="boring")
    assert rec.probs_are_true is False


# --- Defect 3: reproducible state keys -------------------------------------


def test_ops_state_key_is_stable_across_processes() -> None:
    """The ops branch must not depend on PYTHONHASHSEED."""
    prog = (
        "from fuzzer_tool.core.fluctuation import WorkFunctional, TrajectoryRecord;"
        "r=TrajectoryRecord(ops=('byte_flip','havoc'),probs=(0.5,0.5),outcome='x');"
        "print(WorkFunctional.state_key(r))"
    )
    seen = set()
    for seed in ("0", "1", "12345"):
        out = subprocess.run(
            [sys.executable, "-c", prog],
            capture_output=True,
            text=True,
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
            check=True,
        )
        seen.add(out.stdout.strip())
    assert len(seen) == 1, f"state_key varies with PYTHONHASHSEED: {seen}"
    assert next(iter(seen)).startswith("o_")
    assert "-" not in next(iter(seen)), "hex of a negative int leaks a sign"


def test_edge_and_ops_state_keys_do_not_collide() -> None:
    a = TrajectoryRecord(ops=("x",), probs=(1.0,), outcome="b", hit_edges=frozenset({7}))
    b = TrajectoryRecord(ops=("x",), probs=(1.0,), outcome="b")
    assert WorkFunctional.state_key(a) != WorkFunctional.state_key(b)


# --- Crooks retirement ------------------------------------------------------


def test_crooks_is_gone() -> None:
    """It was a ratio of mean works between two arbitrary state buffers.

    No reverse protocol, no matched-W density ratio, no crossing at dF. The
    docstring's own claim -- "for identical work distributions the ratio is
    centered at 1.0" -- is true of any ratio of equal means and tests nothing.
    """
    assert not hasattr(WorkFunctional, "crooks_forward_reverse")


# --- Resume ----------------------------------------------------------------


def test_unpooled_counter_survives_a_resume() -> None:
    wf = WorkFunctional(beta=1.0, window=100)
    wf.observe(
        TrajectoryRecord(ops=("a", "b"), probs=(0.5, 0.5), outcome="boring", state_key="S")
    )
    restored = WorkFunctional()
    restored.restore(wf.snapshot())
    assert restored.stats("S")["unpooled"] == 1
