from __future__ import annotations

import math

from fuzzer_tool.core.fluctuation import TrajectoryRecord, WorkFunctional


def test_step_work_bounds() -> None:
    wf = WorkFunctional()
    assert wf._step_work(0.5) == -math.log(0.5)
    assert wf._step_work(0.0) == -math.log(1e-12)
    assert wf._step_work(1.0) == 0.0
    assert wf._step_work(1e-13) == -math.log(1e-12)


def test_trajectory_work_sums() -> None:
    wf = WorkFunctional()
    record = TrajectoryRecord(
        ops=("bit_flip", "byte_flip"),
        probs=(0.5, 0.25),
        outcome="success",
        hit_edges=frozenset({1, 2, 3}),
        new_edges=2,
    )
    work = wf.observe(record)
    expected = -math.log(0.5) + -math.log(0.25)
    assert math.isclose(work, expected, rel_tol=1e-12)
    assert math.isclose(wf.trajectory_work(), expected, rel_tol=1e-12)


def test_jarzynski_synthetic() -> None:
    wf = WorkFunctional(beta=1.0)
    state_key = "synthetic"
    for p in (0.1, 0.2, 0.3):
        record = TrajectoryRecord(
            ops=("op",),
            probs=(p,),
            outcome="success",
            state_key=state_key,
        )
        wf.observe(record)
    est = wf.jarzynski_estimator(state_key)
    assert est is not None
    assert math.isfinite(est)


def test_crooks_pair_symmetry() -> None:
    wf = WorkFunctional(beta=1.0)
    state_a = "a"
    state_b = "b"
    for _ in range(5):
        wf.observe(
            TrajectoryRecord(ops=("op",), probs=(0.5,), outcome="success", state_key=state_a)
        )
        wf.observe(
            TrajectoryRecord(ops=("op",), probs=(0.5,), outcome="success", state_key=state_b)
        )
    result = wf.crooks_forward_reverse(state_a, state_b)
    assert result["forward"] == 5
    assert result["reverse"] == 5
    assert math.isclose(result["ratio"], 1.0, rel_tol=1e-12)


def test_state_key_stability() -> None:
    rec1 = TrajectoryRecord(
        ops=("a", "b"), probs=(0.5, 0.5), outcome="success", hit_edges=frozenset({1, 2})
    )
    rec2 = TrajectoryRecord(
        ops=("a", "b"), probs=(0.5, 0.5), outcome="boring", hit_edges=frozenset({1, 2})
    )
    assert WorkFunctional.state_key(rec1) == WorkFunctional.state_key(rec2)


def test_window_limits_samples() -> None:
    wf = WorkFunctional(window=10)
    state_key = "win"
    for _ in range(25):
        wf.observe(
            TrajectoryRecord(ops=("op",), probs=(0.5,), outcome="success", state_key=state_key)
        )
    stats = wf.stats(state_key)
    assert stats["samples"] == 10


def test_snapshot_restore_roundtrip() -> None:
    wf = WorkFunctional(beta=2.0, window=50)
    state_key = "roundtrip"
    for p in (0.1, 0.2):
        wf.observe(
            TrajectoryRecord(ops=("op",), probs=(p,), outcome="success", state_key=state_key)
        )
    data = wf.snapshot()
    assert data["beta"] == 2.0
    assert data["window"] == 50

    restored = WorkFunctional()
    restored.restore(data)
    assert restored.beta == 2.0
    assert restored.window == 50
    assert math.isclose(restored.trajectory_work(), wf.trajectory_work(), rel_tol=1e-12)
    est = restored.jarzynski_estimator(state_key)
    assert est is not None
    assert math.isclose(est, wf.jarzynski_estimator(state_key), rel_tol=1e-12)


def test_empty_trajectory() -> None:
    wf = WorkFunctional()
    record = TrajectoryRecord(ops=(), probs=(), outcome="boring", hit_edges=frozenset())
    assert wf.observe(record) == 0.0
    assert wf.trajectory_work() == 0.0
