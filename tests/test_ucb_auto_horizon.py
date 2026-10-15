"""D-UCB / SW-UCB horizons sized from the candidate count.

A fixed horizon in pulls is right for one operator count only: at 246
registered operators SW-UCB's 4000-pull window held ~16 pulls per arm, under
its own 27-pull floor. With ``per_arm`` set the horizon follows the running
maximum candidate-list length K: window = ceil(per_arm * K), gamma =
1 - 1 / (per_arm * K).
"""

from __future__ import annotations

import math
import sys

import pytest

from fuzzer_tool.core.schedulers.op_ducb import DUCB_PER_ARM, DUCBScheduler
from fuzzer_tool.core.schedulers.op_swucb import SWUCB_PER_ARM, SWUCBScheduler

SMALL_K = 12
LARGE_K = 246
FIXED_GAMMA = 0.999
FIXED_WINDOW = 50


def _ops(k: int) -> list[str]:
    return [f"op{i}" for i in range(k)]


def _offer(s, k: int) -> None:
    ops = _ops(k)
    for op in ops:
        s.init_arm(op)
    s.select_op(ops)


def test_swucb_window_follows_k():
    """Falsification: window = ceil(per_arm * K) after K candidates are offered."""
    s = SWUCBScheduler(per_arm=SWUCB_PER_ARM)
    _offer(s, LARGE_K)
    assert s.window == math.ceil(SWUCB_PER_ARM * LARGE_K)


def test_ducb_memory_follows_k():
    """Falsification: effective memory 1/(1-gamma) = per_arm * K."""
    s = DUCBScheduler(per_arm=DUCB_PER_ARM)
    _offer(s, LARGE_K)
    assert 1.0 / (1.0 - s.gamma) == pytest.approx(DUCB_PER_ARM * LARGE_K)


def test_horizon_never_shrinks():
    """Adversarial: a smaller candidate list after a larger one keeps the larger horizon."""
    sw = SWUCBScheduler(per_arm=SWUCB_PER_ARM)
    du = DUCBScheduler(per_arm=DUCB_PER_ARM)
    for s in (sw, du):
        _offer(s, LARGE_K)
        _offer(s, SMALL_K)
    assert sw.window == math.ceil(SWUCB_PER_ARM * LARGE_K)
    assert 1.0 / (1.0 - du.gamma) == pytest.approx(DUCB_PER_ARM * LARGE_K)


def test_explicit_horizon_stays_fixed():
    """Adversarial: an explicit gamma/window ignores K."""
    sw = SWUCBScheduler(window=FIXED_WINDOW)
    du = DUCBScheduler(gamma=FIXED_GAMMA)
    for s in (sw, du):
        _offer(s, LARGE_K)
    assert sw.window == FIXED_WINDOW
    assert du.gamma == FIXED_GAMMA


def test_no_forgetting_before_first_offer():
    """Edge: before any candidate list, auto mode forgets nothing."""
    sw = SWUCBScheduler(per_arm=SWUCB_PER_ARM)
    du = DUCBScheduler(per_arm=DUCB_PER_ARM)
    assert sw.window == sys.maxsize
    assert du.gamma == 1.0


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_non_positive_per_arm_rejected(bad):
    """Adversarial: per_arm must be positive."""
    with pytest.raises(ValueError):
        SWUCBScheduler(per_arm=bad)
    with pytest.raises(ValueError):
        DUCBScheduler(per_arm=bad)


def test_window_evicts_at_auto_size():
    """Falsification: the grown window actually bounds the history."""
    s = SWUCBScheduler(per_arm=1.0)
    _offer(s, SMALL_K)
    for _ in range(3 * SMALL_K):
        s.record("op0", success=False)
    assert sum(s.windowed_counts().values()) == SMALL_K


def test_cli_defaults_are_auto(monkeypatch):
    """Falsification: the shipped parser leaves both horizons on auto (None)."""
    from fuzzer_tool.cli import commands

    captured = {}

    def _spy(args):
        captured["g"], captured["w"] = args.ducb_gamma, args.swucb_window
        return 0

    monkeypatch.setattr(commands, "cmd_fuzz", _spy)
    monkeypatch.setattr(sys, "argv", ["fuzzer-tool", "fuzz", "/bin/true"])
    assert commands.main() == 0
    assert captured == {"g": None, "w": None}
