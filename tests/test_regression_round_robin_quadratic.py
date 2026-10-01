"""Round robin filtered candidates with ``x in list`` per registered arm: O(n^2) per pick.

Measured ~100 ms per pick at 5000 seeds. A counting list makes the scan
visible without timing: membership must go through a set, never the list.
"""

from __future__ import annotations

from fuzzer_tool.core.schedulers.op_round_robin import RoundRobinScheduler
from fuzzer_tool.core.schedulers.seed_round_robin import SeedRoundRobinScheduler


class _ProbeList(list):
    """List that counts membership probes."""

    probes = 0

    def __contains__(self, item):
        _ProbeList.probes += 1
        return super().__contains__(item)


def _ids(n=200):
    _ProbeList.probes = 0
    return _ProbeList(f"k{i}" for i in range(n))


def test_regression_seed_round_robin_no_list_scan():
    ids = _ids()
    s = SeedRoundRobinScheduler()
    for _ in range(5):
        s.select_seed(ids)

    assert _ProbeList.probes == 0


def test_regression_op_round_robin_no_list_scan():
    ops = _ids()
    s = RoundRobinScheduler()
    for op in ops:
        s.init_arm(op)
    for _ in range(5):
        s.select_op(ops)

    assert _ProbeList.probes == 0


def test_regression_round_robin_order_unchanged():
    """Falsification: the fix keeps registration-order cycling over the live subset."""
    s = SeedRoundRobinScheduler()
    s.select_seed(["a", "b", "c"])

    assert [s.select_seed(["c", "a"]) for _ in range(4)] in (["a", "c"] * 2, ["c", "a"] * 2)

    o = RoundRobinScheduler()
    for op in ("x", "y", "z"):
        o.init_arm(op)

    assert [o.select_op(["z", "x"]) for _ in range(4)] in (["x", "z"] * 2, ["z", "x"] * 2)
