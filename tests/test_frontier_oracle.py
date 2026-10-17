"""adapters/frontier_cov.py FrontierOracle: per-seed frontier reach, online."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from fuzzer_tool.adapters.frontier_cov import FrontierOracle, OracleMode, Side
from tests.conftest import requires_clang
from tests.test_frontier_tool import _TARGET, LINE_B, STUB

pytestmark = [
    requires_clang,
    pytest.mark.skipif(
        not (shutil.which("llvm-cov") and shutil.which("llvm-profdata")),
        reason="llvm-cov / llvm-profdata not installed",
    ),
]


@pytest.fixture
def cov_bin(tmp_path):
    src = tmp_path / "t.c"
    src.write_text(_TARGET)
    out = tmp_path / "t_cov"
    cmd = ["clang", "-O0", "-g", "-fprofile-instr-generate", "-fcoverage-mapping"]
    r = subprocess.run([*cmd, "-o", str(out), str(src), str(STUB)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return out


def _oracle(cov_bin, tmp_path, **kw) -> FrontierOracle:
    return FrontierOracle(cov_bin, tmp_path / "oracle", mode=OracleMode.INLINE, **kw)


def _cond_b(oracle) -> tuple:
    (cond,) = [c for c in oracle.frontier() if c[0] == "t.c" and c[1] == LINE_B]
    return cond


def test_reach_and_frontier(cov_bin, tmp_path):
    # 'ax' executes b's condition (false side); 'zz' never gets there.
    o = _oracle(cov_bin, tmp_path)
    o.submit("a", b"ax")
    o.submit("z", b"zz")
    o.drain()
    cond = _cond_b(o)
    assert o.frontier()[cond] is Side.TRUE
    assert cond in o.open_reached("a")
    assert cond not in o.open_reached("z")
    assert o.reachers(cond) == 1


def test_falsification_closed_condition_leaves_reach(cov_bin, tmp_path):
    o = _oracle(cov_bin, tmp_path)
    o.submit("a", b"ax")
    o.drain()
    cond = _cond_b(o)
    o.submit("b", b"ab")  # takes b's true side: the condition closes
    o.drain()
    assert cond not in o.frontier()
    assert cond not in o.open_reached("a")
    assert o.stats()["closed"] >= 1


def test_adversarial_bounded_queue_and_unknown_key(cov_bin, tmp_path):
    o = _oracle(cov_bin, tmp_path, max_pending=1)
    for k in ("a", "b", "c"):
        o.submit(k, b"ax")
    assert o.stats()["dropped"] == 2
    o.drain()
    assert o.open_reached("missing") is None
    assert o.open_reached("b") is None  # dropped, never measured


def test_seed_cap_evicts_oldest(cov_bin, tmp_path):
    o = _oracle(cov_bin, tmp_path, max_seeds=1)
    o.submit("a", b"ax")
    o.submit("b", b"ax")
    o.drain()
    assert o.open_reached("a") is None
    assert o.reachers(_cond_b(o)) == 1


def test_thread_mode_processes_in_background(cov_bin, tmp_path):
    o = FrontierOracle(cov_bin, tmp_path / "oracle", mode=OracleMode.THREAD)
    try:
        o.submit("a", b"ax")
        o.drain()  # waits for the worker
        assert _cond_b(o) in o.open_reached("a")
    finally:
        o.close()
    assert not (tmp_path / "oracle").exists() or not any((tmp_path / "oracle").iterdir())
