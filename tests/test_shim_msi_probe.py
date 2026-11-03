"""Stepped edge-table probe (`-D__AFL_PROBE_MSI=1`) against an independent model.

Reference: home slot, then the same window of PROBE_MAX slots walked with an odd
step derived from the edge id (mask-step-index, Wellons 2022), wrapping at the
table end. A non-power-of-two table falls back to step 1. The default build
stays linear and byte-identical in placement (the control, Hard Rule 46).
"""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

from tests.conftest import requires_clang
from tests.test_regression_shim_audit import SHIM, _build
from tests.test_shim_simd_probe import _DRIVER, FIRES, WINDOW, _homing_on, _model, _run
from tests.test_shim_map_slot import _home

STEP_MUL = 0x85EBCA6B
STEP_SHIFT = 16
U32 = 0xFFFFFFFF


def _step(eid: int, size: int) -> int:
    if size & (size - 1):
        return 1
    return ((((eid * STEP_MUL) & U32) >> STEP_SHIFT) | 1) & (size - 1)


def _msi_model(size: int, ids: list[int]) -> tuple[dict[int, tuple[int, int]], int]:
    """Reference stepped probe: ({edge_id: (slot, count)}, dropped)."""
    slots = [0] * size
    counts: dict[int, int] = {}
    where: dict[int, int] = {}
    dropped = 0
    window = min(WINDOW, size)

    for eid in ids:
        pos, step = _home(eid, size), _step(eid, size)
        for _ in range(window):
            if slots[pos] == eid:
                counts[eid] += 1
                break
            if slots[pos] == 0:
                slots[pos] = eid
                counts[eid] = 1
                where[eid] = pos
                break
            pos = (pos + step) % size
        else:
            dropped += 1

    return {e: (where[e], counts[e]) for e in where}, dropped


@pytest.fixture(scope="module")
def linear(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("lin"), _DRIVER, "-D__AFL_CTX_SENSITIVE=0")


@pytest.fixture(scope="module")
def msi(tmp_path_factory):
    return _build(
        tmp_path_factory.mktemp("msi"), _DRIVER, "-D__AFL_CTX_SENSITIVE=0", "-D__AFL_PROBE_MSI=1"
    )


def _dense(size: int, load: float, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    ids = [int(x) for x in rng.integers(1, 1 << 31, size=int(size * load))]
    return ids + ids[: len(ids) // 3]


@requires_clang
def test_default_build_is_linear(linear):
    """Control: the model of the default build is the linear model."""
    ids = _dense(1024, 0.8, 7)
    _, got, dropped = _run(linear, 1024, ids)
    assert (got, dropped) == _model(1024, ids)


@requires_clang
@pytest.mark.parametrize("size", [1024, 4096])
@pytest.mark.parametrize("load", [0.3, 0.8, 0.95])
def test_msi_matches_model(msi, size, load):
    """Falsification: placement, counts and drops equal the stepped model."""
    ids = _dense(size, load, 11)
    _, got, dropped = _run(msi, size, ids)
    assert (got, dropped) == _msi_model(size, ids)


@requires_clang
def test_msi_is_not_inert(linear, msi):
    """The macro changes placement, or the arm tests nothing."""
    ids = _dense(1024, 0.8, 7)
    assert _run(msi, 1024, ids)[1] != _run(linear, 1024, ids)[1]


@requires_clang
def test_msi_drops_less_at_high_load(linear, msi):
    """Same ids, same window: fewer edges lost (load 0.95, fixed seed)."""
    ids = _dense(1024, 0.95, 5)
    lin_drops = _run(linear, 1024, ids)[2]
    msi_drops = _run(msi, 1024, ids)[2]
    assert msi_drops < lin_drops


@requires_clang
@pytest.mark.parametrize("size", [1000, 1027])
def test_non_pow2_falls_back_to_linear(msi, size):
    """Adversarial: a step is only an odd generator in a power-of-two table."""
    ids = _dense(size, 0.8, 3)
    _, got, dropped = _run(msi, size, ids)
    assert (got, dropped) == _model(size, ids)


@requires_clang
@pytest.mark.parametrize("cluster", [9, 33, WINDOW + 5])
def test_same_home_cluster_matches_model(msi, cluster):
    """Adversarial: ids sharing one home slot, past the window, fired twice."""
    ids = _homing_on(512, 1024, cluster)
    ids = ids * FIRES
    _, got, dropped = _run(msi, 1024, ids)
    assert (got, dropped) == _msi_model(1024, ids)


@requires_clang
@pytest.mark.parametrize("size", [2, 4, WINDOW, 128])
def test_tiny_tables(msi, size):
    """Edge: windows wider than the table, steps equal to the mask."""
    ids = list(range(1, size + 4)) * FIRES
    _, got, dropped = _run(msi, size, ids)
    assert (got, dropped) == _msi_model(size, ids)


@requires_clang
def test_refire_never_duplicates(msi):
    """Lookup retraces insertion: one slot per id, count = fires."""
    ids = _dense(2048, 0.9, 9)[:1800]
    _, got, _ = _run(msi, 2048, ids * 3)
    assert all(c >= 3 for _, c in got.values())
    assert len({s for s, _ in got.values()}) == len(got)


@requires_clang
def test_msi_with_simd_is_rejected(tmp_path):
    """The stepped probe has no contiguous run to vectorize."""
    if not shutil.which("clang"):
        pytest.skip("clang not installed")
    src = tmp_path / "t.c"
    src.write_text("int main(void){return 0;}\n")
    r = subprocess.run(
        ["clang", "-D__AFL_PROBE_MSI=1", "-D__AFL_PROBE_ISA=1", "-include", str(SHIM),
         "-c", str(src), "-o", str(tmp_path / "t.o")],
        capture_output=True, text=True,
    )  # fmt: skip
    assert r.returncode != 0 and "__AFL_PROBE_MSI" in r.stderr
