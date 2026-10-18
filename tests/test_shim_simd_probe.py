"""SIMD edge-table probe (`__AFL_PROBE_ISA`) must place edges exactly like scalar.

The reference is a Python linear-probe model, derived independently of the
shim: home slot, window of PROBE_MAX, wrap at the table end, first empty or
matching slot wins, a full window drops.
"""

from __future__ import annotations

import shutil
import subprocess
from itertools import count, islice
from pathlib import Path

import numpy as np
import pytest

from fuzzer_tool.adapters.shm import _ENTRY_DTYPE, ShmCoverage
from tests.conftest import requires_clang
from tests.test_regression_shim_audit import SHIM, _build, _clean_env
from tests.test_shim_map_slot import _home

COUNT_MASK = 0x00FFFFFF
FIRES = 2
WINDOW = ShmCoverage.PROBE_MAX


class Isa:
    """__AFL_PROBE_ISA values and the extra compiler flags each needs."""

    SCALAR = (0, ())
    SSE2 = (1, ())
    AVX2 = (2, ("-mavx2",))


# argv[1:]: edge ids; prints the compiled probe mode first.
_DRIVER = """
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv) {
    printf("%d\\n", __afl_probe_mode);
    for (int i = 1; i < argc; i++)
        __afl_map_id_raw((uint32_t)strtoul(argv[i], NULL, 10));
    return 0;
}
"""


def _avx2_host() -> bool:
    try:
        return "avx2" in Path("/proc/cpuinfo").read_text()
    except OSError:
        return False


def _builder(tmp_path_factory, isa: tuple[int, tuple[str, ...]]) -> Path:
    if not shutil.which("clang"):
        pytest.skip("clang not installed")
    mode, flags = isa
    d = tmp_path_factory.mktemp(f"isa{mode}")
    return _build(d, _DRIVER, "-D__AFL_CTX_SENSITIVE=0", f"-D__AFL_PROBE_ISA={mode}", *flags)


@pytest.fixture(
    scope="module", params=[Isa.SCALAR, Isa.SSE2, Isa.AVX2], ids=["scalar", "sse2", "avx2"]
)
def exe(request, tmp_path_factory):
    if request.param is Isa.AVX2 and not _avx2_host():
        pytest.skip("host has no AVX2")
    return _builder(tmp_path_factory, request.param), request.param[0]


def _run(exe: Path, size: int, ids: list[int]) -> tuple[int, dict[int, tuple[int, int]], int]:
    """Returns (compiled mode, {edge_id: (slot, count)}, dropped)."""
    cov = ShmCoverage(size=size)
    try:
        env = _clean_env(__AFL_SHM_ID=cov.env_id, AFL_MAP_SIZE=str(size))
        cov.reset_edge_map()
        r = subprocess.run(
            [str(exe), *map(str, ids)], env=env, capture_output=True, text=True, timeout=30
        )
        assert r.returncode == 0, r.stderr
        table = np.frombuffer(cov._map, dtype=_ENTRY_DTYPE, count=cov.num_entries)
        got = {
            int(e): (i, int(c) & COUNT_MASK)
            for i, (e, c) in enumerate(zip(table["edge_id"], table["count"], strict=True))
            if e
        }
        return int(r.stdout.split()[0]), got, cov.read_dropped_edges()
    finally:
        cov.cleanup()


def _model(size: int, ids: list[int]) -> tuple[dict[int, tuple[int, int]], int]:
    """Reference linear probe: ({edge_id: (slot, count)}, dropped)."""
    slots: list[int] = [0] * size
    counts: dict[int, int] = {}
    where: dict[int, int] = {}
    dropped = 0
    window = min(WINDOW, size)

    for eid in ids:
        home = _home(eid, size)
        for k in range(window):
            s = (home + k) % size
            if slots[s] == eid:
                counts[eid] += 1
                break
            if slots[s] == 0:
                slots[s] = eid
                counts[eid] = 1
                where[eid] = s
                break
        else:
            dropped += 1

    return {e: (where[e], counts[e]) for e in where}, dropped


def _homing_on(home: int, size: int, n: int) -> list[int]:
    """First n ids whose home slot is *home*."""
    return list(islice((i for i in count(1) if _home(i, size) == home), n))


@requires_clang
def test_compiled_mode_is_requested(exe):
    """The ISA asked for is the one built (the host supports it)."""
    path, mode = exe
    got_mode, _, _ = _run(path, 1024, [1])
    assert got_mode == mode


@requires_clang
@pytest.mark.parametrize("size", [1000, 1024])
@pytest.mark.parametrize("cluster", [3, 9, 33])
def test_cluster_matches_model(exe, size, cluster):
    """Falsification: a collision chain lands on the model's slots and counts."""
    ids = _homing_on(size // 2, size, cluster)
    ids = ids + ids[::-1]
    _, got, dropped = _run(exe[0], size, ids)
    assert (got, dropped) == _model(size, ids)


@requires_clang
@pytest.mark.parametrize("size", [1000, 1024, 1027])
@pytest.mark.parametrize("back", [1, 2, 3, 5])
def test_chain_straddles_table_end(exe, size, back):
    """Adversarial: chains crossing the end at every vector phase, odd sizes."""
    ids = _homing_on(size - back, size, 11) * FIRES
    _, got, dropped = _run(exe[0], size, ids)
    assert (got, dropped) == _model(size, ids)


@requires_clang
@pytest.mark.parametrize("size", [1, 2, 3, 5, 7, WINDOW, WINDOW + 1])
def test_tiny_and_full_window(exe, size):
    """Edge: tables narrower than a vector, and windows that overflow (drops)."""
    ids = list(range(1, size + 4)) * FIRES
    _, got, dropped = _run(exe[0], size, ids)
    assert (got, dropped) == _model(size, ids)


@requires_clang
def test_random_stream_matches_model(exe):
    """A dense pseudo-random stream at ~0.8 load."""
    size = 1024
    rng = np.random.default_rng(7)
    ids = [int(x) for x in rng.integers(1, 1 << 31, size=800)]
    ids = ids + ids[:300]
    _, got, dropped = _run(exe[0], size, ids)
    assert (got, dropped) == _model(size, ids)


@requires_clang
def test_model_matches_itself_on_scalar(tmp_path_factory):
    """Control (Hard Rule 46): legacy scalar agrees with the model."""
    path = _builder(tmp_path_factory, Isa.SCALAR)
    ids = _homing_on(1020, 1024, 9) * FIRES
    _, got, dropped = _run(path, 1024, ids)
    assert (got, dropped) == _model(1024, ids)


@requires_clang
def test_avx2_without_flag_falls_back(tmp_path_factory):
    """Adversarial: AVX2 requested but not targetable -> scalar, same table."""
    path = _builder(tmp_path_factory, (Isa.AVX2[0], ("-mno-avx2",)))
    ids = _homing_on(5, 1024, 9)
    mode, got, dropped = _run(path, 1024, ids)
    assert mode == Isa.SCALAR[0]
    assert (got, dropped) == _model(1024, ids)


@requires_clang
def test_unknown_isa_fails_build(tmp_path):
    """Adversarial: an out-of-range ISA is a build error, not a silent scalar."""
    c = tmp_path / "t.c"
    c.write_text("int main(void) { return 0; }\n")
    cc = [
        "clang",
        "-D__AFL_PROBE_ISA=7",
        "-include",
        SHIM,
        "-c",
        str(c),
        "-o",
        str(tmp_path / "t.o"),
    ]
    r = subprocess.run(cc, capture_output=True, text=True)
    assert r.returncode != 0
    assert "__AFL_PROBE_ISA must be" in r.stderr
