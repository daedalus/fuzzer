"""Edge-table slot math: no runtime division on the per-edge path.

`edge_id % __afl_map_size` and the probe wrap `(pos + i) % size` divided by a
runtime variable on every edge hit. Power-of-two sizes use a mask; the probe
wraps by subtraction. Placement must match plain `%` / linear probing.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from fuzzer_tool.adapters.shm import _ENTRY_DTYPE, ShmCoverage
from tests.conftest import requires_clang
from tests.test_regression_shim_audit import SHIM, _build, _clean_env

POW2_SIZE = 1024
ODD_SIZE = 1000
TINY_SIZES = [1, 3]
MAX_ID = 0xFFFFFFFF

# argv[1:]: edge ids (decimal), each mapped raw into the table.
_DRIVER = """
#include <stdlib.h>
int main(int argc, char **argv) {
    for (int i = 1; i < argc; i++)
        __afl_map_id_raw((uint32_t)strtoul(argv[i], NULL, 10));
    return 0;
}
"""


@pytest.fixture(scope="module")
def target(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("slot"), _DRIVER, "-D__AFL_CTX_SENSITIVE=0")


def _slots(exe: Path, size: int, ids: list[int]) -> dict[int, int]:
    """Run the driver; return {edge_id: slot index} from the raw table."""
    cov = ShmCoverage(size=size)
    try:
        env = _clean_env(__AFL_SHM_ID=cov.env_id, AFL_MAP_SIZE=str(size))
        cov.reset_edge_map()
        r = subprocess.run(
            [str(exe), *map(str, ids)], env=env, capture_output=True, text=True, timeout=30
        )
        assert r.returncode == 0, r.stderr
        table = np.frombuffer(cov._map, dtype=_ENTRY_DTYPE, count=cov.num_entries)
        return {int(e): i for i, e in enumerate(table["edge_id"]) if e}
    finally:
        cov.cleanup()


@requires_clang
@pytest.mark.parametrize("size", [POW2_SIZE, ODD_SIZE])
def test_home_slot_matches_modulo(target, size):
    ids = [1, size - 1, size, size + 2, 3 * size + 7, MAX_ID - 1]  # distinct homes
    got = _slots(target, size, ids)
    assert got == {i: i % size for i in ids}


@requires_clang
@pytest.mark.parametrize("size", [POW2_SIZE, ODD_SIZE])
def test_probe_wraps_past_table_end(target, size):
    """Adversarial: every id homes on size-2, so probing must wrap to slot 0."""
    home = size - 2
    ids = [home + k * size for k in range(5)]
    got = _slots(target, size, ids)
    assert got == {eid: (home + k) % size for k, eid in enumerate(ids)}


@requires_clang
@pytest.mark.parametrize("size", TINY_SIZES)
def test_tiny_sizes(target, size):
    """Edge: size 1 wraps the reciprocal to 0; size 3 is the smallest odd."""
    eid = MAX_ID - 1
    assert _slots(target, size, [eid]) == {eid: eid % size}


@requires_clang
def test_control_non_pow2_not_masked(target):
    """Falsification: a mask applied to a non-power-of-two size would differ."""
    eid = ODD_SIZE + 5
    assert eid & (ODD_SIZE - 1) != eid % ODD_SIZE
    assert _slots(target, ODD_SIZE, [eid]) == {eid: eid % ODD_SIZE}


def test_hot_path_has_no_runtime_modulo():
    src = Path(SHIM).read_text()
    assert "(pos + i) % __afl_map_size" not in src
    assert "% __afl_map_size" not in src
