"""Manual-only shim builds were sized by disassembly, at the cap.

A wrapper that links the shim but calls only hand-placed ``__afl_map_edge``
sites (system libraries uninstrumented) has no sancov section and no
trace-pc sites, so ``estimate_map_size_detail`` fell through to
branch density. That counts the whole ``.so``: png_read_asan.so read 5,713
"blocks" and asked for 524,288 entries for ~120 live edges.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

from fuzzer_tool.core.elf import MAP_SIZE_DEFAULT, estimate_map_size_detail

ROOT = Path(__file__).resolve().parents[1]
SHIM = ROOT / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"

_WRAPPER = """
#include <stddef.h>
int fuzz_shm_run(const unsigned char *d, size_t n) {
    if (n > 0 && d[0] == 'A') __afl_map_edge(1);
    if (n > 1 && d[1] == 'B') __afl_map_edge(2);
    __afl_map_edge(3);
    return 0;
}
"""


def _build(tmp: Path, name: str, src: str, flags: list[str]) -> str:
    c = tmp / f"{name}.c"
    c.write_text(src)
    out = tmp / f"{name}.so"
    cmd = ["clang", "-O1", "-shared", "-fPIC", *flags, "-o", str(out), str(c)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"{name} failed to build: {r.stderr[:300]}")
    return str(out)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    if shutil.which("clang") is None:
        pytest.skip("no clang")
    tmp = tmp_path_factory.mktemp("manual_shim")
    shim = ["-include", str(SHIM)]
    return {
        "manual": _build(tmp, "manual", _WRAPPER, shim),
        "guards": _build(tmp, "guards", _WRAPPER, [*shim, "-fsanitize-coverage=trace-pc-guard"]),
        "noshim": _build(tmp, "noshim", "int fuzz_shm_run(void) { return 0; }\n", []),
    }


def test_regression_map_size_manual_shim(built):
    est = estimate_map_size_detail(built["manual"])
    assert est.source == "shim_manual"
    assert est.entries == MAP_SIZE_DEFAULT
    assert not est.exact


def test_guard_build_keeps_exact_count(built):
    """Falsification: a sancov build must not be reclassified as manual."""
    est = estimate_map_size_detail(built["guards"])
    assert est.source == "sancov_guards"
    assert est.exact


def test_no_shim_still_uses_branch_density(built):
    """Adversarial: a plain .so, no shim -- the new tier must not fire."""
    assert estimate_map_size_detail(built["noshim"]).source == "branch_density"
