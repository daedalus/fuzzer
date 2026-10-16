"""Clang ``trace-pc`` edge ids must not alias neighbouring blocks.

Clang fed the raw PC offset (``key >> 1``) into ``(prev >> 1) ^ cur``.
Neighbouring blocks share low bits, so distinct edges merged: ~2-3% of
edges on irregular generated code. trace-pc-guard instruments the same
blocks with mixed ids, so its id count is the oracle.
"""

from __future__ import annotations

import os
import subprocess

import pytest

from fuzzer_tool.adapters.shm import VIRGIN_DENSE_MAX, ShmCoverage
from fuzzer_tool.core.elf import detect_edge_id_scheme
from fuzzer_tool.core.rand_pool import RandPool
from tests.conftest import requires_clang
from tests.test_gcc_trace_pc import requires_gcc12

SHIM = os.path.join(os.path.dirname(__file__), "..", "src", "fuzzer_tool", "adapters", "afl_shim.c")
MAP_ENTRIES = 1 << 18
TRACE_PC = "-fsanitize-coverage=trace-pc"
GUARD = "-fsanitize-coverage=trace-pc-guard"
_FUNCS = 500
_SEED = 3
_DRIVER_ITERS = 600
# Mixed 24-bit ids still merge ~E^2 / 2^25 (~1 at E ~ 6k); raw ids lost ~200.
_ALIAS_TOL = 16


def _irregular_source(funcs: int, seed: int) -> str:
    """Random call graph, branch counts and body sizes: irregular block layout."""
    rng = RandPool(seed)
    out = ["#include <stdint.h>\n#include <unistd.h>\nvolatile unsigned sink;"]
    out += [f"static unsigned f{i}(unsigned x, int d);" for i in range(funcs)]

    for i in range(funcs):
        arms = []
        for _ in range(rng.randint(1, 6)):
            callee, k, mod = rng.randrange(funcs), rng.randint(1, 1 << 20), rng.randint(2, 7)
            stmts = " ".join(
                f"sink ^= x * {rng.randint(1, 999)}u;" for _ in range(rng.randint(0, 5))
            )
            arms.append(
                f"if ((x ^ {k}u) % {mod}u == 0) {{ {stmts} "
                f"if (d < 6) x += f{callee}(x * 2654435761u + {k}u, d + 1); }}"
            )
        out.append(
            f"__attribute__((noinline)) static unsigned f{i}(unsigned x, int d) "
            f"{{ {' '.join(arms)} return x + {i}u; }}"
        )

    out.append(
        "__attribute__((noinline)) static int entry(const uint8_t *p, size_t n) {\n"
        f"  (void)p; (void)n; for (unsigned s = 0; s < {_DRIVER_ITERS}u; s++)\n"
        f"    sink += f0(s * 40503u, 0) + f{funcs // 2}(s, 0) + f{funcs - 1}(s ^ 77u, 0);\n"
        "  return 0;\n}\n"
        'int main(void) { __afl_guarded_call(entry, (const uint8_t *)"", 0); _exit(0); }'
    )
    return "\n".join(out)


def _build(tmp_path, name: str, cc: str, *flags: str) -> str:
    src = tmp_path / "irregular.c"
    if not src.exists():
        src.write_text(_irregular_source(_FUNCS, _SEED))
    exe = tmp_path / name
    r = subprocess.run(
        [
            cc,
            "-O1",
            *flags,
            "-D__AFL_CTX_SENSITIVE=0",
            "-include",
            SHIM,
            "-o",
            str(exe),
            str(src),
            "-ldl",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr[-1500:]
    return str(exe)


def _ids(exe: str) -> list[int]:
    cov = ShmCoverage(size=MAP_ENTRIES)
    try:
        env = dict(os.environ, __AFL_SHM_ID=str(cov.shm_id), AFL_MAP_SIZE=str(MAP_ENTRIES))
        env.pop("LD_PRELOAD", None)
        subprocess.run([exe], env=env, capture_output=True, timeout=60, check=True)
        return cov.get_edge_ids()
    finally:
        cov.cleanup()


@pytest.fixture(scope="module")
def builds(tmp_path_factory):
    d = tmp_path_factory.mktemp("tracepc_alias")
    return {
        "trace_pc": _build(d, "tpc", "clang", TRACE_PC),
        "guard": _build(d, "guard", "clang", GUARD),
    }


@pytest.fixture(scope="module")
def ids(builds):
    return {k: _ids(v) for k, v in builds.items()}


@requires_clang
def test_oracle_matches_itself(builds, ids):
    """Control: the guard build rerun reports the same id count."""
    assert len(_ids(builds["guard"])) == len(ids["guard"])


@requires_clang
def test_regression_clang_tracepc_edges_not_merged(ids):
    assert len(ids["trace_pc"]) >= len(ids["guard"]) - _ALIAS_TOL


@requires_clang
def test_clang_tracepc_ids_fit_dense_map(ids):
    assert max(ids["trace_pc"]) < VIRGIN_DENSE_MAX


@requires_clang
def test_clang_tracepc_reports_scheme_3(builds):
    """Old state from raw-PC ids must be refused, not silently mixed."""
    assert detect_edge_id_scheme(builds["trace_pc"]) == 3


@requires_clang
def test_clang_guard_scheme_unchanged(builds):
    """Falsification: guard ids did not change, so their scheme must not."""
    assert detect_edge_id_scheme(builds["guard"]) == 2


@requires_gcc12
def test_gcc_tracepc_scheme_unchanged(tmp_path):
    """gcc trace-pc ids were already mixed; resume of their state stays allowed."""
    assert detect_edge_id_scheme(_build(tmp_path, "gcc_tpc", "gcc", TRACE_PC)) == 2
