"""`edge-dump` / `minimize` on an ASAN `.so`: the loader child needs libasan first.

Only `fuzz`, and since the verify fix `verify`/`replay`, put libasan in
LD_PRELOAD. Every other in-process command died in the loader child with
"ASan runtime does not come first in initial library list" (2026-11-07,
`edge-dump fuzzgoat_read_asan.so`).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import requires_clang
from tests.test_regression_shim_audit import _build, _clean_env

ASAN_ORDER_ERROR = "ASan runtime does not come first"
LOADER_FAILED = "Persistent loader failed to start"

_SO_SRC = """
#include <stddef.h>
int fuzz_shm_run(const unsigned char *d, size_t n) { return n && d[0] == 'Z'; }
"""


@pytest.fixture
def asan_so(tmp_path) -> Path:
    sub = tmp_path / "so"
    sub.mkdir()
    return _build(sub, _SO_SRC, "-fsanitize-coverage=trace-pc-guard", shared=True, asan=True)


@pytest.fixture
def corpus(tmp_path) -> Path:
    seeds = tmp_path / "corpus" / "seeds"
    seeds.mkdir(parents=True)
    for name, data in {"a": b"a", "z": b"Z"}.items():
        (seeds / name).write_bytes(data)
    return seeds.parent


def _cli(tmp: Path, *argv: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "fuzzer_tool", *argv],
        capture_output=True, text=True, timeout=120, cwd=tmp, env=_clean_env(),
    )  # fmt: skip


@requires_clang
def test_regression_edge_dump_asan_so(tmp_path, asan_so, corpus):
    """Falsification: edge-dump of an ASAN .so writes its dump."""
    out = tmp_path / "edges.pkl"
    r = _cli(tmp_path, "edge-dump", str(asan_so), "-d", str(corpus), "-o", str(out), "--inprocess")

    assert ASAN_ORDER_ERROR not in r.stdout + r.stderr
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    assert out.is_file()


@requires_clang
def test_regression_minimize_asan_so(tmp_path, asan_so, corpus):
    """Adversarial: minimize of an ASAN .so keeps its seeds, not 'crashes' on all."""
    r = _cli(tmp_path, "minimize", str(asan_so), "-d", str(corpus), "--inprocess")

    assert ASAN_ORDER_ERROR not in r.stdout + r.stderr
    assert LOADER_FAILED not in r.stdout + r.stderr
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    assert any((corpus / "seeds").iterdir())
