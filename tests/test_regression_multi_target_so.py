"""Multi-target .so runs: every target runs in-process on its own SHM map.

Bug: one InProcessRunner was built for target 0 on the global map. Target
switches only changed ``Fuzzer.target``, so every exec ran target 0 and the
per-target maps the fuzzer read stayed empty: 0 edges for every target.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tests.conftest import requires_clang
from tests.test_regression_shim_audit import _build, _clean_env

# Writes a marker file named after the target on its first call: proof it ran.
_SO_SRC = """
#include <stdio.h>
#include <stddef.h>
static int ran;
int fuzz_shm_run(const unsigned char *d, size_t n) {
    if (!ran) { FILE *f = fopen("ran_%s", "w"); if (f) fclose(f); ran = 1; }
    if (n > 1 && d[0] == '%s') return d[1] == 'Z';
    return 0;
}
"""


def _so(tmp: Path, name: str) -> Path:
    sub = tmp / name
    sub.mkdir()
    return _build(sub, _SO_SRC % (name, name[0]), "-fsanitize-coverage=trace-pc-guard", shared=True)


def _fuzz(tmp: Path, *targets: Path) -> subprocess.CompletedProcess:
    corpus = tmp / "corpus"
    corpus.mkdir()
    (corpus / "seed").write_bytes(b"seed")
    return subprocess.run(
        [sys.executable, "-m", "fuzzer_tool", "fuzz", *map(str, targets), "-d", str(corpus),
         "-o", str(tmp / "crashes"), "--max-execs", "40", "--target-schedule", "round-robin"],
        capture_output=True, text=True, timeout=180, cwd=tmp, env=_clean_env(),
    )  # fmt: skip


@requires_clang
def test_regression_multi_target_so_coverage(tmp_path):
    """Falsification: each target's map records edges at the self-test."""
    r = _fuzz(tmp_path, _so(tmp_path, "alpha"), _so(tmp_path, "beta"))

    assert r.returncode == 0, r.stderr[-2000:]
    assert "recorded no edges" not in r.stdout
    assert "Shim self-test: 2 target(s) OK" in r.stdout


@requires_clang
def test_regression_multi_target_so_runs_each(tmp_path):
    """Adversarial: coverage alone could come from target 0; each must run."""
    r = _fuzz(tmp_path, _so(tmp_path, "alpha"), _so(tmp_path, "beta"))

    assert r.returncode == 0, r.stderr[-2000:]
    assert (tmp_path / "ran_alpha").exists()
    assert (tmp_path / "ran_beta").exists()
