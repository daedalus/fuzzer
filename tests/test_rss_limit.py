"""--rss-limit-mb: bound a spawned target's memory.

No limit existed: an input that makes the target allocate without bound ran
until the kernel OOM killer or the timeout. Plain targets get RLIMIT_AS on
the child; sanitizer targets get hard_rss_limit_mb / malloc_limit_mb (their
shadow memory cannot live under RLIMIT_AS). In-process modes are not limited.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from fuzzer_tool.adapters import process
from fuzzer_tool.adapters.forkserver import ForkserverRunner, _ensure_compiled
from fuzzer_tool.core.sanitizer import SanitizerReport
from fuzzer_tool.services import fuzzer as fuzzer_mod

_LIMIT_MB = 64
_ALLOC_MB = 512
_MALLOC_FAILED = 42

# Reads stdin first, so the limit set right after spawn is in place.
_SRC = rf"""
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
int main(void) {{
    char c;
    if (read(0, &c, 1) != 1 || c != 'M') return 0;
    char *p = malloc((size_t){_ALLOC_MB} << 20);
    if (!p) return {_MALLOC_FAILED};
    memset(p, 1, (size_t){_ALLOC_MB} << 20);
    return p[(unsigned char)c] == 1 ? 0 : 1;  /* observable: no allocation elision */
}}
"""


@pytest.fixture
def hog(tmp_path) -> str:
    if not shutil.which("clang"):
        pytest.skip("clang not installed")
    src = tmp_path / "hog.c"
    src.write_text(_SRC)
    exe = tmp_path / "hog"
    subprocess.run(["clang", "-O0", "-o", str(exe), str(src)], check=True)
    return str(exe)


@pytest.fixture
def limit():
    process.set_rss_limit(_LIMIT_MB)
    yield
    process.set_rss_limit(0)


def _forkserver_rc(target: str, env: dict[str, str]) -> int | None:
    if _ensure_compiled() is None:
        return None
    fs = ForkserverRunner(target, timeout=10.0, env=env)
    try:
        assert fs.start()
        return fs.run_one(b"M")[0]
    finally:
        fs.stop()


def test_rss_limit_stops_allocation(hog, limit) -> None:
    rc, _, _ = process.run_target_stdin(hog, b"M", 10.0)

    assert rc == _MALLOC_FAILED


def test_rss_limit_off_allocation_succeeds(hog) -> None:
    # Falsification: no limit, the same input allocates.
    rc, _, _ = process.run_target_stdin(hog, b"M", 10.0)

    assert rc == 0


def test_rss_limit_reaches_fuzz_loader(hog, monkeypatch) -> None:
    env = {**os.environ, process.RSS_LIMIT_ENV: str(_LIMIT_MB)}
    rc = _forkserver_rc(hog, env)
    if rc is None:
        pytest.skip("fuzz_loader failed to compile")

    assert rc == _MALLOC_FAILED


def test_small_input_unaffected(hog, limit) -> None:
    # Edge: a run under the limit is untouched.
    rc, _, _ = process.run_target_stdin(hog, b"q", 10.0)

    assert rc == 0


# ── Sanitizer targets: runtime options, and their reports ──────────────


def test_regression_shadow_target_gets_no_rlimit(monkeypatch) -> None:
    # Adversarial: RLIMIT_AS on an ASAN target kills it at startup
    # (shadow reservation), so only the runtime options carry the limit.
    for var in (*fuzzer_mod._SANITIZER_OPTION_VARS, process.RSS_LIMIT_ENV):
        monkeypatch.setenv(var, "x")
        monkeypatch.delenv(var)
    fuzzer_mod._export_rss_limit(_LIMIT_MB, fuzzer_mod.Shadow.RESERVED)

    assert process.RSS_LIMIT_ENV not in os.environ
    assert process._rss_limit_bytes == 0
    assert f"hard_rss_limit_mb={_LIMIT_MB}" in os.environ["ASAN_OPTIONS"]


def test_shadow_detected_on_asan_build(tmp_path) -> None:
    if not shutil.which("clang"):
        pytest.skip("clang not installed")
    src = tmp_path / "a.c"
    src.write_text("int main(void) { return 0; }\n")
    exe = tmp_path / "a"
    r = subprocess.run(
        ["clang", "-fsanitize=address", "-o", str(exe), str(src)], capture_output=True
    )
    if r.returncode:
        pytest.skip("ASAN runtime not installed")

    assert fuzzer_mod._shadow_of(str(exe)) is fuzzer_mod.Shadow.RESERVED
    assert fuzzer_mod._shadow_of("/bin/true") is fuzzer_mod.Shadow.NONE


def test_sanitizer_options_carry_limit(monkeypatch) -> None:
    monkeypatch.setattr(process, "_rss_limit_bytes", 0)  # restored after the test
    for var in fuzzer_mod._SANITIZER_OPTION_VARS:
        monkeypatch.setenv(var, "detect_leaks=0")
    monkeypatch.setenv(process.RSS_LIMIT_ENV, "x")
    monkeypatch.delenv(process.RSS_LIMIT_ENV)
    fuzzer_mod._export_rss_limit(_LIMIT_MB, fuzzer_mod.Shadow.NONE)

    for var in fuzzer_mod._SANITIZER_OPTION_VARS:
        opts = os.environ[var].split(":")
        assert "detect_leaks=0" in opts
        assert f"hard_rss_limit_mb={_LIMIT_MB}" in opts
        assert f"malloc_limit_mb={_LIMIT_MB}" in opts
    assert os.environ[process.RSS_LIMIT_ENV] == str(_LIMIT_MB)


def test_rss_limit_zero_exports_nothing(monkeypatch) -> None:
    # Falsification: off by default.
    for var in (*fuzzer_mod._SANITIZER_OPTION_VARS, process.RSS_LIMIT_ENV):
        monkeypatch.setenv(var, "x")
        monkeypatch.delenv(var)
    fuzzer_mod._export_rss_limit(0, fuzzer_mod.Shadow.NONE)

    assert all(v not in os.environ for v in fuzzer_mod._SANITIZER_OPTION_VARS)
    assert process.RSS_LIMIT_ENV not in os.environ


@pytest.mark.parametrize(
    ("header", "want"),
    [
        ("hard rss limit exhausted (64Mb vs 70Mb)", "rss-limit-exhausted"),
        ("out of memory: allocator is trying to allocate 0x20000000 bytes", "out-of-memory"),
    ],
)
def test_regression_asan_memory_report_types(header: str, want: str) -> None:
    report = SanitizerReport.parse(f"==1==ERROR: AddressSanitizer: {header}\n")

    assert report is not None
    assert report.error_type == want
