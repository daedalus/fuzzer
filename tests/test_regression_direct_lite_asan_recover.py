"""Regression: direct_lite died on the first ASAN report of a fuzzgoat run.

The installed `fuzzer-tool` wrapper preloads libasan for an ASAN `.so` with
``halt_on_error=0``, and direct_lite is chosen. ``halt_on_error=0`` only
works for code built with ``-fsanitize-recover=address``; build_targets.sh
never passed it, so every ASAN check was fatal: ASAN's Die() ran
``_exit(1)`` (abort_on_error=0) and took the fuzzer with it -- silently,
since stderr is a capture pipe at that moment. With abort_on_error=1 the
guard's siglongjmp escaped Die() holding ASAN's report lock, and the next
report hit "nested bug" and exited. Measured with `fuzzer-tool fuzz
fuzzgoat_read_asan.so` on the injected-bug seeds: exit 1 during seed
calibration, no stats.

With recover-mode objects ASAN reports and returns; ``suppress_equal_pcs=0``
keeps a long-lived in-process run reporting every repeat of a bug site.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from fuzzer_tool.cli import ldpreload_wrapper as wrapper
from fuzzer_tool.services.fuzzer import _asan_fatal_in_process
from tests.conftest import requires_clang

SCRIPT = Path(__file__).parent.parent / "tools" / "build_targets.sh"

RECOVER = "-fsanitize-recover=address"

BUG_SRC = """
#include <stddef.h>
#include <stdlib.h>
int fuzz_shm_run(const unsigned char *b, size_t n) {
    char *p = malloc(4);
    int v = (n > 4) ? p[n] : 0;   /* heap read the checker must see */
    free(p);
    return v;
}
"""


# ── build: one ASAN flag set, recover mode, used at every site ──────────


@pytest.fixture(scope="module")
def script_text() -> str:
    return SCRIPT.read_text()


def test_regression_asan_cflags_recover(script_text):
    m = re.search(r'^ASAN_CFLAGS="([^"]*)"', script_text, re.M)
    assert m, "ASAN_CFLAGS not defined"
    assert "-fsanitize=address" in m.group(1)
    assert RECOVER in m.group(1)


def test_no_target_pass_bypasses_asan_cflags(script_text):
    """Adversarial: a literal -fsanitize=address on an _asan build pass
    silently reintroduces fatal checks for that pass."""
    offenders = [
        line.strip()
        for line in script_text.splitlines()
        if re.search(r'"_asan[a-z_]*"\s+"-fsanitize=address"', line)
    ]
    assert offenders == []


# ── runtime: wrapper keeps repeat reports ────────────────────────────


class _Exec(Exception):
    """Stands in for execvpe so main() returns control to the test."""


def _wrapper_opts(monkeypatch, tmp_path, preset: str) -> list[str]:
    target = tmp_path / "t.so"
    target.write_bytes(b"")
    monkeypatch.setattr(wrapper.sys, "argv", ["fuzzer-tool", "fuzz", str(target)])
    monkeypatch.setattr(wrapper, "_detect_asan", lambda t: True)
    monkeypatch.setattr(wrapper, "_detect_ubsan", lambda t: False)
    monkeypatch.setattr(wrapper, "_resolve_asan", lambda: None)
    monkeypatch.setattr(wrapper.os, "execvpe", lambda *a: (_ for _ in ()).throw(_Exec()))
    monkeypatch.setenv("ASAN_OPTIONS", preset)
    with pytest.raises(_Exec):
        wrapper.main()
    return wrapper.os.environ["ASAN_OPTIONS"].split(":")


def test_regression_wrapper_reports_repeat_sites(monkeypatch, tmp_path):
    assert "suppress_equal_pcs=0" in _wrapper_opts(monkeypatch, tmp_path, "")


def test_wrapper_keeps_user_suppress_choice(monkeypatch, tmp_path):
    """Falsification: a user-set key wins over the default."""
    opts = _wrapper_opts(monkeypatch, tmp_path, "suppress_equal_pcs=1")
    assert "suppress_equal_pcs=1" in opts
    assert "suppress_equal_pcs=0" not in opts


# ── detection: fatal ASAN checks in an in-process target ─────────────


def _build(tmp_path: Path, name: str, *flag_sets: str) -> Path:
    """Compile BUG_SRC split across objects, one per flag set, into one .so."""
    objs = []
    for i, flags in enumerate(flag_sets):
        src = tmp_path / f"{name}{i}.c"
        src.write_text(BUG_SRC.replace("fuzz_shm_run", f"part{i}") if i else BUG_SRC)
        obj = tmp_path / f"{name}{i}.o"
        r = subprocess.run(
            ["clang", "-O1", "-fPIC", *flags.split(), "-c", str(src), "-o", str(obj)],
            capture_output=True,
        )
        if r.returncode != 0:
            pytest.skip(f"clang cannot build {flags!r}: {r.stderr[-200:]!r}")
        objs.append(str(obj))
    so = tmp_path / f"{name}.so"
    link = ["clang", "-shared", "-o", str(so), *objs]
    if any("address" in f for f in flag_sets):
        link.append("-lasan")
    r = subprocess.run(link, capture_output=True)
    if r.returncode != 0:
        pytest.skip(f"cannot link ASAN .so: {r.stderr[-200:]!r}")
    return so


@requires_clang
def test_regression_plain_asan_is_fatal(tmp_path):
    assert _asan_fatal_in_process(str(_build(tmp_path, "plain", "-fsanitize=address")))


@requires_clang
def test_recover_asan_is_not_fatal(tmp_path):
    """Falsification: a recover build is the supported in-process build."""
    so = _build(tmp_path, "rec", f"-fsanitize=address {RECOVER}")
    assert not _asan_fatal_in_process(str(so))


@requires_clang
def test_mixed_recover_and_fatal_is_fatal(tmp_path):
    """Adversarial: a recover wrapper over a non-recover library object
    still dies on the library's first report."""
    so = _build(tmp_path, "mixed", f"-fsanitize=address {RECOVER}", "-fsanitize=address")
    assert _asan_fatal_in_process(str(so))


@requires_clang
def test_unsanitized_is_not_fatal(tmp_path):
    """Adversarial: no ASAN at all is not an ASAN problem."""
    assert not _asan_fatal_in_process(str(_build(tmp_path, "nosan", "")))


def test_unreadable_target_is_not_fatal(tmp_path):
    """Adversarial: nm failure must not raise or claim fatal."""
    assert not _asan_fatal_in_process(str(tmp_path / "missing.so"))
