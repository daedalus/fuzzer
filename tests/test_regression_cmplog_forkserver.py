"""Regression: cmplog kept every ``fuzzer-tool fuzz`` run off the forkserver.

The CLI always passes ``cmplog=True``, and ``_setup_forkserver`` bailed on
``self._cmplog``, so every executable run took the stdin spawn path
(fuzzgoat ASAN: 41 eps campaign vs 79 eps raw). With the FIFO sink the
cmplog environment is run-invariant, and the shim zeroes its counters before
forking, so the loader can carry it.

Built with clang: edge callbacks need -fsanitize-coverage=trace-pc-guard.
"""

import enum
import os
import shutil
import subprocess

import pytest

from fuzzer_tool.adapters.forkserver import _ensure_compiled
from fuzzer_tool.services import fuzzer as fuzzer_mod
from tests.conftest import requires_clang

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHIM = os.path.join(_ROOT, "src", "fuzzer_tool", "adapters", "afl_shim.c")

# One memcmp per run: the counter vector and the record stream have one
# known source. INPUTS miss and hit it.
TOKEN = b"QXZT"
INPUTS = (b"AAAA", TOKEN, b"AAAA")
_TARGET = """
#include <stdio.h>
#include <string.h>

int main(void) {
    unsigned char buf[64];
    size_t n = fread(buf, 1, sizeof(buf), stdin);
    if (n >= 4 && memcmp(buf, "QXZT", 4) == 0) return 3;
    return 0;
}
"""


@pytest.fixture(scope="module")
def target(tmp_path_factory):
    if not shutil.which("clang"):
        pytest.skip("clang not installed")
    if _ensure_compiled() is None:
        pytest.skip("fuzz_loader failed to compile")
    d = tmp_path_factory.mktemp("cmplog_fsrv")
    src = d / "t.c"
    src.write_text(_TARGET)
    exe = d / "t"
    cmd = [
        "clang",
        "-O0",
        "-fno-builtin",
        "-fsanitize-coverage=trace-pc-guard",
        "-D__AFL_CMPLOG=1",
        "-include",
        SHIM,
        "-o",
        str(exe),
        str(src),
        "-ldl",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"target failed to build: {r.stderr[:300]}")
    return str(exe)


class Sink(enum.Enum):
    """Cmplog record sink; NONE turns cmplog off."""

    FIFO = "fifo"
    FILE = "file"
    NONE = "none"


class Exec(enum.Enum):
    """Executable backend under test."""

    FORKSERVER = "forkserver"
    SPAWN = "spawn"


def _fuzzer(target, tmp_path, sink: Sink, backend: Exec = Exec.FORKSERVER):
    from fuzzer_tool.services.fuzzer import Fuzzer

    f = Fuzzer(
        target=target,
        corpus_dir=str(tmp_path / "corpus"),
        crashes_dir=str(tmp_path / "crashes"),
        max_len=64,
        timeout=2,
        mutations_per_input=1,
        cmplog=sink is not Sink.NONE,
        cmplog_workdir=str(tmp_path / "cmplog"),
        cmplog_fifo_sink=sink is Sink.FIFO,
        forkserver=backend is Exec.FORKSERVER,
    )
    return f


def _teardown(f) -> None:
    """Stop loader + collector and hand os.environ back."""
    if f._forkserver is not None:
        f._forkserver.stop()
    if f._cmplog is not None:
        f._cmplog.stop()
        f._cmplog.restore_env()
    if f.shm_cov is not None:
        f.shm_cov.cleanup()
    fuzzer_mod._restore_environ()


@pytest.fixture
def fifo_fuzzer(target, tmp_path):
    f = _fuzzer(target, tmp_path, Sink.FIFO)
    try:
        yield f
    finally:
        _teardown(f)


@requires_clang
def test_regression_cmplog_fifo_uses_forkserver(fifo_fuzzer):
    """Falsification: with cmplog on (FIFO sink), the loader must be up."""
    f = fifo_fuzzer
    assert f._cmplog is not None, "cmplog not detected on a -D__AFL_CMPLOG=1 build"
    assert f._forkserver is not None and f._forkserver._ready


def _vectors(target, tmp_path, backend: Exec) -> list:
    """Per-exec (rc, fired, asserted) over INPUTS on one backend."""
    f = _fuzzer(target, tmp_path, Sink.FIFO, backend)
    try:
        assert (f._forkserver is not None) is (backend is Exec.FORKSERVER)
        f._cmplog.collect_counts()
        out = []
        for data in INPUTS:
            rc, _err = f._runner.run_target(data)
            out.append((rc, *f._cmplog.collect_counts()))
        return out
    finally:
        _teardown(f)


@requires_clang
def test_cmplog_counts_match_spawn_path(target, tmp_path):
    """Each forked child reports what a fresh spawn reports, per execution.

    Adversarial: the server's pre-fork strcmp("__AFL_FORKSRV") and any
    inherited counter state would add a phantom offset to every vector.
    Oracle is the spawn path; its self-comparison runs first (Hard Rule 46).
    """
    spawn_a = _vectors(target, tmp_path / "a", Exec.SPAWN)
    spawn_b = _vectors(target, tmp_path / "b", Exec.SPAWN)
    assert spawn_a == spawn_b, "control: spawn path not reproducible"
    assert all(fired.get("memcmp") for _rc, fired, _as in spawn_a), spawn_a

    fsrv = _vectors(target, tmp_path / "c", Exec.FORKSERVER)
    assert fsrv == spawn_a


@requires_clang
def test_cmplog_records_reach_collector_via_forkserver(fifo_fuzzer):
    """The record stream (token mining) still arrives from forked children."""
    f = fifo_fuzzer
    assert f._forkserver is not None

    f._runner.run_target(b"AAAA")
    f._cmplog.flush_shims()
    tokens = f._cmplog.collect_tokens()
    seen = set(tokens) | {p for pair in f._cmplog.pairs for p in pair}
    assert TOKEN in seen, (tokens[:8], f._cmplog.pairs[:8])


@requires_clang
def test_regression_startup_leaves_no_phantom_counts(fifo_fuzzer):
    """Adversarial: arming the loader must not preload the shim process-wide.

    It once did (setup_env_for_run), and every tool spawned during start-up
    dumped its own comparisons into the counts file: the first rounds read
    hundreds of thousands of strcmp calls as cmp progress.
    """
    f = fifo_fuzzer
    assert f._cmplog._shim_path not in os.environ.get("LD_PRELOAD", "")
    fired, asserted = f._cmplog.collect_counts()
    assert fired == {} and asserted == {}


@requires_clang
def test_cmplog_file_sink_keeps_spawn_path(target, tmp_path):
    """The file sink truncates per run, so its env is not run-invariant:
    the forkserver must stay off and the stdin path keep serving it."""
    f = _fuzzer(target, tmp_path, Sink.FILE)
    try:
        assert f._cmplog is not None
        assert f._forkserver is None
    finally:
        _teardown(f)


@requires_clang
def test_cmplog_off_still_uses_forkserver(target, tmp_path):
    """Control: no cmplog -> forkserver, as before the change."""
    f = _fuzzer(target, tmp_path, Sink.NONE)
    try:
        assert f._forkserver is not None and f._forkserver._ready
    finally:
        _teardown(f)
