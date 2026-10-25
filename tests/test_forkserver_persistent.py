"""Persistent mode (``__AFL_LOOP``) and deferred init (``__AFL_INIT``).

Every exec forked a child that ran ``main()`` from the top. A harness that
wraps its body in ``while (__AFL_LOOP(n))`` now keeps one child for up to n
inputs: the shim raises SIGSTOP between inputs and the forkserver resumes it
with SIGCONT. ``__AFL_INIT()`` moves the forkserver start past the harness's
own setup. Both macros embed AFL++'s signature strings; the fuzzer finds them
in the binary and only then sets ``__AFL_PERSISTENT`` / ``__AFL_DEFER_FORKSRV``.
"""

from __future__ import annotations

import re
import shutil
import signal
import subprocess
from pathlib import Path

import pytest

from fuzzer_tool.adapters.forkserver import ForkserverRunner, _ensure_compiled, mode_env
from fuzzer_tool.adapters.shm import ShmCoverage
from fuzzer_tool.core.elf import AflMode, detect_afl_modes

REPO = Path(__file__).resolve().parents[1]
SHIM = REPO / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"
TIMEOUT = 2.0
LOOP_MAX = 3
MAP = 8192

# One stderr line per input: pid and the child's own iteration count.
HARNESS = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>

int main(int argc, char **argv) {
    const char *log = getenv("PROBE_SETUP_LOG");
    if (log) {                       /* expensive setup, before __AFL_INIT */
        int fd = open(log, O_WRONLY | O_APPEND | O_CREAT, 0600);
        if (fd >= 0) { write(fd, "s\n", 2); close(fd); }
    }
#ifdef USE_INIT
    __AFL_INIT();
#endif
    unsigned char buf[64];
    int iter = 0;
#ifdef USE_LOOP
    while (__AFL_LOOP(LOOP_MAX)) {
#else
    {
#endif
        ssize_t n = read(0, buf, sizeof buf);
        iter++;
        fprintf(stderr, "pid=%d iter=%d\n", (int)getpid(), iter);
        if (n >= 5 && memcmp(buf, "CRASH", 5) == 0) *(volatile int *)0 = 1;
        if (n >= 4 && memcmp(buf, "HANG", 4) == 0) for (;;) pause();
        if (n > 0 && buf[0] == 'a') { volatile int x = 1; (void)x; }
        if (n > 0 && buf[0] == 'b') { volatile int y = 2; (void)y; }
    }
    return 0;
}
"""


def _build(d: Path, name: str, *defines: str) -> Path:
    src = d / "h.c"
    src.write_text(HARNESS)
    exe = d / name
    cmd = [
        "clang", "-O0", "-fsanitize-coverage=trace-pc-guard", f"-DLOOP_MAX={LOOP_MAX}",
        *(f"-D{x}" for x in defines), "-include", str(SHIM), "-o", str(exe), str(src),
    ]  # fmt: skip
    built = subprocess.run(cmd, capture_output=True, text=True)
    if built.returncode != 0:
        pytest.skip(f"harness did not build: {built.stderr[-300:]}")
    return exe


@pytest.fixture(scope="module")
def exes(tmp_path_factory):
    if shutil.which("clang") is None:
        pytest.skip("clang not available")
    if _ensure_compiled() is None:
        pytest.skip("fuzz_loader failed to compile")
    d = tmp_path_factory.mktemp("persist")
    return {
        "loop": _build(d, "loop", "USE_LOOP"),
        "init": _build(d, "init", "USE_INIT"),
        "plain": _build(d, "plain"),
    }


@pytest.fixture
def runner_for(exes):
    made = []

    def make(
        kind: str,
        env: dict[str, str] | None = None,
        shm: ShmCoverage | None = None,
        extra: dict[str, str] | None = None,
    ):
        exe = exes[kind]
        full = dict(env if env is not None else mode_env(str(exe)))
        full.update(extra or {})
        if shm is not None:
            full.update(__AFL_SHM_ID=shm.env_id, AFL_MAP_SIZE=str(shm.size))
        r = ForkserverRunner(str(exe), timeout=TIMEOUT, env=full)
        made.append(r)
        if not r.start():
            pytest.skip("forkserver failed to start")
        return r

    yield make
    for r in made:
        r.stop()


def _pid_iter(err: str) -> tuple[int, int]:
    m = re.search(r"pid=(\d+) iter=(\d+)", err)
    assert m, f"no probe line in stderr: {err!r}"
    return int(m.group(1)), int(m.group(2))


def test_detects_signatures(exes):
    assert detect_afl_modes(str(exes["loop"])) == {AflMode.PERSISTENT}
    assert detect_afl_modes(str(exes["init"])) == {AflMode.DEFERRED}
    assert detect_afl_modes(str(exes["plain"])) == set()


def test_mode_env(exes):
    assert mode_env(str(exes["loop"])) == {"__AFL_PERSISTENT": "1"}
    assert mode_env(str(exes["init"])) == {"__AFL_DEFER_FORKSRV": "1"}
    assert mode_env(str(exes["plain"])) == {}


def test_regression_persistent_loop_reuses_child(runner_for):
    r = runner_for("loop")
    seen = [_pid_iter(r.run_one(b"x")[1]) for _ in range(LOOP_MAX)]
    assert len({pid for pid, _ in seen}) == 1
    assert [it for _, it in seen] == list(range(1, LOOP_MAX + 1))


def test_loop_without_env_forks_per_exec(runner_for):
    """Falsification: no __AFL_PERSISTENT, one input per child as before."""
    r = runner_for("loop", env={})
    seen = [_pid_iter(r.run_one(b"x")[1]) for _ in range(LOOP_MAX)]
    assert len({pid for pid, _ in seen}) == LOOP_MAX
    assert {it for _, it in seen} == {1}


def test_loop_budget_ends_child(runner_for):
    """After LOOP_MAX inputs the child exits; the next exec gets a fresh one."""
    r = runner_for("loop")
    first = [_pid_iter(r.run_one(b"x")[1]) for _ in range(LOOP_MAX)]
    pid, it = _pid_iter(r.run_one(b"x")[1])
    assert pid != first[0][0] and it == 1


def test_crash_then_fresh_child(runner_for):
    """Adversarial: a crash mid-loop is reported and the next input runs fresh."""
    r = runner_for("loop")
    pid0, _ = _pid_iter(r.run_one(b"x")[1])
    rc, err = r.run_one(b"CRASH")
    # A signal, or the sanitizer runtime's report and exit (as without --loop).
    assert rc == -signal.SIGSEGV or (rc != 0 and "SEGV" in err)
    pid, it = _pid_iter(r.run_one(b"x")[1])
    assert pid != pid0 and it == 1


def test_hang_then_fresh_child(runner_for):
    r = runner_for("loop")
    r.run_one(b"x")
    assert r.run_one(b"HANG")[0] == -1
    _, it = _pid_iter(r.run_one(b"x")[1])
    assert it == 1


def test_iterations_do_not_inherit_run_state(runner_for):
    """Adversarial: same input in iteration 2, after different predecessors.

    The path hash accumulates across edges; one left over from the
    predecessor makes the same path hash differently, and _scan_new's
    known-path cache keys on it. Only middle iterations compare: the first
    also covers main()'s prologue and the last its return path.
    """
    shm = ShmCoverage(size=MAP)
    try:
        r = runner_for("loop", shm=shm)
        edges, hashes = [], []
        # Three children of LOOP_MAX inputs; iteration 2 is index 1, 4, 7.
        for data in (b"x", b"a", b"x", b"b", b"a", b"b", b"x", b"b", b"x"):
            shm.reset_edge_map()
            r.run_one(data)
            edges.append(frozenset(shm.get_edge_ids()))
            hashes.append(shm.read_path_hash())
        assert edges[1] == edges[4]
        assert hashes[1] == hashes[4]
        assert edges[1] != edges[7], "falsification: a different input must differ"
        assert hashes[1] != hashes[7]
    finally:
        shm.cleanup()


def _setup_runs(runner_for, tmp_path: Path, env: dict[str, str] | None) -> int:
    log = tmp_path / "setup.log"
    r = runner_for("init", env=env, extra={"PROBE_SETUP_LOG": str(log)})
    for _ in range(LOOP_MAX):
        assert r.run_one(b"x")[0] == 0
    return log.read_text().count("s")


def test_deferred_init_runs_setup_once(runner_for, tmp_path):
    assert _setup_runs(runner_for, tmp_path, None) == 1


def test_undeferred_runs_setup_per_exec(runner_for, tmp_path):
    """Falsification: without the env, __AFL_INIT is a no-op and setup reruns."""
    assert _setup_runs(runner_for, tmp_path, {}) == LOOP_MAX


def test_fuzzer_wires_mode_env(exes, tmp_path):
    """The Fuzzer's own forkserver gets the env for the harness's macros."""
    from fuzzer_tool.services.fuzzer import Fuzzer

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "s0").write_bytes(b"x")
    f = Fuzzer(
        target=str(exes["loop"]),
        corpus_dir=str(corpus),
        crashes_dir=str(tmp_path / "crashes"),
        use_coverage=True,
        seed=7,
    )
    try:
        fs = f._forkserver
        if fs is None:
            pytest.skip("fuzzer chose no forkserver here")
        assert fs.env_overrides.get("__AFL_PERSISTENT") == "1"
    finally:
        if f._forkserver is not None:
            f._forkserver.stop()
