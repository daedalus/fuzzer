"""The fuzzer checks shim health on its own, in every execution mode.

  1. startup self-test: one execution per target before fuzzing; an
     instrumented target that records no edges is reported
  2. periodic counters: ``__afl_shim_health()`` read every
     SHIM_HEALTH_PERIOD executions in direct in-process mode
  3. stderr scan: ``__afl_shim:`` diagnostics in any execution's stderr
  4. multi-target startup: layout and stale-shim checks per target

Every issue is reported once per run (ShimWatch dedups).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from fuzzer_tool.core.shim_health import (
    SHIM_HEALTH_PERIOD,
    SHIM_STDERR_TAG,
    STDERR_SEEN_MAX,
    Attach,
    ShimField,
    ShimWatch,
    self_test_issues,
    shim_stderr_lines,
)
from fuzzer_tool.services.fuzzer import Fuzzer
from tests.conftest import requires_clang
from tests.test_regression_shim_audit import _build, _clean_env


def _counters(**over: int) -> tuple[int, ...]:
    c = [0] * len(ShimField)
    c[ShimField.ATTACHED] = 1
    c[ShimField.MAP_ENTRIES] = 8192
    for name, v in over.items():
        c[ShimField[name]] = v
    return tuple(c)


# ── core: stderr lines ─────────────────────────────────────────────────


def test_stderr_lines_keep_only_shim_diagnostics():
    text = (
        "libpng warning: x\n"
        f"{SHIM_STDERR_TAG} segment 5 is 8 bytes -- too small\n"
        f"  {SHIM_STDERR_TAG} indented still counts\n"
        "[shim] abort() intercepted\n"
    )

    assert shim_stderr_lines(text) == [
        f"{SHIM_STDERR_TAG} segment 5 is 8 bytes -- too small",
        f"{SHIM_STDERR_TAG} indented still counts",
    ]


def test_stderr_lines_ignore_tag_inside_other_text():
    """Adversarial: target output quoting the tag mid-line is not ours."""
    assert shim_stderr_lines(f"echo: {SHIM_STDERR_TAG} spoof\n") == []
    assert shim_stderr_lines("") == []


# ── core: ShimWatch ────────────────────────────────────────────────────


def test_watch_reports_each_stderr_line_once():
    w = ShimWatch()
    line = f"{SHIM_STDERR_TAG} shmat(3) failed"

    assert w.stderr(line + "\n") == [line]
    assert w.stderr(line + "\n") == []


def test_watch_stderr_memory_is_bounded():
    """Adversarial: a target printing a fresh line per exec cannot grow it."""
    w = ShimWatch()
    reported = sum(len(w.stderr(f"{SHIM_STDERR_TAG} n={i}\n")) for i in range(10 * STDERR_SEEN_MAX))

    assert reported == STDERR_SEEN_MAX
    assert w.seen_count() == STDERR_SEEN_MAX


def test_watch_counters_report_first_occurrence_only():
    w = ShimWatch()

    assert w.counters(_counters(), Attach.EXPECTED) == []
    first = w.counters(_counters(CMPLOG_DROPPED=3), Attach.EXPECTED)
    again = w.counters(_counters(CMPLOG_DROPPED=900), Attach.EXPECTED)

    assert len(first) == 1 and "3 cmplog record" in first[0]
    assert again == []


def test_watch_counters_detach_reported_once():
    w = ShimWatch()
    lost = _counters(ATTACHED=0)

    assert len(w.counters(lost, Attach.EXPECTED)) == 1
    assert w.counters(lost, Attach.EXPECTED) == []
    assert ShimWatch().counters(lost, Attach.NOT_EXPECTED) == []


def test_watch_counters_tolerate_old_and_new_shims():
    """Adversarial: fewer fields (older shim) or more (newer) never raise."""
    w = ShimWatch()

    assert w.counters((1, 8192), Attach.EXPECTED) == []
    assert w.counters((*_counters(), 7, 7), Attach.EXPECTED) == []


# ── core: self-test verdict ────────────────────────────────────────────


def test_self_test_flags_zero_edges_when_expected():
    issues = self_test_issues("t", 0, Attach.EXPECTED)

    assert len(issues) == 1 and "no edges" in issues[0] and "t" in issues[0]


def test_self_test_silent_when_unmeasurable_or_healthy():
    assert self_test_issues("t", None, Attach.EXPECTED) == []
    assert self_test_issues("t", 5, Attach.EXPECTED) == []
    assert self_test_issues("t", 0, Attach.NOT_EXPECTED) == []


# ── services: wiring on a bare Fuzzer ──────────────────────────────────


class _Shm:
    def __init__(self, edges: set[int]):
        self._edges = edges

    def get_edge_ids(self) -> set[int]:
        return set(self._edges)

    def reset_edge_map(self) -> None:
        pass


def _bare_fuzzer(stderr: str = "", counters=None, edges=None, targets=()) -> Fuzzer:
    """A Fuzzer with only the state the shim checks touch."""
    f = Fuzzer.__new__(Fuzzer)
    f.target = targets[0] if targets else "tgt"
    f.multi_targets = list(targets)
    f.use_coverage = True
    f.shm_cov = _Shm(edges if edges is not None else {1})
    f._target_shm_covs = {}
    f._runner = SimpleNamespace(run_target=lambda data: (0, stderr))
    f._inprocess_runner = (
        SimpleNamespace(shim_health=lambda: counters) if counters is not None else None
    )
    f._init_shim_watch()
    return f


def test_run_target_warns_on_shim_stderr_once(capsys):
    line = f"{SHIM_STDERR_TAG} AFL_MAP_SIZE=-5 is not a valid entry count"
    f = _bare_fuzzer(stderr=line + "\n")

    assert f._run_target(b"x") == (0, line + "\n")
    f._run_target(b"x")

    assert capsys.readouterr().out.count(line) == 1


def test_run_target_reads_counters_once_per_period(capsys):
    state = {"reads": 0}

    def health():
        state["reads"] += 1
        return _counters(STRAY_SIGNALS=1)

    f = _bare_fuzzer()
    f._inprocess_runner = SimpleNamespace(shim_health=health)

    for _ in range(2 * SHIM_HEALTH_PERIOD):
        f._run_target(b"x")

    assert state["reads"] == 2
    assert capsys.readouterr().out.count("crash signal") == 1


def test_run_target_without_inprocess_runner_never_reads(capsys):
    """Adversarial: exec/forkserver modes have no counters to read."""
    f = _bare_fuzzer()
    for _ in range(SHIM_HEALTH_PERIOD + 1):
        f._run_target(b"x")

    assert "WARNING" not in capsys.readouterr().out


def test_self_test_warns_on_empty_map(capsys):
    f = _bare_fuzzer(edges=set())
    f._shim_self_test(["tgt"])

    assert "no edges" in capsys.readouterr().out


def test_self_test_ok_line_when_healthy(capsys):
    f = _bare_fuzzer(edges={1, 2}, counters=_counters())
    f._shim_self_test(["tgt"])

    out = capsys.readouterr().out
    assert "Shim self-test: 1 target(s) OK" in out and "WARNING" not in out


def test_self_test_covers_each_target_and_restores_active(capsys):
    f = _bare_fuzzer(targets=("a", "b"))
    f._target_shm_covs = {"a": _Shm({1}), "b": _Shm(set())}
    f._shim_self_test(["a", "b"])

    out = capsys.readouterr().out
    assert "b" in out and "no edges" in out
    assert f.target == "a"


def test_self_test_skips_unmeasurable_target(capsys):
    """Adversarial: no SHM (ptrace, --no-shm) is not evidence of a fault."""
    f = _bare_fuzzer()
    f.shm_cov = None
    f._shim_self_test(["tgt"])

    assert "WARNING" not in capsys.readouterr().out


# ── services: multi-target static checks ───────────────────────────────


def _marker_binary(tmp_path: Path, body: str) -> Path:
    src = tmp_path / "m.c"
    src.write_text(body + "\nint main(void) { return 0; }\n")
    exe = tmp_path / "m"
    subprocess.run(["clang", str(src), "-o", str(exe)], check=True)
    return exe


@requires_clang
def test_multi_target_warns_on_stale_shim(tmp_path, capsys):
    stale = _marker_binary(tmp_path, "__attribute__((used)) const unsigned __afl_shm_layout_3 = 3;")
    f = _bare_fuzzer(targets=(str(stale),))
    f._target_shm_covs = {str(stale): _Shm({1})}
    f._check_target_shims([str(stale)])

    assert "__afl_scoped_crash_handler" in capsys.readouterr().out


@requires_clang
def test_multi_target_refuses_stale_layout(tmp_path):
    old = _marker_binary(tmp_path, "")
    f = _bare_fuzzer(targets=(str(old),))
    f._target_shm_covs = {str(old): _Shm({1})}

    with pytest.raises(RuntimeError, match="SHM layout 1"):
        f._check_target_shims([str(old)])


@requires_clang
def test_multi_target_current_shim_passes(tmp_path, capsys):
    exe = _build(tmp_path, "int main(void) { return 0; }", "-fsanitize-coverage=trace-pc-guard")
    f = _bare_fuzzer(targets=(str(exe),))
    f._target_shm_covs = {str(exe): _Shm({1})}
    f._check_target_shims([str(exe)])

    assert "WARNING" not in capsys.readouterr().out


# ── end to end ─────────────────────────────────────────────────────────


@requires_clang
def test_fuzz_run_self_test_passes_on_healthy_target(tmp_path):
    exe = _build(
        tmp_path,
        """
        #include <stdio.h>
        int main(void) {
            unsigned char b[64];
            size_t n = fread(b, 1, sizeof b, stdin);
            return n > 3 && b[0] == 'F' ? 1 : 0;
        }
        """,
        "-fsanitize-coverage=trace-pc-guard",
    )
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "seed").write_bytes(b"seed")
    r = subprocess.run(
        [sys.executable, "-m", "fuzzer_tool", "fuzz", str(exe), "-d", str(corpus),
         "-o", str(tmp_path / "crashes"), "-n", "20"],
        capture_output=True, text=True, timeout=180, cwd=tmp_path, env=_clean_env(),
    )  # fmt: skip

    assert r.returncode == 0, r.stderr[-2000:]
    assert "Shim self-test: 1 target(s) OK" in r.stdout
    assert SHIM_STDERR_TAG not in r.stdout


@requires_clang
def test_fuzz_run_flags_shim_that_attached_nothing(tmp_path):
    """Adversarial end to end: the environment is clobbered before attach."""
    exe = _build(
        tmp_path,
        """
        #include <stdio.h>
        #include <stdlib.h>
        __attribute__((constructor(101))) static void clobber(void) {
            setenv("AFL_MAP_SIZE", "-5", 1);
        }
        int main(void) {
            unsigned char b[64];
            return fread(b, 1, sizeof b, stdin) > 3 && b[0] == 'F';
        }
        """,
        "-fsanitize-coverage=trace-pc-guard",
    )
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "seed").write_bytes(b"seed")
    r = subprocess.run(
        [sys.executable, "-m", "fuzzer_tool", "fuzz", str(exe), "-d", str(corpus),
         "-o", str(tmp_path / "crashes"), "-n", "20"],
        capture_output=True, text=True, timeout=180, cwd=tmp_path, env=_clean_env(),
    )  # fmt: skip

    assert r.returncode == 0, r.stderr[-2000:]
    assert "recorded no edges" in r.stdout
    assert r.stdout.count("AFL_MAP_SIZE=-5 is not a valid entry count") == 1
    assert "Shim self-test: 1 target(s) OK" not in r.stdout
