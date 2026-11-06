"""bench_paired cells launch through the sanitizer LD_PRELOAD wrapper.

``run_cell`` ran ``python -m fuzzer_tool`` directly, so an ASAN ``.so``
(strong undefined ``__asan_init``) loaded libasan mid-process: 51 eps vs 330
noasan on fuzzgoat (eval_set.py), and no ASAN set could be benched. The
wrapper preloads the runtime only when the target needs it and then execs
the same ``-m fuzzer_tool`` argv, so noasan cells are unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "lib"))

import bench_paired  # noqa: E402
from child_rusage import TimedRun  # noqa: E402

from fuzzer_tool.cli import ldpreload_wrapper  # noqa: E402


def _cmd(monkeypatch) -> list[str]:
    got: dict[str, list[str]] = {}

    def fake_run(cmd, timeout, **kw):
        got["cmd"] = cmd
        return TimedRun(rc=0, log="", wall=0.0, cpu_user=0.0, cpu_sys=0.0)

    monkeypatch.setattr(bench_paired, "run_timed", fake_run)
    bench_paired.run_cell("baseline", "/t/target_asan.so", "", 1, 10, 10)
    return got["cmd"]


def test_regression_cell_launches_via_preload_wrapper(monkeypatch):
    cmd = _cmd(monkeypatch)
    assert cmd[1:3] == ["-m", ldpreload_wrapper.__name__]
    assert cmd[3:5] == ["fuzz", "/t/target_asan.so"]


def test_adversarial_wrapper_reexecs_same_argv(monkeypatch):
    """A noasan cell must reach `-m fuzzer_tool` with its argv untouched."""
    seen: dict[str, list[str]] = {}
    argv = ["fuzz", "/t/plain.so", "-d", "/w", "--max-execs", "10"]
    monkeypatch.setattr(sys, "argv", ["wrapper", *argv])
    monkeypatch.setattr(ldpreload_wrapper, "_has_undefined_symbol", lambda *_: False)
    monkeypatch.setattr(ldpreload_wrapper.os, "execvpe", lambda f, c, e: seen.update(cmd=c))
    ldpreload_wrapper.main()
    assert seen["cmd"] == [sys.executable, "-m", "fuzzer_tool", *argv]
