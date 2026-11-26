"""`replay` picks its .so backend from the same flags as `fuzz`/`minimize`.

  neither                         subprocess loader (a .so cannot be exec'd)
  --inprocess                     subprocess loader (crash-isolated)
  --inprocess-direct              direct_lite ctypes call
  --inprocess --inprocess-direct  direct ctypes call

An executable stays on spawn whatever the flags. The shim binds
`__afl_area` at dlopen, so every direct run builds its own `.so`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from fuzzer_tool.cli import commands
from fuzzer_tool.cli.commands import _replay_exec_mode
from fuzzer_tool.services.minimize import ExecMode

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIM = REPO_ROOT / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"

# 'x' aborts; anything else returns cleanly.
TARGET_SRC = r"""
#include <stddef.h>
#include <stdlib.h>
__attribute__((visibility("default")))
int fuzz_test(const unsigned char *buf, size_t len) {
    if (len && buf[0] == 'x') abort();
    return 0;
}
"""

CRASH = b"x"
BENIGN = b"a"
REPRODUCED = 0
NOT_REPRODUCED = 1

IN_PROCESS_MODES = {
    ExecMode.LOADER: ("--inprocess",),
    ExecMode.DIRECT_LITE: ("--inprocess-direct",),
    ExecMode.DIRECT: ("--inprocess", "--inprocess-direct"),
}


@pytest.fixture(autouse=True)
def _restore_env():
    """Direct modes write os.environ (input path, SHM id); undo it."""
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


@pytest.fixture
def build_so(tmp_path):
    """Factory: a fresh shim-linked target `.so` per call."""
    if shutil.which("clang") is None:
        pytest.skip("clang not available")
    count = 0

    def build(source: str = TARGET_SRC) -> str:
        nonlocal count
        count += 1
        d = tmp_path / f"so{count}"
        d.mkdir()
        src = d / "t.c"
        src.write_text(source)
        so = d / "t.so"
        cmd = ["clang", "-O1", "-fPIC", "-shared", "-o", str(so), str(src), str(SHIM)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            pytest.skip(f"target failed to build: {proc.stderr[-400:]}")
        return str(so)

    return build


def _main(monkeypatch, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["fuzzer-tool", "replay", *argv])
    return commands.main()


def _input(tmp_path: Path, data: bytes) -> str:
    p = tmp_path / f"in_{data.hex() or 'empty'}.bin"
    p.write_bytes(data)
    return str(p)


class TestFlags:
    @pytest.fixture(autouse=True)
    def _capture(self, monkeypatch):
        """Run the real parser; capture the Namespace cmd_replay gets."""
        self.seen = []
        monkeypatch.setattr(commands, "cmd_replay", lambda a: self.seen.append(a) or 0)
        self.monkeypatch = monkeypatch

    def _args(self, target: str, *extra: str):
        assert _main(self.monkeypatch, target, "crash.bin", *extra) == 0
        return self.seen[-1]

    def test_defaults_off(self):
        a = self._args("t.so")
        assert (a.inprocess, a.inprocess_direct) == (False, False)
        assert a.inprocess_func is None  # probe
        assert a.repeat == 1

    @pytest.mark.parametrize("flag", ["-N", "--repeat"])
    def test_repeat_parses(self, flag):
        assert self._args("t.so", flag, "5").repeat == 5

    @pytest.mark.parametrize(
        ("flags", "want"),
        [((), ExecMode.LOADER), *((f, m) for m, f in IN_PROCESS_MODES.items())],
    )
    def test_so_mode_table(self, flags, want):
        assert _replay_exec_mode(self._args("t.so", *flags)) is want

    @pytest.mark.parametrize("flags", [(), *IN_PROCESS_MODES.values()])
    def test_executable_stays_spawn(self, flags):
        # Adversarial: an ET_EXEC cannot be dlopen'd, whatever the flags say.
        exe = shutil.which("true")
        assert exe is not None
        assert _replay_exec_mode(self._args(exe, *flags)) is ExecMode.SPAWN


@pytest.mark.parametrize("mode", list(IN_PROCESS_MODES), ids=lambda m: m.value)
class TestReplay:
    def _replay(self, monkeypatch, tmp_path, so, data, mode, *extra):
        flags = (*IN_PROCESS_MODES[mode], "--inprocess-func", "fuzz_test", *extra)
        return _main(monkeypatch, so, _input(tmp_path, data), *flags)

    def test_crash_reproduced(self, monkeypatch, tmp_path, build_so, mode):
        assert self._replay(monkeypatch, tmp_path, build_so(), CRASH, mode) == REPRODUCED

    def test_benign_not_reproduced(self, monkeypatch, tmp_path, build_so, mode):
        # Falsification: a clean input must not read as a crash.
        assert self._replay(monkeypatch, tmp_path, build_so(), BENIGN, mode) == NOT_REPRODUCED

    def test_control_two_runs_agree(self, monkeypatch, tmp_path, build_so, mode):
        """The oracle against itself: the same input gives the same verdict."""
        runs = [self._replay(monkeypatch, tmp_path, build_so(), CRASH, mode) for _ in "ab"]
        assert runs[0] == runs[1]

    def test_missing_func(self, monkeypatch, tmp_path, build_so, mode, capsys):
        # Adversarial: a wrong --inprocess-func is an error, never "reproduced".
        so = build_so()
        rc = _main(
            monkeypatch,
            so,
            _input(tmp_path, CRASH),
            *IN_PROCESS_MODES[mode],
            "--inprocess-func",
            "no_such_function",
        )
        # Direct modes raise at load; the loader falls back to per-call and exits 1.
        out = capsys.readouterr()
        assert rc == NOT_REPRODUCED
        assert "Crash reproduced" not in out.out


# Scripted runner outcomes: rc 0 clean, 134 SIGABRT (128 + 6).
CLEAN_RUN = (0, "")
ABORT_RUN = (134, "")


class TestRepeat:
    """-N: one runner, N calls, first hit reported, k/N summary."""

    @pytest.fixture
    def scripted(self, monkeypatch):
        """Replace the backend with a scripted run; record call and stop counts."""
        self.calls = 0
        self.stops = 0

        def install(outcomes):
            def run(_data):
                out = outcomes[self.calls]
                self.calls += 1
                return out

            def stop():
                self.stops += 1

            monkeypatch.setattr(commands, "_replay_runner", lambda _args: (run, stop))

        return install

    def _run(self, monkeypatch, tmp_path, *flags):
        exe = shutil.which("true")
        assert exe is not None
        return _main(monkeypatch, exe, _input(tmp_path, CRASH), *flags)

    def test_runs_exactly_n(self, monkeypatch, tmp_path, scripted, capsys):
        outcomes = [CLEAN_RUN, ABORT_RUN, CLEAN_RUN, ABORT_RUN, CLEAN_RUN]
        scripted(outcomes)
        n = len(outcomes)
        hits = outcomes.count(ABORT_RUN)

        assert self._run(monkeypatch, tmp_path, "-N", str(n)) == REPRODUCED
        assert (self.calls, self.stops) == (n, 1)
        assert f"Reproduced {hits}/{n}" in capsys.readouterr().out

    def test_never_reproduced(self, monkeypatch, tmp_path, scripted, capsys):
        # Falsification: N clean runs must not read as a crash.
        n = 3
        scripted([CLEAN_RUN] * n)
        assert self._run(monkeypatch, tmp_path, "-N", str(n)) == NOT_REPRODUCED
        out = capsys.readouterr().out
        assert f"Reproduced 0/{n}" in out
        assert "Crash reproduced" not in out

    def test_late_hit_is_reported(self, monkeypatch, tmp_path, scripted, capsys):
        # Adversarial: a crash only on the last run still wins over earlier clean runs.
        n = 4
        scripted([CLEAN_RUN] * (n - 1) + [ABORT_RUN])
        assert self._run(monkeypatch, tmp_path, "-N", str(n)) == REPRODUCED
        out = capsys.readouterr().out
        assert "Crash reproduced: signal 6" in out
        assert "No crash detected" not in out

    def test_one_run_no_summary(self, monkeypatch, tmp_path, scripted, capsys):
        scripted([ABORT_RUN])
        assert self._run(monkeypatch, tmp_path) == REPRODUCED
        assert "Reproduced 1/1" not in capsys.readouterr().out

    @pytest.mark.parametrize("n", ["0", "-3"])
    def test_rejects_non_positive(self, monkeypatch, tmp_path, scripted, n):
        # Adversarial: no runs, an error exit.
        scripted([])
        assert self._run(monkeypatch, tmp_path, "-N", n) == NOT_REPRODUCED
        assert self.calls == 0


# Aborts on the CALLS-th call: needs state that survives between calls.
STATEFUL_SRC = r"""
#include <stddef.h>
#include <stdlib.h>
#ifndef CALLS
#define CALLS 3
#endif
static int calls;
__attribute__((visibility("default")))
int fuzz_test(const unsigned char *buf, size_t len) {
    (void)buf; (void)len;
    if (++calls == CALLS) abort();
    return 0;
}
"""
STATEFUL_CALLS = 3


class TestRepeatState:
    """Direct modes keep globals across repeats; the loader forks per call."""

    def _hits(self, monkeypatch, tmp_path, build_so, mode, n, capsys) -> str:
        flags = (*IN_PROCESS_MODES[mode], "--inprocess-func", "fuzz_test", "-N", str(n))
        _main(monkeypatch, build_so(STATEFUL_SRC), _input(tmp_path, BENIGN), *flags)
        return capsys.readouterr().out

    @pytest.mark.parametrize("mode", [ExecMode.DIRECT_LITE, ExecMode.DIRECT], ids=lambda m: m.value)
    def test_direct_keeps_state(self, monkeypatch, tmp_path, build_so, mode, capsys):
        # The counter equals CALLS exactly once in N > CALLS calls.
        n = STATEFUL_CALLS + 2
        out = self._hits(monkeypatch, tmp_path, build_so, mode, n, capsys)
        assert f"Reproduced 1/{n}" in out

    @pytest.mark.parametrize("mode", [ExecMode.DIRECT_LITE, ExecMode.DIRECT], ids=lambda m: m.value)
    def test_control_below_threshold(self, monkeypatch, tmp_path, build_so, mode, capsys):
        # Falsification: fewer calls than CALLS never crash, so the hit above is the counter.
        n = STATEFUL_CALLS - 1
        out = self._hits(monkeypatch, tmp_path, build_so, mode, n, capsys)
        assert f"Reproduced 0/{n}" in out

    def test_loader_resets_state(self, monkeypatch, tmp_path, build_so, capsys):
        # Each call runs in a fresh fork: the counter never passes 1.
        n = STATEFUL_CALLS + 2
        out = self._hits(monkeypatch, tmp_path, build_so, ExecMode.LOADER, n, capsys)
        assert f"Reproduced 0/{n}" in out


class TestReplayDir:
    """crash_file is a dir: replay every input in it on one runner."""

    @pytest.fixture
    def scripted(self, monkeypatch):
        """Backend keyed by content: CRASH aborts, anything else is clean."""
        self.seen = []
        self.builds = 0
        self.stops = 0

        def run(data):
            self.seen.append(data)
            return ABORT_RUN if data == CRASH else CLEAN_RUN

        def build(_args):
            self.builds += 1
            return run, stop

        def stop():
            self.stops += 1

        monkeypatch.setattr(commands, "_replay_runner", build)

    def _run(self, monkeypatch, crash_dir, *flags):
        exe = shutil.which("true")
        assert exe is not None
        return _main(monkeypatch, exe, str(crash_dir), *flags)

    def _dir(self, tmp_path, files: dict[str, bytes]) -> Path:
        d = tmp_path / "crashes"
        for name, data in files.items():
            p = d / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        return d

    def test_replays_each_file(self, monkeypatch, tmp_path, scripted, capsys):
        inputs = {"a.bin": BENIGN, "b.bin": CRASH, "sub/c.bin": BENIGN}
        d = self._dir(tmp_path, inputs)

        assert self._run(monkeypatch, d) == REPRODUCED
        assert sorted(self.seen) == sorted(inputs.values())
        assert (self.builds, self.stops) == (1, 1)
        hits = list(inputs.values()).count(CRASH)
        assert f"{hits}/{len(inputs)} files reproduced" in capsys.readouterr().out

    def test_repeat_per_file(self, monkeypatch, tmp_path, scripted):
        inputs = {"a.bin": BENIGN, "b.bin": CRASH}
        n = 3
        self._run(monkeypatch, self._dir(tmp_path, inputs), "-N", str(n))
        assert len(self.seen) == n * len(inputs)

    def test_none_reproduced(self, monkeypatch, tmp_path, scripted, capsys):
        # Falsification: a dir of clean inputs must not read as a crash.
        inputs = {"a.bin": BENIGN, "b.bin": BENIGN}
        assert self._run(monkeypatch, self._dir(tmp_path, inputs)) == NOT_REPRODUCED
        assert f"0/{len(inputs)} files reproduced" in capsys.readouterr().out

    def test_skips_sidecars(self, monkeypatch, tmp_path, scripted):
        # Adversarial: fuzz writes .json/.txt reports beside each crash; never inputs.
        d = self._dir(
            tmp_path, {"crash_x.bin": BENIGN, "crash_x.json": CRASH, "crash_x.txt": CRASH}
        )
        (d / "link.bin").symlink_to(d / "crash_x.json")
        assert self._run(monkeypatch, d) == NOT_REPRODUCED
        assert self.seen == [BENIGN]

    @pytest.mark.parametrize("files", [{}, {"crash_x.json": CRASH}])
    def test_no_inputs(self, monkeypatch, tmp_path, scripted, files, capsys):
        # Adversarial: nothing to replay is an error, and no runner is built.
        d = self._dir(tmp_path, files)
        d.mkdir(exist_ok=True)
        assert self._run(monkeypatch, d) == NOT_REPRODUCED
        assert (self.builds, self.seen) == (0, [])
        assert "No crash inputs" in capsys.readouterr().err


class TestReviewRegressions:
    """PR #462 review: entry name, script targets, ASAN preload for replay."""

    @pytest.fixture
    def seen_func(self, monkeypatch):
        """Capture the func _so_exec receives; never load anything."""
        seen = []

        def fake(_target, _timeout, _mode, func, restore=None):
            seen.append(func)
            return (lambda _d: CLEAN_RUN), (lambda: None)

        monkeypatch.setattr(commands, "_so_exec", fake)
        return seen

    @pytest.mark.parametrize(
        ("flags", "want"),
        [
            ((), None),  # control: no flag still probes
            (("--inprocess-func", "LLVMFuzzerTestOneInput"), "LLVMFuzzerTestOneInput"),
            (("--inprocess-func", "fuzz_test"), "fuzz_test"),
        ],
    )
    def test_regression_func_kept(self, monkeypatch, tmp_path, seen_func, flags, want):
        so = tmp_path / "t.so"
        so.write_bytes(b"")
        so.chmod(0o755)
        _main(monkeypatch, str(so), _input(tmp_path, CRASH), "--inprocess", *flags)
        assert seen_func == [want]

    @pytest.mark.parametrize("flags", list(IN_PROCESS_MODES.values()))
    def test_regression_script(self, monkeypatch, tmp_path, flags):
        # A runnable non-ELF target cannot be dlopen'd: spawn it.
        script = tmp_path / "t.sh"
        script.write_text("#!/bin/sh\nexit 0\n")
        script.chmod(0o755)
        seen = []
        monkeypatch.setattr(commands, "cmd_replay", lambda a: seen.append(a) or 0)
        _main(monkeypatch, str(script), "crash.bin", *flags)
        assert _replay_exec_mode(seen[-1]) is ExecMode.SPAWN

    @pytest.mark.parametrize(
        ("argv", "want"),
        [
            (["fuzz", "t.so", "-d", "c"], "t.so"),  # control: unchanged
            (["replay", "t.so", "crash.bin", "--inprocess-direct"], "t.so"),
            (["rank", "t.so"], None),  # falsification: other commands untouched
            (["replay", "--inprocess-direct"], None),  # adversarial: option, not a path
        ],
    )
    def test_regression_preload(self, argv, want):
        from fuzzer_tool.cli.ldpreload_wrapper import _fuzz_target

        assert _fuzz_target(argv) == want


# InProcessRunner's infrastructure-failure sentinel.
ERROR_RUN = (-2, "runner not initialized")


class TestRunnerError:
    """rc -2 is a runner failure, never a crash."""

    @pytest.fixture
    def scripted(self, monkeypatch):
        def install(outcomes):
            it = iter(outcomes)
            monkeypatch.setattr(
                commands, "_replay_runner", lambda _a: ((lambda _d: next(it)), (lambda: None))
            )

        return install

    def _run(self, monkeypatch, tmp_path, *flags):
        exe = shutil.which("true")
        assert exe is not None
        return _main(monkeypatch, exe, _input(tmp_path, CRASH), *flags)

    def test_regression_err_not_crash(self, monkeypatch, tmp_path, scripted, capsys):
        scripted([ERROR_RUN])
        assert self._run(monkeypatch, tmp_path) == NOT_REPRODUCED
        out = capsys.readouterr()
        assert "Crash reproduced" not in out.out
        assert ERROR_RUN[1] in out.out + out.err

    def test_regression_err_not_hit(self, monkeypatch, tmp_path, scripted, capsys):
        # Adversarial: errors mixed with a real crash count only the crash.
        scripted([ERROR_RUN, ABORT_RUN, ERROR_RUN])
        assert self._run(monkeypatch, tmp_path, "-N", "3") == REPRODUCED
        assert "Reproduced 1/3" in capsys.readouterr().out


class TestSnapshotRestore:
    """`replay --snapshot-restore`: the fuzz run's globals mode, default off."""

    @pytest.fixture(autouse=True)
    def _capture(self, monkeypatch):
        self.seen = []
        monkeypatch.setattr(commands, "cmd_replay", lambda a: self.seen.append(a) or 0)
        self.monkeypatch = monkeypatch

    def _restore(self, *extra: str) -> str:
        assert _main(self.monkeypatch, "t.so", "crash.bin", *extra) == 0
        return self.seen[-1].snapshot_restore

    @pytest.mark.parametrize(
        ("flags", "want"),
        [
            ((), "off"),  # control: #462 semantics, direct repeats keep state
            (("--snapshot-restore", "globals"), "globals"),
            (("--snapshot-restore", "globals", "--no-snapshot-restore"), "off"),
        ],
    )
    def test_parses(self, flags, want):
        assert self._restore(*flags) == want

    def test_rejects_unknown(self):
        """Adversarial: a typo fails loudly."""
        with pytest.raises(SystemExit):
            self._restore("--snapshot-restore", "heap")


@pytest.mark.parametrize("mode", [ExecMode.DIRECT_LITE, ExecMode.DIRECT], ids=lambda m: m.value)
def test_globals_resets_state(monkeypatch, tmp_path, build_so, mode, capsys):
    """Falsification of TestRepeatState: globals restore resets the counter every exec."""
    n = STATEFUL_CALLS + 2
    flags = (*IN_PROCESS_MODES[mode], "--inprocess-func", "fuzz_test", "-N", str(n))
    so = build_so(STATEFUL_SRC)
    _main(monkeypatch, so, _input(tmp_path, BENIGN), *flags, "--snapshot-restore", "globals")
    assert f"Reproduced 0/{n}" in capsys.readouterr().out


class TestPreloadTarget:
    """The wrapper must find the target after options, else ASAN is never preloaded."""

    @pytest.fixture
    def elf(self, tmp_path) -> str:
        p = tmp_path / "t.so"
        p.write_bytes(b"\x7fELF" + b"\0" * 60)
        return str(p)

    def test_regression_options_first(self, elf, tmp_path):
        """`replay --inprocess-direct -N 5 t.so`: target was None, libasan not preloaded."""
        from fuzzer_tool.cli.ldpreload_wrapper import _fuzz_target

        crash = tmp_path / "c.bin"
        crash.write_bytes(b"x")
        argv = ["replay", "--inprocess-direct", "-N", "5", elf, str(crash)]

        assert _fuzz_target(argv) == elf

    def test_option_value_not_target(self, elf, tmp_path):
        """Adversarial: an option value that is an existing non-ELF file is skipped."""
        from fuzzer_tool.cli.ldpreload_wrapper import _fuzz_target

        d = tmp_path / "x.dict"
        d.write_text('"a"\n')

        assert _fuzz_target(["fuzz", "-d", str(d), elf]) == elf


class _OverlapRunner:
    """Fake InProcessRunner: counts calls that overlap another call."""

    overlaps = 0
    active = 0

    def __init__(self, **_kw):
        pass

    def run_one(self, _data):
        import time

        cls = type(self)
        cls.active += 1
        if cls.active > 1:
            cls.overlaps += 1
        time.sleep(0.02)  # hold the call open so an unserialized peer overlaps
        cls.active -= 1
        return CLEAN_RUN

    def stop(self):
        pass


@pytest.mark.parametrize("threads", [1, 4], ids=["control", "concurrent"])
def test_so_exec_serializes(monkeypatch, tmp_path, threads):
    """PR #463 review: direct runs share the .so's globals; calls must not interleave."""
    import threading

    from fuzzer_tool.adapters import inprocess

    monkeypatch.setattr(inprocess, "InProcessRunner", _OverlapRunner)
    _OverlapRunner.overlaps = _OverlapRunner.active = 0
    so = tmp_path / "t.so"
    so.write_bytes(b"")
    run, stop = commands._so_exec(str(so), 1.0, ExecMode.DIRECT_LITE, "fuzz_test")

    workers = [threading.Thread(target=run, args=(b"a",)) for _ in range(threads)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    stop()

    assert _OverlapRunner.overlaps == 0


class TestTargetRole:
    """PR #463 review: the target is the first positional by argparse role, not any ELF."""

    @pytest.fixture
    def elves(self, tmp_path) -> tuple[str, str]:
        out = []
        for name in ("a.so", "b.so"):
            p = tmp_path / name
            p.write_bytes(b"\x7fELF" + b"\0" * 60)
            out.append(str(p))
        return out[0], out[1]

    def test_regression_elf_option_value(self, elves):
        """An ELF given as an option value before the target was picked as the target."""
        from fuzzer_tool.cli.ldpreload_wrapper import _fuzz_target

        value, target = elves

        assert _fuzz_target(["fuzz", "-d", value, target]) == target

    def test_regression_script_target(self, elves, tmp_path):
        """Script target + ELF crash file: the crash file was picked."""
        from fuzzer_tool.cli.ldpreload_wrapper import _fuzz_target

        script = tmp_path / "t.sh"
        script.write_text("#!/bin/sh\n")
        argv = ["replay", "--inprocess-direct", str(script), elves[0]]

        assert _fuzz_target(argv) == str(script)

    def test_invalid_cli_no_target(self, capsys):
        """Adversarial: unparsable argv yields None and prints nothing (the command reports it)."""
        from fuzzer_tool.cli.ldpreload_wrapper import _fuzz_target

        assert _fuzz_target(["replay", "-N"]) is None
        assert capsys.readouterr().err == ""
