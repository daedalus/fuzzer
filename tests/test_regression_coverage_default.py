"""Regressions for coverage-guided mode being the default.

`fuzz` previously required `-c/--coverage`. Without it the SHM bitmap was
never created, so every coverage-guided subsystem ran on a constant-zero
signal and the corpus never grew — measured on this tree at 1500 execs,
2 repeats:

    targets/test_target   coverage on 158.4 eps, corpus 3.0, 3 edges
                          coverage off 160.7 eps, corpus 1.0, 0 edges
    targets/png_read      coverage on 128.1 eps, corpus 14.5, 80.5 edges
                          coverage off 138.9 eps, corpus 1.0, 0 edges

i.e. the default mode of a coverage-guided fuzzer was blind mutation, and
switching it on costs 1-8% throughput. `tests/test_regression_no_coverage_
warning.py` already covered the same trap for in-process (.so) targets
only; this file covers the general case and the new default.
"""

from __future__ import annotations

import subprocess
import sys
from unittest.mock import MagicMock

from fuzzer_tool.services.fuzzer import Fuzzer, afl_instrumentation_status


def _parsed_coverage(monkeypatch, argv: list[str]) -> bool:
    """Return `args.coverage` as the *shipped* parser produces it.

    main() builds its parsers and dispatches in one call, so there is no
    parser object to inspect. Rebuilding the flag locally would make these
    assertions self-referential — they would pass against a copy while the
    real parser said something else, which is the failure mode this whole
    file exists to catch. Instead the real main() runs and `cmd_fuzz` is
    patched out: `set_defaults(func=cmd_fuzz)` resolves the module global
    at parser-build time, which is inside main(), so the patch lands.
    """
    from fuzzer_tool.cli import commands

    captured: dict[str, bool] = {}

    def _spy(args):
        captured["coverage"] = args.coverage
        return 0

    monkeypatch.setattr(commands, "cmd_fuzz", _spy)
    monkeypatch.setattr(sys, "argv", ["fuzzer-tool", "fuzz", "/bin/true", *argv])
    rc = commands.main()
    assert rc == 0
    assert "coverage" in captured, "cmd_fuzz was never reached"
    return captured["coverage"]


class TestCoverageIsOnByDefault:
    """Parsed against the shipped CLI, not a local reconstruction."""

    def _help(self) -> str:
        """--help as a user actually gets it.

        `python -m fuzzer_tool.cli.commands` prints nothing — the module has
        no __main__ block — so this goes through main() the way the console
        script does.
        """
        r = subprocess.run(
            [
                sys.executable,
                "-c",
                "from fuzzer_tool.cli.commands import main; main()",
                "fuzz",
                "--help",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        return r.stdout + r.stderr

    def test_no_coverage_flag_exists(self):
        assert "--no-coverage" in self._help(), (
            "the opt-out must be discoverable in --help, or flipping the "
            "default just removes the user's ability to turn it off"
        )

    def test_short_flag_still_accepted(self):
        """`-c` is in every existing script, README line and doc example."""
        assert "-c" in self._help()

    def test_default_is_on(self, monkeypatch):
        assert _parsed_coverage(monkeypatch, []) is True

    def test_explicit_on_still_works(self, monkeypatch):
        assert _parsed_coverage(monkeypatch, ["-c"]) is True
        assert _parsed_coverage(monkeypatch, ["--coverage"]) is True

    def test_opt_out(self, monkeypatch):
        assert _parsed_coverage(monkeypatch, ["--no-coverage"]) is False

    def test_last_flag_wins(self, monkeypatch):
        """A script passing -c after --no-coverage must get coverage."""
        assert _parsed_coverage(monkeypatch, ["--no-coverage", "-c"]) is True
        assert _parsed_coverage(monkeypatch, ["-c", "--no-coverage"]) is False


class TestInstrumentationStatusIsTriState:
    """`unknown` must not collapse into `absent`.

    A stripped binary reports no symbols at all, so a boolean cannot tell
    "not instrumented" from "symbol table removed" — and a stripped,
    fully-instrumented target is a normal thing to be handed. Warning on it
    is how a warning gets trained out of people.
    """

    def test_instrumented_target_is_present(self, tmp_path):
        # Built by tools/build_targets.sh; skip rather than fail if absent.
        import os

        t = "targets/test_target"
        if not os.path.exists(t):
            import pytest

            pytest.skip("targets/test_target not built")
        assert afl_instrumentation_status(t) == "present"

    def test_uninstrumented_binary_is_absent(self, tmp_path):
        src = tmp_path / "plain.c"
        src.write_text("int main(void){return 0;}\n")
        exe = tmp_path / "plain"
        r = subprocess.run(["gcc", "-O0", "-o", str(exe), str(src)], capture_output=True)
        if r.returncode != 0:
            import pytest

            pytest.skip("no working gcc")
        assert afl_instrumentation_status(str(exe)) == "absent"

    def test_stripped_binary_is_unknown_not_absent(self, tmp_path):
        src = tmp_path / "plain.c"
        src.write_text("int main(void){return 0;}\n")
        exe = tmp_path / "stripped"
        r = subprocess.run(["gcc", "-O0", "-o", str(exe), str(src)], capture_output=True)
        if r.returncode != 0:
            import pytest

            pytest.skip("no working gcc")
        subprocess.run(["strip", str(exe)], capture_output=True)
        assert afl_instrumentation_status(str(exe)) == "unknown"

    def test_missing_file_is_unknown(self):
        assert afl_instrumentation_status("/nonexistent/binary") == "unknown"


_SENTINEL = object()  # stands in for a live ShmCoverage / PtraceCoverage


class _Bare:
    """Minimal object carrying only what _warn_uninstrumented touches."""

    _warn_uninstrumented = Fuzzer._warn_uninstrumented

    def __init__(self, use_coverage=True, shm_cov=_SENTINEL, ptrace_cov=None):
        self.use_coverage = use_coverage
        self.shm_cov = shm_cov
        self.ptrace_cov = ptrace_cov


class TestUninstrumentedWarning:
    def test_warns_when_coverage_on_and_target_bare(self, capsys):
        _Bare(use_coverage=True)._warn_uninstrumented(["/tmp/plain"])
        out = capsys.readouterr().out
        assert "WARNING" in out
        assert "/tmp/plain" in out

    def test_silent_when_coverage_explicitly_off(self, capsys):
        """--no-coverage is a deliberate choice; nagging about it is noise."""
        _Bare(use_coverage=False)._warn_uninstrumented(["/tmp/plain"])
        assert capsys.readouterr().out == ""

    def test_silent_on_the_ptrace_path(self, capsys):
        """ptrace needs no build-time instrumentation, so the premise fails.

        Caught by running `--no-shm` against an uninstrumented target after
        the default flip: coverage was working — 5 breakpoints, edges
        accumulating — while the warning claimed the bitmap would stay empty
        and offered --no-coverage as the remedy.
        """
        _Bare(ptrace_cov=_SENTINEL)._warn_uninstrumented(["/tmp/plain"])
        assert capsys.readouterr().out == ""

    def test_silent_when_not_on_the_shm_path(self, capsys):
        """In-process modes have their own warning (_warn_no_coverage)."""
        _Bare(shm_cov=None)._warn_uninstrumented(["/tmp/plain"])
        assert capsys.readouterr().out == ""

    def test_names_the_consequence_and_the_way_out(self, capsys):
        _Bare()._warn_uninstrumented(["/tmp/plain"])
        out = capsys.readouterr().out.lower()
        assert "corpus growth" in out
        assert "--no-coverage" in out
        assert "build_targets.sh" in out

    def test_warns_once_per_run(self, capsys):
        b = _Bare()
        b._warn_uninstrumented(["/tmp/plain"])
        b._warn_uninstrumented(["/tmp/plain"])
        b._warn_uninstrumented(["/tmp/plain"])
        assert capsys.readouterr().out.count("WARNING") == 1

    def test_sentinel_is_per_instance(self):
        """Parallel workers each build a Fuzzer; each should warn once."""
        a, b = _Bare(), _Bare()
        a._warn_uninstrumented(["/tmp/plain"])
        assert getattr(a, "_uninstrumented_warned", False) is True
        assert getattr(b, "_uninstrumented_warned", False) is False

    def test_multi_target_summarises(self, capsys):
        _Bare()._warn_uninstrumented(["/a", "/b", "/c"])
        assert "3 targets" in capsys.readouterr().out

    def test_logs_at_warning_level(self, caplog):
        with caplog.at_level("WARNING"):
            _Bare()._warn_uninstrumented(["/tmp/plain"])
        assert any(r.levelname == "WARNING" for r in caplog.records)

    def test_does_not_crash_on_partial_object(self):
        """Called from run(), which is reachable in odd states."""
        m = MagicMock()
        m.ptrace_cov = None
        m.shm_cov = _SENTINEL
        Fuzzer._warn_uninstrumented(m, ["/tmp/plain"])  # must not raise

    def test_report_helper_exists_and_is_wired(self):
        import inspect

        assert callable(getattr(Fuzzer, "_report_instrumentation", None))
        src = inspect.getsource(Fuzzer.run)
        assert "_report_instrumentation()" in src, (
            "startup must report instrumentation state; the positive-only "
            "print it replaced is what made a bare target look healthy"
        )


class TestDeadCommonArgsHelperIsGone:
    """`_add_common_args` had zero production call sites.

    It was the obvious place to change `-c` — it is literally named
    "arguments shared by fuzz and subcommands" — and editing it would have
    changed nothing a user could reach while turning test_commands.py red.
    """

    def test_helper_removed(self):
        from fuzzer_tool.cli import commands

        assert not hasattr(commands, "_add_common_args")
