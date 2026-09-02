"""Regression: --trace is on by default, --no-trace disables it, and
--hail-mary must NOT toggle --no-trace (the user-facing contract: tracing
ships on so every saved crash sidecar carries a real GDB backtrace; hail-mary
force-enables *opt-in* features, never *opt-out* features).

The original `--trace` (store_true) shipped tracing off and forced users to
add the flag to every command. That left crash sidecars in the wild carrying
only the signal line (see ``test_regression_trace_probe_finds_fuzz_shm_run``
in ``test_trace.py`` for the matching fuzz-entry-point fix). Switching the
default flips the burden: a fresh run gets a real backtrace, and a user
who wants the old fast behaviour asks for ``--no-trace`` explicitly.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

from fuzzer_tool.cli import commands


def _parse(argv: list[str]):
    """Run the real CLI parser; return the parsed args Namespace."""
    cmd = MagicMock(return_value=0)
    saved_main = commands.cmd_fuzz
    commands.cmd_fuzz = cmd
    try:
        old_argv = sys.argv
        sys.argv = ["fuzzer-tool", *argv]
        rc = commands.main()
        assert rc == 0
        return cmd.call_args[0][0]
    finally:
        commands.cmd_fuzz = saved_main
        sys.argv = old_argv


class TestTraceDefaultOn:
    def test_default_is_on(self):
        args = _parse(["fuzz", "/bin/true"])
        assert args.trace is True, (
            f"--trace must default to True (crash sidecars need a GDB backtrace); got {args.trace!r}"
        )

    def test_no_trace_disables(self):
        args = _parse(["fuzz", "/bin/true", "--no-trace"])
        assert args.trace is False, "--no-trace must turn tracing off"

    def test_no_trace_explicit_does_not_propagate(self):
        # The CLI must NOT accept a stray --trace (renamed to --no-trace).
        try:
            _parse(["fuzz", "/bin/true", "--trace"])
        except SystemExit as e:
            assert e.code != 0, "--trace must be removed (renamed to --no-trace)"
        else:  # pragma: no cover - parser must reject
            raise AssertionError("parser accepted --trace, but the flag was removed")


class TestFuzzerDefault:
    def test_fuzzer_signature_default_is_on(self):
        # The Fuzzer() default must match the CLI default — if they drift,
        # fuzzing without explicit trace_crashes would still skip GDB and
        # leave a Signal-only sidecar.
        import inspect

        from fuzzer_tool.services.fuzzer import Fuzzer

        sig = inspect.signature(Fuzzer.__init__)
        assert sig.parameters["trace_crashes"].default is True, (
            f"Fuzzer trace_crashes default must be True; got {sig.parameters['trace_crashes'].default!r}"
        )


class TestHailMaryDoesNotDisableTrace:
    def test_hail_mary_leaves_trace_at_user_value(self):
        # The opt-out is --no-trace; hail-mary is the *opt-in* enabler, so
        # it must not flip --no-trace to True (i.e. must not disable tracing).
        # trace is removed from _HAIL_MARY_FLAGS for exactly this reason:
        # hail-mary would otherwise "force-enable" an opt-out and silently
        # strip the GDB backtrace off every crash sidecar.
        assert "trace" not in commands._HAIL_MARY_FLAGS, (
            "trace must NOT be in _HAIL_MARY_FLAGS — it's a default-on feature, "
            "hail-mary is the additive opt-in enabler, not the opt-out"
        )

    def test_hail_mary_does_not_flip_trace_to_false(self):
        captured: dict = {}

        def _spy(args):
            captured["args"] = args
            return 0

        saved = commands.cmd_fuzz
        commands.cmd_fuzz = _spy
        try:
            old_argv = sys.argv
            sys.argv = ["fuzzer-tool", "fuzz", "/bin/true", "--hail-mary"]
            rc = commands.main()
        finally:
            commands.cmd_fuzz = saved
            sys.argv = old_argv
        assert rc == 0
        args = captured["args"]
        # Default is True; hail-mary must not turn it off.
        assert args.trace is True, (
            f"--hail-mary must not disable tracing; got args.trace={args.trace!r}"
        )
