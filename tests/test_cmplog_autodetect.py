"""`--cmplog` is always enabled by default.

Cmplog was opt-in, so a target with the instrumentation compiled in still
fuzzed without comparison tracing unless the user remembered the flag --
and magic-value and checksum branches are exactly where edge discovery
plateaus on real formats. `_detect_cmplog()` already identified those
targets reliably, so the flag was removed entirely in favor of always-on
auto-detect.

The constructor's `cmplog` parameter is now a boolean: `True` (default) means
on-with-auto-detect, `False` means force off. The user-facing switch for
`False` is `--no-cmplog` (a store_true opt-out). `--cmplog` is its explicit
counterpart (store_false on the same dest): a no-op by default, it overrides
`--no-cmplog` and `--lightweight`. `--no-cmplog-fifo-sink` is separate: it controls the FIFO drain mode but
not whether cmplog runs.
"""

from __future__ import annotations

import tempfile
from unittest.mock import patch

from fuzzer_tool.services import fuzzer as fuzzer_mod


def _make_fuzzer(monkeypatch, cmplog_arg, detected: bool):
    """Construct a real Fuzzer with cmplog detection stubbed.

    Drives the actual constructor rather than a copy of its logic: a helper
    that re-implements the branch would pass no matter what the constructor
    did.
    """
    from fuzzer_tool.services.fuzzer import Fuzzer

    monkeypatch.setattr(fuzzer_mod, "_detect_cmplog", lambda path: detected)

    class _StubCollector:
        fifo_sink = False  # keeps _setup_forkserver off the stub's env

        def __init__(self, *a, **k):
            pass

        def start(self):
            return True

    monkeypatch.setattr("fuzzer_tool.core.cmplog.CmplogCollector", _StubCollector)

    tmpdir = tempfile.mkdtemp(prefix="cmplog_autodetect_")
    with patch("os.path.isfile", return_value=True), patch("os.access", return_value=True):
        return Fuzzer(
            target="/bin/true",
            corpus_dir=f"{tmpdir}/corpus",
            crashes_dir=f"{tmpdir}/crashes",
            max_len=256,
            timeout=1,
            mutations_per_input=2,
            cmplog=cmplog_arg,
        )


def _cmplog_on(monkeypatch, cmplog_arg, detected: bool) -> bool:
    return _make_fuzzer(monkeypatch, cmplog_arg, detected)._cmplog is not None


class TestCmplogAlwaysOn:
    def test_default_enables_on_a_detected_target(self, monkeypatch):
        """With cmplog=True (default), a detected target gets cmplog."""
        assert _cmplog_on(monkeypatch, True, detected=True) is True

    def test_default_attempts_on_an_uninstrumented_target(self, monkeypatch):
        """With cmplog=True (default), cmplog is attempted even if detection fails;
        the collector prints a warning and disables itself on failure."""
        # The stub collector always succeeds, so cmplog stays on.
        assert _cmplog_on(monkeypatch, True, detected=False) is True

    def test_explicit_false_disables_even_on_a_detected_target(self, monkeypatch):
        """cmplog=False overrides detection; used by tests that need cmplog off."""
        assert _cmplog_on(monkeypatch, False, detected=True) is False


class TestWiring:
    def test_constructor_default_is_true(self):
        """The constructor default is True (always-on auto-detect)."""
        import inspect

        sig = inspect.signature(fuzzer_mod.Fuzzer.__init__)
        assert sig.parameters["cmplog"].default is True

    def test_cmplog_flag_is_explicit_counterpart_of_no_cmplog(self):
        """--cmplog shares the no_cmplog dest as store_false and adds no default of
        its own (SUPPRESS), so --no-cmplog stays the only source of the default."""
        import inspect

        from fuzzer_tool.cli import commands

        src = inspect.getsource(commands)
        idx = src.index('"--cmplog",')
        decl = src[idx : idx + 200]
        assert 'dest="no_cmplog"' in decl
        assert 'action="store_false"' in decl
        assert "argparse.SUPPRESS" in decl
        assert src.index('"--no-cmplog",') < idx  # get_default() must see --no-cmplog first

    def test_cmplog_fifo_sink_is_boolean_optional(self):
        """--no-cmplog-fifo-sink is the opt-out for the FIFO drain mode."""
        import argparse
        import inspect

        from fuzzer_tool.cli import commands

        src = inspect.getsource(commands)
        idx = src.index('"--cmplog-fifo-sink"')
        decl = src[idx : idx + 400]
        assert "BooleanOptionalAction" in decl
        assert "default=True" in decl
        assert hasattr(argparse, "BooleanOptionalAction")


class TestNoCmplogFlag:
    """--no-cmplog is the opt-out of comparison tracing for a whole run."""

    def test_flag_declared_as_store_true_opt_out(self):
        import inspect

        from fuzzer_tool.cli import commands

        src = inspect.getsource(commands)
        idx = src.index('"--no-cmplog",')
        decl = src[idx : idx + 200]
        assert 'dest="no_cmplog"' in decl
        assert 'action="store_true"' in decl

    def test_flag_is_wired_into_fuzzer_construction(self):
        import inspect

        from fuzzer_tool.cli import commands

        src = inspect.getsource(commands)
        assert 'cmplog=not getattr(args, "no_cmplog", False)' in src

    def test_hail_mary_does_not_force_cmplog_off(self):
        from fuzzer_tool.cli import commands

        assert "no_cmplog" not in commands._HAIL_MARY_FLAGS

    def test_disabled_run_announces_itself_and_creates_no_collector(self, monkeypatch, capsys):
        fz = _make_fuzzer(monkeypatch, False, detected=True)
        assert fz._cmplog is None
        assert "Cmplog: disabled (--no-cmplog)" in capsys.readouterr().out
