"""`--cmplog` is always enabled by default.

Cmplog was opt-in, so a target with the instrumentation compiled in still
fuzzed without comparison tracing unless the user remembered the flag --
and magic-value and checksum branches are exactly where edge discovery
plateaus on real formats. `_detect_cmplog()` already identified those
targets reliably, so the flag was removed entirely in favor of always-on
auto-detect.

The constructor's `cmplog` parameter is now a boolean: `True` (default) means
on-with-auto-detect, `False` means force off. There is no user-facing
`--no-cmplog` flag any more; the only cmplog-related opt-out is
`--no-cmplog-fifo-sink`, which controls the FIFO drain mode but not
whether cmplog runs.
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

    def test_no_cmplog_argparse_flag(self):
        """There is no --cmplog flag; cmplog is always enabled when the
        target has instrumentation. The only opt-out is --no-cmplog-fifo-sink."""
        import inspect

        from fuzzer_tool.cli import commands

        src = inspect.getsource(commands)
        assert '"--cmplog"' not in src

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
