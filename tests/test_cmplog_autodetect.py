"""`--cmplog` is tri-state: auto by default, forceable either way.

Cmplog was opt-in, so a target with the instrumentation compiled in still
fuzzed without comparison tracing unless the user remembered the flag --
and magic-value and checksum branches are exactly where edge discovery
plateaus on real formats. `_detect_cmplog()` already identified those
targets reliably; nothing was reading it early enough to decide.

The resolution happens in the constructor. The pre-existing detection call
in the direct_lite path runs only when `self._cmplog` is already non-None,
so it could refine *how* cmplog runs but never *whether* it runs.

`None` -> detect; `True` -> on regardless; `False` -> off regardless. The
explicit-off case matters most: a user who passes `--no-cmplog` is usually
working around something, and auto-detect must not override that.
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

    # CmplogCollector shells out to build a shim; stub it so these tests
    # measure the decision, not the toolchain.
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


class TestTriState:
    def test_none_means_auto_and_enables_on_a_detected_target(self, monkeypatch):
        assert _cmplog_on(monkeypatch, None, detected=True) is True

    def test_none_means_auto_and_stays_off_on_a_plain_target(self, monkeypatch):
        assert _cmplog_on(monkeypatch, None, detected=False) is False

    def test_explicit_true_wins_even_when_detection_fails(self, monkeypatch):
        """--cmplog on a target whose symbols we cannot read must still try.
        Detection is a convenience, not a gate."""
        assert _cmplog_on(monkeypatch, True, detected=False) is True

    def test_explicit_false_wins_even_on_a_detected_target(self, monkeypatch):
        """--no-cmplog is a user override and auto-detect must never
        second-guess it."""
        assert _cmplog_on(monkeypatch, False, detected=True) is False

    def test_detection_is_not_consulted_when_the_flag_is_explicit(self, monkeypatch):
        """Skipping the probe on an explicit flag keeps two `nm` subprocesses
        off the startup path for users who already decided."""
        calls = []
        monkeypatch.setattr(
            fuzzer_mod, "_detect_cmplog", lambda path: (calls.append(path), True)[1]
        )

        class _StubCollector:
            def __init__(self, *a, **k):
                pass

            def start(self):
                return True

        monkeypatch.setattr("fuzzer_tool.core.cmplog.CmplogCollector", _StubCollector)
        tmpdir = tempfile.mkdtemp(prefix="cmplog_autodetect_")
        from fuzzer_tool.services.fuzzer import Fuzzer

        with patch("os.path.isfile", return_value=True), patch("os.access", return_value=True):
            Fuzzer(
                target="/bin/true",
                corpus_dir=f"{tmpdir}/corpus",
                crashes_dir=f"{tmpdir}/crashes",
                max_len=256,
                timeout=1,
                mutations_per_input=2,
                cmplog=False,
            )
        assert calls == []

    def test_auto_resolution_is_recorded(self, monkeypatch):
        """`_cmplog_auto` distinguishes 'user asked for this' from 'we
        guessed', which is what any later diagnostic needs to report."""
        assert _make_fuzzer(monkeypatch, None, detected=True)._cmplog_auto is True
        assert _make_fuzzer(monkeypatch, True, detected=True)._cmplog_auto is False


class TestWiring:
    def test_cli_flag_is_tri_state(self):
        """store_true cannot express 'unset', so the flag has to be
        BooleanOptionalAction with default=None for auto to exist at all."""
        import argparse
        import inspect

        from fuzzer_tool.cli import commands

        src = inspect.getsource(commands)
        idx = src.index('"--cmplog"')
        decl = src[idx : idx + 400]
        assert "BooleanOptionalAction" in decl
        assert "default=None" in decl
        assert hasattr(argparse, "BooleanOptionalAction")

    def test_constructor_default_is_auto_not_off(self):
        """The constructor default has to be None too, or every caller that
        omits the argument silently opts out of detection."""
        import inspect

        sig = inspect.signature(fuzzer_mod.Fuzzer.__init__)
        assert sig.parameters["cmplog"].default is None

    def test_constructor_resolves_before_building_the_collector(self):
        """The decision must precede CmplogCollector construction; resolving
        it later would mean auto-detect could never turn cmplog on."""
        import inspect

        src = inspect.getsource(fuzzer_mod.Fuzzer.__init__)
        decide = src.index("cmplog = _detect_cmplog(self.target)")
        build = src.index("CmplogCollector(")
        assert decide < build
