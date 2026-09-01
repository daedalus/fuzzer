"""Regression: cmplog silently collected nothing under `--hail-mary`.

`--hail-mary` force-enables `--inprocess` and `--inprocess-direct` (see
`_HAIL_MARY_FLAGS`), so a hail-mary run always takes the `elif inprocess:`
branch of `Fuzzer.__init__` rather than the auto-detect `.so` branch just
above it. That auto-detect branch calls `self._cmplog.setup_env_for_run()`
(which sets `_CMPLOG_OUT` in `os.environ`) and, for direct_lite execution,
`self._cmplog.preload_shims()` -- both *before* the target `.so` is loaded.
The explicit-`inprocess` branch built `InProcessRunner` without either
call, so a target with cmplog compiled in (`_detect_cmplog()` returns
True) never received a `_CMPLOG_OUT` path: the compiled-in shim had
nowhere to write, and cmplog collected 0 tokens/pairs for the entire run
even though startup printed "cmplog enabled".

Reproduced against a real ffmpeg `.so` harness built with
`tools/build_ffmpeg_ready.sh` (cmplog compiled in via `-D__AFL_CMPLOG=1`):
a plain `--cmplog` run (auto-detect branch, `--inprocess` left unset)
collected pairs within a few hundred execs; the identical target run
under `--hail-mary` (explicit-`inprocess` branch) stayed at "cmplog: 0t
0p" for tens of thousands of execs, and the fuzzer's own diagnostic
("no comparisons observed ... cmplog is enabled but not reaching the
target") fired.

These tests exercise the constructor directly rather than shelling out to
a real toolchain-built target, so they run fast and never touch the
network -- but they drive the actual `Fuzzer.__init__` branch (not a
reimplementation of its logic), matching the style of
test_cmplog_autodetect.py.
"""

from __future__ import annotations

import tempfile
from unittest.mock import patch

from fuzzer_tool.services import fuzzer as fuzzer_mod


def _build(monkeypatch, *, has_cmplog: bool, inprocess_direct: bool = True):
    """Construct a real Fuzzer with `inprocess=True` (the hail-mary path)
    and cmplog force-enabled, returning (fuzzer, env_calls, preload_calls).
    """
    from fuzzer_tool.services.fuzzer import Fuzzer

    monkeypatch.setattr(fuzzer_mod, "_detect_cmplog", lambda path: has_cmplog)
    monkeypatch.setattr(fuzzer_mod, "_detect_tracecmp_target", lambda path: False)

    env_calls = []
    preload_calls = []

    class _StubCollector:
        def __init__(self, *a, **k):
            pass

        def start(self):
            return True

        def setup_env_for_run(self):
            env_calls.append(True)

        def preload_shims(self):
            preload_calls.append(True)
            return True

    monkeypatch.setattr("fuzzer_tool.core.cmplog.CmplogCollector", _StubCollector)

    runner_kwargs = {}

    class _StubInProcessRunner:
        def __init__(self, *a, **k):
            self._persistent = False
            runner_kwargs.update(k)

    monkeypatch.setattr(fuzzer_mod, "InProcessRunner", _StubInProcessRunner, raising=False)
    monkeypatch.setattr(
        "fuzzer_tool.adapters.inprocess.InProcessRunner", _StubInProcessRunner
    )
    monkeypatch.setattr(
        Fuzzer, "_probe_so_function", lambda self, target: "LLVMFuzzerTestOneInput"
    )

    tmpdir = tempfile.mkdtemp(prefix="cmplog_explicit_inprocess_")
    with patch("os.path.isfile", return_value=True), patch("os.access", return_value=True):
        f = Fuzzer(
            target="/tmp/fake_target.so",
            corpus_dir=f"{tmpdir}/corpus",
            crashes_dir=f"{tmpdir}/crashes",
            max_len=256,
            timeout=1,
            mutations_per_input=2,
            cmplog=True,
            inprocess=True,
            inprocess_direct=inprocess_direct,
        )
    return f, env_calls, preload_calls, runner_kwargs


class TestExplicitInprocessCmplogWiring:
    def test_env_is_set_up_when_cmplog_is_compiled_in(self, monkeypatch):
        """The bug: this call never happened on the explicit-`inprocess`
        (i.e. --hail-mary) path, so _CMPLOG_OUT never reached the shim."""
        _f, env_calls, _preload, _kw = _build(monkeypatch, has_cmplog=True)
        assert env_calls == [True]

    def test_shims_are_preloaded_for_direct_ctypes_mode(self, monkeypatch):
        _f, _env, preload_calls, _kw = _build(
            monkeypatch, has_cmplog=True, inprocess_direct=True
        )
        assert preload_calls == [True]

    def test_env_is_still_set_up_when_not_compiled_in_but_cmplog_requested(
        self, monkeypatch
    ):
        """setup_env_for_run() must run regardless of detection outcome --
        e.g. an externally LD_PRELOAD'd shim in subprocess-loader mode
        still needs _CMPLOG_OUT."""
        _f, env_calls, _preload, _kw = _build(monkeypatch, has_cmplog=False)
        assert env_calls == [True]

    def test_direct_mode_is_declined_when_not_compiled_in_and_not_preloaded(
        self, monkeypatch
    ):
        """Without compiled-in cmplog or an LD_PRELOAD'd shim, forcing
        direct ctypes mode would load a .so with unresolved cmplog
        callbacks; fall back to the subprocess loader instead."""
        f, _env, preload_calls, kw = _build(
            monkeypatch, has_cmplog=False, inprocess_direct=True
        )
        assert preload_calls == []
        assert f._inprocess_runner is not None
        assert kw.get("direct") is False
