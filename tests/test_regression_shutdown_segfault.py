"""Regressions for the intermittent ``Segmentation fault`` on a full suite run.

The crash was not where the output stopped. It happened during interpreter
finalization, *after* every test had already passed, inside
``Z3_del_context`` -- z3's singleton context losing a teardown race with z3's
own module globals. Roughly one run in eight.

Two separate defects kept it invisible for as long as it was:

* ``services/fuzzer.py`` installed a *Python* ``SIGSEGV`` handler at import.
  A Python signal callback is deferred to the eval loop between bytecodes, so
  it can never service a synchronous fault inside a C extension -- and it
  displaced ``faulthandler``, which can. The symptom was a bare
  "Segmentation fault" with no diagnostics at all.
* Nothing kept the z3 context out of finalization.

A third, latent defect was found while investigating and is covered here too:
``ShmCoverage.cleanup()`` left ctypes views bound to the detached address.

See ``docs/handover/suite_segfault_z3_finalization_2026-08-16.md``.
"""

from __future__ import annotations

import atexit
import faulthandler
import pathlib
import re
import signal
import subprocess
import sys
import textwrap

import pytest

from fuzzer_tool.adapters.shm import DistanceTableShm, ShmCoverage
from fuzzer_tool.core import z3_lifecycle
from tests.conftest import requires_z3


class TestFaultHandlerOwnsSigsegv:
    """A native fault must produce a traceback, not silence."""

    def test_sigsegv_is_not_a_python_handler(self):
        import fuzzer_tool.services.fuzzer  # noqa: F401  (installs handlers)

        handler = signal.getsignal(signal.SIGSEGV)
        assert not callable(handler) or handler in (
            signal.SIG_DFL,
            signal.SIG_IGN,
        ), (
            "a Python SIGSEGV handler is installed; it cannot run from a "
            "synchronous fault in a C extension and it suppresses "
            f"faulthandler's dump (got {handler!r})"
        )

    def test_faulthandler_is_enabled(self):
        import fuzzer_tool.services.fuzzer  # noqa: F401

        assert faulthandler.is_enabled(), (
            "faulthandler must own SIGSEGV so a native crash prints every "
            "thread's stack instead of dying silently"
        )

    def test_a_real_fault_prints_a_traceback(self):
        """End-to-end: importing the fuzzer must not swallow a segfault.

        Run in a subprocess -- the whole point is that the process dies.
        """
        script = textwrap.dedent(
            """
            import ctypes
            import fuzzer_tool.services.fuzzer  # installs signal handlers
            ctypes.string_at(1)  # dereference a bad address
            """
        )
        proc = subprocess.run([sys.executable, "-c", script], capture_output=True, timeout=60)
        assert proc.returncode != 0, "the fault did not kill the process"
        assert b"Fatal Python error" in proc.stderr, (
            "a segfault produced no faulthandler traceback; stderr was:\n"
            f"{proc.stderr.decode(errors='replace')}"
        )


class TestZ3ShutdownGuard:
    def test_guard_is_idempotent(self):
        z3_lifecycle.guard_z3_shutdown()
        z3_lifecycle.guard_z3_shutdown()
        # Registering twice would run the hook twice; harmless but sloppy.
        assert z3_lifecycle._registered is True

    def test_disown_is_a_noop_without_z3(self, monkeypatch):
        """Importing the guard must never import z3, or drag in the extra."""
        monkeypatch.delitem(sys.modules, "z3.z3", raising=False)
        z3_lifecycle._disown_main_context()  # must not raise

    @requires_z3
    def test_arming_the_guard_disowns_the_singleton_context(self):
        import z3

        z3.main_ctx()  # force the singleton into existence
        z3_lifecycle.guard_z3_shutdown()
        z3_lifecycle._disown_main_context()

        ctx = sys.modules["z3.z3"]._main_ctx
        assert ctx is not None
        assert ctx.owner is False, (
            "z3's singleton context is still owned, so Context.__del__ will "
            "call Z3_del_context during finalization -- the crash this "
            "guard exists to prevent"
        )
        # Leave it disowned: that is the production end state, and re-owning
        # it here would re-arm the crash for the rest of this process.

    def test_every_module_importing_z3_also_arms_the_guard(self):
        """The invariant, checked structurally rather than spot-checked.

        This assertion used to be made against ``xor_map_solver`` alone, and
        the docstring already said what was wrong with that: *every* z3 entry
        point must arm the guard, not just one of them. When xor_map_solver
        stopped using z3 -- its recovery is GF(2) elimination now -- that
        spot check would have gone vacuous while still passing, which is the
        exact failure mode this file exists to prevent.

        Reading the source rather than importing keeps this honest on a box
        without the optional 'smt' extra, where the arming code cannot run.
        """
        core = pathlib.Path(__file__).resolve().parents[1] / "src" / "fuzzer_tool" / "core"
        offenders = []
        importers = []
        for path in sorted(core.glob("*.py")):
            text = path.read_text()
            if not re.search(r"^\s*import z3\b", text, re.MULTILINE):
                continue
            importers.append(path.name)
            if "guard_z3_shutdown()" not in text:
                offenders.append(path.name)
        assert importers, "no module imports z3; this test has lost its subject"
        assert not offenders, f"modules import z3 without arming the shutdown guard: {offenders}"

    @requires_z3
    def test_solver_use_arms_the_guard(self):
        """The same invariant, executed rather than read.

        Covers the two z3 entry points that expose their import as a
        callable. The others arm inside larger solver functions and are
        covered by the source-level sweep above.
        """
        from fuzzer_tool.core import path_constraints, smt_solver

        for label, entry in (
            ("smt_solver._z3_available", smt_solver._z3_available),
            ("path_constraints._z3", path_constraints._z3),
        ):
            z3_lifecycle._registered = False
            try:
                assert entry() is not None
                assert z3_lifecycle._registered, (
                    f"calling {label} did not arm the z3 shutdown guard"
                )
            finally:
                z3_lifecycle._registered = True
                atexit.register(z3_lifecycle._disown_main_context)

    @requires_z3
    def test_del_context_does_not_run_at_finalization(self, tmp_path):
        """The load-bearing assertion, measured rather than asserted about.

        Spies on ``Z3_del_context`` through a raw fd -- the destructor runs
        so late that ``open`` itself is already gone -- and checks the file
        after the child has fully exited.
        """
        log = tmp_path / "delctx.txt"
        script = textwrap.dedent(
            f"""
            import os, sys
            fd = os.open({str(log)!r}, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            _write = os.write

            import z3.z3core as core
            import z3.z3 as z3mod
            _orig = core.Z3_del_context
            def spy(ctx, *a, **k):
                _write(fd, b"called\\n")
                return _orig(ctx, *a, **k)
            core.Z3_del_context = spy
            z3mod.Z3_del_context = spy

            # A live z3 entry point. This was the XOR-map solver until that
            # module moved to GF(2) elimination and stopped importing z3 --
            # at which point this spy would have measured nothing at all and
            # still reported success.
            from fuzzer_tool.core.path_constraints import _z3
            z3 = _z3()
            assert z3 is not None
            s = z3.Solver()
            x = z3.BitVec("x", 32)
            s.add(x + 7 == 42)
            assert s.check() == z3.sat
            s.model()
            """
        )
        proc = subprocess.run([sys.executable, "-c", script], capture_output=True, timeout=120)
        assert proc.returncode == 0, proc.stderr.decode(errors="replace")
        calls = log.read_text().count("called") if log.exists() else 0
        assert calls == 0, (
            f"Z3_del_context ran {calls}x during finalization; the shutdown "
            "guard is not armed on this path"
        )


class TestShmCleanupDropsItsViews:
    """``cleanup()`` must not leave handles on the detached mapping.

    Not primarily a crash fix. The kernel hands the same address back to the
    next ``shmat``, so a stale view aliases the *next* live segment and
    corrupts another instance's coverage table silently; the segfault is only
    the case where nothing happens to occupy the address.
    """

    @pytest.mark.parametrize(
        "method",
        [
            "get_edge_ids",
            "get_edge_counts",
            "reset_edge_map",
            "read_distance_tail",
            "read_diag",
            "read_path_hash",
            "read_edge_count",
        ],
    )
    def test_use_after_cleanup_raises_instead_of_faulting(self, method):
        cov = ShmCoverage()
        cov.cleanup()
        with pytest.raises((TypeError, AttributeError)):
            getattr(cov, method)()

    def test_views_are_dropped(self):
        cov = ShmCoverage()
        cov.cleanup()
        assert cov._map is None
        assert cov._entries is None
        assert cov._tail is None
        assert cov._ptr is None

    def test_cleanup_is_idempotent(self):
        cov = ShmCoverage()
        cov.cleanup()
        cov.cleanup()  # atexit runs it again after an explicit call

    def test_resize_rebinds_every_view(self):
        """resize() drops the views before shmdt; it must rebind all of them."""
        cov = ShmCoverage(size=1024)
        try:
            cov.resize(4096)
            assert cov._map is not None
            assert cov._entries is not None
            assert cov._tail is not None
            assert cov.num_entries == 4096
            # Must be usable, not merely non-None.
            cov.reset_edge_map()
            assert cov.get_edge_ids() == set()
        finally:
            cov.cleanup()

    def test_distance_table_uses_none_not_zero(self):
        """``0`` makes stale ``_ptr + offset`` address near-null instead of raising."""
        empty = DistanceTableShm({})
        assert empty._ptr is None

        table = DistanceTableShm({0x1000: 1.5, 0x2000: 2.5})
        table.cleanup()
        assert table._ptr is None
        table.cleanup()
