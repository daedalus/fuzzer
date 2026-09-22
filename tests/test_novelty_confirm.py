"""Confirm-on-novelty: a "new" edge must reproduce before anything acts on it.

F2 (handover_edge_id_axis_2026-09-18.md): an execution can report ids that
never reappear. Measured on the default path (one persistent table, one
warm-up, fuzzgoat): ~6.5% of executions carry them, 12-18% of "new coverage"
successes exist only because of them, and 71% of singleton edges are phantom.
`_calibrate_seed_stability` cannot see them -- it compares reruns with each
other, and the phantoms are only in the original run.

`--confirm-novelty` reruns the input once when (and only when) the execution
reported new coverage, and keeps `original & rerun`.
"""

from __future__ import annotations

import tempfile
from unittest.mock import patch

import pytest

from fuzzer_tool.core.novelty_confirm import Confirmation, confirm


def _c(has_new=True, new=(), old_bucket=False, orig=(), rerun=()):
    return confirm(has_new, frozenset(new), old_bucket, orig, rerun)


class TestConfirmPureFunction:
    def test_phantom_only_success_is_withdrawn(self):
        c = _c(new={9}, orig={1, 2, 9}, rerun={1, 2})
        assert c.has_new is False
        assert c.phantoms == {9}
        assert c.edge_ids == {1, 2}

    def test_real_new_edge_survives(self):
        c = _c(new={9}, orig={1, 2, 9}, rerun={1, 2, 9})
        assert c.has_new is True and c.phantoms == frozenset()

    def test_mixed_new_edges_keep_the_real_one_and_drop_the_phantom(self):
        c = _c(new={8, 9}, orig={1, 8, 9}, rerun={1, 8})
        assert c.has_new is True
        assert c.phantoms == {9}
        assert c.edge_ids == {1, 8}

    def test_bucket_novelty_on_an_old_edge_survives_a_phantom_neighbour(self):
        # A real (edge, bucket) event coinciding with a phantom id must not be
        # thrown away with it: that would cost genuine admissions.
        c = _c(new={9}, old_bucket=True, orig={1, 9}, rerun={1})
        assert c.has_new is True and c.phantoms == {9}

    def test_bucket_only_novelty_needs_no_new_id(self):
        c = _c(new=(), old_bucket=True, orig={1, 2}, rerun={1, 2})
        assert c.has_new is True and c.phantoms == frozenset()

    def test_no_novelty_stays_no_novelty(self):
        assert _c(has_new=False, orig={1}, rerun={1}).has_new is False

    def test_an_id_only_the_rerun_saw_is_not_adopted(self):
        # F2 also drops real ids from the first run. They are not "confirmed":
        # confirmation is a subset of the original, never a superset.
        c = _c(new={9}, orig={1, 9}, rerun={1, 9, 5})
        assert 5 not in c.edge_ids

    def test_empty_rerun_withdraws_every_new_id(self):
        c = _c(new={9}, orig={1, 9}, rerun=())
        assert c.has_new is False and c.phantoms == {1, 9}

    @pytest.mark.parametrize(
        "orig,rerun,new",
        [({1, 2, 3}, {2, 3, 4}, {1}), ({5}, {5}, {5}), (set(), {1}, set()), ({1, 2}, set(), {2})],
    )
    def test_partition_and_subset_invariants(self, orig, rerun, new):
        c = _c(new=new, orig=orig, rerun=rerun)
        assert c.edge_ids | c.phantoms == frozenset(orig)
        assert c.edge_ids.isdisjoint(c.phantoms)
        assert c.edge_ids <= frozenset(rerun)

    def test_idempotent(self):
        c = _c(new={9}, orig={1, 2, 9}, rerun={1, 2})
        again = confirm(c.has_new, frozenset(), False, c.edge_ids, c.edge_ids)
        assert again.edge_ids == c.edge_ids and again.phantoms == frozenset()

    def test_result_is_a_confirmation(self):
        assert isinstance(_c(orig={1}, rerun={1}), Confirmation)


_PATH = [0x1000]


def _put(cov, edges: dict[int, int]) -> None:
    """Write one execution's table the way the C shim leaves it.

    ``record_edge`` folds into the seen set itself (it stands in for reader and
    writer), so it cannot produce a scan that reports novelty.
    """
    import ctypes

    cov.reset_edge_map()
    gen = cov.read_generation()
    for slot, (edge, count) in enumerate(edges.items()):
        cov._entries[slot].edge_id = edge
        cov._entries[slot].count = (gen << 24) | count
    ctypes.c_uint64.from_address(cov._ptr + 16).value = len(edges)
    _PATH[0] += 7919
    ctypes.c_uint64.from_address(cov._ptr + 8).value = _PATH[0]


class TestAdapterReportsWhatCausedTheNovelty:
    def _cov(self):
        from fuzzer_tool.adapters.shm import ShmCoverage

        return ShmCoverage()

    def test_new_ids_are_exposed(self):
        cov = self._cov()
        try:
            _put(cov, {11: 1, 12: 1})
            new, _ = cov.is_new_coverage_with_edges()
            assert new is True
            assert cov.last_new_ids == {11, 12}
            assert cov.last_old_bucket_novel is False
        finally:
            cov.cleanup()

    def test_old_edge_reaching_a_new_bucket_is_not_a_new_id(self):
        cov = self._cov()
        try:
            _put(cov, {11: 1})
            cov.is_new_coverage_with_edges()
            _put(cov, {11: 40})  # count bucket 32+ was never visited
            new, _ = cov.is_new_coverage_with_edges()
            assert new is True
            assert cov.last_new_ids == frozenset()
            assert cov.last_old_bucket_novel is True
        finally:
            cov.cleanup()

    def test_phantom_beside_a_bucket_event_reports_both(self):
        cov = self._cov()
        try:
            _put(cov, {11: 1})
            cov.is_new_coverage_with_edges()
            _put(cov, {11: 40, 99: 1})
            cov.is_new_coverage_with_edges()
            assert cov.last_new_ids == {99}
            assert cov.last_old_bucket_novel is True
        finally:
            cov.cleanup()

    def test_a_new_id_alone_is_not_an_old_bucket_event(self):
        # The bucket that comes with a brand-new id must not read as an old
        # edge having moved: that is the whole reason the fold is split.
        cov = self._cov()
        try:
            _put(cov, {11: 1})
            cov.is_new_coverage_with_edges()
            _put(cov, {11: 1, 99: 1})
            cov.is_new_coverage_with_edges()
            assert cov.last_new_ids == {99}
            assert cov.last_old_bucket_novel is False
        finally:
            cov.cleanup()

    def test_flags_reset_on_the_fast_path(self):
        cov = self._cov()
        try:
            _put(cov, {11: 1})
            cov.is_new_coverage_with_edges()
            new, _ = cov.is_new_coverage_with_edges()  # nothing changed
            assert new is False
            assert cov.last_new_ids == frozenset()
            assert cov.last_old_bucket_novel is False
        finally:
            cov.cleanup()

    def test_folding_is_unchanged_by_the_split(self):
        # New-id and old-id subsets both feed the virgin map, so repeating the
        # same execution must report nothing new.
        cov = self._cov()
        try:
            _put(cov, {11: 1, 12: 3, 13: 1})
            cov.is_new_coverage_with_edges()
            _put(cov, {11: 1, 12: 3, 13: 1})
            new, _ = cov.is_new_coverage_with_edges()
            assert new is False
        finally:
            cov.cleanup()

    def test_reject_phantoms_masks_and_retracts_the_count(self):
        cov = self._cov()
        try:
            _put(cov, {11: 1, 99: 1})
            cov.is_new_coverage_with_edges()
            before = cov.cumulative_edges
            assert cov.reject_phantoms({99}) == 1
            assert cov.cumulative_edges == before - 1
            assert 99 in cov.masked_edges
            assert 99 not in cov._last_ids
            assert cov.reject_phantoms({99}) == 0  # twice is not twice
            assert cov.cumulative_edges == before - 1
        finally:
            cov.cleanup()

    def test_a_masked_phantom_cannot_return_as_new(self):
        cov = self._cov()
        try:
            _put(cov, {11: 1, 99: 1})
            cov.is_new_coverage_with_edges()
            cov.reject_phantoms({99})
            _put(cov, {11: 1, 99: 1})
            new, _ = cov.is_new_coverage_with_edges()
            assert new is False
        finally:
            cov.cleanup()


class _FakeShm:
    """Scripted runs; ``last_new_ids`` is what the original scan reported."""

    def __init__(self, runs, new=(), old_bucket=False):
        self._runs = list(runs)
        self._i = 0
        self.last_new_ids = frozenset(new)
        self.last_old_bucket_novel = old_bucket
        self.rejected: set[int] = set()
        self.reruns = 0

    def advance(self):
        self._i = min(self._i + 1, len(self._runs) - 1)
        self.reruns += 1

    def get_edge_ids(self):
        return set(self._runs[self._i])

    def reject_phantoms(self, ids):
        self.rejected |= set(ids)


def _fuzzer(**kw):
    from fuzzer_tool.services.fuzzer import Fuzzer

    tmp = tempfile.mkdtemp(prefix="confirm_")
    with patch("os.path.isfile", return_value=True), patch("os.access", return_value=True):
        return Fuzzer(
            target="/bin/true",
            corpus_dir=f"{tmp}/c",
            crashes_dir=f"{tmp}/x",
            max_len=64,
            timeout=1,
            mutations_per_input=2,
            **kw,
        )


def _wired(shm, **kw):
    f = _fuzzer(**kw)
    f.shm_cov = shm
    f._run_target = lambda data: (shm.advance(), (0, ""))[1]
    return f


class TestFuzzerGate:
    def test_off_by_default_and_costs_nothing(self):
        shm = _FakeShm([{1, 9}, {1}], new={9})
        f = _wired(shm)
        assert f._confirm_new_coverage(b"x", shm, True, {1, 9}) == (True, {1, 9})
        assert shm.reruns == 0

    def test_phantom_only_success_is_withdrawn_and_rejected(self):
        shm = _FakeShm([{1, 9}, {1}], new={9})
        f = _wired(shm, confirm_novelty=True)
        assert f._confirm_new_coverage(b"x", shm, True, {1, 9}) == (False, {1})
        assert shm.rejected == {9} and shm.reruns == 1
        assert f._confirmed_edges == {1}

    def test_real_novelty_costs_exactly_one_rerun(self):
        shm = _FakeShm([{1, 9}, {1, 9}], new={9})
        f = _wired(shm, confirm_novelty=True)
        assert f._confirm_new_coverage(b"x", shm, True, {1, 9}) == (True, {1, 9})
        assert shm.reruns == 1 and shm.rejected == frozenset()

    def test_no_novelty_means_no_rerun(self):
        shm = _FakeShm([{1}, {1}])
        f = _wired(shm, confirm_novelty=True)
        assert f._confirm_new_coverage(b"x", shm, False, {1}) == (False, {1})
        assert shm.reruns == 0 and f._confirmed_edges is None

    def test_crash_or_timeout_is_never_rerun(self):
        # A truncated execution's ids are short, not phantom.
        shm = _FakeShm([{1, 9}, {1}], new={9})
        f = _wired(shm, confirm_novelty=True)
        assert f._confirm_new_coverage(b"x", shm, True, {1, 9}, skip=True) == (True, {1, 9})
        assert shm.reruns == 0

    def test_a_rerun_that_raises_leaves_the_verdict_alone(self):
        shm = _FakeShm([{1, 9}], new={9})
        f = _fuzzer(confirm_novelty=True)
        f.shm_cov = shm

        def boom(data):
            raise OSError("cannot spawn")

        f._run_target = boom
        assert f._confirm_new_coverage(b"x", shm, True, {1, 9}) == (True, {1, 9})
        assert shm.rejected == frozenset()

    def test_confirmed_edges_reset_each_call(self):
        shm = _FakeShm([{1, 9}, {1}, {1}], new={9})
        f = _wired(shm, confirm_novelty=True)
        f._confirm_new_coverage(b"x", shm, True, {1, 9})
        f._confirm_new_coverage(b"x", shm, False, {1})
        assert f._confirmed_edges is None

    def test_stats_count_reruns_and_phantoms(self):
        shm = _FakeShm([{1, 8, 9}, {1, 8}], new={8, 9})
        f = _wired(shm, confirm_novelty=True)
        f._confirm_new_coverage(b"x", shm, True, {1, 8, 9})
        assert f._confirm_stats == {"reruns": 1, "withdrawn": 0, "phantom_ids": 1}

    def test_none_shm_is_a_passthrough(self):
        f = _fuzzer(confirm_novelty=True)
        assert f._confirm_new_coverage(b"x", None, True, {1}) == (True, {1})


class TestFalsification:
    """Hard Rule 23: the gate must be observable, so the test must be able to fail."""

    def test_phantom_ids_reach_the_tracker_only_without_the_gate(self):
        def run(enabled):
            shm = _FakeShm([{1, 2, 9}, {1, 2}], new={9})
            f = _wired(shm, confirm_novelty=enabled)
            has_new, ids = f._confirm_new_coverage(b"x", shm, True, {1, 2, 9})
            return has_new, ids

        assert run(False) == (True, {1, 2, 9})  # phantom admitted
        assert run(True) == (False, {1, 2})  # phantom withdrawn


class TestWiring:
    def test_constructor_flag_defaults_off_and_is_not_last(self):
        import inspect

        from fuzzer_tool.services.fuzzer import Fuzzer

        params = inspect.signature(Fuzzer.__init__).parameters
        assert params["confirm_novelty"].default is False
        # The kruskal-count wiring test pins seed_round_robin_scheduler as last.
        assert list(params)[-1] == "seed_round_robin_scheduler"

    def test_cli_passes_flag_to_every_construction(self):
        import ast
        import inspect

        from fuzzer_tool.cli import commands

        def kws(fn, callee):
            tree = ast.parse(inspect.getsource(fn))
            calls = [
                n
                for n in ast.walk(tree)
                if isinstance(n, ast.Call) and getattr(n.func, "id", None) == callee
            ]
            assert calls
            return [{k.arg for k in c.keywords} for c in calls]

        assert all("confirm_novelty" in k for k in kws(commands.cmd_fuzz, "Fuzzer"))

    def test_parser_declares_flag(self):
        import ast
        import inspect

        from fuzzer_tool.cli import commands
        from tests.test_regression_cli_fuzzer_kwargs import _fuzz_parser_dests

        assert "confirm_novelty" in _fuzz_parser_dests(ast.parse(inspect.getsource(commands)))


class TestSummaryLine:
    def _reporter(self):
        from fuzzer_tool.services.stats import StatsReporter

        return StatsReporter.__new__(StatsReporter)

    def test_line_reports_reruns_withdrawals_and_phantoms(self, capsys):
        from types import SimpleNamespace

        f = SimpleNamespace(
            _confirm_novelty=True,
            _confirm_stats={"reruns": 12, "withdrawn": 2, "phantom_ids": 5},
        )
        self._reporter()._print_summary_confirm(f)
        out = capsys.readouterr().out
        assert "12 reruns, 2 successes withdrawn, 5 phantom ids rejected" in out

    def test_silent_when_off_or_absent(self, capsys):
        from types import SimpleNamespace

        r = self._reporter()
        r._print_summary_confirm(SimpleNamespace(_confirm_novelty=False))
        r._print_summary_confirm(SimpleNamespace())
        assert capsys.readouterr().out == ""


class TestCallSitesRerunTheInputThatRan:
    """Regression. The first wiring passed ``data`` (the parent seed) at the
    fuzz_one site while the executed input is ``mutated``. Every rerun then
    measured a different input, "confirmed" almost nothing, and a campaign
    that discovered 178 edges found 60. The helper's own tests could not see
    it, so the call sites are pinned to the variable that was executed."""

    @staticmethod
    def _calls(fn_name, attr):
        import ast
        import inspect

        from fuzzer_tool.services import fuzzer as mod

        fn = getattr(mod.Fuzzer, fn_name)
        tree = ast.parse(__import__("textwrap").dedent(inspect.getsource(fn)))
        return [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == attr
        ]

    @staticmethod
    def _first_arg(call):
        import ast

        return ast.unparse(call.args[0])

    def test_fuzz_one_reruns_the_mutated_input(self):
        runs = self._calls("fuzz_one", "_run_target")
        confirms = self._calls("fuzz_one", "_confirm_new_coverage")
        assert len(confirms) == 1
        executed = self._first_arg(runs[0])
        assert executed == "mutated"
        assert self._first_arg(confirms[0]) == executed

    def test_calibration_reruns_the_seed_that_ran(self):
        runs = self._calls("_calibrate_seed_baselines", "_run_target")
        confirms = self._calls("_calibrate_seed_baselines", "_confirm_new_coverage")
        assert len(confirms) == 1
        assert self._first_arg(confirms[0]) == self._first_arg(runs[0])


class TestHelp:
    def test_fuzz_help_renders(self):
        """Regression: an unescaped percent sign in a help string makes
        ``fuzz --help`` raise TypeError from argparse's %-formatting."""
        import subprocess
        import sys

        out = subprocess.run(
            [sys.executable, "-m", "fuzzer_tool", "fuzz", "--help"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert out.returncode == 0, out.stderr[-300:]
        assert "--confirm-novelty" in out.stdout
