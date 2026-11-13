"""Perf-counter signals: campaign-max novelty, fault budget, wiring."""

from types import SimpleNamespace

from fuzzer_tool.core.hw_signals import (
    HW_MAX_GROWTH,
    PAGE_BYTES,
    FaultBudget,
    HwMax,
)

MIB = 1 << 20


# ── HwMax ─────────────────────────────────────────────────────────────


def test_first_sighting_is_baseline_not_novelty():
    hm = HwMax()
    assert not hm.observe({"instructions": 1000, "branches": 100})
    assert hm.hits == 0


def test_growth_past_factor_reports():
    hm = HwMax()
    hm.observe({"instructions": 1000})
    assert hm.observe({"instructions": int(1000 * HW_MAX_GROWTH) + 1})
    assert hm.hits == 1


def test_jitter_inside_factor_never_reports():
    """Falsification: noise below the growth factor is not novelty."""
    hm = HwMax()
    hm.observe({"instructions": 1000})
    for v in (1001, 1020, int(1000 * HW_MAX_GROWTH)):
        assert not hm.observe({"instructions": v})
    assert hm.hits == 0


def test_any_tracked_counter_reports():
    hm = HwMax()
    hm.observe({"instructions": 1000, "branches": 100})
    assert hm.observe({"instructions": 1000, "branches": 200})


def test_missing_or_zero_counter_ignored():
    """Adversarial: no PMU (deltas lack the counters) never reports."""
    hm = HwMax()
    for d in ({}, {"instructions": 0}, {"task_clock": 10**9}, {"task_clock": 10**12}):
        assert not hm.observe(d)
    assert hm.hits == 0


def test_reports_bounded_by_growth():
    """Each report admits an input: a doubling stream reports once per step,
    a 10x range at most log_growth(10) + 1 times."""
    import math

    hm = HwMax()
    for v in range(1000, 10_001):
        hm.observe({"instructions": v})
    assert hm.hits <= math.log(10) / math.log(HW_MAX_GROWTH) + 1


# ── FaultBudget ───────────────────────────────────────────────────────


def test_budget_trips_past_limit():
    fb = FaultBudget(limit_mb=1)
    pages = MIB // PAGE_BYTES
    assert not fb.spend(pages)
    assert fb.spend(1)


def test_budget_reset_starts_over():
    fb = FaultBudget(limit_mb=1)
    fb.spend(10**9)
    fb.reset()
    assert not fb.spend(1)


def test_budget_steady_target_never_trips():
    """Falsification: a non-leaking target (no new pages) never restarts."""
    fb = FaultBudget(limit_mb=1)
    assert not any(fb.spend(0) for _ in range(100_000))


# ── FuzzRound admission ───────────────────────────────────────────────


def _round(hw_max, deltas, timeout=False, crash=False):
    from fuzzer_tool.services.fuzz_round import FuzzRound

    f = SimpleNamespace(
        _exec_perplexity=SimpleNamespace(due=lambda: False),
        _doppler=None,
        _perf_novelty=False,
        _perf_novelty_hits=0,
        _hw_max=hw_max,
        _last_perf_deltas=deltas,
    )
    r = FuzzRound(f, b"seed")
    r._is_timeout = timeout
    r._is_crash = crash
    r._scanned_shm = None
    r._observe_counts()
    return r


def test_hw_max_admits_and_credits():
    hm = HwMax()
    hm.observe({"instructions": 1000})
    r = _round(hm, {"instructions": 5000})
    assert r._is_hw_max
    assert r._admits()

    r._judge()
    assert r._success


def test_hw_max_ignores_truncated_runs():
    """Adversarial: a timeout's counts are not a new max."""
    hm = HwMax()
    hm.observe({"instructions": 1000})
    assert not _round(hm, {"instructions": 10**9}, timeout=True)._is_hw_max
    assert not _round(hm, {"instructions": 10**9}, crash=True)._is_hw_max
    assert hm.hits == 0


def test_no_hw_max_without_counters():
    r = _round(None, {"instructions": 10**9})
    assert not r._is_hw_max


# ── Instruction-count hang verdict ────────────────────────────────────


def _limited(limit, instr, rc=0):
    from fuzzer_tool.services.fuzzer import Fuzzer

    f = SimpleNamespace(
        _hw_instr_limit=limit,
        _last_perf_deltas={"instructions": instr},
    )
    return Fuzzer._instr_verdict(f, (rc, "err"))


def test_instr_limit_turns_clean_exit_into_hang():
    assert _limited(1000, 1001) == (-1, "err")


def test_instr_limit_within_budget_untouched():
    assert _limited(1000, 1000) == (0, "err")


def test_instr_limit_never_masks_crash():
    """Falsification: a crash stays a crash however long it ran."""
    assert _limited(1000, 10**9, rc=-11) == (-11, "err")


def test_instr_limit_cli(monkeypatch):
    from fuzzer_tool.cli import commands

    seen = {}
    monkeypatch.setattr(commands, "cmd_fuzz", lambda args: seen.setdefault("a", args) and 0)
    argv = ["ft", "fuzz", "/bin/true", "--hw-instr-limit", "5000"]
    monkeypatch.setattr(commands.sys, "argv", argv)
    commands.main()
    assert seen["a"].hw_instr_limit == 5000


# ── Schedule: cycles factor ───────────────────────────────────────────


def test_cycles_factor_boosts_slow_seed():
    from fuzzer_tool.core.schedules import SeedScorer

    base = dict(
        exec_us=100,
        avg_exec_us=100,
        bitmap_size=50,
        avg_bitmap_size=50,
        handicap=0,
        depth=1,
        fuzz_level=1,
        n_fuzz=1,
        total_execs=100,
    )

    def score(cycles):
        s = SeedScorer("base")
        s.score(**base, hw_cycles=1000)
        return s.score(**base, hw_cycles=cycles)

    assert score(5000) > score(1000)
    assert score(100) < score(1000)


def test_seed_hw_carries_cycles():
    from fuzzer_tool.core.edge_tracker import EdgeTracker

    et = EdgeTracker(map_size=1024)
    et.record_hw("k", {"instructions": 10, "branches": 2, "cpu_cycles": 7})
    assert et.seed_hw("k") == {"hw_instructions": 10, "hw_branches": 2, "hw_cycles": 7}

    # Survives the state round trip.
    et2 = EdgeTracker(map_size=1024)
    et2.from_dict(et.to_dict())
    assert et2.seed_hw("k")["hw_cycles"] == 7


# ── Stats ─────────────────────────────────────────────────────────────


def _stats(totals):
    from fuzzer_tool.adapters.perf_event import PerfCounters
    from fuzzer_tool.services.stats import StatsReporter

    pc = PerfCounters(counter_names=list(totals))
    pc._read_count = 1
    pc._totals = dict(totals)
    return StatsReporter._print_stats_sw_str(None, SimpleNamespace(_perf_counters=pc))


def test_sw_field_format():
    line = _stats({"task_clock": 12_300_000, "page_faults": 1234, "context_switches": 5})
    assert line == " | sw: 12.3ms 1,234pf 5cs"


def test_sw_field_absent_when_nothing_counted():
    assert _stats({"task_clock": 0, "page_faults": 0}) == ""


def test_cache_field_format():
    from fuzzer_tool.adapters.perf_event import PerfCounters
    from fuzzer_tool.services.stats import StatsReporter

    pc = PerfCounters(counter_names=["llc_read_access", "llc_read_miss", "dtlb_read_miss"])
    pc._read_count = 1
    pc._totals = {"llc_read_access": 1000, "llc_read_miss": 250, "dtlb_read_miss": 9}
    line = StatsReporter._print_stats_cache_str(None, SimpleNamespace(_perf_counters=pc))
    assert line == " | llc: 1,000r 250m 25.00% tlb: 9d 0i"


# ── Persistent restart on fault budget ────────────────────────────────


class _Persist:
    def __init__(self):
        self.starts = self.stops = 0

    def run_one(self, data):
        return 0, ""

    def start(self):
        self.starts += 1
        return True

    def stop(self):
        self.stops += 1

    alive = True

    def restart(self):
        self.stop()
        return self.start()


def _persistent_run(faults_per_exec, n):
    from fuzzer_tool.services.runner import TargetRunner

    pr = _Persist()
    f = SimpleNamespace(
        multi_targets=False,
        shm_cov=None,
        _cmplog=None,
        _inprocess_runner=None,
        _persistent_runner=pr,
        _perf_counters=SimpleNamespace(read_and_reset=lambda: {"page_faults": faults_per_exec}),
        _fault_budget=FaultBudget(limit_mb=1),
        _last_perf_deltas={},
    )
    runner = TargetRunner(f)
    for _ in range(n):
        runner.run_target(b"x")
    return pr


def test_leaking_persistent_target_restarts():
    pages = MIB // PAGE_BYTES
    # 2 budgets' worth of faults in 4 execs: restarted twice.
    pr = _persistent_run(pages // 2 + 1, 4)
    assert (pr.stops, pr.starts) == (2, 2)


def test_steady_persistent_target_not_restarted():
    """Falsification: no faults, no restart."""
    pr = _persistent_run(0, 1000)
    assert (pr.stops, pr.starts) == (0, 0)


def test_perf_field_absent_without_pmu():
    """Adversarial: software-only counters must not print a zero perf: field."""
    from fuzzer_tool.adapters.perf_event import PerfCounters
    from fuzzer_tool.services.stats import StatsReporter

    pc = PerfCounters(counter_names=["task_clock"])
    pc._read_count = 1
    pc._totals = {"task_clock": 5}
    line = StatsReporter._print_stats_perf_str(None, SimpleNamespace(_perf_counters=pc))
    assert line == ""


# ── Fuzzer wiring ─────────────────────────────────────────────────────


def _fz(tmp_path, **kw):
    from fuzzer_tool.services.fuzzer import Fuzzer

    corpus = tmp_path / "c"
    corpus.mkdir()
    return Fuzzer(
        target="/bin/true",
        corpus_dir=str(corpus),
        crashes_dir=str(tmp_path / "k"),
        use_coverage=False,
        **kw,
    )


def test_fuzzer_arms_signals_with_counters(tmp_path):
    from fuzzer_tool.adapters.perf_event import PerfCounters

    if not PerfCounters(counter_names=["task_clock"]).available:
        import pytest

        pytest.skip("perf_event_open unavailable")
    f = _fz(tmp_path, hw_perf_counters=["task_clock"], rss_limit_mb=64)
    try:
        assert isinstance(f._hw_max, HwMax)
        assert f._fault_budget._limit == 64 * MIB // PAGE_BYTES
    finally:
        f._perf_counters.close()


def test_fuzzer_without_counters_has_no_signals(tmp_path):
    f = _fz(tmp_path)
    assert f._hw_max is None
    assert f._fault_budget is None
    assert f._hw_instr_limit == 0


def test_instr_limit_dropped_without_pmu(tmp_path, monkeypatch, capsys):
    """Adversarial: a limit with no instruction counter is refused loudly."""
    from fuzzer_tool.adapters import perf_event

    monkeypatch.setattr(perf_event.PerfCounters, "_check_available", lambda self: True)
    f = _fz(tmp_path, hw_instr_limit=1000)
    try:
        assert f.hw_perf
        assert f._hw_instr_limit == 0
        assert "--hw-instr-limit" in capsys.readouterr().out
    finally:
        f._perf_counters.close()
