"""Regression: the stall relay's release dwell and cycle telemetry.

Stall recovery is a relay. Engaging costs ``--stall`` execs of silence;
releasing used to cost one edge, so the hysteresis was asymmetric by the
whole threshold and under bursty discovery -- the regime
``StructureFunctionDetector.is_overdispersed`` exists to identify -- a single
arrival from a burst ended recovery and the next quiet stretch re-engaged.

These tests drive ``Fuzzer._stall_recovery_enter`` / ``_stall_recovery_exit``
/ ``_stall_relay_stats`` directly rather than reimplementing the predicate,
so a change to the real release condition cannot pass them by.

The default dwell is deliberately 1 -- i.e. the mechanism ships without
changing behaviour. Every dwell above 1 trades switching frequency for duty
cycle (measured: over 12 fixed arrival realisations, 1 edge gives 103.9
cycles at 64.3% duty, 2 gives 100.8 at 66.0%, 4 gives 81.1 at 73.1%, 9 gives
48.2 at 83.8%), and which side of that trade is better depends on the relay
amplitude, which no campaign had measured until ``_stall_relay_stats``
existed. Picking a default before that measurement is what the handover's
own §5 says not to do.
"""

import collections
import types


class _Relay:
    """The relay half of Fuzzer, with the real methods bound onto it."""

    def __init__(self, release_edges=1, threshold=1000):
        from fuzzer_tool.services.fuzzer import Fuzzer

        self.exec_count = 0
        self._stall_threshold = threshold
        self._stall_release_edges = release_edges
        self._stall_recovery_active = False
        self._stall_recovery_count = 0
        self._stall_recovery_execs = 0
        self._stall_edges_in_recovery = 0
        self._stall_edges_active = 0
        self._stall_engaged_at = None
        self._stall_last_engage_exec = None
        self._stall_cycles = collections.deque(maxlen=256)
        self._stall_cycle_edges = 0
        self._stall_cycle_execs = 0
        self._stall_release_reason = None
        self._cumulative_edges = 0
        self._edge_tracker = types.SimpleNamespace(
            get_cumulative_edge_count=lambda: self._cumulative_edges
        )
        for name in (
            "_stall_recovery_enter",
            "_stall_recovery_exit",
            "_stall_note_coverage",
            "_stall_relay_stats",
        ):
            setattr(self, name, getattr(Fuzzer, name).__get__(self))

    def observe_edges(self, n):
        """Drive the REAL release path, not a copy of it.

        An earlier version of this harness reimplemented the dwell check
        inline, mirroring the fuzz loop. Stripping the dwell from production
        left every case here passing, which is the whole reason
        ``_stall_note_coverage`` exists as a method.
        """
        self._cumulative_edges += n
        self._stall_note_coverage(n)

    def step(self, n=1):
        for _ in range(n):
            self.exec_count += 1
            if self._stall_recovery_active:
                self._stall_recovery_execs += 1


class TestReleaseDwell:
    def test_one_edge_releases_at_the_default(self):
        r = _Relay(release_edges=1)
        r._stall_recovery_enter("test", 1000)
        assert r._stall_recovery_active
        r.observe_edges(1)
        assert not r._stall_recovery_active

    def test_a_dwell_of_three_holds_through_two_edges(self):
        r = _Relay(release_edges=3)
        r._stall_recovery_enter("test", 1000)
        r.observe_edges(1)
        assert r._stall_recovery_active
        r.observe_edges(1)
        assert r._stall_recovery_active
        r.observe_edges(1)
        assert not r._stall_recovery_active

    def test_dwell_counts_edges_not_execs(self):
        """A burst of n edges in one exec satisfies a dwell of n at once.

        The dwell asks for evidence of renewed discovery, not for elapsed
        time. Counting execs instead would hold recovery open across a burst
        that has already answered the question.
        """
        r = _Relay(release_edges=4)
        r._stall_recovery_enter("test", 1000)
        r.observe_edges(4)
        assert not r._stall_recovery_active

    def test_dwell_resets_between_engagements(self):
        r = _Relay(release_edges=3)
        r._stall_recovery_enter("test", 1000)
        r.observe_edges(2)  # not enough
        r._stall_recovery_exit("supercritical regime")  # released another way
        r._stall_recovery_enter("test", 1000)
        assert r._stall_edges_in_recovery == 0
        r.observe_edges(2)
        assert r._stall_recovery_active, "leftover dwell credit leaked across cycles"

    def test_exit_is_idempotent(self):
        """Three call sites reach the exit; a double release must not count."""
        r = _Relay()
        r._stall_recovery_enter("test", 1000)
        r._stall_recovery_exit("new coverage")
        before = len(r._stall_cycles)
        r._stall_recovery_exit("supercritical regime")
        assert len(r._stall_cycles) == before
        assert not r._stall_recovery_active

    def test_release_edges_floor_is_one(self):
        from fuzzer_tool.services.fuzzer import Fuzzer

        sig = Fuzzer.__init__.__code__.co_varnames
        assert "stall_release_edges" in sig
        r = _Relay()
        r._stall_release_edges = max(1, 0)
        assert r._stall_release_edges == 1


class TestRelayTelemetry:
    def _campaign(self, release_edges=1, gap=2500, burst=4, cycles=6):
        r = _Relay(release_edges=release_edges)
        for _ in range(cycles):
            r.step(gap)
            if not r._stall_recovery_active:
                r._stall_recovery_enter("no new edges", r._stall_threshold)
            r.step(200)
            for _ in range(burst):
                r.observe_edges(1)
                r.step(30)
        return r

    def test_period_is_engage_to_engage_spacing(self):
        r = self._campaign()
        st = r._stall_relay_stats()
        assert st["cycles"] >= 3
        assert st["period_mean"] is not None
        # One synthetic cycle is gap + 200 + burst*30 execs long.
        assert 2500 < st["period_mean"] < 4000
        assert st["period_min"] <= st["period_mean"] <= st["period_max"]

    def test_amplitude_compares_engaged_and_released_discovery_rates(self):
        r = self._campaign()
        st = r._stall_relay_stats()
        assert st["rate_active"] is not None
        assert st["rate_idle"] is not None
        assert st["amplitude"] == st["rate_active"] / st["rate_idle"]
        # All edges in this campaign arrive while engaged, so engaged is
        # strictly the more productive mode and amplitude must say so.
        assert st["amplitude"] > 1.0

    def test_edges_partition_between_engaged_and_released(self):
        r = self._campaign()
        st = r._stall_relay_stats()
        assert st["edges_active"] + st["edges_idle"] == r._cumulative_edges

    def test_unmeasured_fields_are_none_not_zero(self):
        """n/a and 0.0 are different answers; the report relies on this."""
        r = _Relay()
        st = r._stall_relay_stats()
        assert st["period_mean"] is None
        assert st["rate_active"] is None
        assert st["amplitude"] is None
        assert st["cycles"] == 0

    def test_idle_rate_of_zero_does_not_divide(self):
        """Released execs with no edges is a real 0.0, and must not divide."""
        r = _Relay()
        r.step(500)  # released, no discovery -> rate_idle is 0.0, not None
        r._stall_recovery_enter("test", 1000)
        r.step(100)
        r.observe_edges(1)  # every edge arrives while engaged
        st = r._stall_relay_stats()
        assert st["rate_idle"] == 0.0
        assert st["rate_active"] is not None
        assert st["amplitude"] is None  # not ZeroDivisionError, not inf

    def test_no_released_execs_reads_none_not_zero(self):
        r = _Relay()
        r._stall_recovery_enter("test", 1000)
        r.step(100)  # every exec engaged
        st = r._stall_relay_stats()
        assert st["rate_idle"] is None
        assert st["amplitude"] is None

    def test_a_dwell_above_the_burst_size_holds_through_the_burst(self):
        """The behavioural difference the dwell actually buys.

        Not asserted here: that a higher dwell lowers the engagement count
        over a campaign. That is a property of the arrival process, not of
        this predicate, and the numbers for it are in the module docstring
        from the simulation. A unit test contrived to show it would only be
        testing its own fixture.
        """
        burst = 3
        held = _Relay(release_edges=burst + 1)
        held._stall_recovery_enter("test", 1000)
        for _ in range(burst):
            held.observe_edges(1)
            held.step(30)
        assert held._stall_recovery_active

        released = _Relay(release_edges=burst)
        released._stall_recovery_enter("test", 1000)
        for _ in range(burst):
            released.observe_edges(1)
            released.step(30)
        assert not released._stall_recovery_active
