"""Wall detection: comparisons the campaign reaches constantly and never passes.

A wall does not look like a plateau. The campaign is not stuck -- it is
reaching the comparison over and over and failing it -- and from the
coverage map that is indistinguishable from never reaching it at all. High
``fired``, near-zero ``asserted``, flat over the campaign is the signature,
and nothing else in the fuzzer can see it.

The direction of the fire rate is the part that changes what to do. A stall
whose fire count is rising means the campaign still drives into the wall and
keeps failing there; a stall whose fire count is falling means it stopped
reaching the parser depth it used to. Those want opposite remedies and the
edge signal is identical in both.
"""

from __future__ import annotations

from fuzzer_tool.core.cmplog import (
    CMP_RATE_TREND_MARGIN,
    CMP_WALL_MIN_FIRED,
    CmplogCollector,
)


def _c(fired: dict[str, int], asserted: dict[str, int] | None = None) -> CmplogCollector:
    c = CmplogCollector()
    c.cmp_fired = dict(fired)
    c.cmp_asserted = dict(asserted or {})
    return c


class TestWallCriteria:
    def test_never_satisfied_and_hot_is_a_wall(self):
        c = _c({"memcmp": 500_000}, {})
        assert "memcmp" in c.comparison_walls()

    def test_below_the_evidence_floor_is_not_a_wall(self):
        """Entered twice and satisfied never is an accident."""
        c = _c({"memcmp": CMP_WALL_MIN_FIRED - 1}, {})
        assert c.comparison_walls() == {}

    def test_a_comparison_being_passed_is_not_a_wall(self):
        c = _c({"memcmp": 100_000}, {"memcmp": 50_000})
        assert c.comparison_walls() == {}

    def test_a_trickle_of_asserts_still_counts_as_a_wall(self):
        """The checksum-gate shape: millions of fires, a handful of passes."""
        c = _c({"memcmp": 10_000_000}, {"memcmp": 5})
        assert "memcmp" in c.comparison_walls()

    def test_walls_are_ordered_by_fire_count(self):
        c = _c({"memcmp": 5_000, "strcmp": 90_000, "bcmp": 20_000})
        assert list(c.comparison_walls()) == ["strcmp", "bcmp", "memcmp"]

    def test_thresholds_are_overridable(self):
        c = _c({"memcmp": 50}, {})
        assert c.comparison_walls() == {}
        assert "memcmp" in c.comparison_walls(min_fired=10)


class TestFireTrend:
    def _drive(self, c: CmplogCollector, counts: list[int], name: str = "memcmp") -> None:
        for n in counts:
            c.last_fired = {name: n}
            c.cmp_fired[name] = c.cmp_fired.get(name, 0) + n
            c._update_rates()

    def test_unknown_before_any_observation(self):
        assert _c({"memcmp": 5_000}).fire_trend("memcmp") == "unknown"

    def test_a_steady_rate_reads_flat(self):
        c = _c({})
        self._drive(c, [40] * 200)
        assert c.fire_trend("memcmp") == "flat"

    def test_first_observation_seeds_both_rates(self):
        """Or every callback reads as rising while the EWMAs warm up."""
        c = _c({})
        self._drive(c, [40])
        assert c.cmp_rate_fast["memcmp"] == c.cmp_rate_slow["memcmp"] == 40.0
        assert c.fire_trend("memcmp") == "flat"

    def test_a_climbing_rate_reads_rising(self):
        c = _c({})
        self._drive(c, [10] * 200)
        self._drive(c, [400] * 30)
        assert c.fire_trend("memcmp") == "rising"

    def test_a_collapsing_rate_reads_falling(self):
        c = _c({})
        self._drive(c, [400] * 200)
        self._drive(c, [10] * 30)
        assert c.fire_trend("memcmp") == "falling"

    def test_a_callback_that_stopped_firing_decays(self):
        """Absence, not a zero, is how "stopped reaching it" presents."""
        c = _c({})
        self._drive(c, [400] * 200)
        for _ in range(40):
            c.last_fired = {"strcmp": 1}
            c.cmp_fired["strcmp"] = c.cmp_fired.get("strcmp", 0) + 1
            c._update_rates()
        assert c.fire_trend("memcmp") == "falling"

    def test_an_empty_drain_does_not_move_the_rates(self):
        """A redundant second drain of an already-drained boundary."""
        c = _c({})
        self._drive(c, [40] * 100)
        before = dict(c.cmp_rate_fast)
        c.last_fired = {}
        c._update_rates()
        assert c.cmp_rate_fast == before

    def test_margin_governs_the_dead_band(self):
        c = _c({})
        c.cmp_rate_slow["memcmp"] = 100.0
        c.cmp_rate_fast["memcmp"] = 100.0 * (1 + CMP_RATE_TREND_MARGIN / 2)
        assert c.fire_trend("memcmp") == "flat"
        c.cmp_rate_fast["memcmp"] = 100.0 * (1 + CMP_RATE_TREND_MARGIN * 2)
        assert c.fire_trend("memcmp") == "rising"


class TestWallSummary:
    def test_empty_when_nothing_is_walled(self):
        assert _c({"memcmp": 100}, {"memcmp": 50}).wall_summary() == ""

    def test_names_the_worst_walls_with_their_counts(self):
        c = _c({"memcmp": 90_000, "strcmp": 20_000})
        summary = c.wall_summary()
        assert summary.startswith("walls: ")
        assert "memcmp 90,000x" in summary
        assert summary.index("memcmp") < summary.index("strcmp")

    def test_caps_the_list(self):
        c = _c({f"cb{i}": 10_000 + i for i in range(9)})
        assert c.wall_summary().count(",") <= 5  # 3 entries, thousands separators
