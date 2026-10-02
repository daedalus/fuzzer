"""CampaignLedger: cross-campaign preset choice (UCB1) and history priors.

Port of Stardust's ``Opponent::selectOpeningUCB1`` / ``minValueInPreviousGames``.
Expected values are derived here from the formula, not read back from the code.
"""

from __future__ import annotations

import gzip
import math
import pickle
from pathlib import Path

import pytest

from fuzzer_tool.core.campaign_ledger import (
    LEDGER_SECTION,
    CampaignLedger,
)
from fuzzer_tool.core.state_store import STATE_FILENAME

BUILD_X = "build-x"
BUILD_Y = "build-y"


def _ucb_reference(records, presets, build, decay, same_build_weight):
    """Independent UCB1 over decayed, build-weighted, max-normalised scores."""
    best = max(r["score"] for r in records) or 1.0
    reward = dict.fromkeys(presets, 0.0)
    potential = dict.fromkeys(presets, 0.0)
    for age, rec in enumerate(reversed(records)):
        if rec["preset"] not in reward:
            continue
        w = math.exp(-decay * (age + 1))
        if rec["build"] == build:
            w *= same_build_weight
        potential[rec["preset"]] += w
        reward[rec["preset"]] += w * rec["score"] / best
    log_total = math.log(max(sum(potential.values()), 1.0))
    scores = {
        p: reward[p] / potential[p] + math.sqrt(2.0 * log_total / potential[p]) for p in presets
    }
    return max(presets, key=lambda p: (scores[p], -presets.index(p)))


def _ledger(tmp_path: Path, rows, **kw) -> CampaignLedger:
    led = CampaignLedger(tmp_path, **kw)
    for preset, score, build in rows:
        led.record(preset, score, build)
    return led


class TestSelect:
    def test_untried_preset_first(self, tmp_path: Path) -> None:
        led = _ledger(tmp_path, [("a", 10, BUILD_X), ("b", 20, BUILD_X)])
        assert led.select(["a", "b", "c"], BUILD_X) == "c"

    def test_fewest_trials_below_minimum(self, tmp_path: Path) -> None:
        rows = [("a", 1, BUILD_X), ("b", 1, BUILD_X), ("b", 1, BUILD_X), ("c", 1, BUILD_X)]
        led = _ledger(tmp_path, rows)
        # a and c have 1 trial (< 2); a comes first in ballot order.
        assert led.select(["a", "b", "c"], BUILD_X, min_trials=2) == "a"

    def test_matches_reference_formula(self, tmp_path: Path) -> None:
        rows = [
            ("a", 100, BUILD_X),
            ("b", 70, BUILD_Y),
            ("a", 90, BUILD_Y),
            ("b", 40, BUILD_X),
            ("a", 30, BUILD_X),
            ("b", 95, BUILD_X),
        ]
        led = _ledger(tmp_path, rows)
        records = [{"preset": p, "score": s, "build": b} for p, s, b in rows]
        chosen = set()
        for decay in (0.0, 0.1, 0.5, 3.0):
            # Control: the reference agrees with itself before it judges the code.
            ref = _ucb_reference(records, ["a", "b"], BUILD_X, decay, 2.0)
            assert ref == _ucb_reference(records, ["a", "b"], BUILD_X, decay, 2.0)
            assert led.select(["a", "b"], BUILD_X, decay=decay, min_trials=1) == ref
            chosen.add(ref)
        # The scenario must exercise both outcomes or it cannot catch a constant.
        assert chosen == {"a", "b"}

    def test_same_build_bonus_flips_choice(self, tmp_path: Path) -> None:
        # a shines on X, b on Y; equal potentials, so only the bonus decides.
        rows = [("a", 100, BUILD_X), ("a", 50, BUILD_Y), ("b", 50, BUILD_X), ("b", 100, BUILD_Y)]
        led = _ledger(tmp_path, rows)
        assert led.select(["a", "b"], BUILD_X, decay=0.0, min_trials=1) == "a"
        assert led.select(["a", "b"], BUILD_Y, decay=0.0, min_trials=1) == "b"

    def test_decay_follows_recent_results(self, tmp_path: Path) -> None:
        # Old history favours a; the last four campaigns favour b.
        old = [("a", 100, BUILD_X), ("b", 10, BUILD_X)] * 4
        new = [("a", 10, BUILD_X), ("b", 100, BUILD_X)] * 2
        led = _ledger(tmp_path, old + new)
        assert led.select(["a", "b"], BUILD_X, decay=0.0, min_trials=1) == "a"
        assert led.select(["a", "b"], BUILD_X, decay=1.0, min_trials=1) == "b"

    def test_regression_heavy_decay_no_domain_error(self, tmp_path: Path) -> None:
        """Total potential < 1 made log(total) < 0 and sqrt raise (Stardust: NaN)."""
        led = _ledger(tmp_path, [("a", 10, BUILD_Y), ("b", 90, BUILD_Y)])
        # Exploration term clamps to 0, so the higher mean wins.
        assert led.select(["a", "b"], BUILD_X, decay=3.0, min_trials=1) == "b"

    def test_single_preset_short_circuits(self, tmp_path: Path) -> None:
        assert CampaignLedger(tmp_path).select(["only"], BUILD_X) == "only"

    def test_empty_ballot_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            CampaignLedger(tmp_path).select([], BUILD_X)

    def test_falsification_converges_on_better_preset(self, tmp_path: Path) -> None:
        """A ledger that ignores history picks a and b equally; UCB1 must not."""
        led = CampaignLedger(tmp_path)
        # The better preset is second on the ballot, so "always first" fails too.
        yields = {"a": 40, "b": 100}
        picks = dict.fromkeys(yields, 0)
        for _ in range(30):
            preset = led.select(["a", "b"], BUILD_X)
            picks[preset] += 1
            led.record(preset, yields[preset], BUILD_X)
        assert picks["b"] > 2 * picks["a"]


class TestRecord:
    def test_history_is_bounded(self, tmp_path: Path) -> None:
        cap = 5
        led = CampaignLedger(tmp_path, max_records=cap)
        for i in range(3 * cap):
            led.record("a", i, BUILD_X)
        assert len(led) == cap
        # Oldest dropped: the survivors are the last `cap` scores.
        assert led.min_value("score", -1, cap, 0) == 2 * cap

    def test_round_trip(self, tmp_path: Path) -> None:
        led = _ledger(tmp_path, [("a", 7, BUILD_X)])
        led.record("b", 9, BUILD_Y, {"max_gap": 1234})
        assert led.save()

        again = CampaignLedger(tmp_path)
        again.load()
        assert len(again) == 2
        assert again.min_value("max_gap", -1, 10, 0) == 1234

    def test_adversarial_tampered_ledger(self, tmp_path: Path) -> None:
        """Junk rows, NaN/negative scores and a non-list section never crash select."""
        junk = [
            "not-a-dict",
            {"preset": "a"},
            {"preset": "a", "score": float("nan"), "build": BUILD_X},
            {"preset": "b", "score": -5, "build": BUILD_X},
            {"preset": "a", "score": "x", "build": BUILD_X},
            {"preset": 3, "score": 1, "build": BUILD_X},
        ]
        with gzip.open(tmp_path / STATE_FILENAME, "wb") as fh:
            pickle.dump({LEDGER_SECTION: {"version": 1, "records": junk}}, fh)
        led = CampaignLedger(tmp_path)
        led.load()
        # Only the negative-score row survives sanitising (clamped to 0).
        assert len(led) == 1
        assert led.select(["a", "b"], BUILD_X) == "a"

        with gzip.open(tmp_path / STATE_FILENAME, "wb") as fh:
            pickle.dump({LEDGER_SECTION: "garbage"}, fh)
        led2 = CampaignLedger(tmp_path)
        led2.load()
        assert len(led2) == 0


class TestMinValue:
    def test_min_over_recent_window(self, tmp_path: Path) -> None:
        led = CampaignLedger(tmp_path)
        for gap in (500, 100, 900, 300):
            led.record("a", 1, BUILD_X, {"max_gap": gap})
        # Newest three: 300, 900, 100.
        assert led.min_value("max_gap", -1, 3, 0) == min(300, 900, 100)

    def test_too_few_records_returns_default(self, tmp_path: Path) -> None:
        led = CampaignLedger(tmp_path)
        led.record("a", 1, BUILD_X, {"max_gap": 5})
        assert led.min_value("max_gap", -1, 10, 2) == -1

    def test_missing_key_is_version_boundary(self, tmp_path: Path) -> None:
        led = CampaignLedger(tmp_path)
        led.record("a", 1, BUILD_X, {"max_gap": 10})  # beyond the boundary: ignored
        led.record("a", 1, BUILD_X)  # older version: no max_gap
        led.record("a", 1, BUILD_X, {"max_gap": 40})
        led.record("a", 1, BUILD_X, {"max_gap": 70})
        # Stops at the key-less record; two values are enough for min_count 2.
        assert led.min_value("max_gap", -1, 10, 2) == 40
        # ...but not for min_count 3.
        assert led.min_value("max_gap", -1, 10, 3) == -1

    def test_no_history_returns_default(self, tmp_path: Path) -> None:
        assert CampaignLedger(tmp_path).min_value("max_gap", 77, 10, 0) == 77
