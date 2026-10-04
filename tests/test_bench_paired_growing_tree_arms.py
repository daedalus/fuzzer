"""Growing Tree sweep arms for bench_paired (handover_maze_algorithms item 3).

An arm is only evidence if its flags reach the real fuzz parser and it
differs from its declared baseline by one knob. Baselines: ``baseline`` is
the default weighted picker; ``seed-round-robin`` is the oldest-first policy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "lib"))

from bench_paired import ARM_BASELINES, ARMS, GROWING_TREE_ARMS, SEED_NEWEST_PCTS  # noqa: E402

from fuzzer_tool.cli import commands  # noqa: E402


def _parse(monkeypatch, flags: list[str]):
    captured: dict[str, object] = {}

    def _spy(args):
        captured["args"] = args
        return 0

    monkeypatch.setattr(commands, "cmd_fuzz", _spy)
    monkeypatch.setattr(sys, "argv", ["fuzzer-tool", "fuzz", "/bin/true", *flags])
    assert commands.main() == 0
    return captured["args"]


class TestRegistered:
    def test_sweep_is_the_handover_grid(self):
        assert SEED_NEWEST_PCTS == (0, 30, 60, 100)

    def test_every_sweep_arm_exists_with_a_baseline(self):
        for arm in GROWING_TREE_ARMS:
            assert arm in ARMS, arm
            assert ARM_BASELINES[arm] in ARMS

    def test_round_robin_arm_is_in_the_group(self):
        assert "seed-round-robin" in GROWING_TREE_ARMS
        assert ARMS["seed-round-robin"] == ["--seed-round-robin-scheduler"]

    def test_one_arm_per_grid_point(self):
        newest = [a for a in GROWING_TREE_ARMS if a.startswith("seed-newest-p")]

        assert len(newest) == len(SEED_NEWEST_PCTS)
        assert len(set(newest)) == len(newest)


class TestParse:
    @pytest.mark.parametrize("pct", SEED_NEWEST_PCTS)
    def test_flags_reach_the_real_parser(self, monkeypatch, pct):
        args = _parse(monkeypatch, ARMS[f"seed-newest-p{pct}"])

        assert args.seed_newest_scheduler is True
        assert args.seed_newest_p == pytest.approx(pct / 100)

    @pytest.mark.parametrize("pct", SEED_NEWEST_PCTS)
    def test_arm_is_only_the_seed_arm_and_its_p(self, pct):
        """Single-variable: no --elo, so the arm is the sole seed strategy."""
        flags = ARMS[f"seed-newest-p{pct}"]

        assert flags[0] == "--seed-newest-scheduler"
        assert "--elo" not in flags
        assert len(flags) == 3

    def test_default_p_matches_the_scheduler_default(self, monkeypatch):
        args = _parse(monkeypatch, ["--seed-newest-scheduler"])

        assert args.seed_newest_p == 0.5


class TestAdversarial:
    def test_out_of_range_p_fails_at_construction_not_mid_campaign(self):
        from fuzzer_tool.core.schedulers.seed_newest import SeedNewestScheduler

        with pytest.raises(ValueError):
            SeedNewestScheduler(rng=object(), p_newest=1.5)

    def test_hail_mary_does_not_enable_the_unmeasured_arm(self, monkeypatch):
        args = _parse(monkeypatch, ["--hail-mary"])

        assert not getattr(args, "seed_newest_scheduler", False)
