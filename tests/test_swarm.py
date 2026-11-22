"""Swarm testing (``--swarm``): per-round random operator subset."""

from __future__ import annotations

from pathlib import Path

import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.swarm import DEFAULT_SWARM_PROB, MIN_SWARM_OPS, check_prob, swarm_filter

_OPS = [f"op{i}" for i in range(20)]


def _rng(seed: int = 1) -> RandPool:
    return RandPool(seed=seed)


class TestSwarmFilter:
    def test_subset_preserves_order(self):
        out = swarm_filter(_OPS, _rng())
        assert out == [op for op in _OPS if op in out]

    def test_rounds_differ(self):
        rng = _rng()
        subsets = {tuple(swarm_filter(_OPS, rng)) for _ in range(50)}
        assert len(subsets) > 40

    def test_inclusion_rate_tracks_prob(self):
        rng = _rng()
        kept = sum(len(swarm_filter(_OPS, rng, 0.25)) for _ in range(2000))
        assert kept / (2000 * len(_OPS)) == pytest.approx(0.25, abs=0.03)

    def test_pair_absence_reachable(self):
        # The point of swarm: "op0 on, op1 off" occurs in ~1/4 of rounds.
        rng = _rng()
        hits = 0
        for _ in range(2000):
            out = swarm_filter(_OPS, rng, DEFAULT_SWARM_PROB)
            hits += "op0" in out and "op1" not in out
        assert 0.18 < hits / 2000 < 0.32

    def test_deterministic_for_seed(self):
        assert swarm_filter(_OPS, _rng(7)) == swarm_filter(_OPS, _rng(7))


class TestAdversarial:
    def test_never_empty_at_tiny_prob(self):
        rng = _rng()
        for _ in range(500):
            out = swarm_filter(_OPS, rng, 0.001)
            assert len(out) >= 1
            assert set(out) <= set(_OPS)

    def test_prob_one_keeps_all(self):
        assert swarm_filter(_OPS, _rng(), 1.0) == _OPS

    @pytest.mark.parametrize("n", range(MIN_SWARM_OPS))
    def test_short_lists_untouched(self, n):
        ops = _OPS[:n]
        assert swarm_filter(ops, _rng()) == ops

    def test_input_not_mutated(self):
        ops = list(_OPS)
        swarm_filter(ops, _rng())
        assert ops == _OPS

    def test_duplicates_and_empty(self):
        assert swarm_filter([], _rng()) == []
        out = swarm_filter(["a"] * 10, _rng())
        assert out and set(out) == {"a"}


class TestWiring:
    def test_cli_flags_present(self):
        from fuzzer_tool.cli import commands

        src = Path(commands.__file__).read_text()
        assert '"--swarm"' in src and '"--swarm-prob"' in src

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5, float("nan")])
    def test_check_prob_rejects(self, bad):
        with pytest.raises(ValueError, match="swarm_prob"):
            check_prob(bad)

    def test_check_prob_accepts(self):
        assert check_prob(0.3) == 0.3

    def _fuzzer(self, tmp_path, **kw):
        from fuzzer_tool.adapters.filesystem import save_to_corpus
        from fuzzer_tool.services.fuzzer import Fuzzer

        corpus, crashes = tmp_path / "c", tmp_path / "k"
        (corpus / "seeds").mkdir(parents=True)
        crashes.mkdir()
        save_to_corpus(b"A" * 64, corpus, set(), None)
        return Fuzzer(
            target="targets/test_target",
            corpus_dir=str(corpus),
            crashes_dir=str(crashes),
            max_len=4096,
            **kw,
        )

    def test_off_by_default(self, tmp_path):
        assert self._fuzzer(tmp_path)._swarm is False

    def test_bad_prob_rejected_by_fuzzer(self, tmp_path):
        with pytest.raises(ValueError, match="swarm_prob"):
            self._fuzzer(tmp_path, swarm=True, swarm_prob=1.0)

    def test_mutate_offers_subset(self, tmp_path, monkeypatch):
        f = self._fuzzer(tmp_path, swarm=True)
        seen: list[int] = []
        full = len(f._operators.build_ops(b"A" * 64))
        real = f._operators.select_op

        def spy(ops):
            seen.append(len(ops))
            return real(ops)

        monkeypatch.setattr(f._operators, "select_op", spy)
        for _ in range(20):
            f._operators.mutate(b"A" * 64)

        assert seen and max(seen) < full
        assert len(set(seen)) > 1
