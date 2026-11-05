"""Tests for tools/edge_hit_spectrum.py: per-edge hit spectrum of a corpus replay.

Replay runs through injected callables, so no target is built here.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS))

import edge_hit_spectrum as ehs  # noqa: E402


def _spectrum(totals) -> "ehs.Spectrum":
    sp = ehs.Spectrum()
    for eid, total in enumerate(totals, 1):
        sp.add({eid: int(total)})
    return sp


class TestSpectrum:
    def test_first_seen_order_and_totals(self):
        sp = ehs.Spectrum()
        sp.add({5: 2, 9: 1})
        sp.add({9: 3, 7: 1})
        assert sp.ids().tolist() == [5, 9, 7]
        assert sp.totals().tolist() == [2, 4, 1]

    def test_slope_recovers_exponent(self):
        # Exact power law along discovery order: total_r = 1e9 * r^-1.5.
        ranks = np.arange(1, 2001)
        sp = _spectrum(np.round(1e9 * ranks**-1.5))
        assert sp.discovery_slope() == pytest.approx(1.5, abs=0.01)

    def test_flat_spectrum_has_zero_slope(self):
        # Control: no ordering effect, no exponent.
        assert _spectrum([7] * 500).discovery_slope() == pytest.approx(0.0, abs=1e-9)

    def test_hot_first_is_negative_rank_corr(self):
        assert _spectrum(range(1000, 0, -1)).hot_first_corr() == pytest.approx(-1.0)
        assert _spectrum(range(1, 1001)).hot_first_corr() == pytest.approx(1.0)

    def test_too_small_raises(self):
        with pytest.raises(ValueError):
            _spectrum([3]).discovery_slope()

    def test_save_load_roundtrip(self, tmp_path):
        sp = _spectrum([9, 4, 4, 1])
        path = tmp_path / "spec.pkl"
        sp.save(path)
        back = ehs.Spectrum.load(path)
        assert back.ids().tolist() == sp.ids().tolist()
        assert back.totals().tolist() == sp.totals().tolist()


class TestCollect:
    def test_collects_counts_per_input(self):
        # Fake target: input byte b touches edge b, b times; edge 1 every time.
        state: dict[int, int] = {}

        def run(data: bytes) -> None:
            state.clear()
            state.update({1: 1, data[0]: data[0]})

        sp = ehs.collect([b"\x03", b"\x05", b"\x03"], run, lambda: dict(state))
        assert sp.ids().tolist() == [1, 3, 5]
        assert sp.totals().tolist() == [3, 6, 5]  # edge 1: once per input

    def test_empty_corpus(self):
        sp = ehs.collect([], lambda d: None, dict)
        assert sp.ids().size == 0
