"""Regression: seed picker's Markov gate disagreed with the plateau flag.

``_pick_markov_seed`` gated with strict ``<`` and, at threshold 0, fell back
to a KS critical value. The plateau flag uses ``<=`` against the JS null, so a
chain still learning brand-new contexts (null 0, JS > 0) was flagged as
learning but given the plateau generation rate.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from fuzzer_tool.services.seed_picker import SeedPicker
from tests.support.scripted_rng import ScriptedRng

PLATEAU_RATE = 0.03
LEARNING_RATE = 0.15
# A draw between the two rates: generates via the perplexity loop only
# when the gate reads "learning".
BETWEEN = (PLATEAU_RATE + LEARNING_RATE) / 2


class _Markov:
    def __init__(self, js: float, threshold: float):
        self.last_js_divergence = js
        self.last_plateau_threshold = threshold
        self._contexts_seen = 10_000
        self.perplexity_calls = 0

    def generate(self, n):
        return b"x" * n

    def perplexity(self, _data):
        self.perplexity_calls += 1
        return 1.0


def _gated_learning(js: float, threshold: float) -> bool:
    markov = _Markov(js, threshold)
    f = SimpleNamespace(
        markov=markov,
        _rng=ScriptedRng(randoms=[BETWEEN], randints=[8]),
        exec_count=1,
        corpus=[],
        max_len=64,
    )
    picker = SeedPicker.__new__(SeedPicker)
    picker.f = f
    picker._last_corpus_pp = 50.0
    picker._pick_markov_seed()
    return markov.perplexity_calls > 0


@pytest.mark.parametrize(
    ("js", "threshold", "learning"),
    [
        (0.3, 0.0, True),  # new contexts only: null 0, JS > 0
        (0.0, 0.0, False),  # deterministic model / no decision yet
        (0.01, 0.01, False),  # equality is a plateau, as in the flag
        (0.02, 0.01, True),
    ],
)
def test_gate_matches_plateau_flag(js, threshold, learning):
    assert _gated_learning(js, threshold) is learning
