"""Regression: the startup "Raw target speed" probe times corpus seeds.

It ran ``b"\\x00" * 64`` 100 times. ffmpeg rejects that in 0.15 ms and
printed 2,148 eps, while the same build averages 2.13 ms (469 eps) on the
corpus it fuzzes -- a 4.6x overstated ceiling that made the fuzzer's own
overhead look 4.6x worse than it was.
"""

from __future__ import annotations

from types import SimpleNamespace

from fuzzer_tool.services.fuzzer import RAW_PROBE_FALLBACK, Fuzzer


def _probe(corpus: list[bytes]) -> tuple[list[bytes], float]:
    seen: list[bytes] = []
    fake = SimpleNamespace(corpus=corpus, _run_target=seen.append)
    eps = Fuzzer._probe_raw_speed(fake)
    return seen, eps


def test_regression_raw_speed_probe_uses_corpus():
    corpus = [bytes([i % 256]) * (i + 1) for i in range(500)]
    seen, eps = _probe(corpus)
    assert seen, "probe ran nothing"
    assert set(seen) <= set(corpus)
    assert eps > 0


def test_regression_raw_speed_probe_spreads_over_corpus():
    """Falsification: not just the first seeds (a sorted corpus skews by size)."""
    corpus = [i.to_bytes(2, "little") for i in range(1000)]
    seen, _ = _probe(corpus)
    assert len(set(seen)) == len(seen)
    assert max(corpus.index(s) for s in seen) >= len(corpus) // 2


def test_regression_raw_speed_probe_adversarial():
    """Empty corpus falls back; a one-seed corpus still probes the minimum."""
    seen, _ = _probe([])
    assert seen and set(seen) == {RAW_PROBE_FALLBACK}

    seen, _ = _probe([b"only"])
    assert seen and set(seen) == {b"only"}
