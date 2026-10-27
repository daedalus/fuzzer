"""Regression: the raw-speed probe must not time seeds stored under seeds/timeouts/.

Inputs saved to ``seeds/timeouts/`` are loaded into the live corpus (as
irreplaceable seeds) and each one burns the whole execution timeout. In a
corpus where they were ~40% of the seeds the probe averaged ~0.4 s per run and
printed "Raw target speed: 2 eps", which says nothing about the target on the
inputs it normally sees.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fuzzer_tool.adapters.filesystem import hash_data
from fuzzer_tool.services.fuzzer import RAW_PROBE_FALLBACK, Fuzzer


def _store(corpus_dir: Path, tree: str, data: bytes) -> None:
    h = hash_data(data)
    dest = corpus_dir / "seeds" / tree / h[:2] / f"id_{h}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)


def _probe(corpus: list[bytes], corpus_dir: Path) -> list[bytes]:
    seen: list[bytes] = []
    fake = SimpleNamespace(corpus=corpus, corpus_dir=corpus_dir, _run_target=seen.append)
    Fuzzer._probe_raw_speed(fake)
    return seen


def test_regression_raw_speed_probe_skips_timeout_seeds(tmp_path):
    normal = [b"ok%04d" % i for i in range(300)]
    slow = [b"slow%04d" % i for i in range(300)]
    for s in normal:
        _store(tmp_path, "ab", s)
    for s in slow:
        _store(tmp_path, "timeouts", s)
    corpus = [s for pair in zip(normal, slow, strict=True) for s in pair]
    seen = _probe(corpus, tmp_path)
    assert seen, "probe ran nothing"
    assert set(seen) <= set(normal)


def test_regression_raw_speed_probe_keeps_other_protected_seeds(tmp_path):
    """Falsification: only timeouts/ is skipped, not irreplaceable/ or crashing/."""
    keep = [b"irr%03d" % i for i in range(20)]
    for s in keep:
        _store(tmp_path, "irreplaceable", s)
    seen = _probe(keep, tmp_path)
    assert seen and set(seen) <= set(keep)


def test_regression_raw_speed_probe_all_timeouts_uses_fallback(tmp_path):
    only = [b"slow%02d" % i for i in range(30)]
    for s in only:
        _store(tmp_path, "timeouts", s)
    seen = _probe(only, tmp_path)
    assert seen and set(seen) == {RAW_PROBE_FALLBACK}


def test_regression_raw_speed_probe_no_timeouts_dir(tmp_path):
    corpus = [b"x%03d" % i for i in range(50)]
    seen = _probe(corpus, tmp_path)
    assert seen and set(seen) <= set(corpus)
