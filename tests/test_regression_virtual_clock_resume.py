"""Regression: the virtual clock survives resume.

Virtual ``time()`` is ``VIRTUAL_EPOCH + ticks * VIRTUAL_EXEC_S`` and ticks
restarted at 0 on ``--resume``, while restored ``added_at`` stamps kept the old
run's ticks. Ages went negative; at exactly -100 s the seed picker's
``(1 + age * 0.01)`` denominator hit 0 and ``_cdf_pick`` raised
"Total of weights must be finite". Seen on gstreamer_read after 116k execs.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fuzzer_tool.core.clock import Clock, ClockMode
from fuzzer_tool.services.corpus_manager import CorpusManager

from .test_regression_cost_ledger_resume import _make_fuzzer, _meta

TICKS = 5000


def _ticked(n):
    clock = Clock(ClockMode.VIRTUAL)
    for _ in range(n):
        clock.tick()
    return clock


def _resume(tmp, seed, saved, mutate=None):
    """save_state from *saved*, optionally edit the raw state, load fresh."""
    CorpusManager(saved).save_state()
    if mutate:
        raw = saved._state_store.get("corpus")
        mutate(raw)
        saved._state_store.set("corpus", raw)

    loaded = _make_fuzzer(tmp, [seed], {}, resume=True)
    loaded._clock = Clock(ClockMode.VIRTUAL)
    CorpusManager(loaded).init_seed_metadata()
    return loaded


def test_regression_virtual_clock_resume():
    """Restored seeds never read as added in the future."""
    seed = b"late-seed"
    clock = _ticked(TICKS)
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        saved = _make_fuzzer(tmp, [seed], {seed: _meta(added_at=clock.time())})
        saved._clock = clock
        saved.exec_count = TICKS

        loaded = _resume(tmp, seed, saved)
        assert loaded._clock.time() == clock.time()
        assert loaded._clock.time() - loaded.seed_meta[seed]["added_at"] >= 0


def test_legacy_state_falls_back_to_exec_count():
    """Falsification: a state without the tick count still resumes ahead."""
    seed = b"legacy"
    clock = _ticked(TICKS)
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        saved = _make_fuzzer(tmp, [seed], {seed: _meta(added_at=clock.time())})
        saved._clock = clock
        saved.exec_count = TICKS

        loaded = _resume(tmp, seed, saved, lambda raw: raw.pop("clock_ticks", None))
        assert loaded._clock.time() - loaded.seed_meta[seed]["added_at"] >= 0


def test_corrupt_ticks_never_rewind():
    """Adversarial: a negative or non-integer tick count leaves time unchanged."""
    seed = b"corrupt"
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        for bad in (-7, "x", None):
            saved = _make_fuzzer(tmp, [seed], {seed: _meta()})
            saved._clock = Clock(ClockMode.VIRTUAL)

            loaded = _resume(tmp, seed, saved, lambda raw, b=bad: raw.update(clock_ticks=b))
            assert loaded._clock.time() == Clock(ClockMode.VIRTUAL).time()
