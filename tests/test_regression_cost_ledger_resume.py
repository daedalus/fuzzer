"""Regression: the per-seed cost ledger survives resume.

``meta["total_time"]`` is written on every timed execution in
``Fuzzer.fuzz_one``, and all three of its readers divide it to recover a mean
``exec_us``.  It was absent from ``CorpusManager.save_state``/``load_state``
while ``fuzz_count`` was persisted, so a resumed seed carried a large restored
count against a zero numerator, hit the ``max(1.0, ...)`` floor in every
reader, and read as the cheapest seed in the corpus — for the rest of the
campaign, since the numerator restarts at zero while the count does not.

Measured on png_read before the fix: after a 200-execution resume, 116 of 216
fuzzed seeds carried ``total_time == 0.0`` against 407 restored fuzzes.

``cost_samples`` travels with it because ``fuzz_count`` is the wrong
denominator even within a single run: the initial seed replay in ``Fuzzer.run``
increments the count without crediting time.  Measured on the same campaign,
the two disagreed on 126 of 147 timed seeds.
"""

from __future__ import annotations

import tempfile
from array import array
from pathlib import Path
from types import SimpleNamespace

from fuzzer_tool.core.cost_ledger import seed_exec_us
from fuzzer_tool.core.state_store import StateStore
from fuzzer_tool.services.corpus_manager import CorpusManager, seed_key
from fuzzer_tool.services.operators import OperatorEngine


class _StubTracker:
    def to_dict(self):
        return {}

    def from_dict(self, data):
        pass


class _StubIO:
    def save(self):
        return {}

    def load(self, data):
        pass


class _StubSeedQuality:
    def state_dict(self):
        return {}

    def load_state_dict(self, data):
        pass


def _make_fuzzer(tmp: Path, corpus, seed_meta, resume=False):
    return SimpleNamespace(
        corpus_dir=tmp,
        target=str(tmp / "no-such-target"),
        _state_store=StateStore(tmp),
        corpus=list(corpus),
        seed_meta=dict(seed_meta),
        map_size=8192,
        resume=resume,
        _use_elo=False,
        _edge_tracker=_StubTracker(),
        _sensitivity=_StubIO(),
        _crash_mi=_StubIO(),
        _seed_quality=_StubSeedQuality(),
        exec_count=0,
        crash_count=0,
        timeout_count=0,
        crash_sigs={},
        crash_frames={},
        crash_min_sizes={},
        op_counts={},
        op_success={},
        op_edges={},
        _operators=OperatorEngine(None),
        _corpus_size_history=array("I"),
    )


def _meta(**kw):
    base = {
        "fuzz_count": 0,
        "coverage_edges": 0,
        "momentum": 0.0,
        "redqueen_offsets": [],
        "added_at": 0.0,
    }
    base.update(kw)
    return base


def _round_trip(seed_meta):
    """save_state then load_state into a fresh fuzzer; return the new meta."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        corpus = list(seed_meta)
        saved = _make_fuzzer(tmp, corpus, seed_meta)
        CorpusManager(saved).save_state()

        loaded = _make_fuzzer(tmp, corpus, {}, resume=True)
        cm = CorpusManager(loaded)
        cm.init_seed_metadata()
        return loaded.seed_meta


class TestLedgerRoundTrip:
    def test_total_time_and_samples_are_restored(self):
        seed = b"expensive-seed"
        meta = _round_trip({seed: _meta(fuzz_count=400, total_time=12.5, cost_samples=380)})
        assert meta[seed]["total_time"] == 12.5
        assert meta[seed]["cost_samples"] == 380

    def test_restored_seed_no_longer_reads_as_free(self):
        """The bug itself: restored count, dropped numerator, 1us floor."""
        seed = b"expensive-seed"
        meta = _round_trip({seed: _meta(fuzz_count=400, total_time=12.5, cost_samples=380)})
        corpus_mean_us = 160.0
        exec_us = seed_exec_us(meta[seed], corpus_mean_us)
        # 12.5s over 380 samples is ~32.9ms per execution.
        assert exec_us > 30_000
        # Before the fix this was exactly the floor.
        assert exec_us != 1.0

    def test_ledger_is_not_recoverable_from_fuzz_count(self):
        """A cheap and an expensive seed with equal counts must stay distinct."""
        cheap, pricey = b"cheap", b"pricey"
        meta = _round_trip(
            {
                cheap: _meta(fuzz_count=100, total_time=0.02, cost_samples=100),
                pricey: _meta(fuzz_count=100, total_time=2.0, cost_samples=100),
            }
        )
        assert seed_exec_us(meta[pricey], 160.0) > 50 * seed_exec_us(meta[cheap], 160.0)


class TestLegacyState:
    def test_state_without_the_ledger_restores_as_unmeasured(self):
        """An old state file has neither key; that must mean unknown, not free."""
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            seed = b"legacy-seed"
            saved = _make_fuzzer(tmp, [seed], {seed: _meta(fuzz_count=400)})
            CorpusManager(saved).save_state()

            # Simulate a pre-ledger state file by deleting the two keys.
            raw = saved._state_store.get("corpus")
            entry = raw["seed_meta"][seed_key(seed)]
            entry.pop("total_time", None)
            entry.pop("cost_samples", None)
            saved._state_store.set("corpus", raw)

            loaded = _make_fuzzer(tmp, [seed], {}, resume=True)
            CorpusManager(loaded).init_seed_metadata()

            m = loaded.seed_meta[seed]
            assert m["fuzz_count"] == 400
            assert m["cost_samples"] == 0
            # Zero samples means "no measurement": the reader substitutes the
            # corpus mean rather than the floor.
            assert seed_exec_us(m, 160.0) == 160.0
