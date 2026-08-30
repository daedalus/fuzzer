"""Regression: persisted seed_meta survives resume for seeds of any size.

``CorpusManager.save_state`` keyed ``seed_meta`` by ``seed.hex()`` and
skipped any key >= 256 chars. 289c85f added that guard to drop corrupted
tracker JSON that had been loaded as corpus seeds, on the stated
assumption that "seed keys should be hex hashes (< 256 chars)". They were
not hashes -- ``seed.hex()`` is the hex of the seed's whole content -- so
the threshold was really a seed length of 128 bytes, and what it dropped
was the entire metadata entry for every seed above it.

Measured on a live png campaign before the fix: the corpus held seeds up
to 2858 bytes, the longest persisted key was 234 chars (117 bytes), and
no key >= 256 chars existed in the state at all. Across ``--resume`` those
seeds lost fuzz_count, coverage_edges, added_at, momentum, lineage_depth,
the redqueen offsets and matches, and the cost ledger -- so b393145,
landed specifically to keep the cost ledger across resume, was defeated
for essentially every realistic seed.

tests/test_regression_cost_ledger_resume.py covers that ledger round trip
and passed throughout, because its seeds are a dozen bytes long and never
reach the cliff. Seed size is the axis that was missing, so it is the axis
these tests vary.
"""

from __future__ import annotations

import tempfile
from array import array
from pathlib import Path
from types import SimpleNamespace

from fuzzer_tool.core.state_store import StateStore
from fuzzer_tool.services.corpus_manager import CorpusManager, seed_key
from fuzzer_tool.services.operators import OperatorEngine

# Past the old 128-byte cliff by a wide margin: hex of this is 4096 chars,
# so the >= 256 guard skipped it outright.
BIG_SEED = bytes(range(256)) * 8
# Comfortably under it -- the size every existing state test happens to use.
SMALL_SEED = b"tiny-seed"


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


def _saved_state(seed_meta):
    """save_state, then return the raw persisted corpus section."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        f = _make_fuzzer(tmp, list(seed_meta), seed_meta)
        CorpusManager(f).save_state()
        return f._state_store.get("corpus")


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


class TestKeyScheme:
    def test_key_length_does_not_grow_with_the_seed(self):
        """The property 289c85f's guard was reaching for.

        Hashing gives it outright, rather than buying it by refusing to
        persist anything large.
        """
        assert len(seed_key(BIG_SEED)) == len(seed_key(SMALL_SEED))
        assert len(seed_key(BIG_SEED)) < 256

    def test_the_old_key_for_this_seed_was_over_the_guard(self):
        """Pin the defect, so the reason for the change stays legible."""
        assert len(BIG_SEED.hex()) >= 256


class TestLargeSeedSurvives:
    def test_large_seed_is_persisted_at_all(self):
        state = _saved_state({BIG_SEED: _meta(fuzz_count=41)})
        assert seed_key(BIG_SEED) in state["seed_meta"], (
            "metadata for a seed over 128 bytes was dropped at save time"
        )

    def test_both_sizes_are_persisted(self):
        state = _saved_state({SMALL_SEED: _meta(fuzz_count=1), BIG_SEED: _meta(fuzz_count=2)})
        assert len(state["seed_meta"]) == 2

    def test_fuzz_count_restored_for_a_large_seed(self):
        meta = _round_trip({BIG_SEED: _meta(fuzz_count=41)})
        assert meta[BIG_SEED]["fuzz_count"] == 41

    def test_cost_ledger_restored_for_a_large_seed(self):
        """The exact loss that defeated b393145.

        total_time without cost_samples reads as a seed that consumed
        time for free, so the pair has to make the trip together.
        """
        meta = _round_trip({BIG_SEED: _meta(fuzz_count=400, total_time=12.5, cost_samples=380)})
        assert meta[BIG_SEED]["total_time"] == 12.5
        assert meta[BIG_SEED]["cost_samples"] == 380

    def test_large_and_small_seeds_stay_distinct(self):
        meta = _round_trip(
            {
                SMALL_SEED: _meta(fuzz_count=3, total_time=0.5, cost_samples=3),
                BIG_SEED: _meta(fuzz_count=400, total_time=12.5, cost_samples=380),
            }
        )
        assert meta[SMALL_SEED]["fuzz_count"] == 3
        assert meta[BIG_SEED]["fuzz_count"] == 400


class TestLegacyState:
    def test_hex_keyed_state_still_loads(self):
        """State written before the change must still restore.

        Those files only ever held seeds under 128 bytes, so the fallback
        recovers exactly what is there and reinterprets nothing.
        """
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            f = _make_fuzzer(tmp, [SMALL_SEED], {SMALL_SEED: _meta(fuzz_count=77)})
            CorpusManager(f).save_state()

            state = f._state_store.get("corpus")
            entry = state["seed_meta"][seed_key(SMALL_SEED)]
            state["seed_meta"] = {SMALL_SEED.hex(): entry}  # old on-disk shape
            f._state_store.set("corpus", state)
            f._state_store.save()

            loaded = _make_fuzzer(tmp, [SMALL_SEED], {}, resume=True)
            cm = CorpusManager(loaded)
            cm.init_seed_metadata()
            assert loaded.seed_meta[SMALL_SEED]["fuzz_count"] == 77
