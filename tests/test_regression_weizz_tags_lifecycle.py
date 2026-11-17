"""Weizz tag lifecycle: collected tags must land, survive, and reach every seed.

Four defects kept ``--weizz-tags`` operators from ever arming in a real run:

1. ``attach_tags_to_meta`` copied the meta dict, so the collector's result
   was discarded.
2. Collection ran on the round's mutant before ``save_to_corpus`` replaced
   its meta wholesale; initial seeds were never collected at all.
3. A length-changing Weizz op marked the *parent's* map dirty for good: the
   parent's bytes never change, so nothing re-collected it.
4. ``save_state`` dropped the ``weizz_tags_*`` keys, losing tags on resume.
"""

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from fuzzer_tool.core.weizz_tags import (
    ByteTag,
    StructureMap,
    attach_tags_to_meta,
    load_tags_from_meta,
)
from fuzzer_tool.services.corpus_manager import CorpusManager
from fuzzer_tool.services.fuzz_round import FuzzRound
from fuzzer_tool.services.fuzzer import Fuzzer
from fuzzer_tool.services.operators import OperatorEngine
from tests.test_regression_seed_meta_persistence import _make_fuzzer, _meta

SEED = b"AAAABBBBCCCC"
FIELD = 4


def _smap(data: bytes = SEED) -> StructureMap:
    # Three 4-byte top-level fields: cmp_id 1, 2, 3.
    n = len(data) // FIELD
    tags = [ByteTag(cmp_id=1 + i // FIELD) for i in range(n * FIELD)]
    return StructureMap(tags=tags, ntypes=n, input_len=len(data), from_cmplog=True)


# ── 1. collector result lands in seed_meta ─────────────────────────────


def test_regression_weizz_collect_stores_tags():
    meta = _meta()
    f = MagicMock()
    f.colorize = False
    f.seed_meta = {SEED: meta}
    f.weizz_tags_max_len = 4096

    with patch("fuzzer_tool.core.weizz_tags.collect_structure_map", return_value=_smap()):
        Fuzzer._maybe_collect_weizz_tags(f, SEED)

    stored = load_tags_from_meta(f.seed_meta[SEED])
    assert stored is not None
    assert stored.field_spans() == _smap().field_spans()


def test_attach_mutates_in_place():
    """Falsification: the documented contract is "mutates and returns"."""
    meta = _meta()
    out = attach_tags_to_meta(meta, _smap())
    assert out is meta
    assert meta["weizz_tags_len"] == len(SEED)


def test_empty_map_not_stored():
    """Adversarial: a map with no tagged types must not arm the ops."""
    f = MagicMock()
    f.colorize = False
    f.seed_meta = {SEED: _meta()}
    empty = StructureMap(tags=[], ntypes=0, input_len=len(SEED))

    with patch("fuzzer_tool.core.weizz_tags.collect_structure_map", return_value=empty):
        Fuzzer._maybe_collect_weizz_tags(f, SEED)

    assert "weizz_tags_rle" not in f.seed_meta[SEED]


# ── 2. every seed (initial, admitted, resumed) gets collected ─────────


def _feed_round(fuzz_count: int, meta_extra: dict | None = None, data: bytes = SEED):
    f = MagicMock()
    f.weizz_tags = True
    f.weizz_tags_max_len = 4096
    r = FuzzRound(f, data)
    r._meta = {"fuzz_count": fuzz_count, **(meta_extra or {})}
    r._mutated = b"mutant-bytes"
    r._has_new_coverage = False
    r._feed_weizz()
    return f


def test_regression_weizz_initial_seed_collected():
    """First round of a seed with no map collects for the seed, not the mutant."""
    f = _feed_round(1)
    f._maybe_collect_weizz_tags.assert_called_once_with(SEED)


@pytest.mark.parametrize("fuzz_count", [2, 4, 8, 1024])
def test_retry_on_power_of_two_rounds(fuzz_count):
    """An empty pool on round 1 must not leave the seed untagged forever."""
    _feed_round(fuzz_count)._maybe_collect_weizz_tags.assert_called_once_with(SEED)


@pytest.mark.parametrize("fuzz_count", [0, 3, 5, 1023])
def test_no_collect_off_backoff(fuzz_count):
    """Falsification: retries are bounded to O(log rounds) per seed."""
    _feed_round(fuzz_count)._maybe_collect_weizz_tags.assert_not_called()


def test_clean_map_not_recollected():
    clean = attach_tags_to_meta({}, _smap())
    _feed_round(1, clean)._maybe_collect_weizz_tags.assert_not_called()


def test_dirty_map_recollected():
    """Adversarial: a child that inherited a dirty map gets its own."""
    dirty = attach_tags_to_meta({}, _smap())
    dirty["weizz_tags_dirty"] = True
    _feed_round(1, dirty)._maybe_collect_weizz_tags.assert_called_once_with(SEED)


def test_oversize_seed_skipped():
    big = b"\x00" * 4097
    _feed_round(1, data=big)._maybe_collect_weizz_tags.assert_not_called()


def test_no_meta_skipped():
    f = MagicMock()
    f.weizz_tags = True
    r = FuzzRound(f, SEED)
    r._meta = None
    r._feed_weizz()
    f._maybe_collect_weizz_tags.assert_not_called()


# ── 3. length-changing ops stale the map for the rest of the stack only ─


def _engine():
    rng = SimpleNamespace(
        choice=lambda seq: seq[0],
        sample=lambda seq, k: list(seq)[:k],
        randint=lambda a, b: a,
    )
    f = SimpleNamespace(
        max_len=4096,
        weizz_tags=True,
        _rng=rng,
        seed_meta={SEED: attach_tags_to_meta({}, _smap())},
    )
    return OperatorEngine(f), f


def test_regression_weizz_dirty_not_permanent():
    eng, f = _engine()
    assert eng._op_weizz_chunk_dup(bytearray(SEED), 0, SEED) is not None

    assert not f.seed_meta[SEED].get("weizz_tags_dirty")
    assert eng._weizz_structure_map(SEED, bytearray(SEED)) is not None


def test_stale_within_round():
    """Falsification: later ops in the same stack must not use shifted offsets."""
    eng, _ = _engine()
    grown = eng._op_weizz_chunk_dup(bytearray(SEED), 0, SEED)
    assert eng._op_weizz_field_havoc(bytearray(grown), 0, SEED) is None


def test_equal_swap_keeps_map():
    """Adversarial: an equal-width swap shifts nothing; the map stays usable."""
    eng, _ = _engine()
    swapped = eng._op_weizz_chunk_swap(bytearray(SEED), 0, SEED)
    assert len(swapped) == len(SEED)
    assert eng._weizz_structure_map(SEED) is not None


# ── 4. tags survive save_state / load_state ───────────────────────────


def _round_trip(meta: dict) -> dict:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        saved = _make_fuzzer(tmp, [SEED], {SEED: meta})
        CorpusManager(saved).save_state()

        loaded = _make_fuzzer(tmp, [SEED], {}, resume=True)
        CorpusManager(loaded).init_seed_metadata()
        return loaded.seed_meta[SEED]


def test_regression_weizz_tags_survive_resume():
    meta = attach_tags_to_meta(_meta(fuzz_count=5), _smap())
    restored = _round_trip(meta)

    smap = load_tags_from_meta(restored)
    assert smap is not None
    assert smap.field_spans() == _smap().field_spans()
    assert restored["weizz_tags_dirty"] is False


def test_dirty_flag_survives_resume():
    meta = attach_tags_to_meta(_meta(), _smap())
    meta["weizz_tags_dirty"] = True
    assert _round_trip(meta)["weizz_tags_dirty"] is True


def test_untagged_seed_resumes_untagged():
    """Adversarial: no tag keys must not restore as an empty (armed) map."""
    restored = _round_trip(_meta(fuzz_count=2))
    assert load_tags_from_meta(restored) is None
    assert "weizz_tags_rle" not in restored
