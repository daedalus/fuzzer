"""Regression tests for the 2026-11-16 bloom audit.

* give-up path executed a negative-bloom member;
* ``from_dict`` accepted payloads that crash or lose keys at query time;
* corpus save paths ran a redundant bloom pre-check before the exact set;
* a generational wipe zeroed the cumulative query counters;
* ``fold_to`` past ``error_rate`` left ``capacity`` below ``n_added``;
* ``build_batched`` buffered every digest even when extending a base.
"""

import gzip
import inspect
import pickle
from types import SimpleNamespace

import pytest

from fuzzer_tool.adapters import filesystem
from fuzzer_tool.core import negative_bloom
from fuzzer_tool.core.bloom import BloomFilter, ProbeCount
from fuzzer_tool.services.fuzzer import EXEC_DEDUP_RETRIES, Fuzzer

PARENT = b"parent-seed"
KNOWN = b"known-input"
NOVEL = b"novel-mutant"


def _flip(mutated: bytes, _data: bytes) -> bytes:
    return bytes([mutated[0] ^ 0x01]) + mutated[1:] if mutated else mutated


class _Stub:
    """Only what _dedup_mutate touches; ``mutate`` pops a scripted list."""

    def __init__(self, mutants, known=()):
        self._mutants = list(mutants)
        self._dedup_execs = True
        self._exec_bloom = BloomFilter(capacity=1000, error_rate=1e-3)
        self._dedup_hits = 0
        self._dedup_gaveup = 0
        self._orbit_specs = []
        self._orbit_folded = 0
        self.cuckoo_seed_filter = None
        self._cuckoo_recovered: set[str] = set()
        self._operators = SimpleNamespace(force_change=_flip)
        self._neg_bloom = negative_bloom.build(list(known)) if known else None
        self._neg_bloom_hits = 0

    def mutate(self, data):
        return self._mutants.pop(0)

    _dedup_mutate = Fuzzer._dedup_mutate
    _is_repeat = Fuzzer._is_repeat
    _avoid_pruned = Fuzzer._avoid_pruned


# ── 1. negative bloom on the give-up path ────────────────────────────


def test_regression_neg_bloom_giveup_not_executed():
    f = _Stub([KNOWN] * (EXEC_DEDUP_RETRIES + 1), known=[KNOWN])

    out = f._dedup_mutate(PARENT)

    assert f._dedup_gaveup == 1
    assert out == _flip(KNOWN, PARENT)
    assert not f._neg_bloom.query_bytes(out)


# Falsification: a give-up mutant outside the negative bloom runs unchanged.
def test_neg_bloom_giveup_unknown_passes():
    f = _Stub([NOVEL] * (EXEC_DEDUP_RETRIES + 1), known=[KNOWN])
    f._exec_bloom.update_bytes(NOVEL)

    assert f._dedup_mutate(PARENT) == NOVEL
    assert f._dedup_gaveup == 1


# Adversarial: an empty known mutant cannot be changed; no crash, returned.
def test_neg_bloom_giveup_empty_mutant():
    f = _Stub([b""] * (EXEC_DEDUP_RETRIES + 1), known=[b""])

    assert f._dedup_mutate(PARENT) == b""


# ── 2. from_dict validation ──────────────────────────────────────────


def _state(**over):
    bf = BloomFilter(capacity=100, error_rate=0.01)
    bf.add_bytes(b"x")
    state = bf.to_dict()
    state.update(over)
    return state


@pytest.mark.parametrize(
    "over",
    [
        {"k": 0},
        {"k": True},
        {"k": -3},
        {"m": 0, "bits": b""},
        {"m": 1, "bits": b"\x00"},
        {"m": 3 * 128, "bits": bytes(48)},
        {"m": 1024.0},
        {"double": 1},
        {"double": True},
        {"n_added": -1},
        {"n_queries": "7"},
    ],
    ids=lambda o: ",".join(f"{k}={v!r}"[:24] for k, v in o.items()),
)
def test_regression_from_dict_rejects_bad_state(over):
    with pytest.raises(ValueError):
        BloomFilter.from_dict(_state(**over))


# Falsification: an untouched payload still round-trips with its keys.
def test_from_dict_valid_roundtrip():
    bf = BloomFilter.from_dict(_state())

    assert bf.query_bytes(b"x")


# Adversarial: a tampered negative-bloom file fails at load, not mid-run.
def test_regression_neg_bloom_tampered_k(tmp_path):
    path = tmp_path / "neg.pkl.gz"
    path.write_bytes(gzip.compress(pickle.dumps(_state(k=0))))

    with pytest.raises(negative_bloom.NegativeBloomError):
        negative_bloom.load(path)


# ── 3. corpus save paths decide on the exact set alone ───────────────

_SAVE_FUNCS = (
    filesystem.save_to_corpus,
    filesystem.save_irreplaceable,
    filesystem.save_crashing_seed,
    filesystem.save_timeout_seed,
    filesystem.load_corpus,
)


@pytest.mark.parametrize("fn", _SAVE_FUNCS, ids=lambda fn: fn.__name__)
def test_regression_corpus_paths_have_no_bloom(fn):
    assert "bloom" not in inspect.signature(fn).parameters


def test_regression_fuzzer_has_no_corpus_bloom():
    assert "self.bloom" not in inspect.getsource(Fuzzer.__init__)


# Falsification: dedup still holds without the pre-check.
def test_save_to_corpus_dedups_by_set(tmp_path):
    seen: set[str] = set()

    assert filesystem.save_to_corpus(b"a", tmp_path, seen) is True
    assert filesystem.save_to_corpus(b"a", tmp_path, seen) is False


# Adversarial: a hash in the set but never seen by any filter is a duplicate.
def test_save_to_corpus_preloaded_set(tmp_path):
    seen = {filesystem.hash_data(b"old")}

    assert filesystem.save_to_corpus(b"old", tmp_path, seen) is False


# ── 4. generational wipe keeps cumulative counters ───────────────────


def test_regression_reset_keeps_counters():
    bf = BloomFilter(capacity=2, error_rate=0.01)
    for key in (b"a", b"b", b"c", b"c"):
        bf.update_bytes(key, reset_on_full=True)

    assert (bf.n_queries, bf.n_accepted, bf.n_rejected) == (4, 3, 1)
    assert bf.n_added == 1  # wiped once, then "c" added


def test_regression_reset_digest_keeps_counters():
    bf = BloomFilter(capacity=1, error_rate=0.01)
    for key in (b"a", b"b"):
        bf.update_digest(BloomFilter.digest_bytes(key), reset_on_full=True)

    assert (bf.n_queries, bf.n_accepted) == (2, 2)


# Falsification: an explicit clear() still zeroes everything.
def test_clear_zeroes_counters():
    bf = BloomFilter(capacity=4, error_rate=0.01)
    bf.update_bytes(b"a")
    bf.clear()

    assert (bf.n_added, bf.n_queries, bf.n_accepted, bf.n_rejected) == (0, 0, 0, 0)


# ── 5. fold_to past error_rate ───────────────────────────────────────


def _oversized(n: int) -> BloomFilter:
    bf = BloomFilter(capacity=10 * n, error_rate=1e-3, probes=ProbeCount.FROM_RATE)
    for i in range(n):
        bf.add_bytes(b"k%d" % i)
    return bf


def test_regression_loose_fold_keeps_capacity():
    n = 1000
    bf = _oversized(n)

    assert bf.fold_to(5e-2) > 0
    assert bf.capacity > bf.n_added

    bf.update_bytes(b"new", reset_on_full=True)
    assert all(bf.query_bytes(b"k%d" % i) for i in range(n))


# Falsification: a fold at the filter's own rate leaves error_rate alone.
def test_tight_fold_keeps_error_rate():
    bf = _oversized(1000)
    bf.fold_to(1e-3)

    assert bf.error_rate == 1e-3


# Adversarial: a looser target never lowers a filter's own rate.
def test_fold_never_tightens_rate():
    bf = _oversized(1000)
    bf.fold_to(1e-4)

    assert bf.error_rate == 1e-3


# ── 6. build_batched streams into a base ─────────────────────────────


def test_regression_batched_base_streams():
    base = BloomFilter(capacity=100, error_rate=1e-3)

    def batches():
        yield [b"a", b"b"]
        assert base.query_bytes(b"a") and base.query_bytes(b"b")
        yield [b"c"]

    out = negative_bloom.build_batched(batches(), base)

    assert out is base
    assert all(base.query_bytes(k) for k in (b"a", b"b", b"c"))


# Falsification: without a base the filter is still sized from the count.
def test_batched_no_base_sized():
    bf = negative_bloom.build_batched([[b"a", b"b"], [b"c"]])

    assert bf.capacity == negative_bloom.NEG_BLOOM_FACTOR * 3


# Adversarial: all-empty batches extending a base still raise.
def test_batched_base_empty_raises():
    base = BloomFilter(capacity=100, error_rate=1e-3)

    with pytest.raises(negative_bloom.NegativeBloomError):
        negative_bloom.build_batched([[], []], base)
