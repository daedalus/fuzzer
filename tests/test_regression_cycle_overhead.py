"""Regression: per-cycle fuzzer overhead on ffmpeg (tgt ~1.3 ms vs cyc ~8.6 ms).

py-spy, minimal ffmpeg 9.0.1: ``versifier_generate`` re-scanned the whole
corpus (Python per-byte printable count) and re-tokenized the same newest
text seed on every admission, 1.14 ms/exec; crash-MI ``all_mi`` rebuilt a
65,536-entry dict, sorted list and prefix list every 50 execs for
``weighted_position``, 0.27 ms/exec.
"""

from __future__ import annotations

import bisect
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from fuzzer_tool.core.analyzers.analyzer_crash_eta import CrashMITracker
from fuzzer_tool.core.mutations import generic
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.operators import OperatorEngine
from tests.support.scripted_rng import ScriptedRng

TEXT_A = b"a=1;b=2;c=3"
TEXT_B = b"x=9;y=8;z=7"
BINARY = bytes(range(256))


# ── versifier ───────────────────────────────────────────────────────────


def _old_texty(data: bytes) -> bool:
    """The pre-change predicate, kept as the oracle."""
    printable = sum(1 for b in data if 0x20 <= b < 0x7F)
    return printable >= len(data) * 9 // 10


@pytest.mark.parametrize(
    "data",
    [
        b"",
        TEXT_A,
        BINARY,
        b"\x7f" * 10,
        b"\x1f" + b"a" * 9,  # exactly 9/10 printable
        b"\x1f\x1f" + b"a" * 9,  # just under
        b"line\nline\n",
        b" " * 3 + b"\x00" * 1,
    ],
)
def test_texty_matches_old_predicate(data):
    assert generic._is_texty(data) is _old_texty(data)


def _engine(corpus):
    eng = OperatorEngine.__new__(OperatorEngine)
    eng._ctx_cache = SimpleNamespace(corpus=corpus, _rng=RandPool(seed=1), max_len=4096)
    return eng


def _calls(eng, corpus_states):
    """Run the op once per corpus state; return the seeds _build_verse saw."""
    seen = []
    real = generic._build_verse

    def spy(raw, rng):
        seen.append(bytes(raw))
        return real(raw, rng)

    with patch.object(generic, "_build_verse", spy):
        for corpus in corpus_states:
            eng._ctx_cache.corpus = corpus
            eng._op_versifier_generate(bytearray(b"seed"), 0, b"seed")
    return seen


class TestVersifierReuse:
    def test_regression_binary_admissions_do_not_rebuild(self):
        """Falsification: appending binary seeds re-tokenized TEXT_A each time."""
        base = [TEXT_A]
        states = [base, [*base, BINARY], [*base, BINARY, BINARY[::-1]]]
        seen = _calls(_engine(base), states)
        assert seen.count(TEXT_A) == 1

    def test_new_text_seed_is_picked_up(self):
        states = [[TEXT_A], [TEXT_A, TEXT_B]]
        eng = _engine(states[0])
        seen = _calls(eng, states)
        assert seen[-1] == TEXT_B
        assert eng._versifier_src is states[1][1]

    def test_evicted_source_falls_back(self):
        """Adversarial: the cached seed left the corpus; next newest text wins."""
        old_src = bytes(TEXT_B)
        states = [[TEXT_A, old_src], [TEXT_A, BINARY, BINARY[::-1]]]
        eng = _engine(states[0])
        seen = _calls(eng, states)
        assert seen[-1] == TEXT_A

    def test_all_binary_corpus_returns_none(self):
        """Adversarial: no texty seed; op declines, nothing tokenized."""
        eng = _engine([BINARY])
        assert _calls(eng, [[BINARY], [BINARY, BINARY[::-1]]]) == []


# ── crash-MI weighted_position ──────────────────────────────────────────


def _tracker() -> CrashMITracker:
    t = CrashMITracker(max_positions=64, min_observations=1)
    for i in range(40):
        data = bytes((i * 13 + p * 7) & 0xFF for p in range(10 + i % 30))
        t.record(data, is_crash=(i % 4 == 0))
    return t


def _old_pick(t: CrashMITracker, input_length: int, rng) -> int | None:
    """The pre-change list/bisect sampler over all_mi(), kept as the oracle."""
    mi = t.all_mi()
    positions = sorted(mi)
    weights = [max(mi[p], 0.01) for p in positions]
    prefix, run = [], 0.0
    for w in weights:
        run += w
        prefix.append(run)
    if not positions:
        return None
    idx = bisect.bisect_left(positions, input_length)
    if idx == 0:
        return None
    eligible = prefix[idx - 1]
    if eligible <= 0.0:
        return positions[rng.randrange(idx)]
    r = rng.random() * eligible
    i = bisect.bisect_left(prefix, r, 0, idx)
    return positions[min(i, idx - 1)]


DRAWS = [0.0, 0.13, 0.5, 0.77, 0.999999]
LENGTHS = [1, 5, 17, 39, 64, 1000]


class TestCrashMiSampler:
    def _picks(self, pick, t):
        return [pick(t, n, ScriptedRng(randoms=[d])) for n in LENGTHS for d in DRAWS]

    def test_control_oracle_matches_itself(self):
        """Hard Rule 46: the oracle must agree with a second run of itself."""
        t = _tracker()
        assert self._picks(_old_pick, t) == self._picks(_old_pick, t)

    def test_regression_vector_sampler_matches_oracle(self):
        t = _tracker()
        want = self._picks(_old_pick, t)

        def new(tr, n, rng):
            tr._rng = rng
            return tr.weighted_position(n)

        assert self._picks(new, _tracker()) == want

    def test_sampling_does_not_build_mi_dict(self):
        """Falsification: the per-50-exec rebuild materialised a 65k-entry dict."""
        t = _tracker()
        t._rng = ScriptedRng(randoms=[0.5])
        t.weighted_position(20)
        assert not t._mi_dict_valid

    def test_all_mi_still_exact(self):
        t = _tracker()
        rows = sorted(t.all_mi())
        assert t.all_mi() == dict(zip(rows, t._mi_rows(rows), strict=True))

    def test_empty_tracker(self):
        """Adversarial: nothing observed."""
        t = CrashMITracker()
        assert t.weighted_position(10) is None
        assert t.all_mi() == {}

    def test_no_crashes_is_uniform_floor(self):
        """Adversarial: all MI zero -> every weight is the 0.01 floor."""
        t = CrashMITracker(max_positions=8, min_observations=1)
        for i in range(5):
            t.record(bytes([i] * 8), is_crash=False)
        t._rng = ScriptedRng(randoms=[0.5])
        assert t.weighted_position(8) == _old_pick(t, 8, ScriptedRng(randoms=[0.5]))


# ── in-process input buffer ─────────────────────────────────────────────

import ctypes  # noqa: E402

from fuzzer_tool.adapters import inprocess  # noqa: E402


class TestInputBuffer:
    @pytest.mark.parametrize("n", [0, 1, 255, 2048, 20000])
    def test_regression_exact_size_and_content(self, n):
        """Exact size keeps ASAN's redzone right after the input (overreads caught)."""
        data = bytes((i * 31) & 0xFF for i in range(n))
        buf = inprocess._input_buf(data)
        assert ctypes.sizeof(buf) == n
        assert bytes(buf) == data

    @pytest.mark.parametrize("kind", [bytearray, memoryview])
    def test_buffer_protocol_inputs(self, kind):
        """Adversarial: mutators hand bytearrays; views must copy, not alias."""
        src = bytearray(b"\x00\x01\xfe\xff")
        buf = inprocess._input_buf(kind(src))
        src[0] = 0x7F
        assert bytes(buf) == b"\x00\x01\xfe\xff"


# ── checksum-learner cmplog pairs ───────────────────────────────────────

from fuzzer_tool.core.analyzers.analyzer_checksum_learner import ChecksumLearner  # noqa: E402


def _old_pairs(pool, data):
    """Pre-change per-pair bytes.find loop, kept as the oracle."""
    out = []
    for a, b in pool:
        if len(a) == 4 and len(b) == 4:
            pa, pb = data.find(a), data.find(b)
            if pa >= 0 and pb >= 0 and pa != pb:
                out.append((a if pa < pb else b, int.from_bytes(b if pa < pb else a, "big")))
    return out


def _learner(pool):
    return ChecksumLearner(SimpleNamespace(_cmplog=SimpleNamespace(pairs=pool)))


DATA = bytes((i * 37 + (i >> 3)) & 0xFF for i in range(600)) + b"ABCDABCD" + b"\x00\x01\x02\x03"
POOL = [
    (DATA[10:14], DATA[100:104]),  # data before checksum
    (DATA[200:204], DATA[50:54]),  # reversed order
    (b"ABCD", b"BCDA"),  # repeated token: first occurrence wins
    (b"ABCD", b"ABCD"),  # same position -> dropped
    (DATA[0:4], b"\xde\xad\xbe\xef"),  # one side absent
    (b"\x00\x01", DATA[5:9]),  # non-4-byte operand
    (DATA[-4:], DATA[3:7]),  # operand at the very end
    (DATA[596:600], DATA[597:601]),  # overlapping windows
]


class TestChecksumCmplogPairs:
    def test_control_oracle_matches_itself(self):
        assert _old_pairs(POOL, DATA) == _old_pairs(list(POOL), DATA)

    def test_regression_matches_find_oracle(self):
        assert _learner(POOL).extract_cmplog_pairs(DATA) == _old_pairs(POOL, DATA)

    def test_regression_no_per_pair_find(self):
        """Falsification: the old loop called bytes.find twice per pair."""

        class NoFind(bytes):
            def find(self, *a, **k):
                raise AssertionError("per-pair find")

        assert _learner(POOL).extract_cmplog_pairs(NoFind(DATA)) == _old_pairs(POOL, DATA)

    def test_pool_growth_and_replacement_invalidate(self):
        """Adversarial: extend (same list) and eviction (new list) both seen."""
        pool = list(POOL[:2])
        learner = _learner(pool)
        assert learner.extract_cmplog_pairs(DATA) == _old_pairs(pool, DATA)

        pool.extend(POOL[2:])
        assert learner.extract_cmplog_pairs(DATA) == _old_pairs(pool, DATA)

        learner.f._cmplog.pairs = list(POOL[5:7])
        assert learner.extract_cmplog_pairs(DATA) == _old_pairs(POOL[5:7], DATA)

    @pytest.mark.parametrize("data", [b"", b"abc", b"abcd"])
    def test_short_inputs(self, data):
        """Adversarial: fewer than two 4-byte windows."""
        pool = [(b"abcd", b"abcd"), (b"abcd", b"zzzz")]
        assert _learner(pool).extract_cmplog_pairs(data) == _old_pairs(pool, data)

    def test_empty_pool(self):
        assert _learner([]).extract_cmplog_pairs(DATA) == []


# ── cmplog per-drain asserted view ──────────────────────────────────────

import tracemalloc  # noqa: E402

import numpy as np  # noqa: E402

from fuzzer_tool.core.cmplog import _SiteCounters  # noqa: E402


def _counters(n_sites: int) -> _SiteCounters:
    sc = _SiteCounters(lambda _n: None)
    sc._sites = [("cb", i) for i in range(n_sites)]
    return sc


def _old_last_asserted(sc, site, asserted, start=None):
    """Pre-change dense per-site accumulation, kept as the oracle."""
    last = dict(start or {})
    hit = asserted > 0
    if not hit.any():
        return last
    per_site = np.zeros(len(sc._sites), np.int64)
    np.add.at(per_site, site[hit], asserted[hit])
    ids = np.flatnonzero(per_site)
    for key, n in zip(
        map(sc._sites.__getitem__, ids.tolist()), per_site[ids].tolist(), strict=True
    ):
        last[key] = last.get(key, 0) + n
    return last


SITE = np.array([7, 3, 7, 0, 9, 3, 3], np.int64)
ASSERTED = np.array([2, 0, 5, 1, 0, 4, 1], np.int64)


class TestLastAsserted:
    def test_control_oracle_matches_itself(self):
        sc = _counters(16)
        assert _old_last_asserted(sc, SITE, ASSERTED) == _old_last_asserted(sc, SITE, ASSERTED)

    def test_regression_matches_dense_oracle_with_order(self):
        """Same totals and the same insertion order (dicts are ordered)."""
        sc = _counters(16)
        start = {("cb", 9): 1, ("cb", 3): 2}
        sc.last_asserted = dict(start)
        sc._add_last_asserted(SITE, ASSERTED)
        want = _old_last_asserted(sc, SITE, ASSERTED, start)
        assert list(sc.last_asserted.items()) == list(want.items())

    def test_regression_empty_view_bulk_insert(self):
        """Falsification: the per-drain view starts empty; no per-site get()."""

        class NoGet(dict):
            def get(self, *a, **k):
                raise AssertionError("per-site merge on an empty view")

        sc = _counters(16)
        sc.last_asserted = NoGet()
        sc._add_last_asserted(SITE, ASSERTED)
        want = _old_last_asserted(sc, SITE, ASSERTED)
        assert list(sc.last_asserted.items()) == list(want.items())

    def test_no_hits_is_noop(self):
        """Adversarial: every count zero."""
        sc = _counters(4)
        sc._add_last_asserted(np.array([1, 2], np.int64), np.zeros(2, np.int64))
        assert sc.last_asserted == {}

    def test_regression_memory_independent_of_site_count(self):
        """Falsification: the dense version zeroed len(sites) int64s per drain."""
        sc = _counters(1_000_000)
        site = np.array([5, 999_999, 5], np.int64)
        asserted = np.array([1, 2, 3], np.int64)

        tracemalloc.start()
        try:
            sc._add_last_asserted(site, asserted)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert sc.last_asserted == {("cb", 5): 4, ("cb", 999_999): 2}
        assert peak < 1_000_000  # dense: 8 MB


# ── cmplog pool eviction ────────────────────────────────────────────────

from fuzzer_tool.core.cmplog import _drop_indices  # noqa: E402

POOL10 = [f"t{i}".encode() for i in range(10)]


@pytest.mark.parametrize(
    "victims",
    [{0}, {0, 1, 2}, {0, 2, 5}, {9}, set(range(10)), {3, 9}],
)
def test_drop_indices_matches_comprehension(victims):
    want = [t for i, t in enumerate(POOL10) if i not in victims]
    assert _drop_indices(list(POOL10), victims) == want


def test_regression_drop_does_not_walk_tail():
    """Falsification: the comprehension iterated all 10k tokens per drain."""

    class NoIter(list):
        def __iter__(self):
            raise AssertionError("walked the whole pool")

    pool = NoIter(POOL10)
    assert _drop_indices(pool, {0, 1}) == POOL10[2:]


# ── skip-det bit counting ───────────────────────────────────────────────

from fuzzer_tool.core.skipdet import SkipDetector  # noqa: E402


def _old_count(virgin, trace, map_size):
    n = 0
    for i in range(min(len(trace) * 8, map_size)):
        if i >> 3 < len(trace) and (trace[i >> 3] >> (i & 7)) & 1 and not virgin[i]:
            n += 1
    return n


def _old_mark(virgin, trace, map_size):
    for i in range(min(len(trace) * 8, map_size)):
        if i >> 3 < len(trace) and (trace[i >> 3] >> (i & 7)) & 1:
            virgin[i] = 1


TRACES = [
    bytearray(),
    bytearray(b"\x01"),
    bytearray(b"\xff\x00\xa5"),
    bytearray((i * 29) & 0xFF for i in range(200)),  # longer than map/8 below
]


class TestSkipDetBits:
    @pytest.mark.parametrize("map_size", [13, 64, 1024])
    def test_regression_count_and_mark_match_loop(self, map_size):
        sd = SkipDetector(map_size)
        virgin = bytearray(map_size)
        for trace in TRACES:
            assert sd._count_new_bits(trace) == _old_count(virgin, trace, map_size)
            sd._mark_det_bits(trace)
            _old_mark(virgin, trace, map_size)
            assert sd.virgin_det_bits == virgin
            assert isinstance(sd.virgin_det_bits, bytearray)

    def test_regression_no_per_bit_indexing(self):
        """Falsification: the loops indexed virgin_det_bits once per bit (524k)."""

        class NoIndex(bytearray):
            def __getitem__(self, i):
                raise AssertionError("per-bit index")

            def __setitem__(self, i, v):
                raise AssertionError("per-bit store")

        sd = SkipDetector(64)
        sd.virgin_det_bits = NoIndex(64)
        trace = bytearray(b"\x0f" * 8)
        assert sd._count_new_bits(trace) == 32
        sd._mark_det_bits(trace)
        assert sd._count_new_bits(trace) == 0
