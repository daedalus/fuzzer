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
