"""Regression: the PRNG learner stops paying for windows that cannot be a stream.

ffmpeg, 4,000 execs in-process: 200 recovery attempts, 0 recovered, 12.1 s.
The windows were byte-shift chains -- each word the previous one shifted by a
byte plus a new byte (0x1e08359b, 0x08359bc9, 0x359bc9bf, ...): a demuxer's
sync-word accumulator reading input, not a generator. Each cost a ~35 ms
family sweep.

Two cuts, neither able to discard a real stream for good:
  * a byte-shift chain window is refused before any family solve;
  * after a streak of refuted windows, observations are skipped on an
    exponential back-off; a recovery resets it.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from fuzzer_tool.core import lcg_recovery, prng_state_recovery
from fuzzer_tool.core.analyzers.analyzer_prng_state_learner import (
    PRNG_BACKOFF_AFTER,
    PRNG_BACKOFF_MAX,
    PRNGStateLearner,
    _byte_shift_chain,
)
from fuzzer_tool.core.lcg_recovery import LCG_FAMILIES
from fuzzer_tool.core.prng_state_recovery import FAMILIES, output_word, step_state
from fuzzer_tool.core.rand_pool import RandPool

# Observed on ffmpeg_read_9.0.2 (8-byte compare sites, values 32-bit sized).
FFMPEG_BE_CHAIN = [0x1E08359B, 0x08359BC9, 0x359BC9BF, 0x9BC9BFBF, 0xC9BFBF06, 0xBFBF06F7]
FFMPEG_STEP_CHAIN = [0xD200D300, 0x00D300D4, 0xD300D400, 0x00D400D5, 0xD400D500, 0x00D500D6]


def _le_chain(seed: int, n: int) -> list[int]:
    """Little-endian accumulator: each word shifted right a byte, new top byte."""
    out, w = [], seed
    for i in range(n):
        out.append(w)
        w = (w >> 8) | (((i * 37 + 11) & 0xFF) << 24)
    return out


def _stream(spec, state: tuple[int, ...], n: int) -> list[int]:
    out, s = [], state
    for _ in range(n):
        s = step_state(s, spec)
        out.append(output_word(s, spec))
    return out


def _lcg(spec, x: int, n: int) -> list[int]:
    out = []
    for _ in range(n):
        out.append((x >> spec.shift) & ((1 << spec.out_bits) - 1))
        x = (spec.a * x + spec.c) % spec.m
    return out


class _Cond:
    def __init__(self, word: int, width: int, pc: int = 0x1000) -> None:
        self.base = MagicMock(op_a=word.to_bytes(width, "little"), op_b=b"\0" * width, pc=pc)


def _learner(words: list[int], width: int = 4) -> PRNGStateLearner:
    f = MagicMock()
    f._inprocess_runner = MagicMock()
    f._cmplog = MagicMock(last_conds=[_Cond(w, width) for w in words])
    return PRNGStateLearner(f)


def _payload(width: int) -> bytes:
    return b"\0" * width + b"PAYLOAD"


class TestByteShiftChain:
    def test_regression_prng_chain_control(self):
        """Hard Rule 46: the detector agrees with itself on a copy."""
        assert _byte_shift_chain(list(FFMPEG_BE_CHAIN)) == _byte_shift_chain(FFMPEG_BE_CHAIN)

    @pytest.mark.parametrize(
        "window", [FFMPEG_BE_CHAIN, FFMPEG_STEP_CHAIN, _le_chain(0xA1B2C3D4, 8)]
    )
    def test_regression_prng_chain_detects_accumulators(self, window):
        assert _byte_shift_chain(window)

    def test_regression_prng_chain_spares_every_family(self):
        """Falsification: no shipped generator's stream reads as a chain,
        tiny-seed warm-up included."""
        rp = RandPool(seed=29)
        for spec in FAMILIES.values():
            seeds = [tuple(range(1, spec.n_slots + 1))]
            seeds += [
                tuple(rp.randrange(1 << 32) | 0xFFFFF for _ in range(spec.n_slots))
                for _ in range(20)
            ]
            for st in seeds:
                words = _stream(spec, st, 16)
                if len(set(words)) < 2:
                    continue  # degenerate all-zero state: nothing to recover anyway
                assert not _byte_shift_chain(words), (spec.name, st)
        for spec in LCG_FAMILIES.values():
            for _ in range(20):
                assert not _byte_shift_chain(_lcg(spec, rp.randrange(1 << 40) | 1, 16)), spec.name

    def test_regression_prng_chain_window_skips_the_solve(self, monkeypatch):
        calls: list[Any] = []

        def _no_solve(*args, **kwargs):
            calls.append(args)
            raise AssertionError("family solve ran on a byte-shift chain")

        for driver in (prng_state_recovery, lcg_recovery):
            monkeypatch.setattr(driver, "recover_state", _no_solve)
        learner = _learner(FFMPEG_BE_CHAIN, width=8)
        assert learner.observe_execution(_payload(8)) is False
        assert calls == []


class TestRefuteBackoff:
    def _noise_learner(self) -> tuple[PRNGStateLearner, RandPool]:
        rp = RandPool(seed=5)
        return _learner([rp.randrange(1 << 32) for _ in range(16)]), rp

    def _refute(self, learner: PRNGStateLearner, rp: RandPool) -> None:
        learner.f._cmplog.last_conds = [_Cond(rp.randrange(1 << 32), 4) for _ in range(16)]
        learner.observe_execution(b"\0" * 4 + rp.randbytes(8))

    def test_regression_prng_backoff_skips_after_streak(self, monkeypatch):
        learner, rp = self._noise_learner()
        extracted: list[int] = []
        real = learner._extract_by_site
        monkeypatch.setattr(learner, "_extract_by_site", lambda d: extracted.append(1) or real(d))

        for _ in range(PRNG_BACKOFF_AFTER):
            self._refute(learner, rp)
        assert len(extracted) == PRNG_BACKOFF_AFTER

        before = len(extracted)
        for _ in range(learner._backoff):
            self._refute(learner, rp)
        assert len(extracted) == before, "back-off window still extracted"

    def test_regression_prng_backoff_grows_and_caps(self):
        learner, rp = self._noise_learner()
        seen = []
        for _ in range(PRNG_BACKOFF_AFTER + 4 * PRNG_BACKOFF_MAX):
            self._refute(learner, rp)
            seen.append(learner._backoff)
        assert max(seen) == PRNG_BACKOFF_MAX
        grown = [b for b in dict.fromkeys(seen) if b]
        assert grown == sorted(grown), "back-off must only grow during a refute streak"

    def test_regression_prng_backoff_resets_on_recovery(self):
        """Adversarial: a real stream after a long refute streak still recovers."""
        learner, rp = self._noise_learner()
        for _ in range(PRNG_BACKOFF_AFTER + 3 * PRNG_BACKOFF_MAX):
            self._refute(learner, rp)
        assert learner._backoff > 0

        # A full window at the noise site replaces every noise draw in it.
        words = _stream(FAMILIES["xorshift32"], (0xACE12345,), 16)
        learner.f._cmplog.last_conds = [_Cond(w, 4) for w in words]
        recovered = False
        for _ in range(PRNG_BACKOFF_MAX + 1):
            recovered = learner.observe_execution(_payload(4)) or recovered
            if learner.has_state():
                break
        assert learner.has_state()
        assert learner._backoff == 0
