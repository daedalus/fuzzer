"""Regression: ``_id_shift`` was cubic in the input.

It counted, for every 4-byte window of every message, the messages holding
that window (``sum(substr in m for m in msgs)``). A 64 KiB GStreamer seed that
splits into 15.8k messages took 84 s per call; ``seq_mutate`` was 83% of wall
under ``--elo all`` (20 eps). The rewrite counts 4-gram document frequency
once and must draw the same candidate, so outputs stay byte-identical.
"""

from __future__ import annotations

import time

import pytest

from fuzzer_tool.core.mutations.seq_mutate import _id_shift
from fuzzer_tool.core.rand_pool import RandPool

SEEDS = range(40)
BIG_MSGS = 5000
BIG_BUDGET_S = 2.0


def _oracle(msgs, rng, _donor):
    """Pre-fix ``_id_shift``, verbatim."""
    if len(msgs) < 2:
        return None
    candidates = []
    for _i, msg in enumerate(msgs):
        for j in range(len(msg) - 3):
            substr = msg[j : j + 4]
            if len(substr) == 4:
                count = sum(1 for m in msgs if substr in m)
                if count >= 2:
                    candidates.append((substr, j, count))
    if not candidates:
        return None
    substr, offset, _ = rng.choice(candidates)
    delta = rng.randint(-255, 255)
    if delta == 0:
        delta = 1
    for i in range(len(msgs)):
        pos = msgs[i].find(substr)
        while pos >= 0:
            old_val = int.from_bytes(msgs[i][pos : pos + 4], "big")
            new_val = (old_val + delta) & 0xFFFFFFFF
            msgs[i] = msgs[i][:pos] + new_val.to_bytes(4, "big") + msgs[i][pos + 4 :]
            pos = msgs[i].find(substr, pos + 4)
    return msgs


def _cases():
    """Message lists: shared ids, short messages, repeats, >256 candidates."""
    gen = RandPool(seed=7)
    many = [b"ID\x00\x01" + gen.randbytes(12) for _ in range(80)]
    return [
        [b"user=AAAA;", b"pass=AAAA;"],
        [b"ab", b"abcd", b"xabcdx", b""],
        [b"\x00\x00\x00\x00\x00"] * 3,
        [gen.randbytes(40) + b"KEEP" for _ in range(6)],
        many,
        [b"ab", b"", b"Q" * 600 + b"ABCD", b"xABCD", b"Q"],  # numpy path, short msgs
    ]


def _run(fn, msgs, seed):
    rng = RandPool(seed=seed)
    out = fn(list(msgs), rng, None)
    return out, rng.randint(0, 2**31)


@pytest.mark.parametrize("case", range(len(_cases())))
def test_oracle_matches_itself(case):
    """Control (Hard Rule 46): the oracle is deterministic per seed."""
    msgs = _cases()[case]
    for s in SEEDS:
        assert _run(_oracle, msgs, s) == _run(_oracle, msgs, s)


@pytest.mark.parametrize("case", range(len(_cases())))
def test_regression_id_shift_speed(case):
    """Same output and same RNG position as the pre-fix code."""
    msgs = _cases()[case]
    for s in SEEDS:
        assert _run(_id_shift, msgs, s) == _run(_oracle, msgs, s)


def test_unshared_window_is_no_candidate():
    """Falsification: a window repeated inside one message only is not shared."""
    assert _id_shift([b"ZZZZZZZZ", b"abcdefgh"], RandPool(seed=1), None) is None


def test_many_messages_finish_in_budget():
    """Adversarial: 5k messages (the 84 s shape, scaled down) stay cheap."""
    gen = RandPool(seed=3)
    msgs = [gen.randbytes(8) + b"SAME" for _ in range(BIG_MSGS)]
    t0 = time.perf_counter()
    out = _id_shift(msgs, RandPool(seed=1), None)
    assert out is not None
    assert time.perf_counter() - t0 < BIG_BUDGET_S
