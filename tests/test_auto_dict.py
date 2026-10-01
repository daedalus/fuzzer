"""Auto-dictionary: atomic tokens harvested from byteflip trace hashes.

AFL's auto extras (lcamtuf, "making up grammar with a dictionary in hand",
2015): a run of bytes where flipping any one yields the same path, distinct
from its neighbours' paths, is an atomic check -- ``memcmp(p, "IHDR", 4)``.
The whole run becomes a dictionary token.
"""

from array import array
from pathlib import Path

import pytest

from fuzzer_tool.core.auto_dict import (
    MAX_AUTO_TOKEN,
    MIN_AUTO_TOKEN,
    harvest_tokens,
)
from fuzzer_tool.core.rand_pool import RandPool

PNG_HEAD = bytes.fromhex("89504e470d0a1a0a0000000d49484452000000200000002002030000000e1492")
IHDR = slice(12, 16)

H_MAGIC = 0xAAAA
H_IHDR = 0xBBBB


def _hashes(n: int, runs: dict[tuple[int, int], int]) -> array:
    """Zero (unchanged/unmeasured) everywhere but the given [lo, hi) runs."""
    h = array("Q", bytes(8 * n))
    for (lo, hi), value in runs.items():
        for i in range(lo, hi):
            h[i] = value
    return h


def _reference(data: bytes, hashes: array) -> list[bytes]:
    """Scalar oracle: the walk AFL's maybe_add_auto feeds, written plainly."""
    out = []
    n = min(len(data), len(hashes))
    i = 0
    while i < n:
        j = i
        while j + 1 < n and hashes[j + 1] == hashes[i]:
            j += 1
        tok = data[i : j + 1]
        size = j + 1 - i
        if hashes[i] and MIN_AUTO_TOKEN <= size <= MAX_AUTO_TOKEN and len(set(tok)) > 1:
            out.append(tok)
        i = j + 1
    return out


class TestHarvest:
    def test_png_magic_and_chunk_type(self):
        h = _hashes(len(PNG_HEAD), {(0, 8): H_MAGIC, (12, 16): H_IHDR})
        assert harvest_tokens(PNG_HEAD, h) == [PNG_HEAD[0:8], PNG_HEAD[IHDR]]

    def test_adjacent_runs_split_on_hash_change(self):
        data = b"ABCDEFGH"
        h = _hashes(8, {(0, 4): 1, (4, 8): 2})
        assert harvest_tokens(data, h) == [b"ABCD", b"EFGH"]

    def test_run_at_buffer_end(self):
        data = b"xxxxxIHDR"
        h = _hashes(len(data), {(5, 9): H_IHDR})
        assert harvest_tokens(data, h) == [b"IHDR"]

    def test_matches_scalar_reference(self):
        rng = RandPool(seed=7)
        for _ in range(64):
            n = rng.randint(0, 96)
            data = bytes(rng.randint(0, 255) for _ in range(n))
            h = array("Q", (rng.randint(0, 3) for _ in range(n)))
            assert harvest_tokens(data, h) == _reference(data, h)


class TestFalsification:
    """Inputs that look token-ish but must not produce a token."""

    def test_unchanged_trace_is_never_a_token(self):
        assert harvest_tokens(b"ABCDEFGH", _hashes(8, {})) == []

    def test_short_run_rejected(self):
        h = _hashes(8, {(2, 2 + MIN_AUTO_TOKEN - 1): 9})
        assert harvest_tokens(b"ABCDEFGH", h) == []

    def test_long_run_rejected(self):
        n = MAX_AUTO_TOKEN + 1
        data = bytes(range(n))
        assert harvest_tokens(data, _hashes(n, {(0, n): 9})) == []

    def test_uniform_bytes_rejected(self):
        # Padding / zero fill gated by one length check is not a token.
        assert harvest_tokens(b"\x00" * 8, _hashes(8, {(0, 8): 9})) == []


class TestAdversarial:
    def test_empty(self):
        assert harvest_tokens(b"", array("Q")) == []

    def test_hashes_shorter_than_data(self):
        # A truncated map must not index past its end.
        data = b"ABCDEFGH"
        h = _hashes(5, {(0, 5): 3})
        assert harvest_tokens(data, h) == [b"ABCDE"]

    def test_max_u64_hash(self):
        h = _hashes(4, {(0, 4): 2**64 - 1})
        assert harvest_tokens(b"WXYZ", h) == [b"WXYZ"]


# ── engine wiring ────────────────────────────────────────────────────────

TARGET = str(Path(__file__).resolve().parent.parent / "targets" / "test_target")


@pytest.fixture
def det_fuzzer(tmp_path):
    from fuzzer_tool.services.fuzzer import Fuzzer

    def build(seed: bytes):
        corpus = tmp_path / "corpus"
        crashes = tmp_path / "crashes"
        (corpus / "seeds").mkdir(parents=True)
        crashes.mkdir()
        (corpus / "seeds" / "seed1").write_bytes(seed)
        f = Fuzzer(
            target=TARGET,
            corpus_dir=str(corpus),
            crashes_dir=str(crashes),
            max_len=4096,
            deterministic=True,
        )
        key = f._seed_key(seed)
        f._favored = {key}
        f._edge_tracker.seed_edges[key] = {1, 2, 3}
        return f

    return build


def _drain(f, data: bytes, hash_of) -> None:
    """Run the deterministic queue, answering byteflips with hash_of(pos)."""
    ops = f._operators
    while ops.maybe_deterministic_mutation(data) is not None:
        pending = ops._det_pending
        if pending is None:
            continue
        h = hash_of(pending[1])
        ops.note_deterministic_result(h != 0, h)


class TestEngine:
    def test_drained_map_feeds_dictionary(self, det_fuzzer):
        data = PNG_HEAD
        f = det_fuzzer(data)
        f.dictionary = []
        _drain(f, data, lambda i: H_IHDR if IHDR.start <= i < IHDR.stop else 0)
        assert PNG_HEAD[IHDR] in f.dictionary

    def test_existing_token_not_duplicated(self, det_fuzzer):
        data = PNG_HEAD
        f = det_fuzzer(data)
        f.dictionary = [PNG_HEAD[IHDR]]
        _drain(f, data, lambda i: H_IHDR if IHDR.start <= i < IHDR.stop else 0)
        assert f.dictionary.count(PNG_HEAD[IHDR]) == 1

    def test_learned_tokens_are_capped(self, det_fuzzer):
        from fuzzer_tool.services import operators

        data = bytes(range(64))
        f = det_fuzzer(data)
        f.dictionary = []
        f._operators._auto_tokens = operators.AUTO_DICT_CAP - 1
        # Every 4-byte block is its own token: 16 candidates, room for 1.
        _drain(f, data, lambda i: 1 + i // 4)
        assert len(f.dictionary) == 1

    def test_note_effector_forwards_path_hash(self, det_fuzzer):
        data = PNG_HEAD
        f = det_fuzzer(data)
        key = f._seed_key(data)
        ops = f._operators
        for _ in range(8 * len(data) + 1):  # past bitflip, first byteflip pending
            ops.maybe_deterministic_mutation(data)
        idx = ops._det_pending[1]
        f._edge_tracker.seed_path_hash[key] = 1

        class _Shm:
            def read_path_hash(self):
                return 0xC0FFEE

        f.shm_cov = _Shm()
        f._note_det_effector()
        assert ops._det_eff[key].hashes[idx] == 0xC0FFEE
