"""Tests for sequence-level mutation (core/mutations/seq_mutate.py)."""

from __future__ import annotations

import struct

import pytest

from fuzzer_tool.core.mutations.seq_mutate import seq_mutate, split_msgs
from fuzzer_tool.core.operator_registry import REGISTRY
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.operators import OperatorEngine
from tests.support.operator_env import make_minimal_fuzzer
from tests.support.scripted_rng import ScriptedRng


def _frames(payloads, fmt=">H", inclusive=False):
    width = struct.calcsize(fmt)
    out = b""
    for p in payloads:
        n = len(p) + (width if inclusive else 0)
        out += struct.pack(fmt, n) + p
    return out


class TestSplit:
    @pytest.mark.parametrize("fmt", [">H", "<H", ">I", "<I"])
    @pytest.mark.parametrize("inclusive", [False, True])
    def test_length_prefixed(self, fmt, inclusive):
        data = _frames([b"abc", b"hello", b"xy"], fmt, inclusive)
        msgs = split_msgs(data)
        assert msgs is not None and b"".join(msgs) == data
        assert len(msgs) == 3

    def test_crlf_delimited(self):
        data = b"GET /a HTTP/1.1\r\nHost: x\r\n\r\nGET /b HTTP/1.1\r\nHost: y\r\n\r\n"
        msgs = split_msgs(data)
        cut = data.index(b"GET /b")
        assert msgs == [data[:cut], data[cut:]]

    def test_newline_delimited_keeps_tail(self):
        msgs = split_msgs(b"USER a\nPASS b\nQUIT")
        assert msgs == [b"USER a\n", b"PASS b\n", b"QUIT"]

    def test_join_is_identity(self):
        data = b"a\nb\nc\nd"
        assert b"".join(split_msgs(data)) == data

    # falsification: unframed data is not a message sequence
    @pytest.mark.parametrize("data", [b"", b"x", b"abcdefghijklmnop", bytes(range(1, 40))])
    def test_unframed_is_none(self, data):
        assert split_msgs(data) is None

    # adversarial
    def test_length_overruns_buffer(self):
        assert split_msgs(struct.pack(">H", 500) + b"abc" * 3) is None

    def test_truncated_second_frame(self):
        assert split_msgs(_frames([b"abc", b"hello"])[:-1]) is None

    def test_all_zero_not_framed(self):
        assert split_msgs(bytes(16)) is None

    def test_single_frame_is_not_a_sequence(self):
        assert split_msgs(_frames([b"abcdef"])) is None

    def test_inclusive_length_below_header(self):
        data = struct.pack(">H", 1) + b"abcdef" * 2
        assert split_msgs(data) is None


class TestSeqMutate:
    DATA = _frames([b"one", b"two22", b"three"])

    def test_output_is_whole_messages(self):
        base = set(split_msgs(self.DATA))
        for seed in range(30):
            out = seq_mutate(self.DATA, RandPool(seed=seed), 4096)
            assert out is not None and out != self.DATA
            msgs = split_msgs(out)
            assert msgs is not None and set(msgs) <= base

    def test_modes_reachable(self):
        sizes = set()
        for seed in range(60):
            out = seq_mutate(self.DATA, RandPool(seed=seed), 4096)
            sizes.add(len(split_msgs(out)) - 3)
        assert sizes == {-1, 0, 1}

    def test_scripted_swap(self):
        data = b"A\nB\nC\n"
        rng = ScriptedRng(randints=[0, 0, 2])
        assert seq_mutate(data, rng, 4096) == b"C\nB\nA\n"

    def test_scripted_drop(self):
        data = b"A\nB\nC\n"
        rng = ScriptedRng(randints=[2, 1])
        assert seq_mutate(data, rng, 4096) == b"A\nC\n"

    def test_scripted_dup(self):
        rng = ScriptedRng(randints=[1, 0, 2])
        assert seq_mutate(b"A\nB\nC\n", rng, 4096) == b"A\nB\nA\nC\n"

    def test_scripted_splice_from_donor(self):
        rng = ScriptedRng(randints=[3, 1, 0])
        assert seq_mutate(b"A\nB\nC\n", rng, 4096, donor=b"X\nY\nZ\n") == b"Y\nA\nB\nC\n"

    def test_splice_skipped_without_donor(self):
        data = b"A\nB\nC\n"
        for seed in range(30):
            out = seq_mutate(data, RandPool(seed=seed), 4096)
            assert out is not None and out != data

    def test_unterminated_tail_never_fuses(self):
        for seed in range(60):
            out = seq_mutate(b"A\nB\nQUIT", RandPool(seed=seed), 4096)
            assert out is not None
            for fused in (b"QUITA", b"QUITB", b"QUITQUIT"):
                assert fused not in out

    def test_declines_unframed(self):
        assert seq_mutate(b"abcdefghij", RandPool(seed=1), 4096) is None

    def test_identical_messages_cannot_change_by_swap(self):
        data = b"A\nA\nA\n"
        for seed in range(30):
            out = seq_mutate(data, RandPool(seed=seed), 4096)
            assert out is not None and out != data

    def test_respects_max_len(self):
        for seed in range(30):
            out = seq_mutate(self.DATA, RandPool(seed=seed), len(self.DATA))
            assert out is None or len(out) <= len(self.DATA)

    def test_drop_never_empties(self):
        data = _frames([b"A", b"B"])
        for seed in range(40):
            out = seq_mutate(data, RandPool(seed=seed), 4096)
            assert out


class TestWiring:
    def test_registered_structural(self):
        assert "seq_mutate" in REGISTRY.names()
        assert REGISTRY.category_of("seq_mutate") == "structural"

    def test_gate_follows_input_only(self):
        f = make_minimal_fuzzer()
        assert "seq_mutate" in REGISTRY.available(f, b"A\nB\nC\n")
        assert "seq_mutate" not in REGISTRY.available(f, b"abcdefghijkl")

    def test_dispatch_mutates(self):
        f = make_minimal_fuzzer()
        f._rng = RandPool(seed=5)
        data = b"A\nB\nC\n"
        out = REGISTRY.dispatch(OperatorEngine(f))["seq_mutate"](bytearray(data), 0, data)
        assert out is not None and out != bytearray(data)

    def test_dispatch_declines(self):
        f = make_minimal_fuzzer()
        f._rng = RandPool(seed=5)
        buf = bytearray(b"abcdefghij")
        OperatorEngine(f)._op_seq_mutate(buf, 0, bytes(buf))
        assert bytes(buf) == b"abcdefghij"
        assert f._op_declines == {"seq_mutate": 1}
