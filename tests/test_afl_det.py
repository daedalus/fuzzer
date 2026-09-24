"""T1-1: AFL deterministic stages as an arbitrated seed-factory arm.

``core/mutations/afl_det.py`` addresses AFL's deterministic sweep (bitflip
1/4/8/32 -> arith 8/16/32 -> interest 8/16/32) by index, so the operator
walks a per-parent cursor one variant per call. Coverage-new outputs become
corpus seeds through the normal admission path.
"""

import struct

import pytest
import xxhash

from fuzzer_tool.core.mutations import afl_det
from fuzzer_tool.core.mutations.generic import INTERESTING_8, INTERESTING_16, INTERESTING_32
from fuzzer_tool.core.operator_registry import REGISTRY
from fuzzer_tool.services.operators import OperatorEngine

from .support.operator_env import make_minimal_fuzzer

SEED = bytes([0x10, 0x20, 0x30, 0x40, 0x50])


def _sweep(data):
    out = []
    k = 0
    while (v := afl_det.det_variant(data, k)) is not None:
        out.append(v)
        k += 1
    return out


def _stage_start(n, name):
    names = [s.name for s in afl_det.STAGES]
    return sum(afl_det.stage_sizes(n)[: names.index(name)])


# --------------------------------------------------------------------------
# index addressing
# --------------------------------------------------------------------------


def test_stage_order_matches_afl():
    assert [s.name for s in afl_det.STAGES] == [
        "flip1",
        "flip4",
        "flip8",
        "flip32",
        "arith8",
        "arith16",
        "arith32",
        "interest8",
        "interest16",
        "interest32",
    ]


def test_stage_sizes_formula():
    n = len(SEED)
    d = 2 * afl_det.ARITH_MAX
    expected = [
        8 * n,
        8 * n - 3,
        n,
        n - 3,
        n * d,
        (n - 1) * d * 2,
        (n - 3) * d * 2,
        n * len(INTERESTING_8),
        (n - 1) * len(INTERESTING_16) * 2,
        (n - 3) * len(INTERESTING_32) * 2,
    ]
    assert afl_det.stage_sizes(n) == expected
    assert afl_det.det_total(n) == sum(expected)


def test_total_matches_enumeration():
    assert len(_sweep(SEED)) == afl_det.det_total(len(SEED))


def test_flip1_is_every_single_bit_msb_first():
    outs = _sweep(SEED)[: 8 * len(SEED)]
    assert outs[0] == bytes([SEED[0] ^ 0x80]) + SEED[1:]
    assert len(set(outs)) == len(outs)
    for v in outs:
        diff = sum(bin(a ^ b).count("1") for a, b in zip(v, SEED, strict=True))
        assert diff == 1


def test_arith8_plus_then_minus():
    k = _stage_start(len(SEED), "arith8")
    assert afl_det.det_variant(SEED, k)[0] == SEED[0] + 1
    assert afl_det.det_variant(SEED, k + 1)[0] == SEED[0] - 1


def test_arith16_big_endian_half():
    k = _stage_start(len(SEED), "arith16")
    le = afl_det.det_variant(SEED, k)
    be = afl_det.det_variant(SEED, k + 2 * afl_det.ARITH_MAX)
    base_le = struct.unpack_from("<H", SEED, 0)[0]
    base_be = struct.unpack_from(">H", SEED, 0)[0]
    assert struct.unpack_from("<H", le, 0)[0] == (base_le + 1) & 0xFFFF
    assert struct.unpack_from(">H", be, 0)[0] == (base_be + 1) & 0xFFFF


def test_interest32_writes_listed_values():
    k = _stage_start(len(SEED), "interest32")
    v = afl_det.det_variant(SEED, k)
    assert v[:4] == struct.pack("<I", INTERESTING_32[0] & 0xFFFFFFFF)
    assert v[4:] == SEED[4:]


def test_falsification_differs_from_existing_stream_widths():
    """The in-loop det stage has no flip4/flip32/16-/32-bit passes; this must."""
    sizes = dict(zip([s.name for s in afl_det.STAGES], afl_det.stage_sizes(8), strict=True))
    for name in ("flip4", "flip32", "arith16", "arith32", "interest16", "interest32"):
        assert sizes[name] > 0


def test_adversarial_short_and_empty_inputs():
    assert afl_det.det_total(0) == 0
    assert afl_det.det_variant(b"", 0) is None
    assert afl_det.det_variant(b"\x00", -1) is None
    # Wide stages vanish below their width instead of going negative.
    assert all(s >= 0 for s in afl_det.stage_sizes(1))
    assert len(_sweep(b"\x00")) == afl_det.det_total(1)
    assert afl_det.det_variant(b"\x00", afl_det.det_total(1)) is None


def test_lengths_preserved():
    assert {len(v) for v in _sweep(SEED)} == {len(SEED)}


# --------------------------------------------------------------------------
# operator wiring
# --------------------------------------------------------------------------


class TestOperator:
    def setup_method(self):
        self.f = make_minimal_fuzzer(0x5EED)
        self.f.op_afl_det = True
        self.engine = OperatorEngine(self.f)

    def test_registered_and_dispatchable(self):
        assert REGISTRY.category_of("afl_det") == REGISTRY.category_of("skipdet_probe")
        assert "afl_det" in self.engine.build_dispatch()

    def test_gated_on_flag(self):
        assert "afl_det" in REGISTRY.available(self.f, SEED)
        self.f.op_afl_det = False
        assert "afl_det" not in REGISTRY.available(self.f, SEED)

    def test_cursor_walks_per_parent(self):
        first = self.engine._op_afl_det(bytearray(SEED), 0, SEED)
        second = self.engine._op_afl_det(bytearray(SEED), 0, SEED)
        other = self.engine._op_afl_det(bytearray(b"abcd"), 0, b"abcd")
        assert bytes(first) == afl_det.det_variant(SEED, 0)
        assert bytes(second) == afl_det.det_variant(SEED, 1)
        assert bytes(other) == afl_det.det_variant(b"abcd", 0)

    def test_skips_no_op_variants(self):
        # interest8 value 0 on a zero byte is a no-op; the cursor must pass it.
        data = b"\x00"
        k = _stage_start(1, "interest8") + INTERESTING_8.index(0)
        self.engine._afl_det_cursor[xxhash.xxh3_64_intdigest(data)] = k
        out = self.engine._op_afl_det(bytearray(data), 0, data)
        assert bytes(out) != data
        assert self.engine._afl_det_cursor[xxhash.xxh3_64_intdigest(data)] == k + 2

    def test_declines_when_sweep_done(self):
        data = b"\x00"
        self.engine._afl_det_cursor[xxhash.xxh3_64_intdigest(data)] = afl_det.det_total(1)
        out = self.engine._op_afl_det(bytearray(data), 0, data)
        assert bytes(out) == data
        assert self.f._op_declines.get("afl_det") == 1

    def test_cursor_store_is_bounded(self, monkeypatch):
        monkeypatch.setattr("fuzzer_tool.services.operators._AFL_DET_CURSOR_CAP", 3)
        for i in range(10):
            d = bytes([i, i])
            self.engine._op_afl_det(bytearray(d), 0, d)
        assert len(self.engine._afl_det_cursor) == 3


def test_cli_flag_wired():
    import ast
    import inspect

    from fuzzer_tool.cli import commands
    from tests.test_regression_cli_fuzzer_kwargs import _fuzz_parser_dests

    tree = ast.parse(inspect.getsource(commands.cmd_fuzz))
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "Fuzzer"
    ]
    assert all("op_afl_det" in {k.arg for k in c.keywords} for c in calls)
    assert "op_afl_det" in _fuzz_parser_dests(ast.parse(inspect.getsource(commands)))
    assert "op_afl_det" in commands._HAIL_MARY_FLAGS


@pytest.mark.parametrize("n", [1, 2, 3, 4, 7])
def test_every_index_below_total_yields(n):
    data = bytes(range(n))
    total = afl_det.det_total(n)
    assert all(afl_det.det_variant(data, k) is not None for k in range(total))
