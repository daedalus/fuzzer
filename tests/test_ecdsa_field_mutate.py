"""Tests for core/mutations/secp_fields.ecdsa_field_mutate (secp256k1_read.c)."""

import pytest

from fuzzer_tool.core.mutations.secp_fields import (
    MODES,
    PAYLOAD_LENS,
    PREFIXES,
    SCALAR_EDGES,
    ecdsa_field_mutate,
)
from fuzzer_tool.core.rand_pool import RandPool
from tests.support.scripted_rng import ScriptedRng

# Independent copies of the curve constants (SEC 2, secp256k1).
_N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
_P = 2**256 - 2**32 - 977
_HEAD = b"\x10\x00"  # mode: compact ECDSA, recid byte
_MAX = 4096


def _mode(name):
    return [m.__name__ for m in MODES].index(name)


def _be(v):
    return v.to_bytes(32, "big")


def test_scalar_edge_writes_order_into_slot():
    data = _HEAD + bytes(64)
    rng = ScriptedRng(choice_idxs=[_mode("scalar_edge"), 1, SCALAR_EDGES.index(_N)])
    assert ecdsa_field_mutate(data, rng, _MAX) == _HEAD + bytes(32) + _be(_N)


def test_scalar_edges_cover_high_s_boundary_and_field_prime():
    assert (_N - 1) // 2 in SCALAR_EDGES
    assert (_N + 1) // 2 in SCALAR_EDGES
    assert _P in SCALAR_EDGES


def test_prefix_edge():
    data = _HEAD + b"\x02" + bytes(32)
    rng = ScriptedRng(choice_idxs=[_mode("prefix_edge"), PREFIXES.index(0x06)])
    assert ecdsa_field_mutate(data, rng, _MAX) == _HEAD + b"\x06" + bytes(32)


def test_s_negate_flips_low_to_high_s():
    s = 5
    data = _HEAD + _be(7) + _be(s)
    out = ecdsa_field_mutate(data, ScriptedRng(choice_idxs=[_mode("s_negate")]), _MAX)
    assert out == _HEAD + _be(7) + _be(_N - s)


def test_payload_resize_pads_and_truncates():
    data = _HEAD + b"\x02\x01"
    rng = ScriptedRng(choice_idxs=[_mode("payload_resize"), PAYLOAD_LENS.index(33)])
    assert ecdsa_field_mutate(data, rng, _MAX) == _HEAD + b"\x02\x01" + bytes(31)

    rng = ScriptedRng(choice_idxs=[_mode("payload_resize"), PAYLOAD_LENS.index(0)])
    assert ecdsa_field_mutate(data, rng, _MAX) == _HEAD


# Falsification: too-short payloads decline instead of writing past the end.
def test_regression_short_payload_declines():
    assert ecdsa_field_mutate(_HEAD + bytes(31), ScriptedRng(choice_idxs=[0]), _MAX) is None
    assert ecdsa_field_mutate(_HEAD + bytes(63), ScriptedRng(choice_idxs=[2]), _MAX) is None
    assert ecdsa_field_mutate(b"\x10", ScriptedRng(), _MAX) is None


def test_s_negate_of_zero_is_noop_and_declines():
    data = _HEAD + bytes(64)
    assert ecdsa_field_mutate(data, ScriptedRng(choice_idxs=[_mode("s_negate")]), _MAX) is None


def test_resize_over_budget_declines():
    rng = ScriptedRng(choice_idxs=[_mode("payload_resize"), PAYLOAD_LENS.index(97)])
    assert ecdsa_field_mutate(_HEAD, rng, 64) is None


@pytest.mark.parametrize("max_len", [1, 2, 8, 66, 4096])
def test_adversarial_never_exceeds_max_len(max_len):
    blobs = [b"\xff" * 200, bytes(200), bytes(range(256)), _HEAD + _be(_N) * 3]
    for seed in range(40):
        rng = RandPool(seed=seed)
        for blob in blobs:
            out = ecdsa_field_mutate(blob[:max_len], rng, max_len)
            assert out is None or len(out) <= max_len
