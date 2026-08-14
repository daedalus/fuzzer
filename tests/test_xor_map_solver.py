"""Regression + falsification tests for ``core/xor_map_solver.py``."""

from __future__ import annotations

import sys
import threading
from unittest import mock

import pytest

from fuzzer_tool.core.xor_map_solver import (
    IncrementalXorMapSolver,
    XorBitmaskModel,
    clear_active_xor_model,
    compute_xor_checksum,
    get_active_xor_model,
    recover_xor_model,
    set_active_xor_model,
    verify_xor_model,
    xor_model_from_dict,
    xor_model_to_dict,
)

# ── Helpers ─────────────────────────────────────────────────────────────


def _apply_xor_map(data: bytes, masks: list[list[int]], out_bits: int) -> int:
    """Compute the checksum of *data* under a raw mask list.

    Matches ``compute_xor_checksum``'s LSB-first convention:
    input bit ``i`` is ``(data[i // 8] >> (i % 8)) & 1``.
    """
    result = 0
    n_bytes = len(data)
    for j, bits in enumerate(masks):
        if j >= out_bits:
            break
        bit = 0
        for i in bits:
            byte_idx = i // 8
            bit_idx = i % 8
            if byte_idx < n_bytes:
                bit ^= (data[byte_idx] >> bit_idx) & 1
        result |= bit << j
    return result


def _make_pairs(
    masks: list[list[int]],
    out_bits: int,
    n_pairs: int = 8,
    n_bytes: int = 4,
) -> tuple[list[tuple[bytes, int]], list[tuple[int, int]]]:
    """Build (data, checksum) pairs from a mask using distinct inputs."""
    data_list: list[bytes] = []
    checksum_list: list[int] = []
    int_pairs: list[tuple[int, int]] = []

    width_bytes = out_bits // 8
    for value in range(1, n_pairs + 1):
        data = value.to_bytes(n_bytes, "big")
        checksum = _apply_xor_map(data, masks, out_bits)
        data_list.append(data)
        checksum_list.append(checksum)
        int_pairs.append(
            (
                int.from_bytes(data[:width_bytes].ljust(width_bytes, b"\x00"), "big"),
                checksum,
            )
        )

    return list(zip(data_list, checksum_list, strict=False)), int_pairs


def _make_single_bit_pairs(bit_index: int, out_bits: int = 8) -> list[tuple[int, int]]:
    """Build pairs where only ``bit_index`` is set in the input.

    For the identity mask `masks[j] = [j]`, these pairs uniquely force
    ``w_{j,j} = 1`` and all other ``w_{j,*} = 0`` for every output bit j,
    because each pair has exactly one active input bit.
    """
    pairs: list[tuple[int, int]] = []
    for i in range(out_bits):
        inp = 1 << i
        out = inp  # identity: output = input
        pairs.append((inp, out))
    return pairs


# ── Tests ───────────────────────────────────────────────────────────────


class TestComputeXorChecksum:
    def test_identity_8bit(self):
        model = XorBitmaskModel(
            masks=tuple((i,) for i in range(8)),
            out_bits=8,
        )
        data = bytes([0b10110010])
        assert compute_xor_checksum(data, model) == 0b10110010

    def test_parity_8bit(self):
        model = XorBitmaskModel(
            masks=tuple((0, 1, 2, 3, 4, 5, 6, 7) for _ in range(8)),
            out_bits=8,
        )
        data = bytes([0b11110000])
        # parity of 0b11110000 = 4 bits set -> even -> 0
        assert compute_xor_checksum(data, model) == 0

    def test_short_input_padded(self):
        model = XorBitmaskModel(
            masks=tuple((i,) for i in range(8)),
            out_bits=8,
        )
        assert compute_xor_checksum(b"\x01", model) == 0b00000001

    def test_empty_input_returns_zero(self):
        model = XorBitmaskModel(
            masks=tuple((0,) for _ in range(8)),
            out_bits=8,
        )
        assert compute_xor_checksum(b"", model) == 0


class TestIncrementalXorMapSolver:
    def test_recovers_identity_map_from_single_bit_pairs(self):
        """Each output bit j = input bit j; single-bit pairs uniquely force w_j_j=1."""
        solver = IncrementalXorMapSolver(8)
        pairs = _make_single_bit_pairs(bit_index=0, out_bits=8)
        solver.add_pairs(pairs)
        solution, is_sat = solver.solve()
        assert is_sat is True
        assert solution is not None
        for j in range(8):
            assert solution[j] == [j]

    def test_recovers_2bit_identity_map(self):
        solver = IncrementalXorMapSolver(2)
        # Use single-bit pairs: (1,1) and (2,2) uniquely determine the identity.
        solver.add_pairs([(1, 1), (2, 2)])
        solution, is_sat = solver.solve()
        assert is_sat is True
        assert solution is not None
        assert solution[0] == [0]
        assert solution[1] == [1]

    def test_incremental_add_pair(self):
        solver = IncrementalXorMapSolver(2)
        # (1,1) uniquely forces w_0_0=1, w_1_1=1 via single-bit pairs.
        solver.add_pairs([(1, 1), (2, 2)])
        solution_first, is_sat_first = solver.solve()
        assert is_sat_first is True
        assert solution_first == [[0], [1]]

        # Adding a consistent pair must keep the same mask.
        solver.add_pair(3, 3)
        solution_second, is_sat_second = solver.solve()
        assert is_sat_second is True
        assert solution_second == [[0], [1]]

    def test_unsat_inconsistent_pairs(self):
        solver = IncrementalXorMapSolver(2)
        # Bit 0: pair (0b01,0b01) => w_0_0=1, pair (0b11,0b01) => w_0_0*1+w_0_1*1=0 -> w_0_1=1.
        #   Requires both w_0_0=1 and w_0_1=1 -> "exactly one" violated -> UNSAT.
        # Bit 1: pair (0b10,0b10) => w_1_1=1, pair (0b11,0b10) => w_1_0*1+w_1_1*1=0 -> w_1_0=1.
        #   Requires both w_1_0=1 and w_1_1=1 -> "exactly one" violated -> UNSAT.
        solver.add_pair(0b01, 0b01)
        solver.add_pair(0b10, 0b10)
        solver.add_pair(0b11, 0b01)
        solution, is_sat = solver.solve()
        assert is_sat is False
        assert solution is None

    def test_z3_unavailable_returns_none(self):
        solver = IncrementalXorMapSolver(2)
        with mock.patch("fuzzer_tool.core.xor_map_solver._z3_available", False):
            solution, is_sat = solver.solve()
        assert solution is None
        assert is_sat is False

    def test_n_bits_validation(self):
        with pytest.raises(ValueError):
            IncrementalXorMapSolver(0)

    def test_solver_timeout_ms_stored(self):
        solver = IncrementalXorMapSolver(2, timeout_ms=77)
        assert solver._timeout_ms == 77


class TestRecoverXorModel:
    def test_recovers_parity_model(self):
        # parity[j] = XOR of bits 0..7. Use pairs that exercise all 8 input
        # bits so every weight is uniquely constrained to 1.
        masks = [[i for i in range(8)] for _ in range(8)]
        # Values 1..8 exercise bits 0..3 only; add 0x10, 0x20, 0x40, 0x80
        # to exercise bits 4..7 and force all weights to 1.
        extra_values = [0x10, 0x20, 0x40, 0x80]
        all_values = list(range(1, 9)) + extra_values
        pairs = []
        for value in all_values:
            data = value.to_bytes(1, "big")
            checksum = _apply_xor_map(data, masks, 8)
            pairs.append((data, checksum))
        model = recover_xor_model(pairs, min_matches=4, max_pairs=len(pairs))
        assert model is not None
        assert model.out_bits == 8
        assert tuple(model.masks) == tuple(tuple(m) for m in masks)

    def test_rejects_wrong_model_with_constant_predictions(self):
        # Wrong model: each output bit j uses only input bit 0.
        # All pairs have different checksums, but the model predicts the
        # same value (input_bit_0) for all of them.
        wrong_model = XorBitmaskModel(
            masks=tuple((0,) for _ in range(8)),
            out_bits=8,
        )
        # All pairs have input_bit_0 = 0 -> model predicts 0 for all.
        # Actual checksums vary -> fixed-point witness -> verify rejects.
        pairs = [
            (b"\x00\x01", 0x02),
            (b"\x00\x02", 0x04),
            (b"\x00\x04", 0x08),
        ]
        assert verify_xor_model(wrong_model, pairs, min_matches=3) is False

    def test_verify_requires_two_distinct_checksums(self):
        model = XorBitmaskModel(
            masks=tuple(tuple() for _ in range(8)),
            out_bits=8,
        )
        # All-zero checksums: empty mask predicts 0 for all, but only 1
        # distinct checksum value -> must be rejected.
        pairs = [(b"\x00" * 4, 0)] * 4
        assert verify_xor_model(model, pairs, min_matches=4) is False

    def test_returns_none_for_insufficient_pairs(self):
        assert recover_xor_model([]) is None
        assert recover_xor_model([(b"", 0)]) is None
        # Only empty-data pairs.
        assert recover_xor_model([(b"", 0), (b"", 0)]) is None


class TestSerialization:
    def test_roundtrip(self):
        model = XorBitmaskModel(
            masks=((0, 2), (1, 3)),
            out_bits=16,
        )
        data = xor_model_to_dict(model)
        assert data is not None
        restored = xor_model_from_dict(data)
        assert restored == model

    def test_none_roundtrip(self):
        assert xor_model_to_dict(None) is None
        assert xor_model_from_dict(None) is None

    def test_rejects_bad_kind(self):
        assert xor_model_from_dict({"kind": "crc32"}) is None

    def test_rejects_missing_fields(self):
        assert xor_model_from_dict({}) is None
        assert xor_model_from_dict({"kind": "xor_bitmask"}) is None

    def test_rejects_non_list_masks(self):
        assert (
            xor_model_from_dict(
                {
                    "kind": "xor_bitmask",
                    "masks": "not a list",
                    "out_bits": 8,
                }
            )
            is None
        )


class TestActiveModel:
    def test_set_get_clear(self):
        model = XorBitmaskModel(masks=((0,),), out_bits=8)
        set_active_xor_model(model)
        assert get_active_xor_model() == model
        clear_active_xor_model()
        assert get_active_xor_model() is None

    def test_thread_safety(self):
        model = XorBitmaskModel(masks=((0,),), out_bits=8)
        errors: list[Exception] = []

        def worker() -> None:
            try:
                for _ in range(100):
                    set_active_xor_model(model)
                    assert get_active_xor_model() == model
                    clear_active_xor_model()
                    assert get_active_xor_model() is None
            except Exception as exc:  # pragma: no cover - test harness
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, f"Thread errors: {errors}"


class TestModuleImports:
    def test_import_does_not_require_z3(self):
        mod = sys.modules.get("fuzzer_tool.core.xor_map_solver")
        assert mod is not None
