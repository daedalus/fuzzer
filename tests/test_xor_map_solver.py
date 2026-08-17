"""Regression + falsification tests for ``core/xor_map_solver.py``."""

from __future__ import annotations

import random
import subprocess
import sys
import textwrap
import threading
import time
import zlib

import pytest

from fuzzer_tool.core.xor_map_solver import (
    IncrementalXorMapSolver,
    XorBitmaskModel,
    clear_active_xor_model,
    compute_xor_checksum,
    get_active_xor_model,
    invert_xor_model,
    recover_xor_model,
    recover_xor_preimage,
    set_active_xor_model,
    verify_xor_model,
    xor_model_from_dict,
    xor_model_to_dict,
)

# These tests carried @requires_z3 for the reason recorded here: with the
# SAT-backed solver, solve() returned (None, False) on a box without the
# optional 'smt' extra, so the positive tests failed and
# test_unsat_inconsistent_pairs passed for entirely the wrong reason --
# asserting exactly the (None, False) that a missing solver produced
# unconditionally.
#
# The solver is now GF(2) elimination in pure Python and z3 is not on this
# path at all, so the guards are gone and the whole file runs everywhere.
# The vacuous-pass hazard the guard existed for is gone with it: an
# inconsistent system is now rejected by a rank/consistency proof rather
# than by the solver being absent.

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
    """Build (data, checksum) pairs from a mask using distinct inputs.

    ``int_pairs`` packs the window little-endian to match
    ``_extract_fixed_width``: bit ``i`` of the integer must be the bit
    ``_apply_xor_map``/``compute_xor_checksum`` read as
    ``data[i // 8] >> (i % 8)``, which big-endian packing byte-reverses for
    any window wider than one byte.
    """
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
                int.from_bytes(data[:width_bytes].ljust(width_bytes, b"\x00"), "little"),
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

    def test_solver_does_not_import_z3(self):
        """Recovery must work without the optional 'smt' extra.

        The inverse of the old ``test_z3_unavailable_returns_none``, which
        asserted the solver was dead without z3. Runs in a subprocess with
        ``z3`` blocked at import so a module already imported by an earlier
        test cannot mask a re-introduced dependency.
        """
        script = textwrap.dedent(
            """
            import sys

            class Blocker:
                def find_module(self, name, path=None):
                    if name == "z3" or name.startswith("z3."):
                        raise ImportError("z3 blocked for this test")
                    return None

            sys.meta_path.insert(0, Blocker())

            from fuzzer_tool.core.xor_map_solver import recover_xor_model

            # Identity map. The values must span all 8 input bits, or the
            # system is underdetermined and recovery correctly abstains.
            pairs = [(bytes([v]), v) for v in range(1, 256)]
            model = recover_xor_model(pairs, max_pairs=len(pairs))
            assert model is not None, "no model recovered without z3"
            assert model.out_bits == 8
            assert "z3" not in sys.modules, "solver imported z3"
            print("ok")
            """
        )
        proc = subprocess.run([sys.executable, "-c", script], capture_output=True, timeout=120)
        assert proc.returncode == 0, proc.stderr.decode(errors="replace")
        assert b"ok" in proc.stdout

    def test_n_bits_validation(self):
        with pytest.raises(ValueError):
            IncrementalXorMapSolver(0)

    def test_solver_timeout_ms_accepted_but_inert(self):
        """``timeout_ms`` survives as a no-op for backward compatibility.

        Elimination is O(pairs * rank) and always terminates, so there is
        nothing to time out. A caller passing an absurdly small value must
        still get a correct answer rather than a truncated one -- the
        failure mode the old per-bit Z3 timeout had.
        """
        solver = IncrementalXorMapSolver(2, timeout_ms=77)
        assert solver._timeout_ms == 77

        solver = IncrementalXorMapSolver(8, timeout_ms=0)
        solver.add_pairs(_make_single_bit_pairs(bit_index=0, out_bits=8))
        solution, is_sat = solver.solve()
        assert is_sat is True
        assert solution == [[j] for j in range(8)]


class TestGF2Elimination:
    """The properties elimination gives that the SAT path did not."""

    def test_rank_and_determinacy_track_independent_pairs(self):
        solver = IncrementalXorMapSolver(8)
        assert solver.rank == 0
        assert solver.is_determined is False
        for i in range(8):
            solver.add_pair(1 << i, 1 << i)
            assert solver.rank == i + 1
        assert solver.is_determined is True

    def test_dependent_pairs_do_not_raise_rank(self):
        """A pair in the span of earlier ones adds no information."""
        solver = IncrementalXorMapSolver(8)
        solver.add_pair(0b0001, 0b0001)
        solver.add_pair(0b0010, 0b0010)
        assert solver.rank == 2
        solver.add_pair(0b0011, 0b0011)  # = row0 XOR row1
        assert solver.rank == 2
        assert solver.is_determined is False

    def test_dependent_pair_that_contradicts_is_unsat(self):
        """The same dependency with a wrong RHS is a contradiction, not noise."""
        solver = IncrementalXorMapSolver(8)
        solver.add_pair(0b0001, 0b0001)
        solver.add_pair(0b0010, 0b0010)
        solver.add_pair(0b0011, 0b0000)  # should be 0b0011
        solution, is_sat = solver.solve()
        assert is_sat is False
        assert solution is None

    def test_solution_reproduces_every_pair_even_when_underdetermined(self):
        """Free variables are pinned to 0; the fitted pairs still hold.

        This is the property that makes agreement-with-fitting-pairs
        worthless as evidence, and therefore why recover_xor_model gates on
        rank instead.
        """
        rng = random.Random(5)
        pairs = [(v, rng.randrange(256)) for v in (0b0001, 0b0010, 0b0100)]
        solver = IncrementalXorMapSolver(8)
        solver.add_pairs(pairs)
        solution, is_sat = solver.solve()
        assert is_sat is True
        assert solver.is_determined is False
        model = XorBitmaskModel(tuple(tuple(m) for m in solution), 8)
        for inp, out in pairs:
            assert compute_xor_checksum(bytes([inp]), model) == out

    def test_thirty_two_bit_solve_is_fast_and_exact(self):
        """The rung that was abandoned on cost.

        The SAT path needed ~4.7 s *per output bit* here and returned
        (None, False) at the 200 ms budget. Elimination is bounded well
        under a second for the whole 32-bit map; the assertion is loose
        because CI boxes are noisy, and it would still catch a regression
        to anything SAT-shaped by three orders of magnitude.
        """
        rng = random.Random(17)
        masks = [sorted(rng.sample(range(32), 7)) for _ in range(32)]
        bodies = sorted({bytes(rng.randrange(256) for _ in range(4)) for _ in range(120)})
        pairs = [(b, _apply_xor_map(b, masks, 32)) for b in bodies]

        start = time.perf_counter()
        model = recover_xor_model(pairs, max_pairs=len(pairs))
        elapsed = time.perf_counter() - start

        assert model is not None, "no 32-bit model recovered from exactly linear data"
        assert model.out_bits == 32
        assert tuple(model.masks) == tuple(tuple(m) for m in masks), (
            "a full-rank system has one solution; it must be the true map"
        )
        assert elapsed < 1.0, f"32-bit recovery took {elapsed:.3f}s"


class TestRecoverXorModel:
    def test_recovers_sixteen_bit_model(self):
        """The width ladder must actually reach past 8 bits.

        ``_extract_fixed_width`` packed the input window big-endian while
        the solver indexes that integer LSB-first and
        ``compute_xor_checksum`` reads ``data[i // 8] >> (i % 8)``. Those
        agree only for a 1-byte window, so at 16 and 32 bits the solver fit
        a mask set under one convention and ``verify_xor_model`` rejected it
        under the other -- every candidate wider than a byte failed, and
        ``recover_xor_model``'s ``(8, 16, 32)`` ladder was ``(8,)`` in
        practice for every input.

        An 8-bit-only test cannot see this: that is exactly the width where
        the two conventions coincide.
        """
        rng = random.Random(11)
        masks = [sorted(rng.sample(range(16), 3)) for _ in range(16)]
        bodies = sorted({bytes(rng.randrange(256) for _ in range(2)) for _ in range(150)})
        pairs = [(b, _apply_xor_map(b, masks, 16)) for b in bodies]

        model = recover_xor_model(pairs, max_pairs=len(pairs))
        assert model is not None, "no 16-bit model recovered from exactly linear data"
        assert model.out_bits == 16
        assert all(compute_xor_checksum(b, model) == c for b, c in pairs)

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

    def test_underdetermined_width_is_not_accepted(self):
        """The hazard that came with reaching the 32-bit rung.

        A checksum that is *not* a linear map of the window -- here CRC-32,
        which is affine because of its init and final XOR, and Adler-32,
        which is not GF(2) linear at all -- still yields a consistent
        32-bit system when there are fewer independent pairs than unknowns.
        Every solution of that system reproduces all the fitting pairs by
        construction, so verify_xor_model cannot reject it: measured 100/100
        accepted-and-wrong at 24 pairs before the rank gate. Activating such
        a model is worse than having none, because every input the fuzzer
        "repairs" then carries a wrong checksum.
        """
        rng = random.Random(23)
        bodies = sorted({bytes(rng.randrange(256) for _ in range(4)) for _ in range(24)})
        for name, fn in (("crc32", zlib.crc32), ("adler32", zlib.adler32)):
            pairs = [(b, fn(b)) for b in bodies]
            assert recover_xor_model(pairs, max_pairs=len(pairs)) is None, (
                f"accepted a linear model for {name} on an underdetermined system"
            )
            # Same data, gate off: the model comes back, which is what makes
            # the gate load-bearing rather than decorative.
            assert (
                recover_xor_model(pairs, max_pairs=len(pairs), require_determined=False) is not None
            ), f"{name} fixture no longer underdetermined; the test proves nothing"

    def test_accepted_model_predicts_unseen_inputs(self):
        """Acceptance must mean the map generalizes, not that it fits.

        Fits on a full-rank pair set and scores the result on inputs the
        solver never saw.
        """
        rng = random.Random(29)
        masks = [sorted(rng.sample(range(16), 5)) for _ in range(16)]
        seen = sorted({bytes(rng.randrange(256) for _ in range(2)) for _ in range(80)})
        model = recover_xor_model(
            [(b, _apply_xor_map(b, masks, 16)) for b in seen], max_pairs=len(seen)
        )
        assert model is not None and model.out_bits == 16

        held = [
            b
            for b in ({bytes(rng.randrange(256) for _ in range(2)) for _ in range(200)})
            if b not in set(seen)
        ]
        assert len(held) >= 50, "held-out set too small to be meaningful"
        assert all(compute_xor_checksum(b, model) == _apply_xor_map(b, masks, 16) for b in held)

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


class TestInvertXorModel:
    """Wiring tests for the gf2_linalg-backed square-model inverse."""

    def test_square_model_round_trips(self):
        # A tiny fixed-width scramble: out_bit0 = in0^in1, out_bit1 = in1.
        # Square (out_bits == in_bits == 2), so it must be invertible.
        model = XorBitmaskModel(masks=((0, 1), (1,)), out_bits=2)
        inv = invert_xor_model(model)
        assert inv is not None
        for v in range(4):
            checksum = compute_xor_checksum(v.to_bytes(1, "big"), model)
            recovered = recover_xor_preimage(model, checksum)
            assert recovered == v

    def test_non_square_model_returns_none(self):
        # Typical checksum shape: out_bits (8) narrower than the input
        # domain the masks reference (indices up to 31) -- e.g. a CRC-8
        # over a 4-byte buffer. Not invertible in the square sense.
        model = XorBitmaskModel(
            masks=tuple((i, i + 8, i + 16, i + 24) for i in range(8)), out_bits=8
        )
        assert invert_xor_model(model) is None
        assert recover_xor_preimage(model, 0x42) is None

    def test_singular_square_model_returns_none(self):
        # out_bit0 and out_bit1 both equal in0 -- rank-deficient.
        model = XorBitmaskModel(masks=((0,), (0,)), out_bits=2)
        assert invert_xor_model(model) is None
        assert recover_xor_preimage(model, 0b01) is None

    def test_recover_xor_preimage_matches_forward(self):
        # 4-bit invertible scramble built from a known permutation+XOR mix.
        model = XorBitmaskModel(
            masks=((0, 1), (1, 2), (2, 3), (3,)),
            out_bits=4,
        )
        inv = invert_xor_model(model)
        assert inv is not None
        for v in range(16):
            checksum = compute_xor_checksum(v.to_bytes(1, "big")[:1], model)
            recovered = recover_xor_preimage(model, checksum)
            assert recovered is not None
            assert compute_xor_checksum(recovered.to_bytes(1, "big"), model) == checksum
