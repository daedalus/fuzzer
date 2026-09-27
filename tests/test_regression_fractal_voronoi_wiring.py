"""Regression: fractal_voronoi ran without sub-operators and ignored rng.

* ``_register()`` built the mutator with no ``cell_ops``, so the registered
  instance always took the XOR fallback; the per-cell operator premise never
  shipped (handover fractal-voronoi-integration.md §1).
* ``mutate()`` never read ``rng``: output was a pure function of the input, so
  every re-pick of a seed re-executed the same mutant (§2).
"""

from __future__ import annotations

import hashlib
import math

import pytest

from fuzzer_tool.core.mutations.fractal_voronoi import FractalVoronoiMutator
from fuzzer_tool.core.operator_registry import REGISTRY
from tests.support.scripted_rng import ScriptedRng

DATA = bytes(range(256)) * 2
SALT_MAX = 0xFFFFFFFF


def _registered() -> FractalVoronoiMutator:
    return next(m for m in REGISTRY.mutators() if m.name == FractalVoronoiMutator.name)


def _oracle(m: FractalVoronoiMutator, data: bytes, salt: int) -> bytes:
    """Salted mutate, rebuilt from the geometry plan and the documented rules."""
    side = max(1, int(math.sqrt(len(data))))
    out = bytearray(data)
    ops = m.cell_ops
    for idx, (root_hash, on_boundary, root, px, py) in enumerate(m._plan(side, len(data))):
        h = root_hash ^ salt
        if ops and (h + idx) % 7 == 0:
            out[idx] = ops[h % len(ops)](bytes([data[idx]]))[0]
        elif not ops and (h + idx) % 5 == 0:
            out[idx] ^= h & 0xFF
        if on_boundary:
            digest = hashlib.sha256(f"boundary:{root}:{px:.6f}:{py:.6f}".encode()).hexdigest()
            bh = int(digest, 16) ^ salt
            if (bh + idx) % 3 == 0:
                out[idx] ^= (bh >> 8) & 0xFF
    return bytes(out)


class TestCellOpsWired:
    def test_regression_registered_instance_has_cell_ops(self):
        assert len(_registered().cell_ops) >= 2

    def test_regression_cells_use_different_ops(self):
        """Distinct roots map to distinct sub-operators on the registered instance."""
        m = _registered()
        side = int(math.sqrt(len(DATA)))
        n_ops = len(m.cell_ops)
        used = {rh % n_ops for rh, *_ in m._plan(side, len(DATA))}
        assert len(used) >= 2

    def test_registered_output_is_not_the_xor_fallback(self):
        """Falsification: pre-fix the registered output equalled the bare mutator's."""
        salt = 0
        wired = _registered().mutate(DATA, ScriptedRng(randints=[salt]))
        bare = FractalVoronoiMutator().mutate(DATA, ScriptedRng(randints=[salt]))
        assert wired != bare
        assert wired == _oracle(_registered(), DATA, salt)

    def test_cell_ops_are_single_byte_bijections(self):
        """Adversarial: each default op maps every byte value to a distinct byte."""
        ops = _registered().cell_ops
        assert ops
        for op in ops:
            images = [op(bytes([b])) for b in range(256)]
            assert all(len(i) == 1 for i in images)
            assert len(set(images)) == 256


class TestRngSalt:
    @pytest.mark.parametrize("salt", [1, 0xDEADBEEF, SALT_MAX])
    def test_regression_salt_changes_output(self, salt):
        m = FractalVoronoiMutator()
        a = m.mutate(DATA, ScriptedRng(randints=[0]))
        b = m.mutate(DATA, ScriptedRng(randints=[salt]))
        assert a != b
        assert b == _oracle(m, DATA, salt)

    def test_control_same_salt_same_output(self):
        m = _registered()
        assert m.mutate(DATA, ScriptedRng(randints=[7])) == m.mutate(
            DATA, ScriptedRng(randints=[7])
        )

    def test_salt_zero_is_the_legacy_output(self):
        m = FractalVoronoiMutator()
        assert m.mutate(DATA, ScriptedRng(randints=[0])) == m.mutate(DATA, None)

    def test_adversarial_draws_exactly_once(self):
        """One salt per call: a second draw would exhaust the script."""
        rng = ScriptedRng(randints=[3])
        _registered().mutate(DATA, rng)
        with pytest.raises(StopIteration):
            rng.randint(0, SALT_MAX)

    def test_adversarial_short_input_draws_nothing(self):
        rng = ScriptedRng(randints=[])
        assert _registered().mutate(b"x" * 15, rng) is None
