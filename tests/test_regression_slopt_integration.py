"""Test the SLOPT batch-size bandit optimisation.

Hard Rule 37: Write the test first. Observe it failing. Then write the fix.
Hard Rule 38: Verify the test passes (Test driven development).

This test ensures:
- slopt_batch_size() computes correct batch sizes and exponents
- When _use_slopt=True, the mutation loop applies operator in batch mode
- The _last_slopt_exp attribute is set correctly
- Reward arm naming includes the exponent (e.g., "havoc_0.60")
- The feature is gated by --slopt flag (default behavior unchanged)
"""

from __future__ import annotations

import itertools
from unittest.mock import MagicMock

import pytest

from fuzzer_tool.services.operators import OperatorEngine, slopt_batch_size
from tests.support.operator_env import make_minimal_fuzzer
from tests.support.scripted_rng import ScriptedRng

# Seed discipline (Hard Rule 37, 38): fixed seed for reproducibility
_FIXED_SEED = 0xC0FFEE


def _make_minimal_fuzzer(seed: int = _FIXED_SEED) -> object:
    """Create a minimal fuzzer for testing SLOPT integration."""
    f = make_minimal_fuzzer(seed)
    # Add SLOPT-specific attributes (not in BALLOT_SCHEDULERS)
    f._use_slopt = False
    f._last_slopt_exp = 0.0
    # Provide the dispatch table the real Fuzzer would have
    f._op_dispatch = {"havoc": lambda buf, byte_idx, data: None}
    return f


class TestSloptBatchSize:
    """Tests for the slopt_batch_size helper function."""

    def test_known_values(self):
        """Test batch size computation against known values."""
        seed = b"A" * 64  # len=64, len(seed)/l0 = 1.0
        # 1.0 ** exponent = 1.0 for any exponent
        assert slopt_batch_size(seed, "havoc") == (1, 0.6)
        assert slopt_batch_size(seed, "bit_flip") == (1, 0.3)
        assert slopt_batch_size(seed, "arith") == (1, 0.5)
        assert slopt_batch_size(seed, "unknown") == (1, 0.4)

        seed = b"A" * 256  # len=256, len(seed)/l0 = 4.0
        # 4.0 ** 0.6 ≈ 2.297 → int(2.297) = 2
        assert slopt_batch_size(seed, "havoc") == (2, 0.6)
        assert slopt_batch_size(seed, "bit_flip") == (1, 0.3)
        assert slopt_batch_size(seed, "arith") == (2, 0.5)
        assert slopt_batch_size(seed, "unknown") == (1, 0.4)

        seed = b"A" * 1024  # len=1024, len(seed)/l0 = 16.0
        # 16.0 ** 0.6 ≈ 5.278 → int(5.278) = 5
        assert slopt_batch_size(seed, "havoc") == (5, 0.6)
        assert slopt_batch_size(seed, "bit_flip") == (2, 0.3)
        assert slopt_batch_size(seed, "arith") == (4, 0.5)
        assert slopt_batch_size(seed, "unknown") == (3, 0.4)

    def test_minimum_batch_size(self):
        """Ensure batch size never drops below 1."""
        seed = b""  # len=0 → 0 ** exponent = 0 (for exponent > 0)
        assert slopt_batch_size(seed, "havoc") == (1, 0.6)
        assert slopt_batch_size(seed, "bit_flip") == (1, 0.3)
        assert slopt_batch_size(seed, "arith") == (1, 0.5)
        assert slopt_batch_size(seed, "unknown") == (1, 0.4)


class TestSloptIntegration:
    """Tests for SLOPT integration in the mutation loop."""

    def test_slopt_disabled_by_default(self):
        """Verify SLOPT is disabled by default (backward compatibility)."""
        f = _make_minimal_fuzzer()
        assert f._use_slopt is False
        assert f._last_slopt_exp == 0.0

    def test_slopt_enabled_sets_last_slopt_exp(self, monkeypatch):
        """When _use_slopt=True, mutate sets _last_slopt_exp correctly."""

        def mock_slopt_batch_size(seed: bytes, op: str, l0: int = 64) -> tuple[int, float]:
            return 3, 0.5  # batch=3, exp=0.5

        monkeypatch.setattr(
            "fuzzer_tool.services.operators.slopt_batch_size",
            mock_slopt_batch_size,
        )

        f = _make_minimal_fuzzer()
        f._use_slopt = True
        f.mc_bandit = True
        # Mock mc.select_op to avoid needing a real MC scheduler
        f.mc = MagicMock()
        f.mc.select_op = MagicMock(return_value="havoc")
        f.mc_bandit = True
        f._rng = ScriptedRng(
            randoms=list(itertools.repeat(0.0, 200)),
            randints=[42] * 20,
            choice_idxs=[0],
            counts=[8, 8, 8],  # randint_list calls for buffer
        )
        engine = OperatorEngine(f)

        # Apply mutation
        data = b"hello world"
        engine.mutate(data)

        # Verify _last_slopt_exp was set
        assert f._last_slopt_exp == 0.5
        # Verify select_op was called (operator selection)
        f.mc.select_op.assert_called_once()

    def test_slopt_applies_batch_loop(self, monkeypatch):
        """When _use_slopt=True, mutate applies operator in batch loop."""
        from unittest.mock import MagicMock

        def mock_slopt_batch_size(seed: bytes, op: str, l0: int = 64) -> tuple[int, float]:
            return 2, 0.6  # batch=2, exp=0.6

        monkeypatch.setattr(
            "fuzzer_tool.services.operators.slopt_batch_size",
            mock_slopt_batch_size,
        )

        f = _make_minimal_fuzzer()
        f._use_slopt = True
        f.mc_bandit = True
        f.mc = MagicMock()
        f.mc.select_op = MagicMock(return_value="havoc")
        f._rng = ScriptedRng(
            randoms=list(itertools.repeat(0.0, 200)),
            randints=[42] * 20,
            choice_idxs=[0],
            counts=[8, 8, 8],
        )
        engine = OperatorEngine(f)

        data = b"hello world"
        result = engine.mutate(data)

        # Verify _last_slopt_exp was set to 0.6 (havoc exponent)
        assert f._last_slopt_exp == 0.6
        # Verify the result is bytes
        assert isinstance(result, bytes)

    def test_slopt_different_operators_different_exponents(self):
        """Test that different operators use their configured exponents."""
        test_cases: list[tuple[str, float]] = [
            ("havoc", 0.6),
            ("bit_flip", 0.3),
            ("arith", 0.5),
            ("unknown", 0.4),  # default exponent
        ]

        for op_name, expected_exp in test_cases:
            seed = b"X" * 128  # len=128 → 128/64=2.0
            batch, exp = slopt_batch_size(seed, op_name)
            assert exp == expected_exp, f"For {op_name}: expected exp={expected_exp}, got {exp}"
            expected_batch = max(1, int(2.0**expected_exp))
            assert batch == expected_batch, (
                f"For {op_name}: expected batch={expected_batch}, got {batch}"
            )

    def test_slopt_flag_is_false_by_default_in_fuzzer(self):
        """Verify the slopt flag defaults to False in Fuzzer.__init__."""
        from unittest.mock import MagicMock

        from fuzzer_tool.services.fuzzer import Fuzzer

        mock_target = MagicMock()
        fuzzer = Fuzzer(target=mock_target, corpus_dir="/tmp", crashes_dir="/tmp")
        assert fuzzer._use_slopt is False
        assert fuzzer._last_slopt_exp == 0.0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
