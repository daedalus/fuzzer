"""Regression: primitive-root search must not loop forever.

``find_primitive_root`` was unbounded rejection sampling; ``GF2n`` guarded
its modulus with an ``assert`` that ``python -O`` strips, so a reducible
modulus (no primitive element) spun forever.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from fuzzer_tool.core.gf2_common import GF2n, find_primitive_root
from fuzzer_tool.core.rand_pool import RandPool
from tests.support.scripted_rng import ScriptedRng

# x^8 + 1 = (x + 1)^8 over GF(2): reducible, so GF(2)[x]/(m) is not a field.
_REDUCIBLE_DEG8 = 0x101
_HANG_TIMEOUT_S = 5
_SRC = str(Path(__file__).resolve().parent.parent / "src")


def test_regression_root_first_hit_returned() -> None:
    """Normal search returns the first accepted draw, scripted."""
    rng = ScriptedRng(randints=[4, 6, 3])

    assert find_primitive_root(7, lambda a: a == 3, rng) == 3


@pytest.mark.timeout(5)
def test_regression_root_draws_capped() -> None:
    """A predicate that never accepts raises instead of spinning."""
    with pytest.raises(RuntimeError):
        find_primitive_root(255, lambda a: False, RandPool(seed=0))


@pytest.mark.timeout(5)
def test_regression_reducible_modulus_rejected() -> None:
    with pytest.raises(ValueError):
        GF2n(8, modulus=_REDUCIBLE_DEG8)


@pytest.mark.timeout(10)
def test_regression_reducible_rejected_under_O() -> None:
    """``python -O`` strips asserts; the check must survive it."""
    code = f"from fuzzer_tool.core.gf2_common import GF2n\nGF2n(8, modulus={_REDUCIBLE_DEG8})\n"
    proc = subprocess.run(
        [sys.executable, "-O", "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": _SRC},
        timeout=_HANG_TIMEOUT_S,
    )

    assert proc.returncode != 0
    assert "ValueError" in proc.stderr
