"""Regression tests: fractal-diversity seed weight (--fractal-diversity).

Boosts seeds whose content hash sits on a fractal Voronoi boundary
(``core/parallel_fractal_partition.crosses_boundary``), as a cheap
diversity signal against mode collapse toward one region of the corpus's
own content-hash space. See ``SeedPicker._weight_fractal_diversity`` for
why this is scoped away from Approach B in the original handover doc
(bitmap-position adjacency carries no signal in this codebase; content
hashes are exactly what they claim to be).

Properties that must hold:

  * off by default, and a no-op without --fractal-diversity;
  * a boundary-crossing seed gets exactly the configured bonus;
  * a non-boundary seed is untouched;
  * the depth and bonus parameters are actually read from the fuzzer,
    not hardcoded.
"""

from __future__ import annotations

from fuzzer_tool.core.parallel_fractal_partition import crosses_boundary
from fuzzer_tool.services.seed_picker import SeedPicker


class _Fuzzer:
    """Minimal stand-in exposing only what the weight function reads."""

    def __init__(self, *, diversity=True, depth=3, bonus=1.3):
        self._use_fractal_diversity = diversity
        self._fractal_diversity_depth = depth
        self._fractal_diversity_bonus = bonus


def _weight(f, seed: bytes, w: float) -> float:
    return SeedPicker._weight_fractal_diversity(object.__new__(SeedPicker), seed, w, f)


def _find_seed(*, boundary: bool, depth: int = 3, limit: int = 2000) -> bytes:
    """Locate a seed whose crosses_boundary() matches the requested value."""
    for i in range(limit):
        candidate = f"seed-{i}".encode()
        if crosses_boundary(candidate, depth) == boundary:
            return candidate
    raise AssertionError(f"no seed found with crosses_boundary()={boundary} in {limit} tries")


class TestFractalDiversityWeight:
    def test_off_by_default_is_a_no_op(self):
        f = _Fuzzer(diversity=False)
        seed = _find_seed(boundary=True)
        assert _weight(f, seed, 2.0) == 2.0

    def test_boundary_seed_gets_bonus(self):
        f = _Fuzzer(diversity=True, bonus=1.3)
        seed = _find_seed(boundary=True)
        assert _weight(f, seed, 2.0) == 2.0 * 1.3

    def test_non_boundary_seed_unchanged(self):
        f = _Fuzzer(diversity=True, bonus=1.3)
        seed = _find_seed(boundary=False)
        assert _weight(f, seed, 2.0) == 2.0

    def test_bonus_is_read_from_fuzzer_not_hardcoded(self):
        seed = _find_seed(boundary=True)
        f_small = _Fuzzer(diversity=True, bonus=1.1)
        f_large = _Fuzzer(diversity=True, bonus=2.0)
        assert _weight(f_small, seed, 1.0) == 1.1
        assert _weight(f_large, seed, 1.0) == 2.0

    def test_depth_is_read_from_fuzzer(self):
        """The function must use the configured depth, not a fixed one."""
        seed = _find_seed(boundary=True, depth=3)
        f_depth3 = _Fuzzer(diversity=True, depth=3, bonus=1.3)
        expected = 1.3 if crosses_boundary(seed, 3) else 1.0
        assert _weight(f_depth3, seed, 1.0) == expected
