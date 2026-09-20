"""Tests for core/fractal_partition.py (Approach C).

The worker-assignment half of this module went with the -j N mode; what
is left is the partition itself and the coastline predicate
``--fractal-diversity`` reads. ``root_cell`` has no production caller
now, but it is the definition ``crosses_boundary`` is stated in terms of,
so it stays as the oracle those tests check the predicate against.
"""

from __future__ import annotations

import pytest

from fuzzer_tool.core.fractal_partition import crosses_boundary, root_cell


class TestRootCell:
    def test_deterministic(self):
        seed = b"reproducible content"
        assert root_cell(seed, depth=3) == root_cell(seed, depth=3)

    def test_rejects_negative_depth(self):
        with pytest.raises(ValueError, match="depth must be >= 0"):
            root_cell(b"x", depth=-1)

    def test_depth_zero_is_well_defined(self):
        # Depth 0 has no parent chain to trace; should not raise.
        root_cell(b"x", depth=0)

    def test_depends_on_content(self):
        # Collisions are legal, so this only asserts the partition is not
        # a constant function of the seed bytes.
        cells = {root_cell(f"distinct-{i}".encode(), depth=3) for i in range(100)}
        assert len(cells) > 1


class TestCrossesBoundary:
    def test_deterministic(self):
        seed = b"boundary test seed"
        assert crosses_boundary(seed, depth=3) == crosses_boundary(seed, depth=3)

    def test_some_seeds_are_boundary_some_are_not(self):
        """Over enough seeds, both interior and boundary cases should occur."""
        results = {crosses_boundary(f"s-{i}".encode(), depth=3) for i in range(300)}
        assert results == {True, False}

    def test_depth_zero_makes_every_seed_a_boundary_seed(self):
        """Degenerate configuration: at depth 0 each cell is its own root.

        Every 8-neighbour is then a different root by construction, so the
        predicate is constantly True and ``--fractal-diversity-depth 0``
        multiplies every seed's weight by the same bonus -- i.e. it is a
        no-op that looks like a setting. Pinned so it stays visible.
        """
        assert all(crosses_boundary(f"s-{i}".encode(), depth=0) for i in range(50))

    def test_deeper_partitions_have_proportionally_fewer_coastline_seeds(self):
        """The depth knob trades boundary seeds for interior ones.

        Cells shrink as depth grows, but a root cell gathers more of them,
        so a smaller share of seeds sit next to a differently-rooted
        neighbour. Measured over 300 seeds: 300 at depth 1, 202 at 3, 84
        at 5 -- the ordering is what the bonus's selectivity rests on.
        """
        seeds = [f"s-{i}".encode() for i in range(300)]
        counts = [sum(crosses_boundary(s, depth=d) for s in seeds) for d in (1, 3, 5)]
        assert counts[0] > counts[1] > counts[2]
