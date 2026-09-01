"""Tests for core/parallel_fractal_partition.py (Approach C)."""

from __future__ import annotations

import pytest

from fuzzer_tool.core.parallel_fractal_partition import (
    accept_for_worker,
    assign_worker,
    crosses_boundary,
    root_cell,
)


class TestAssignWorker:
    def test_deterministic(self):
        seed = b"some seed content here"
        assert assign_worker(seed, 4) == assign_worker(seed, 4)

    def test_in_range(self):
        for i in range(50):
            seed = f"seed-{i}".encode()
            w = assign_worker(seed, 6)
            assert 0 <= w < 6

    def test_rejects_non_positive_workers(self):
        with pytest.raises(ValueError, match="n_workers must be positive"):
            assign_worker(b"x", 0)

    def test_distributes_across_workers(self):
        """Enough distinct seeds should not all pile onto one worker."""
        seen = {assign_worker(f"seed-{i}".encode(), 4) for i in range(200)}
        assert len(seen) > 1

    def test_different_content_can_differ(self):
        # Not a strict requirement (collisions are legal) but a sanity check
        # that assignment actually depends on content, not a constant.
        assignments = {assign_worker(f"distinct-{i}".encode(), 8) for i in range(100)}
        assert len(assignments) > 1


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


class TestCrossesBoundary:
    def test_deterministic(self):
        seed = b"boundary test seed"
        assert crosses_boundary(seed, depth=3) == crosses_boundary(seed, depth=3)

    def test_some_seeds_are_boundary_some_are_not(self):
        """Over enough seeds, both interior and boundary cases should occur."""
        results = {crosses_boundary(f"s-{i}".encode(), depth=3) for i in range(300)}
        assert results == {True, False}


class TestAcceptForWorker:
    def test_owner_always_accepts_own_seed(self):
        seed = b"owned seed content"
        owner = assign_worker(seed, 4)
        assert accept_for_worker(seed, owner, 4) is True

    def test_non_owner_accepts_only_if_boundary(self):
        seed = b"some arbitrary seed"
        owner = assign_worker(seed, 4)
        other = (owner + 1) % 4
        expected = crosses_boundary(seed, 3)
        assert accept_for_worker(seed, other, 4) == expected

    def test_every_worker_accepts_at_least_the_owner(self):
        for i in range(30):
            seed = f"seed-{i}".encode()
            owner = assign_worker(seed, 5)
            assert accept_for_worker(seed, owner, 5) is True
