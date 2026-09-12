"""Drops drove a resize only from inside stall recovery.

`_maybe_trigger_stall_recovery` was the sole consumer of the drop counter as
a magnitude, and it does not run until `--stall` executions have passed with
no new edge (default 1,000), and only when `--resize-map-on-stall` is set. So
the one honest saturation signal in the system was read late and
conditionally -- and while it was a 16-bit field packed into the diag word,
it had already pinned at 65,535 by execution 34 on a saturating target, which
is to say every value that consumer ever read there was the ceiling.

A drop is not a symptom awaiting confirmation. It is the fuzzer being told,
by the only component that can know, that coverage it will never see has
already been discarded. Waiting for a stall inverts cause and effect: the
stall is downstream of the lost coverage.

`_maybe_resize_on_drops` acts on it directly, on the stats interval.
"""

from __future__ import annotations

import tempfile
from unittest.mock import patch

import pytest


class _FakeShm:
    """Minimal stand-in: a drop count, a size, and a recording resize()."""

    def __init__(self, size: int, dropped: int):
        self._size = size
        self._dropped = dropped
        self._ptr = 0
        self.env_id = "1234"
        self.resized_to: list[int] = []
        self.reset_calls = 0

    @property
    def size(self) -> int:
        return self._size

    def read_dropped_edges(self) -> int:
        return self._dropped

    def resize(self, new_size: int) -> None:
        self.resized_to.append(new_size)
        self._size = new_size

    def reset_dropped_edges(self) -> None:
        self.reset_calls += 1
        self._dropped = 0


def _make_fuzzer(**kwargs):
    from fuzzer_tool.services.fuzzer import Fuzzer

    tmpdir = tempfile.mkdtemp(prefix="dropresize_")
    with patch("os.path.isfile", return_value=True), patch("os.access", return_value=True):
        return Fuzzer(
            target="/bin/true",
            corpus_dir=f"{tmpdir}/corpus",
            crashes_dir=f"{tmpdir}/crashes",
            max_len=256,
            timeout=1,
            mutations_per_input=2,
            **kwargs,
        )


def _wire(f, shm, observed_edges: int):
    f.shm_cov = shm
    f._edge_tracker.map_size = shm.size
    for e in range(observed_edges):
        f._edge_tracker._global_edge_hits[e] = 1
    f.exec_count = 10_000
    f._drop_resize_checked_at = 0
    return f


class TestResizeOnDrops:
    def test_drops_alone_trigger_a_resize(self):
        """No stall, no plateau — just drops."""
        shm = _FakeShm(size=8192, dropped=20_000)
        f = _wire(_make_fuzzer(), shm, observed_edges=6000)
        f._maybe_resize_on_drops()
        assert shm.resized_to, "drops did not trigger a resize"
        assert shm.resized_to[0] > 8192

    def test_zero_drops_changes_nothing(self):
        shm = _FakeShm(size=8192, dropped=0)
        f = _wire(_make_fuzzer(), shm, observed_edges=6000)
        f._maybe_resize_on_drops()
        assert shm.resized_to == []

    def test_the_new_table_starts_on_fresh_evidence(self):
        """Drops against the old table say nothing about the new one; leaving
        them standing would latch every drop-driven decision on forever."""
        shm = _FakeShm(size=8192, dropped=20_000)
        f = _wire(_make_fuzzer(), shm, observed_edges=6000)
        f._maybe_resize_on_drops()
        assert shm.reset_calls == 1
        assert shm.read_dropped_edges() == 0

    def test_rate_limited_between_checks(self):
        """One resize must not be re-proposed on every tick while the new
        table's own evidence is still accumulating."""
        shm = _FakeShm(size=8192, dropped=20_000)
        f = _wire(_make_fuzzer(), shm, observed_edges=6000)
        f._maybe_resize_on_drops()
        first = len(shm.resized_to)
        shm._dropped = 20_000  # pretend the new table drops too
        f._maybe_resize_on_drops()  # same exec_count: inside the interval
        assert len(shm.resized_to) == first

    def test_respects_the_resize_opt_out(self):
        shm = _FakeShm(size=8192, dropped=20_000)
        f = _wire(_make_fuzzer(resize_map_on_stall=False), shm, observed_edges=6000)
        f._maybe_resize_on_drops()
        assert shm.resized_to == []

    def test_no_shm_is_not_an_error(self):
        f = _make_fuzzer()
        f.shm_cov = None
        f._maybe_resize_on_drops()  # must not raise


class TestAtTheCap:
    def test_saturation_at_the_cap_is_reported_once(self, capsys):
        """A run losing coverage it cannot size its way out of is something
        the operator has to be told. Previously the only place that was ever
        said was inside stall recovery."""
        from fuzzer_tool.core.elf import _map_size_max

        cap = _map_size_max()
        shm = _FakeShm(size=cap, dropped=500_000)
        f = _wire(_make_fuzzer(), shm, observed_edges=200_000)
        f._maybe_resize_on_drops()
        out = capsys.readouterr().out
        assert "saturated at" in out
        assert shm.resized_to == [], "resized past the configured maximum"

        # Second call, past the rate limit, must not repeat the warning.
        f.exec_count += 10_000
        f._maybe_resize_on_drops()
        assert "saturated at" not in capsys.readouterr().out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
