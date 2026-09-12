"""Host cache hierarchy and the coverage-map residency line.

The line is informational, so the tests are mostly about not lying: skip
instruction caches (the segment cannot live there), report unknown topology
as unknown rather than as "no cache", and phrase residency as an edge
capacity rather than implying the allocation's size sets its speed.

That last point is measured, not assumed. With the touched set held at
32 KiB and only the segment size varied across two cache boundaries
(32 KiB -> 32 MiB), per-fire cost was 5.36, 5.28, 5.18, 5.21, 5.18,
5.25 ns -- no trend, the smallest segment fractionally slowest. Holding the
segment at 8 MiB and varying distinct edges instead does move it, 5.18 ns
at 512 distinct to 5.98 at 524,288. Hence a line about edges, not speed.
"""

from fuzzer_tool.core.cpu_cache import (
    DEFAULT_LINE_SIZE,
    CacheLevel,
    _count_cpu_list,
    _parse_size,
    data_cache_levels,
    describe_map_residency,
    smallest_level_holding,
)

L1D = CacheLevel(level=1, size_bytes=48 * 1024, line_size=64, shared_by=1)
L2 = CacheLevel(level=2, size_bytes=2 * 1024**2, line_size=64, shared_by=1)
L3 = CacheLevel(level=3, size_bytes=32 * 1024**2, line_size=64, shared_by=8)
LEVELS = [L1D, L2, L3]


class TestSizeParsing:
    def test_sysfs_spellings(self):
        assert _parse_size("48K") == 48 * 1024
        assert _parse_size("2048K") == 2048 * 1024
        assert _parse_size("266240K") == 266240 * 1024
        assert _parse_size("8M") == 8 * 1024**2
        assert _parse_size("64") == 64

    def test_unparseable_is_none_not_zero(self):
        """Zero would make every footprint 'exceed' the level silently."""
        assert _parse_size("") is None
        assert _parse_size("unknown") is None
        assert _parse_size("K") is None


class TestCpuListCounting:
    def test_forms(self):
        assert _count_cpu_list("0") == 1
        assert _count_cpu_list("0-3") == 4
        assert _count_cpu_list("0,2-4") == 4
        assert _count_cpu_list("0-1,4-5") == 4

    def test_empty_counts_as_one_not_zero(self):
        """shared_by=0 would make `shared` arithmetic nonsense downstream."""
        assert _count_cpu_list("") == 1
        assert _count_cpu_list("   ") == 1


class TestLevelSelection:
    def test_picks_the_smallest_level_that_holds_it(self):
        assert smallest_level_holding(1024, LEVELS) is L1D
        assert smallest_level_holding(48 * 1024, LEVELS) is L1D
        assert smallest_level_holding(48 * 1024 + 1, LEVELS) is L2
        assert smallest_level_holding(30 * 1024**2, LEVELS) is L3

    def test_beyond_every_level_is_none(self):
        """None must mean 'reaches memory', never 'fits the biggest'."""
        assert smallest_level_holding(64 * 1024**2, LEVELS) is None


class TestRealHost:
    def test_levels_are_ascending_and_growing(self):
        levels = data_cache_levels()
        if not levels:
            return  # no sysfs here; unknown is a valid answer
        assert [lvl.level for lvl in levels] == sorted(lvl.level for lvl in levels)
        sizes = [lvl.size_bytes for lvl in levels]
        assert sizes == sorted(sizes), f"cache levels not monotonic: {sizes}"
        assert all(lvl.size_bytes > 0 for lvl in levels)

    def test_instruction_cache_is_not_reported(self):
        """L1i cannot hold the segment, so offering it invites comparing a
        data footprint against a cache it can never use."""
        levels = data_cache_levels()
        if not levels:
            return
        l1s = [lvl for lvl in levels if lvl.level == 1]
        assert len(l1s) <= 1, "both L1d and L1i were reported"
        for lvl in levels:
            assert lvl.line_size > 0

    def test_naming(self):
        assert L1D.name == "L1d"
        assert L2.name == "L2"
        assert L3.name == "L3"


class TestResidencyLine:
    def test_names_the_level_that_holds_the_segment(self):
        out = describe_map_residency(64 * 1024, 8192, LEVELS)
        assert out is not None
        assert "fits L2" in out
        assert "64 KiB segment" in out

    def test_says_exceeds_when_no_level_holds_it(self):
        out = describe_map_residency(512 * 1024**2, 1 << 26, LEVELS)
        assert out is not None
        assert "exceeds L3" in out

    def test_notes_a_shared_level(self):
        """L3's nominal size is not what one process gets to itself."""
        out = describe_map_residency(30 * 1024**2, 1 << 22, LEVELS)
        assert out is not None
        assert "shared by 8 CPUs" in out

    def test_capacity_list_stops_at_the_first_level_holding_the_table(self):
        """Repeating the table size for every larger level says nothing."""
        out = describe_map_residency(64 * 1024, 8192, LEVELS)
        assert out is not None
        assert "L2 all 8,192" in out
        assert "L3" not in out.split("edges resident")[1]

    def test_deepest_capacity_is_qualified_when_the_table_does_not_fit(self):
        out = describe_map_residency(512 * 1024**2, 1 << 26, LEVELS)
        assert out is not None
        assert "of 67,108,864" in out

    def test_unknown_topology_yields_no_line(self):
        """Better to print nothing than to print a guessed hierarchy."""
        assert describe_map_residency(64 * 1024, 8192, []) is None

    def test_line_is_about_edges_not_speed(self):
        """Guards the wording the measurements justify.

        The segment's size does not set access cost -- that is measured in
        this module's docstring -- so the line must not claim or imply it.
        """
        out = describe_map_residency(64 * 1024, 8192, LEVELS)
        assert out is not None
        lowered = out.lower()
        for banned in ("faster", "fast", "slow", "speed", "ns/", "throughput", "cached"):
            assert banned not in lowered, f"line implies a performance claim: {banned!r}"
        assert "edges resident" in lowered

    def test_line_size_falls_back_when_sysfs_omits_it(self):
        odd = [CacheLevel(level=1, size_bytes=32 * 1024, line_size=0, shared_by=1)]
        out = describe_map_residency(16 * 1024, 4096, odd)
        assert out is not None
        assert f"{DEFAULT_LINE_SIZE} B/edge" in out


class TestRealSegmentSizes:
    def test_matches_a_real_shm_segment(self):
        from fuzzer_tool.adapters.shm import ShmCoverage

        levels = data_cache_levels()
        if not levels:
            return
        cov = ShmCoverage(size=8192)
        try:
            out = describe_map_residency(cov.shm_bytes, cov.num_entries, levels)
            assert out is not None
            assert "Coverage map:" in out
        finally:
            cov.cleanup()


class TestInstructionCachesAreSkipped:
    """Checked against a synthetic sysfs tree, not the real host.

    The host cannot falsify this. data_cache_levels keeps the largest entry
    per level, so on any machine where L1d > L1i -- the usual case, and true
    where this was written (48K vs 32K) -- L1d wins whether or not the
    instruction cache was filtered. A host with a larger L1i would silently
    report an instruction cache as the data L1 and a host-based assertion
    would still pass. Confirmed: deleting the `type == "Instruction"` skip
    left the whole host-based suite green.
    """

    @staticmethod
    def _write_index(root, n, *, level, kind, size, line=64, cpus="0"):
        d = root / f"index{n}"
        d.mkdir()
        (d / "level").write_text(f"{level}\n")
        (d / "type").write_text(f"{kind}\n")
        (d / "size").write_text(f"{size}\n")
        (d / "coherency_line_size").write_text(f"{line}\n")
        (d / "shared_cpu_list").write_text(f"{cpus}\n")
        return d

    def test_larger_instruction_cache_does_not_displace_l1d(self, tmp_path):
        root = tmp_path / "cache"
        root.mkdir()
        # Deliberately inverted against the common case: L1i bigger than L1d.
        self._write_index(root, 0, level=1, kind="Data", size="32K")
        self._write_index(root, 1, level=1, kind="Instruction", size="512K")
        self._write_index(root, 2, level=2, kind="Unified", size="2048K")

        levels = data_cache_levels(root=root)
        by_level = {lvl.level: lvl for lvl in levels}
        assert by_level[1].size_bytes == 32 * 1024, (
            "the instruction cache was reported as the data L1 — a data "
            "footprint would be compared against a cache it cannot occupy"
        )
        assert by_level[2].size_bytes == 2048 * 1024

    def test_unified_levels_are_kept(self, tmp_path):
        """Data does live in unified levels; only pure-instruction is excluded."""
        root = tmp_path / "cache"
        root.mkdir()
        self._write_index(root, 0, level=1, kind="Data", size="48K")
        self._write_index(root, 1, level=2, kind="Unified", size="1024K")
        assert [lvl.level for lvl in data_cache_levels(root=root)] == [1, 2]

    def test_shared_cpu_list_is_read(self, tmp_path):
        root = tmp_path / "cache"
        root.mkdir()
        self._write_index(root, 0, level=3, kind="Unified", size="8M", cpus="0-15")
        (lvl,) = data_cache_levels(root=root)
        assert lvl.shared_by == 16
        assert lvl.shared

    def test_unreadable_entry_is_skipped_not_fatal(self, tmp_path):
        root = tmp_path / "cache"
        root.mkdir()
        self._write_index(root, 0, level=1, kind="Data", size="48K")
        (root / "index1").mkdir()  # empty: no level/type/size files
        assert [lvl.level for lvl in data_cache_levels(root=root)] == [1]

    def test_missing_root_is_unknown(self, tmp_path):
        assert data_cache_levels(root=tmp_path / "nope") == []


class TestBannerWiring:
    """The helper existing is not the same as the banner printing it.

    Every test above calls describe_map_residency directly, so all of them
    pass with the call site removed from the startup banner.
    """

    def test_the_banner_calls_it(self):
        import inspect

        from fuzzer_tool.services.fuzzer import Fuzzer

        src = inspect.getsource(Fuzzer)
        assert "self._report_map_cache_residency()" in src, (
            "_report_map_cache_residency is defined but never called"
        )

    def test_it_prints_next_to_the_edge_bitmap_line(self):
        """Pinned to the bitmap line: the map's size is the context that
        makes the residency figures meaningful."""
        import inspect

        from fuzzer_tool.services.fuzzer import Fuzzer

        lines = inspect.getsource(Fuzzer).splitlines()
        anchor = next(i for i, ln in enumerate(lines) if "Edge bitmap:" in ln)
        window = "\n".join(lines[anchor : anchor + 3])
        assert "_report_map_cache_residency()" in window

    def test_no_segment_prints_nothing(self, capsys):
        from fuzzer_tool.services.fuzzer import Fuzzer

        stub = object.__new__(Fuzzer)
        stub.shm_cov = None
        Fuzzer._report_map_cache_residency(stub)
        assert capsys.readouterr().out == ""

    def test_a_real_segment_prints_one_line(self, capsys):
        from fuzzer_tool.adapters.shm import ShmCoverage
        from fuzzer_tool.services.fuzzer import Fuzzer

        if not data_cache_levels():
            return  # unknown topology: printing nothing is correct
        cov = ShmCoverage(size=8192)
        try:
            stub = object.__new__(Fuzzer)
            stub.shm_cov = cov
            Fuzzer._report_map_cache_residency(stub)
            out = capsys.readouterr().out
        finally:
            cov.cleanup()
        assert out.count("\n") == 1, f"expected exactly one line, got {out!r}"
        assert "Coverage map:" in out
