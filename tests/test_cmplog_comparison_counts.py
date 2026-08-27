"""Per-callback comparison counters: the $_CMPLOG_COUNTS sidecar.

The CMP record stream cannot report how many comparisons fired or how many
were satisfied -- records carry no callback identity, the collector dedups
multiplicity away, and __afl_cmplog_bytes drops result == 0 on purpose, so a
satisfied layer-1 comparison is never written at all. The shim counts in the
interceptors instead and dumps deltas to a sidecar file; these tests cover
the Python end of that channel.
"""

import os

from fuzzer_tool.core.cmplog import CMPLOG_COUNTS_MAX_BYTES, CmplogCollector


def _collector(tmp_path):
    c = CmplogCollector(workdir=str(tmp_path))
    c.log_path = os.path.join(str(tmp_path), "fuzz_cmplog_deadbeef.cmplog")
    c.counts_path = c._counts_path_for(c.log_path)
    return c


def _dump(collector, *lines):
    """Append a shim dump, the way the shim's O_APPEND fd would."""
    with open(collector.counts_path, "a") as f:
        for line in lines:
            f.write(line + "\n")


class TestCountsPath:
    def test_sidecar_sits_beside_the_log(self, tmp_path):
        c = _collector(tmp_path)
        assert c.counts_path.endswith("fuzz_cmplog_deadbeef.counts")
        assert os.path.dirname(c.counts_path) == os.path.dirname(c.log_path)

    def test_extensionless_log_still_gets_a_sidecar(self, tmp_path):
        c = CmplogCollector(workdir=str(tmp_path))
        assert c._counts_path_for("/tmp/noext") == "/tmp/noext.counts"


class TestDeltaAccumulation:
    def test_dumps_are_summed(self, tmp_path):
        """Each dump is a delta -- the shim zeroes its counters as it writes.

        Summing is what makes the channel mode-independent: a subprocess run
        contributes one dump per exec, a direct_lite run many dumps from one
        process, and neither side has to know which is happening.
        """
        c = _collector(tmp_path)
        _dump(c, "CNT memcmp 1 1")
        c.collect_counts()
        _dump(c, "CNT memcmp 2 0")
        c.collect_counts()
        _dump(c, "CNT memcmp 1 1")
        c.collect_counts()
        assert c.cmp_fired["memcmp"] == 4
        assert c.cmp_asserted["memcmp"] == 2

    def test_multiple_callbacks_kept_apart(self, tmp_path):
        c = _collector(tmp_path)
        _dump(c, "CNT memcmp 10 3", "CNT trace_switch 5 3", "CNT strstr 4 4")
        c.collect_counts()
        assert c.comparison_stats() == {
            "memcmp": (10, 3),
            "strstr": (4, 4),
            "trace_switch": (5, 3),
        }

    def test_offset_read_does_not_replay(self, tmp_path):
        """Read from a saved offset, not read-and-truncate.

        A subprocess target can dump between the read and the truncate, and
        those counts would vanish. The offset makes a second drain with no
        new dump a no-op instead of a double count.
        """
        c = _collector(tmp_path)
        _dump(c, "CNT memcmp 7 2")
        c.collect_counts()
        c.collect_counts()
        c.collect_counts()
        assert c.cmp_fired["memcmp"] == 7

    def test_totals(self, tmp_path):
        c = _collector(tmp_path)
        _dump(c, "CNT memcmp 10 3", "CNT strcmp 5 5")
        c.collect_counts()
        assert c.total_comparisons() == (15, 8)

    def test_no_file_is_not_an_error(self, tmp_path):
        c = _collector(tmp_path)
        c.collect_counts()
        assert c.total_comparisons() == (0, 0)


class TestMalformedInput:
    def test_junk_lines_ignored(self, tmp_path):
        c = _collector(tmp_path)
        _dump(
            c,
            "CMP 41424344 45464748 -1 4",  # a record, not a count
            "CNT memcmp 3 1",
            "CNT truncated 5",  # short field count
            "CNT memcmp x y",  # non-numeric
            "",
        )
        c.collect_counts()
        assert c.comparison_stats() == {"memcmp": (3, 1)}

    def test_partial_final_line_does_not_desync(self, tmp_path):
        """A dump interrupted mid-line must not corrupt the totals.

        The shim writes with write(2) and can be cut short by ENOSPC or a
        signal, so a trailing partial line is reachable in practice.
        """
        c = _collector(tmp_path)
        with open(c.counts_path, "a") as f:
            f.write("CNT memcmp 3 1\nCNT strc")
        c.collect_counts()
        assert c.comparison_stats() == {"memcmp": (3, 1)}


class TestOversizedSidecar:
    def test_truncated_past_the_cap(self, tmp_path):
        c = _collector(tmp_path)
        _dump(c, "CNT memcmp 5 5")
        c.collect_counts()
        with open(c.counts_path, "w") as f:
            f.write("CNT memcmp 1 1\n" * (CMPLOG_COUNTS_MAX_BYTES // 15 + 100))
        c.collect_counts()
        assert os.path.getsize(c.counts_path) == 0
        assert c._counts_offset == 0
        # Totals already banked survive the truncation.
        assert c.cmp_fired["memcmp"] == 5
        # And the channel keeps working afterwards.
        _dump(c, "CNT memcmp 2 0")
        c.collect_counts()
        assert c.cmp_fired["memcmp"] == 7


class TestEnvWiring:
    def test_subprocess_env_carries_the_sidecar(self, tmp_path):
        c = CmplogCollector(workdir=str(tmp_path))
        c._shim_path = str(tmp_path / "shim.so")
        env = c.setup_env({})
        assert env["_CMPLOG_COUNTS"] == c.counts_path
        assert env["_CMPLOG_OUT"] == c.log_path

    def test_inprocess_env_is_restored(self, tmp_path, monkeypatch):
        monkeypatch.delenv("_CMPLOG_COUNTS", raising=False)
        c = CmplogCollector(workdir=str(tmp_path))
        c.setup_env_for_run()
        assert os.environ["_CMPLOG_COUNTS"] == c.counts_path
        c.restore_env()
        assert "_CMPLOG_COUNTS" not in os.environ


class TestCollectTokensIntegration:
    def test_counts_drained_even_with_no_records(self, tmp_path):
        """A target whose comparisons all pass writes no layer-1 records.

        That is exactly the target whose asserted counts matter most, so the
        drain must not sit behind the record stream's existence check.
        """
        c = _collector(tmp_path)
        _dump(c, "CNT memcmp 9 9")
        assert not os.path.exists(c.log_path)
        assert c.collect_tokens() == []
        assert c.comparison_stats() == {"memcmp": (9, 9)}
