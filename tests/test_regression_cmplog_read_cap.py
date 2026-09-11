"""Regression: the collect_tokens() read cap must cap, not zero.

``collect_tokens`` iterated the log with ``for line in f``, broke out of the
loop at the 10,000-line cap, and then called ``f.tell()``. Iterating a
text-mode file fills a read-ahead buffer and disables ``tell()``, so
abandoning the iterator mid-buffer -- precisely what the cap does -- made
that call raise ``OSError: telling position disabled by next() call``. The
handler set ``new_lines = []``, and the unconditional truncate at the end of
the method then destroyed the records.

The effect was a cliff, not a cap: a drain carrying 10,000 lines harvested
10,000 tokens and a drain carrying 10,001 harvested *zero*. Nothing was
deferred either, because the file is emptied on the way out. On a
comparison-dense target under the 20-execution collection throttle the cap
is reached routinely, and the failure was silent (``log.debug`` only).

The same read also decoded with the ambient locale, so a single non-UTF-8
byte anywhere in the stream raised ``UnicodeDecodeError`` -- a ``ValueError``,
which the surrounding ``except OSError`` does not catch -- out of
``collect_tokens`` and into ``fuzz_one``.

No test exercised a drain larger than the cap.
"""

import os

from fuzzer_tool.core.cmplog import CMPLOG_MAX_LINES_PER_READ, CmplogCollector


def _collector(tmp_path):
    c = CmplogCollector(workdir=str(tmp_path))
    c.log_path = os.path.join(str(tmp_path), "cmplog.txt")
    return c


def _fill(collector, n):
    """Write n distinct single-operand CMP records, as the shim would."""
    with open(collector.log_path, "w") as f:
        for i in range(n):
            operand = (b"A%07d" % i).hex()
            f.write(f"CMP {operand} {operand} 0 8\n")


class TestReadCapDoesNotDiscardTheBatch:
    def test_exactly_at_cap(self, tmp_path):
        c = _collector(tmp_path)
        _fill(c, CMPLOG_MAX_LINES_PER_READ)
        assert len(c.collect_tokens()) == CMPLOG_MAX_LINES_PER_READ

    def test_one_line_past_cap_still_yields_a_full_batch(self, tmp_path):
        """The cliff: this harvested 0 tokens before the fix."""
        c = _collector(tmp_path)
        _fill(c, CMPLOG_MAX_LINES_PER_READ + 1)
        assert len(c.collect_tokens()) == CMPLOG_MAX_LINES_PER_READ

    def test_far_past_cap_still_yields_a_full_batch(self, tmp_path):
        c = _collector(tmp_path)
        _fill(c, CMPLOG_MAX_LINES_PER_READ * 3)
        assert len(c.collect_tokens()) == CMPLOG_MAX_LINES_PER_READ

    def test_harvested_tokens_are_the_first_records_not_a_sample(self, tmp_path):
        c = _collector(tmp_path)
        _fill(c, CMPLOG_MAX_LINES_PER_READ + 50)
        tokens = set(c.collect_tokens())
        assert b"A0000000" in tokens
        assert b"A%07d" % (CMPLOG_MAX_LINES_PER_READ - 1) in tokens
        # Past the cap, dropped by design -- the file is truncated below.
        assert b"A%07d" % CMPLOG_MAX_LINES_PER_READ not in tokens

    def test_oversized_drain_still_empties_the_log(self, tmp_path):
        """The remainder is dropped, not left to re-read: offset back to 0."""
        c = _collector(tmp_path)
        _fill(c, CMPLOG_MAX_LINES_PER_READ + 50)
        c.collect_tokens()
        assert os.path.getsize(c.log_path) == 0
        assert c._read_offset == 0

    def test_under_cap_reads_everything(self, tmp_path):
        c = _collector(tmp_path)
        _fill(c, 37)
        assert len(c.collect_tokens()) == 37

    def test_empty_log_is_harmless(self, tmp_path):
        c = _collector(tmp_path)
        _fill(c, 0)
        assert c.collect_tokens() == []


class TestDrainToleratesNonUtf8:
    def test_stray_byte_does_not_abort_the_drain(self, tmp_path):
        c = _collector(tmp_path)
        with open(c.log_path, "wb") as f:
            f.write(b"CMP 4141 4242 0 2\n")
            f.write(b"\xff\xfe not a record\n")
            f.write(b"CMP 4343 4444 0 2\n")
        tokens = set(c.collect_tokens())
        assert {b"AA", b"BB", b"CC", b"DD"} <= tokens

    def test_high_bytes_inside_a_record_do_not_abort_the_drain(self, tmp_path):
        """A torn write can leave arbitrary bytes on a line; keep draining."""
        c = _collector(tmp_path)
        with open(c.log_path, "wb") as f:
            f.write(b"CMP \x80\x81\x82 zz 0 2\n")
            f.write(b"CMP 4545 4646 0 2\n")
        tokens = set(c.collect_tokens())
        assert {b"EE", b"FF"} <= tokens
