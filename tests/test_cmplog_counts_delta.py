"""The per-execution comparison vector: what one drain of the counters saw.

``collect_counts`` folds the shim's deltas into run totals. Those totals
answer "what did this campaign compare" and cannot answer "what did *this
input* compare", which is the question the reward path needs -- and the
totals were the only thing the collector exposed.

The distinction is not academic. ``collect_counts`` was reachable only
through ``collect_tokens``, which throttles itself to every 5th and then
every 20th iteration once the pair pool saturates. On that schedule a
delta is the sum over up to twenty executions and attributes to one input
what nineteen others did.
"""

from __future__ import annotations

from fuzzer_tool.core.cmplog import CmplogCollector


def _write(path, *lines: str) -> None:
    """Append CNT lines the way the shim does: O_APPEND, deltas, no rewrite."""
    with open(path, "a") as fh:
        for line in lines:
            fh.write(line + "\n")


def _collector(tmp_path) -> CmplogCollector:
    c = CmplogCollector()
    c.counts_path = str(tmp_path / "counts.txt")
    open(c.counts_path, "w").close()
    return c


def test_drain_returns_only_what_it_read(tmp_path):
    c = _collector(tmp_path)
    _write(c.counts_path, "CNT memcmp 4 1", "CNT strcmp 2 0")
    fired, asserted = c.collect_counts()
    assert fired == {"memcmp": 4, "strcmp": 2}
    assert asserted == {"memcmp": 1}


def test_consecutive_drains_partition_the_totals(tmp_path):
    """Each drain is one execution's vector; the totals are their sum."""
    c = _collector(tmp_path)
    _write(c.counts_path, "CNT memcmp 4 1")
    assert c.collect_counts()[0] == {"memcmp": 4}

    _write(c.counts_path, "CNT memcmp 3 2")
    fired, asserted = c.collect_counts()
    assert fired == {"memcmp": 3}, "second drain must not re-report the first"
    assert asserted == {"memcmp": 2}

    assert c.comparison_stats() == {"memcmp": (7, 3)}


def test_several_dumps_in_one_drain_are_summed(tmp_path):
    """A direct_lite execution can sync more than once before the drain."""
    c = _collector(tmp_path)
    _write(c.counts_path, "CNT memcmp 1 1", "CNT memcmp 2 0", "CNT strcmp 1 1")
    fired, asserted = c.collect_counts()
    assert fired == {"memcmp": 3, "strcmp": 1}
    assert asserted == {"memcmp": 1, "strcmp": 1}


def test_empty_drain_clears_the_previous_vector(tmp_path):
    """A stale vector read as the current execution's is the whole hazard.

    An execution that compares nothing writes no CNT lines at all -- the
    shim skips sites whose fired count is zero -- so "no new lines" is a
    normal outcome, not an error, and it has to mean an empty vector rather
    than the last one that happened to be non-empty.
    """
    c = _collector(tmp_path)
    _write(c.counts_path, "CNT memcmp 4 1")
    assert c.collect_counts()[0] == {"memcmp": 4}

    fired, asserted = c.collect_counts()
    assert fired == {}
    assert asserted == {}
    assert c.last_fired == {}
    assert c.last_asserted == {}


def test_missing_counts_file_yields_an_empty_vector(tmp_path):
    """cmplog off, or the shim never opened the sidecar."""
    c = CmplogCollector()
    c.counts_path = str(tmp_path / "absent.txt")
    assert c.collect_counts() == ({}, {})


def test_malformed_lines_do_not_poison_the_vector(tmp_path):
    c = _collector(tmp_path)
    _write(
        c.counts_path,
        "CMP 4142 4344 -1 2",  # a record line, wrong channel
        "CNT memcmp notanumber 1",
        "CNT truncated 3",
        "CNT bcmp 5 5",
    )
    fired, asserted = c.collect_counts()
    assert fired == {"bcmp": 5}
    assert asserted == {"bcmp": 5}
