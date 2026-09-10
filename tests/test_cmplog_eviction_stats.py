"""Tests for the CmplogCollector eviction counters.

The collector caps the number of unique operand tokens and operand pairs. When the
caps are exceeded the oldest/lowest‑scoring entries are evicted and the counters
`evicted_token_count` and `evicted_pair_count` are incremented. This test forces a
clear eviction scenario and verifies the counters.
"""

from fuzzer_tool.core.cmplog import CmplogCollector


def test_cmplog_eviction_counters():
    # Small caps to trigger eviction quickly
    collector = CmplogCollector(max_tokens=2, max_pairs=2)

    # Each CMP line introduces two distinct token bytes and one distinct pair.
    # Four lines give 8 tokens and 4 pairs, which exceeds both caps.
    lines = [
        "CMP 01 02 0 1",
        "CMP 03 04 0 1",
        "CMP 05 06 0 1",
        "CMP 07 08 0 1",
    ]

    # Directly parse the lines – this runs the eviction logic.
    collector._parse_lines(lines)

    evicted_tokens, evicted_pairs = collector.get_eviction_stats()

    # 8 tokens added, cap 2 => 6 evicted
    assert evicted_tokens == 6
    # 4 pairs added, cap 2 => 2 evicted
    assert evicted_pairs == 2

    # The collector should now hold only the capped number of items.
    assert len(collector.tokens) == 2
    assert len(collector.pairs) == 2
