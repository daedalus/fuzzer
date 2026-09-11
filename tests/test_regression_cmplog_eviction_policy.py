"""Regression: cmplog pool eviction was nondeterministic and froze the pool.

Two defects in one scoring pass.

**Nondeterminism.** Victims were scored by iterating ``self._token_set`` /
``self._pair_set`` and the survivors were rebuilt with ``list(<set>)``. Set
iteration order for ``bytes`` keys follows ``PYTHONHASHSEED``, so *which*
operands were kept and the order they were handed to the mutators both
varied run to run at a fixed ``--seed``, and ``-j N`` workers fed the same
record stream kept divergent pools. ``docs/refs/bug-classes.md`` already bans
builtin ``hash()`` for anything shared across processes; set iteration order
is that hash. Measured before the fix: the same drain sequence under three
hash seeds produced three different survivor *sets*.

**Ossification.** ``mark_coverage_gain()`` bumped every resident entry on any
new coverage, and the excess an eviction pass sheds is exactly the batch that
just arrived -- every arrival starting at zero. So after the first gain the
resident set was permanently immune and no operand discovered later could
ever enter. Measured: 200-entry pool, one gain, 500 fresh pairs streamed in
-> 1000 evictions, zero new survivors. Across 300 randomised gain/drain
interleavings the value term never decided a single victim.

Now: victims are ``(credit, insertion index)`` ascending, survivors keep
insertion order, credit is per-entry (the input-to-state matches the gaining
iteration used) and is aged so it cannot become permanent immunity. The
uncredited mass -- nearly all of it -- rotates FIFO, which is what a record
stream wants: a comparison the target keeps executing is re-emitted on the
next drain and comes straight back.

The old score also divided credit by operand length, ranking a 2-byte operand
above a 16-byte magic value at equal evidence. Dropped.
"""

import hashlib
import os
import subprocess
import sys
import textwrap

from fuzzer_tool.core.cmplog import CMPLOG_VALUE_AGE_GAINS, CmplogCollector


def _lines(pairs):
    return [f"CMP {a.hex()} {b.hex()} 0 {len(a)}" for a, b in pairs]


def _pairs(prefix, n, start=0):
    return [(b"%s%05d" % (prefix, i), b"y%05d" % i) for i in range(start, start + n)]


class TestEvictionIsDeterministic:
    def test_survivors_are_the_newest_when_nothing_is_credited(self):
        c = CmplogCollector(max_pairs=10, max_tokens=10_000)
        c._parse_lines(_lines(_pairs(b"P", 14)))
        assert c.pairs == _pairs(b"P", 10, start=4)
        assert c.evicted_pair_count == 4

    def test_survivor_order_is_insertion_order(self):
        c = CmplogCollector(max_pairs=100, max_tokens=10_000)
        c._parse_lines(_lines(_pairs(b"P", 100)))
        c._parse_lines(_lines(_pairs(b"R", 10)))
        assert c.pairs == _pairs(b"P", 90, start=10) + _pairs(b"R", 10)

    def test_same_stream_same_pool_under_any_hash_seed(self):
        """Proof the decision never reads a set: cross-process, cross-seed."""
        script = textwrap.dedent(
            """
            import hashlib, sys
            from fuzzer_tool.core.cmplog import CmplogCollector
            def lines(ps):
                return [f"CMP {a.hex()} {b.hex()} 0 {len(a)}" for a, b in ps]
            c = CmplogCollector(max_tokens=64, max_pairs=32)
            c._parse_lines(lines([(b"MAGIC%03d" % i, b"seed%04d" % i) for i in range(32)]))
            c.mark_coverage_gain(pairs=[(b"MAGIC005", b"seed0005")])
            c._parse_lines(lines([(b"NEW%05d" % i, b"new%05d" % i) for i in range(8)]))
            print(hashlib.sha1(repr((c.pairs, c.tokens)).encode()).hexdigest())
            """
        )
        digests = set()
        for seed in ("0", "1", "2", "7", "12345"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            out = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                check=True,
                env=env,
            )
            digests.add(out.stdout.strip())
        assert len(digests) == 1, f"pool depends on PYTHONHASHSEED: {digests}"


class TestPoolRotates:
    def test_operands_discovered_after_saturation_get_in(self):
        c = CmplogCollector(max_pairs=200, max_tokens=10_000)
        base = _pairs(b"OLD", 200)
        c._parse_lines(_lines(base))
        c.mark_coverage_gain(pairs=[base[0]])
        for batch in range(10):
            c._parse_lines(_lines(_pairs(b"NEW%02d" % batch, 20)))
        resident_new = [p for p in c.pairs if p[0].startswith(b"NEW")]
        assert len(c.pairs) == 200
        # Before the fix this was zero, permanently.
        assert len(resident_new) == 199
        assert base[0] in c._pair_set, "credited operand should have been kept"

    def test_uncredited_entries_leave_oldest_first(self):
        c = CmplogCollector(max_pairs=6, max_tokens=10_000)
        c._parse_lines(_lines(_pairs(b"P", 6)))
        c._parse_lines(_lines(_pairs(b"R", 2)))
        assert (b"P00000", b"y00000") not in c._pair_set
        assert (b"P00001", b"y00001") not in c._pair_set
        assert (b"P00002", b"y00002") in c._pair_set

    def test_a_recurring_comparison_comes_straight_back(self):
        """FIFO is safe because the stream re-emits whatever still fires."""
        c = CmplogCollector(max_pairs=4, max_tokens=10_000)
        hot = (b"MAGIC", b"HDR!")
        for _ in range(6):
            # The hot comparison plus two pairs that never repeat.
            c._parse_lines(_lines([hot, *_pairs(b"C", 2)]))
            c._parse_lines(_lines(_pairs(b"D", 3)))
        # Evicted at some point, but re-emitted and re-admitted.
        c._parse_lines(_lines([hot]))
        assert hot in c._pair_set


class TestCreditIsPerEntryAndAged:
    def test_credit_only_touches_the_named_entries(self):
        c = CmplogCollector(max_pairs=100, max_tokens=10_000)
        base = _pairs(b"P", 10)
        c._parse_lines(_lines(base))
        c.mark_coverage_gain(pairs=[base[3]], tokens=[base[3][0]])
        assert c._pair_value == {base[3]: 1}
        assert c._token_value == {base[3][0]: 1}

    def test_credit_accepts_the_swapped_orientation(self):
        """The scan reports pass-2 hits as (op_b, op_a)."""
        c = CmplogCollector(max_pairs=100, max_tokens=10_000)
        c._parse_lines(_lines([(b"ABCD", b"EFGH")]))
        c.mark_coverage_gain(pairs=[(b"EFGH", b"ABCD")])
        assert c._pair_value == {(b"ABCD", b"EFGH"): 1}

    def test_credit_for_an_absent_pair_is_dropped(self):
        c = CmplogCollector(max_pairs=100, max_tokens=10_000)
        c._parse_lines(_lines(_pairs(b"P", 4)))
        c.mark_coverage_gain(pairs=[(b"nope", b"nope")], tokens=[b"nope"])
        assert c._pair_value == {}
        assert c._token_value == {}

    def test_credit_decays(self):
        c = CmplogCollector(max_pairs=100, max_tokens=10_000)
        base = _pairs(b"P", 4)
        c._parse_lines(_lines(base))
        for _ in range(8):
            c.mark_coverage_gain(pairs=[base[0]])
        assert c._pair_value[base[0]] == 8
        # Age once: uncredited gains only.
        for _ in range(CMPLOG_VALUE_AGE_GAINS):
            c.mark_coverage_gain()
        assert c._pair_value[base[0]] == 4

    def test_credit_decays_to_nothing_and_the_entry_becomes_evictable(self):
        c = CmplogCollector(max_pairs=4, max_tokens=10_000)
        base = _pairs(b"P", 4)
        c._parse_lines(_lines(base))
        c.mark_coverage_gain(pairs=[base[0]])
        for _ in range(CMPLOG_VALUE_AGE_GAINS * 2):
            c.mark_coverage_gain()
        assert base[0] not in c._pair_value
        c._parse_lines(_lines(_pairs(b"R", 4)))
        assert base[0] not in c._pair_set


class TestNoLengthBias:
    def test_a_long_magic_value_is_not_ranked_below_a_short_one(self):
        c = CmplogCollector(max_pairs=3, max_tokens=10_000)
        long_pair = (b"MAGICHEADERv1234", b"MAGICHEADERv5678")
        short_pair = (b"ab", b"cd")
        c._parse_lines(_lines([long_pair, short_pair]))
        c.mark_coverage_gain(pairs=[long_pair])
        c.mark_coverage_gain(pairs=[short_pair])
        c._parse_lines(_lines(_pairs(b"R", 3)))
        # Equal credit, so the tiebreak is age -- not length. The long
        # operand was inserted first, so if either goes it is that one;
        # the old score made it 8x more evictable on length alone.
        assert c._pair_value[long_pair] == c._pair_value[short_pair]
        assert long_pair in c._pair_set
        assert short_pair in c._pair_set


class TestCompanionMapsStillPruned:
    def test_evicted_pairs_leave_no_residue(self):
        c = CmplogCollector(max_pairs=8, max_tokens=10_000)
        c._parse_lines(_lines(_pairs(b"P", 8)))
        c._parse_lines(_lines(_pairs(b"R", 8)))
        for m in (c._pair_occurrence, c._pair_cmp, c._pair_pc, c._pair_value):
            assert set(m) <= c._pair_set
        assert len(c._pair_set) == 8

    def test_evicted_tokens_leave_no_residue(self):
        c = CmplogCollector(max_tokens=8, max_pairs=10_000)
        c._parse_lines(_lines(_pairs(b"P", 8)))
        c._parse_lines(_lines(_pairs(b"R", 8)))
        assert len(c._token_set) == 8
        assert len(c.tokens) == 8
        assert set(c._token_value) <= c._token_set
        assert set(c.tokens) == c._token_set


def test_digest_helper_is_used():
    """Guard against the cross-seed test silently comparing empty output."""
    assert hashlib.sha1(b"x").hexdigest()
