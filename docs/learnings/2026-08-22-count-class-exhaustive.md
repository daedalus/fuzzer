# Exhaustive enumeration over `count_class`: the boundary bug under the boundary tests

Date: 2026-08-22

Added: `tests/test_count_class_exhaustive.py`.
Changed: `src/fuzzer_tool/core/count_class.py`, `tests/test_count_class.py`.

Port items P1-5 / P2-6 from the TigerBeetle "A Tale Of Four Fuzzers" survey
(merged into `docs/port-backlog.md`; both items are shipped, so only their open
siblings survive there), applied to the smallest subsystem in the repo that has
a fully enumerable input space.

## Why this module

Every entry point in `core/count_class.py` has a domain of at most 2**16:

| function | domain |
|---|---|
| `_classify_byte`, `classify_single`, `bucket_bit` | 256 |
| `_build_u16_table` | 65 536 |
| `classify_counts` | 256 values x 18 lengths |
| `new_bits` | 256 x 256 byte pairs, x offset |

It had 50 hand-picked tests and no enumeration. Walking the whole space costs
1.7s.

## What it found

`new_bits` gave **different answers for the same byte pair depending on where
the byte sat in the buffer**. It had a word loop for whole 8-byte chunks and a
byte loop for the remainder, and the two implemented different contracts:

```
trace = 0x02, virgin = 0x01
  at offset 0 of an 8-byte buffer  -> 2   (word loop: t & ~v != 0)
  as a 1-byte buffer               -> 1   (byte loop: t and v, so "overlap")
```

Buffer position is not an input to "does this trace contain new coverage", so
no answer may depend on one. Confirmed at every offset 0..15 and every length
1..17.

The eighteen existing `new_bits` tests all passed. Two of them were named
`test_8byte_boundary_overlap` and `test_8byte_new_coverage` — they exercised
the word loop with byte values on which the two paths happen to agree
(`0x01`/`0x01`, `0x01`/`0x00`). **The file's two most reassuring test names sat
directly on top of its only bug.** That is the argument for enumeration in one
line: the examples an author picks to test a boundary are drawn from the same
mental model that produced the boundary.

## The second defect, which the first was hiding

Fixing the disagreement forces the question of which path was right, and the
answer is neither, quite. The word loop returned 2 for any bit the map lacked,
conflating "new bucket on a known edge" with "edge never seen". The byte loop
returned **1 for mere overlap** — so a byte-identical replay against a map that
already contained it reported 1, i.e. a function named `new_bits` reported new
bits for an input contributing none. Four of the old tests pinned exactly that,
including `new_bits(bytes([1]), bytes([1])) == 1`.

The intended contract is AFL's `has_new_bits`, and the module already says so
in two places: the docstring's own Returns block (`2 = new edge — trace has
bits where virgin is 0`) and, more decisively, the `BUCKET_BIT_TABLE` comment,
which exists *because* a virgin map accumulates by OR and needs disjoint bits
per bucket. Under that representation `trace & ~virgin` is exactly "bucket bits
this run contributed", which makes the ladder unambiguous:

```
0  nothing the map does not already have
1  a known edge landed in a bucket the map had not recorded
2  an edge whose slot was still 0
```

`new_bits` has **no production caller** — it is exported from `core/__init__.py`
and used only by its tests — so this semantics change cannot regress a
campaign. That is also why it was safe to leave broken for so long, and why it
was worth fixing now rather than later: the first caller would have inherited
a novelty signal that fires on replays.

## Performance, since the implementation changed

Rewritten from the hand-rolled word loop to a numpy pass (numpy is already an
unconditional import in this module). 65 536-byte map, mean of 100–200 calls:

| case | old | new | |
|---|---|---|---|
| replay, nothing new (full scan) | 4067 us | 14.5 us | **281x** |
| new edge at byte 64 | 4.5 us | 31.6 us | 0.14x |
| new edge at byte 4096 | 237 us | 32.0 us | 7.4x |
| new edge at byte 32768 | 1875 us | 31.5 us | 60x |
| new edge at byte 65000 | 3755 us | 45.1 us | 83x |

The old code beats the new one only when a new edge appears in roughly the
first 550 bytes of the map, because it could return early and the numpy pass
cannot. A geometric-chunk version recovers that case (7.5 us) at the cost of
the full scan (10.6 -> 19.5 us); it was measured and rejected, because the case
it optimises is the rare one. In a campaign the overwhelmingly common outcome
is "nothing new", which is the full-scan column — and there the old
implementation would have capped a 64 KB-map fuzzer at ~250 exec/s on this
function alone.

## Docstring corrections found by the same sweep

Not bugs, but each one would mislead the next reader:

- The module claimed **8** classes and listed `32-127` as one. `_classify_byte`
  splits 32-63 from 64-127, giving **10**. `bucket_bit` genuinely merges them,
  to stay bit-identical to AFL's `count_class_lookup8`. The two ladders are
  *supposed* to differ; the docstring asserted the coarse one for both.
- `classify_single` listed nine possible outputs and omitted `64`, which is
  reachable from any count in 64..127.
- The module referenced a function `classify_and_new_bits` that does not exist
  anywhere in the tree.

Both ladders are now pinned separately and in both directions, so neither can
be quietly "fixed" into the other.

## On the tests that were deleted

`tests/test_count_class.py` went from 50 tests to 4. The 44 that went are
subsumed pointwise by the enumeration; what remains is the part enumeration
cannot express — *when* `LOOKUP_U16` is built (lazily, on first attribute
access) and what it costs (`array('H')`, not `list[int]`).

Keeping both would have been worse than keeping one. The sparse copy is the one
that fails first on an edit, and its failure message names one example instead
of the offending input.

Two oracle rules made the enumeration worth writing, and are stated at the top
of the new file so they survive the next edit:

1. **The oracle must be written differently from the implementation.** The
   implementation computes `1 << (val.bit_length() - 1)`; the oracle walks a
   literal ladder transcribed from AFL. An oracle sharing the implementation's
   formula agrees with it on the wrong inputs too, and proves only determinism.
2. **Where there are two paths, assert they agree across the whole domain.**
   Not at the points where they were known to agree. This is the rule that
   found the bug.

## Verification

- 74 new tests, 1.7s. Against the unfixed module, 34 of them fail; the
  classification and bucket-bit families pass unchanged, isolating the fault to
  `new_bits`.
- Full suite: 4848 passed, 180 skipped, 1 xfailed.
- `ruff check` and `ruff format --check` clean on all changed files.
