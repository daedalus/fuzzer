# A correct fix on a dead path is still an open bug (2026-08-23)

Three defects fixed this session. Two of them had already been *diagnosed*
correctly, in writing, in this repository — and were still live, because the
diagnosis and the fix landed somewhere nothing calls.

## The pattern

`afl_shim.c:786` has carried the anti-wrap table wipe since generation
tagging landed, complete with a comment measuring the ghost edges at
N = 256, 512, 768 and pricing the memset at ~0.34us amortised. Every word of
it is right. It is inside `__afl_map_reset()`, which has never had a caller —
and `inprocess.py:404` says so, in its own docstring, in a note left by the
header-clobber fix.

So the tree contained: the bug, the correct fix, the measurement justifying
it, and a second file observing that the function holding it is dead. The bug
was still live. `reset_edge_map()` in `shm.py` is the reset that actually
runs, and it bumped the tag without ever wiping.

Reproducing it took four lines:

```python
cov.record_edge(4242)
[n for n in range(1, 600)
   if (cov.reset_edge_map() or 4242 in cov.get_edge_ids())]   # -> [256, 512]
```

**Grep for callers, not for the fix.** "Is this handled?" and "does the
handling run?" are different questions, and the audits that produced
`docs/bugreport_2026-08-21_merged.md` answered the first one. A dead-path fix
is worse than no fix: it reads as done in every subsequent review.

## Corollary: fix where the live path is, not where the dead fix is

The obvious repair was to call `__afl_map_reset()` from Python. That would
have been wrong. The shim writes the front header from C, and the
header-clobber fix (`df06056`) had just established that the header has
exactly one writer. Wiring up a second C-side writer to fix a Python-side
reset would have reopened the bug that commit closed.

## Third instance of the assertion-that-pins-the-defect

`test_real_segment_round_trips` failed once `read_bitmap()` began returning
the edge table rather than the segment base — off by exactly 24 bytes,
because the test wrote its payload at offset 0 and asserted it came back from
offset 0. It was recording what the code did.

After `test_hex_escape` (grammar) and `test_different_stderr` (differential),
this is the third. The new wrinkle: it was added *by the fix for a different
bug in the same subsystem*, three commits earlier. Its sibling
`test_real_segment_zeroes_the_whole_table` uses the correct header+table
fixture, because `reset_bitmap()` was corrected at that time and
`read_bitmap()` was not. **When fixing one half of a layout bug, the tests you
write for that half will encode the other half's defect as a premise.**

The fixture was also too small for the layout it implied — `MAP_SIZE` bytes
for `MAP_SIZE` entries, an 8x shortfall — so a *correct* read of it runs off
the end of the segment. A test fixture sized against a buggy reader is a
second copy of the bug.

## On the inert bug

The `read_bitmap()` offset/length error and the caller's bytes-against-entries
bounds check are unobservable today: `coverage_env_id` comes from
`shm_cov.env_id`, so source and destination are the same segment and the
memmove is a self-copy. Nothing end to end can detect it, which is exactly
why the regression tests assert the offset and the length *directly*, and why
the last one drives a separate source segment. An inert bug needs tests
written against the invariant, not against behaviour — behaviour is correct
by accident, and will stay correct right up until it doesn't.

## Method note

The full suite ran green at 5170 passed / 169 skipped / 1 xfailed, but the
first run showed two failures, one of which
(`test_regression_default_timeout_method_spawns_no_thread`) was caused by
passing `--timeout-method=thread` on the command line. The test was doing its
job. Worth remembering before blaming a suite: check the invocation before
the code.
