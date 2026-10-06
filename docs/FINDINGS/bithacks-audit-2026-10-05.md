# Bit Twiddling Hacks audit (graphics.stanford.edu/~seander/bithacks.html)

Harness: `targets/bithacks_diff.c`, built by `tools/build_targets.sh` (executable target `bithacks_diff`).
Coverage: 110 hacks vs reference; ~2^26 sequential/descending/hashed 32-bit inputs,
~2^26 random 64-bit, 8/16-bit generic popcount and 3-op sign-extend exhaustive.
Not a full 2^32 sweep.

## Logic bug
- Mod by (1<<s)-1, "less portable" `m = m & -((signed)(m - d) >> s)`: wrong for all s>=2
  (yields m&1). Correct: `m & ((signed)(m-d) >> 31)` or `m == d ? 0 : m`.

## Undefined behaviour / out of bounds
- `BitReverseTable256[..] << 24` and `MortonTable256[..] << 17`: int-promotion overflow.
- Parallel ctz `-signed(v)`: INT_MIN negation (and C++-only syntax).
- abs / cond-negate / min-max quick&dirty overflow at INT_MIN/INT_MAX.
- rank with pos=0 shifts by 64; var sign-extend with b=32 shifts by 32.
- Parallel mod tables have 6 entries/row; s=1 reads past the row, s=3 uses all 6 steps.
- Float type-punning violates strict aliasing.
Everything else matched the reference.

## Purpose of the target

`bithacks_diff` is a ground-truth benchmark and differential oracle, not a
bug-finding target for new software.

Useful for:
- **Differential oracle.** Detects wrong results, which crash-only fuzzing cannot see
  (the modulus-formula bug above is only visible this way).
- **Known answers.** Verified bugs and UB give a fixed set to measure schedulers and
  mutation operators against (rediscovery rate / time to find).
- **Arithmetic-heavy code.** Magic constants, multiply-and-mask tricks and de Bruijn
  lookups exercise comparison tracing and the Hacker's Delight operators
  (see `tests/test_regression_bithacks.py`, `tests/test_regression_hackers_delight.py`).
- **Cheap and deterministic.** No dependencies, fast execs, suited to A/B runs of fuzzer changes.

## Limitations (as committed)

- The default fuzz entry runs only the 64-bit/multi-argument checks (`check64`).
  `check32` is excluded because it trips known UB immediately, so the logic bug
  above is NOT reachable by the fuzzer; it was found by the sequential sweep
  (`bithacks_diff exh`) plus UBSan.
- A 120 s clang run (trace-pc-guard) found 74 edges and no crashes. Coverage is
  shallow and there is no deep state, so it will not separate good fuzzers from
  mediocre ones; random inputs do about as well.
- The sweep covered ~2^26 samples, not the full 2^32 space.
- Bugs are in a reference web page, not shipped software.

## Planned improvement

Add `check32` to the fuzz entry behind a skip-list for known-UB hacks, so the
modulus bug becomes a rediscovery test with a measurable time to find.
