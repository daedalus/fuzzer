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

## Fuzz entry

Default mode (stdin or file arg, >= 24 bytes) runs `check32((uint32_t)w)` and
`check64(v, w, p, q)`; any mismatch or UBSan error aborts. Hacks with documented
caveats / out-of-spec inputs ("expect", "caveat", "VIOLATED", "OUTSIDE", and the
broken `m&-((signed)(m-d)>>s)` modulus variant) are recorded but not fatal.
The checks themselves use unsigned arithmetic so the fuzzer finds wrong results,
not the known signed-overflow/shift UB (those are listed above and probed in
`exh` mode). The parallel-mod table loop is capped at 6 steps; the s=1 row
overrun is reported as an "expect fail" check instead of reading out of bounds.

## Limitations

- Clang trace-pc-guard run, 128 s: ~33k execs, 437 edges, no crashes. The logic bug
  above is fatal-exempt by design, so the fuzzer cannot "rediscover" it as a crash
  yet; making it a measurable rediscovery needs a per-hack fatal toggle.
- Coverage is shallow and there is no deep state, so it will not separate good
  fuzzers from mediocre ones; random inputs do about as well.
- The sweep covered ~2^26 samples, not the full 2^32 space.
- Bugs are in a reference web page, not shipped software.

## Fuzzer bug found while building this

`snoob_prev` (Hacker's Delight operator) used an O(x) linear search and hung the
mutator on 8-byte windows; fixed upstream in 9546a82c.
