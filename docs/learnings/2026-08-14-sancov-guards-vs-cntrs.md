# The exact map-sizing path had never run

## What happened

`estimate_map_size()` documents a three-tier priority:

1. sancov guard count, when the counter section is present (**exact**)
2. `TargetProfile.total_branches`
3. branch-density estimation over `.text`

Tier 1 has never fired. Its parser, `parse_sancov_offsets()`, matches

```
__start___sancov_cntrs / __stop___sancov_cntrs
```

which is the section emitted by `-fsanitize-coverage=inline-8bit-counters`.
Every target in this tree is built with `-fsanitize-coverage=trace-pc-guard`,
which emits `__sancov_guards` and does not emit `__sancov_cntrs` at all. So
tier 1 returned `None` on every binary and every map size in the project's
history came from tier 3.

Checked directly, on the full built matrix:

```
nm targets/test_target | grep __start___sancov
__start___sancov_guards        # present
__start___sancov_cntrs         # absent, on all 16 targets
```

## Why it stayed hidden

The failure is silent by construction: tier 3 always returns *a* number, and
that number is plausible. There was no log line distinguishing "sized from 91
guards" from "sized from an estimated 3200 branches", and no test asserted the
tier — `tests/test_elf.py::TestParseSancovOffsets::test_real_binary` was
written to tolerate `None` ("May or may not find sancov symbols — just verify
no crash") against `targets/png_read_afl.so`, a filename that no longer exists.
A test that passes when the function under test returns nothing is not a test
of that function.

The second-order effect is worse than the sizing itself. The estimate ran 4–16x
high, and an over-sized map is exactly what makes the per-exec `memset` look
expensive — which is the entire premise of the edge-coverage analysis §2's
generation-tagged reset, a hot-path change to the shim plus every numpy reader
on the Python side. The defect was manufacturing the evidence for its own
follow-up work.

## Measured

| target | guards | tier 3 estimate | exact | reset/exec |
|--------|-------:|----------------:|------:|-----------:|
| test_target | 91 | 131,072 | 8,192 | 40.8 → 4.1 µs |
| gzip_read | 527 | 131,072 | 8,192 | 40.8 → 4.1 µs |
| png_read_nosan | 292 | 32,768 | 8,192 | 11.3 → 4.1 µs |
| proto_target_nosan | 94 | 16,384 | 8,192 | 6.9 → 4.1 µs |

The 36.7 µs/exec given back is invisible under fork+exec — measured 0.997x on
`test_target`, where a `posix_spawn` exec costs 7.5 ms — and is worth about 12%
of a 305 µs forkserver exec. Sizing fixes are worth what the exec they sit
inside is worth, which is a good argument for measuring them after the
forkserver rather than before.

## The census this unblocked

`apt-get install clang` now succeeds in the sandbox (18.1.3), so
`tools/build_targets.sh --clang-scov` builds a real trace-pc-guard matrix for
the first time. Note that instrumentation is opt-in: a plain
`tools/build_targets.sh` yields hand-placed `__afl_map_edge()` calls only, and
`test_target`/`proto_target` have none of those, so they report **zero** edges.
That is documented in README, but it is worth knowing before reading any
benchmark taken on a default build.

With `--clang-scov`, all 16 instrumented targets size to the floor (8192). The
largest is `gzip_read` at 527 guards. `MAP_SIZE_MAX` is 262144. Dropped edges
are zero everywhere; load factor ~13%; average probe depth ~1.07.

So both open halves of §2 — bounding the probe window, and the O(1) reset —
address costs that no target in this matrix pays. They are not wrong, they are
unmotivated, and the number that motivated them was an artifact.

The caveat that keeps §2 open rather than closed: the six library-backed
targets (ffmpeg, fgrep, secp256k1, lz4, jpeg, unrar) do not build here —
`vendor/ffmpeg` is an unbuilt source tree — and those are the ones whose guard
counts could plausibly reach the cap. Guard counts are also per-binary, not per
`edge_id`; a CTX build multiplies distinct ids by call-graph fan-in, and every
target measured was `ctx=0`.

## Generalisation

This is the third instance of the same shape in a week, after `__AFL_FORKSRV`
(gate landed, nothing ever set the variable) and `favored` (threaded through
every scorer, never passed). A code path that cannot report whether it ran will
eventually not run. The cheap defences, in order of cost:

- Log the tier, not just the result. `estimate_map_size()` now emits the block
  count, the source, whether that source is exact or estimated, and whether the
  cap bound, on every call.
- Return the tier, so it can be asserted rather than grepped.
  `estimate_map_size_detail()` gives a `MapSizeEstimate(entries, blocks,
  source, ctx_bits, capped)` with an `.exact` property;
  `TestEstimateMapSizeProvenance` asserts `source == "sancov_guards"` on the
  built matrix, which fails if sizing silently reverts to estimation.
- Never write a test whose assertions are all inside `if result is not None`.

The general form: a fallback chain needs to name the link it stopped at. Every
tier here returns an `int`, so the type system cannot distinguish 91 blocks read
out of a section from 3200 blocks inferred from instruction density, and neither
can any caller. Making provenance part of the return value is what turns a
silent degradation into a visible one.
