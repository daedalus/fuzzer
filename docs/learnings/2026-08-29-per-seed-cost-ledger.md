# Per-seed execution cost is not clustered — the cost ledger carries signal

**Date.** 2026-08-29. **Base.** `14054a5`. **Why.**
`docs/handover/handover_persistence_mechanics_2026-08-29.md` §1 gated all three
of its proposed changes on one measurement: *is `total_time / fuzz_count`
tightly clustered across a real corpus?* If yes, the count criterion and the
cost criterion agree, none of §1a/1b/1c is worth writing, and the handover gets
deleted. This is that measurement.

## Method

Real campaigns, 20,000 executions each, against targets built by
`tools/build_targets.sh --fast`, seeded corpora from `tools/corpus_png.py` and
a generated zlib/gzip corpus spanning compression ratios. Fixed seed `-s 42`.
The per-seed ledger was dumped at end of run and the distribution of
`total_time / cost_samples` taken across every seed that was fuzzed at least
once.

Two questions, not one. The spread of mean execution cost is the headline, but
the decision-relevant quantity is whether the count criterion and the cost
criterion *select different seeds*. A distribution can be visibly spread and
still induce the same ordering.

## Result

| target | p10 / p50 / p90 mean exec_us | p90/p10 | max/min | CV |
|---|---|---|---|---|
| `png_read.so` `-m 65536` | 105.1 / 212.4 / 454.0 | 4.32x | 48.0x | 1.455 |
| `png_read.so` `-m 4096` | 105.5 / 141.7 / 211.4 | 2.00x | 6.51x | 0.387 |
| `gzip_read.so` `-m 65536` | 64.2 / 72.0 / 84.7 | 1.32x | 1.60x | 0.106 |

Criterion disagreement, comparing `fuzz_count >= 50` against the equal-sized
top-N-by-`total_time` set drawn from the same barren pool:

| target | count-stale seeds | disagree | Jaccard | Kendall tau(fuzz_count, total_time) |
|---|---|---|---|---|
| `png_read.so` `-m 65536` | 13 | 10 | 0.130 | 0.458 |
| `png_read.so` `-m 4096` | 145 | 44 | 0.534 | 0.518 |
| `gzip_read.so` | 14 | 0 | 1.000 | 0.830 |

On `png_read` at a realistic `max_len`, the seeds the count criterion flagged
as exhausted had burned between **6.9 ms and 116.9 ms** of target time for the
same verdict — a 16.9x spread in budget behind one word.

## Reading

**Not clustered. §1 survives, and the handover's guess about which target was
backwards.** It predicted spread on `ffmpeg_read` and possible clustering on
`png_read`. `png_read` is the spread one: mutation varies image dimensions and
filter type, so decode cost varies with the payload. The flat decompressor is
the clustered one, because at these input sizes its cost is dominated by
process and harness overhead rather than by anything the input controls.

The generalisable form is not "png is spread and gzip is not" but: **cost
disperses where the input controls how much work the target does, and
concentrates where a fixed overhead dominates.** That is a property of the
target and the corpus jointly, not of either alone — the same target moved from
2.00x to 4.32x purely by raising `max_len`, because the cap was truncating away
the expensive inputs.

This is why the consumers were written against `effective_fuzz_count`, which
expresses the ledger in average-cost executions rather than in seconds: under
uniform cost it equals the sample count exactly, so on a `gzip_read`-shaped
target the change is arithmetically a no-op instead of a behaviour change that
happens to be small. The falsification condition lives in the code, not in a
comment that has to be re-checked per target.

## Two defects the measurement turned up

**The ledger did not survive resume.** The handover asserted it did; it does
not. `total_time` was absent from `CorpusManager.save_state`/`load_state` while
`fuzz_count` was persisted. Measured on `png_read`: after a 200-execution
resume, **116 of 216 fuzzed seeds carried `total_time == 0.0` against 407
restored fuzzes**. All three readers divide by `fuzz_count` and floor at 1
microsecond, so every restored seed read as the cheapest seed in the corpus,
and the bias did not wash out — the numerator restarts at zero while the count
does not.

**`fuzz_count` was the wrong denominator even within a single run.** The
initial seed replay in `Fuzzer.run` increments the count with no time credited.
Measured on the same campaign, `cost_samples` and `fuzz_count` disagreed on
**126 of 147** timed seeds. The error is small per seed and systematic in one
direction.

Both are fixed by `core/cost_ledger.py`, which makes "unmeasured" and
"measured as free" distinguishable — they were the same value before, because a
zero numerator and the floor met at 1 microsecond.

## Reproducing

The harness is not committed: it monkeypatches `Fuzzer.run` to dump
`seed_meta` and is three dozen lines. The measurement is cheap to redo from
scratch and expensive to keep working against a moving `Fuzzer.__init__`. What
matters for re-deriving it is the shape: fix the seed, hold execution count
constant, dump `total_time` and `cost_samples` per seed at end of run, and
compare *selected sets*, not just distribution width.
