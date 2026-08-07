# State-JSON saved as seeds + per-byte concolic model: two fuzzer OOM drivers

**Date:** 2026-08-07
**Context:** fuzzer-new; an `eps dropped to 4` investigation on a `png_read` concolic/cmplog command that turned out to also OOM the fuzzer.

## Problem
A heavy fuzz command (persistent `.so`, cmplog, `--mod-solving concolic`) slowed to ~4 eps and then OOM'd. The throughput fix (memoizing a failed per-iteration computation) only made the loop fast enough to *reach* the memory failure sooner; the memory problem itself was separate and pre-existing.

## Rejected
- **"The eps fix is what OOM'd it"** — plausible at first glance (failure surfaced right after the fix); dropped because that fix only raises executions/sec and allocates nothing; RSS had a real independent cause that needed its own diagnosis.
- **"The corpus root files are being loaded as seeds"** — checked first because of the `corpus/` vs `corpus/seeds/` complaint; `load_corpus` only scans subdirectories, so root `id_*` files are skipped. Part of the story, not the whole cause.

## Approach
Both bombs found by sampling the *live* process RSS (not just `ru_maxrss`) and isolating one variable per run:

1. **Serialized state saved as seeds.** `corpus/*/seeds/` contained 68 `id_*` files that were the fuzzer's own JSON state (a Bayesian/ELO `alpha` map keyed by seed hash), largest 82 MB. Loading them made the fuzzer fuzz MB-scale synthetic inputs → ~5 GB runaway. Quarantined into `corpus/<name>/pruned/state_junk/` (a directory `load_corpus` always excludes) — moved, not deleted, honoring the corpus rules. Also fixed `tools/corpus_png.py`, which wrote seeds flat into a corpus root that `load_corpus` never reads.

2. **Per-byte concolic z3 model.** `ConicTrace.solve()` builds one `z3.BitVec` per *input byte*; over multi-MB inputs that model alone transiently exceeds 1 GB (measured ~1.3 GB spikes). Capped at `_CONCOLIC_MAX_BYTES` (32 KiB): oversized inputs skip the whole-input solve; per-pair cmplog solving is unaffected.

## Key insight
A transient spike that resolves (~1–2 GB for ~a second) still OOMs a tight container even though `ru_maxrss` reports a big peak while live RSS looks fine. Root cause required A/B-isolating one flag at a time while sampling the live worker's `VmRSS` — `ru_maxrss` told us nothing about which subsystem spiked.

## Verification
Live `VmRSS` of the real worker across runs: 5 GB runaway (state-JSON loaded) → 1.3 GB transients (concolic and big inputs) → flat ~500 MB after cleanup + cap. Full suite 3611 passed; regression test shows the same trace solves a small input and returns None on an oversized one.

## Generalizes to
- A state serializer and the corpus loader must never share a path: anything that writes state must not write where a loader globs for inputs. Spot-check a seeds/ dir for huge non-format files before fuzzing.
- Any model sized per unit of attacker-controlled input (a z3 var per byte, a table entry per seed hash) is a memory bomb — cap it and skip beyond the cap; don't let "it eventually frees" count as green. Watch live RSS flatness, not just whether the process was killed.
