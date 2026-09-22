# Perlin-noise `lru_cache`-on-method leak (2026-09-22)

## Symptom
Long-running campaigns showed a sudden, steady climb in RSS (not
plateauing within tens of thousands of execs) with a matching decline in
eps. Real campaigns against `test_target` (plain gcc build, no shim)
confirmed it end to end: RSS 50MB -> 96MB over ~25k execs while corpus
size stayed flat at 3, eps falling 251 -> ~100.

## Method
- `tools/memray_wrapper.py` on a short campaign gave a high-watermark
  allocation summary but wasn't decisive by itself (a lot of the top
  entries -- `CrashMITracker`'s dense numpy arrays, the cmplog
  Aho-Corasick automaton -- turned out to be legitimate, already-bounded
  lazy growth toward documented caps, not leaks).
- A `tracemalloc`-snapshot probe (monkeypatching `Fuzzer.fuzz_one` to
  snapshot every N execs and diff consecutive snapshots) isolated which
  allocation sites grew in *every* window rather than plateauing.
  `core/mutations/perlin_noise.py` and `fractal_voronoi.py` stood out.
- `fractal_voronoi.py` had already been fixed for exactly this pattern in
  `6b3b7c3b` ("hold the geometry caches on the instance") -- an
  `lru_cache` on an instance method keys on `self`, so the cache pins
  every instance that ever populated an entry alive indefinitely (ruff
  `B019`). `ruff check --select B019 src/` found one remaining instance:
  `PerlinNoise1D._gradient`.

## Root cause
`PerlinNoiseMutator._noise_for` builds a fresh `PerlinNoise1D(seed=...)`
almost every call (the seed is content-hash XOR an rng draw, so it's
effectively unique per call) and keeps only the most recent 64 in its own
bounded `_noise_cache`. But `_gradient` was `@lru_cache(maxsize=8192)` on
the instance method, a *class-level* cache shared across every
`PerlinNoise1D` ever constructed, keyed on `(self, node)`. That cache
held up to 8192 entries' worth of `self` references alive regardless of
what `_noise_cache` had already dropped -- hundreds of otherwise-dead
instances pinned at once, continuously replaced as new seeds arrived.

Confirmed directly (`/home/claude/prove_leak.py`, not part of this
patch): 20,000 `mutate()` calls left **435** live `PerlinNoise1D`
instances before the fix (`_noise_cache` itself only holds 32) vs. **32**
after -- zero leaked.

## Fix
Same pattern as the `fractal_voronoi.py` fix: move the cached computation
to a module-level function keyed on `(seed, node)` instead of a bound
method keyed on `(self, node)`. Preserves the cache-hit behavior
(repeated lattice points within one noise field still hit) without
retaining the instance. `ruff check --select B019 src/` is clean
afterward (0 remaining). No behavior change: all 16
`test_perlin_noise_mutator.py` tests and all 12
`test_fractal_voronoi_mutator.py` tests pass unchanged.

## Note for future sessions
Grep/lint for this class of bug directly rather than re-deriving it from
a profiling session: `ruff check --select B019 src/` should stay clean.
Worth adding to whatever pre-commit/CI lint selection this repo runs, if
`B019` isn't already in it.
