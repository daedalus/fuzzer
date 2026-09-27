# Handover: Fractal Jittered Voronoi Integration into fuzzer-tool

Original: 2026-09-01. Pruned 2026-09-26 to open items; full original:
`git show 1c689e8a^:docs/handover/fractal-voronoi-integration.md`.
Verified against `4c021daa`.

Operator: `core/mutations/fractal_voronoi.py:FractalVoronoiMutator` (`structural`,
self-registers via `_register()`). Source algorithm: Boris the Brave, *Fractal
Jittered Voronoi Partitions*, 2026-08-29.

---

## 1. Sub-operators act per byte, not per cell span

**Wiring fixed 2026-09-27:** `_register()` passes `DEFAULT_CELL_OPS` (six
single-byte bijections: invert, ±1, nibble swap, MSB flip, rotate); the bare
constructor keeps the XOR fallback. Test:
`tests/test_regression_fractal_voronoi_wiring.py`.

Still open: each sub-op sees one byte (`op(bytes([data[idx]]))`), not the cell's
byte span, and the ops are core-local rather than the engine's `_op_*`
handlers (those take `(buf, byte_idx, data)`; an adapter belongs in the
services layer, Hard Rule 36).

- **Do:** hand each cell's gathered bytes to its sub-op, scatter back
  length-preserved; optionally adapt engine handlers via services.
- **Accept:** test shows a multi-byte op (e.g. reverse) applied to a whole cell.

## 2. `mutate()` ignored `rng` — fixed 2026-09-27

One `rng.randint(0, 0xFFFFFFFF)` salt per call XORs into every hash-driven
choice (op, byte gate, XOR value, boundary jitter); `_plan` geometry stays
cached. `rng=None` or salt 0 reproduces the legacy output. Tested with
`ScriptedRng` in `tests/test_regression_fractal_voronoi_wiring.py`.

## 3. A/B never run (pending §E4)

Shipped without measurement. No arm in `tools/lib/bench_paired.py`; no CLI flag
disables a single operator, so an off-arm needs one (or a registry-exclusion
seam) first.

- **Do:** paired run (`tools/benchmark.py paired`) on a structured target
  (`png_read`, `ffmpeg_read`), on vs off. Metric: new edges, Elo/op-stats share,
  exec/s. Wiring and salt (§1–§2) landed 2026-09-27; measure on or after that.
- **Accept:** result recorded under `docs/sweeps/`.

## 4. Depth tuning (contingent on §3)

`max_depth=4` fixed. If §3 shows cost without coverage, try 3/5 or adaptive
`depth = min(4, int(log2(len(data)/4)))`.

## 5. Unbounded instance caches

`_boundary_cache` and `_root_hash_cache` are plain dicts on the instance; only
`_plan_cache` is an `LRUCache` (`_PLAN_CACHE_MAX = 16`). Tracked in
`docs/TODO.md` (2026-09-25). Bound both with `core/lru.py:LRUCache`
(Hard Rule 54).
