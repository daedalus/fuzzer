# Handover: Fractal Jittered Voronoi Integration into fuzzer-tool

Original: 2026-09-01. Pruned 2026-09-26 to open items; full original:
`git show 1c689e8a^:docs/handover/fractal-voronoi-integration.md`.
Verified against `4c021daa`.

Operator: `core/mutations/fractal_voronoi.py:FractalVoronoiMutator` (`structural`,
self-registers via `_register()`). Source algorithm: Boris the Brave, *Fractal
Jittered Voronoi Partitions*, 2026-08-29.

---

## 1. Sub-operator composition is not live

The premise — each Voronoi cell runs a different sub-operator, boundaries blend
them — is not what ships. `_register()` builds `FractalVoronoiMutator()` with no
`cell_ops`, so `mutate()` always takes the XOR fallback (`(root_hash + idx) % 5`).
The `cell_ops` branch is reached only in tests
(`tests/test_fractal_voronoi_plan_cache.py`), and even there applies the
sub-op to one byte at a time (the "A full integration would pass a sub-region"
comment in `mutate()`).

- **Do:** pass sub-operators at registration (bit/byte/block band), and hand
  each cell's byte span to its sub-op rather than single bytes.
- **Open:** registry has no `get_operators(bands)`; sub-ops are `OperatorEngine`
  `_op_*` handlers with `(buf, byte_idx, data)` signatures, not `bytes -> bytes`.
  Needs an adapter at the services layer (Hard Rule 36: no layer punching).
- **Accept:** registered instance has non-empty `cell_ops`; test shows two cells
  with different roots mutated by different sub-ops.

## 2. `mutate()` ignores `rng`

Output is a pure function of `data` (plan keyed on `(side, n, max_depth)`,
every choice from SHA-256 of root/boundary). Each seed yields exactly one
mutant; every re-selection re-executes the same input.

- **Do:** draw a per-call salt from `rng` / `context.rand_pool` (Hard Rule 16)
  and mix it into cell→op choice and apply mask. Keep geometry (`_plan`) cached.
- **Accept:** scripted RNG (`tests/support/scripted_rng.py`) drives two
  different salts → two different, exactly-asserted outputs; same salt →
  identical output.

## 3. A/B never run (pending §E4)

Shipped without measurement. No arm in `tools/lib/bench_paired.py`; no CLI flag
disables a single operator, so an off-arm needs one (or a registry-exclusion
seam) first.

- **Do:** paired run (`tools/benchmark.py paired`) on a structured target
  (`png_read`, `ffmpeg_read`), on vs off. Metric: new edges, Elo/op-stats share,
  exec/s. Do after §1–§2, or the result measures the XOR fallback.
- **Accept:** result recorded under `docs/sweeps/`.

## 4. Depth tuning (contingent on §3)

`max_depth=4` fixed. If §3 shows cost without coverage, try 3/5 or adaptive
`depth = min(4, int(log2(len(data)/4)))`.

## 5. Unbounded instance caches

`_boundary_cache` and `_root_hash_cache` are plain dicts on the instance; only
`_plan_cache` is an `LRUCache` (`_PLAN_CACHE_MAX = 16`). Tracked in
`docs/TODO.md` (2026-09-25). Bound both with `core/lru.py:LRUCache`
(Hard Rule 54).
