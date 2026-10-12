# Handover: `--shared-virgin` (2026-10-12)

Base: 0ce384af. Gate: `--shared-virgin`, multi-target only, off by default, excluded from `--hail-mary`.

## What
Per-target `ShmCoverage` tables (one SHM region and generation each) share one novelty store, `SharedVirgin` (`adapters/shm.py`): seen ids, masked ids, (edge, bucket) virgin map. An id or bucket seen through any target is old for all.

## Why not "sum the counts"
One target runs per exec, so nothing sums within an exec. Summing cumulative counts would invent buckets no exec produced. Novelty is the union of per-exec buckets.

## Kept per target
- `cumulative_edges` (`_local_ids`): `_inv_edge_weights` and `_cross_target_gap` weigh targets by what each reaches. Sharing it would read a target whose edges were found elsewhere as empty.
- Per-edge maxima (PerfFuzz): raw trip counts only compare within one binary.

## Identity caveat
Guard ids are `mix(salt ^ counter) & mask`, salt per module. Same id in two binaries is a chance collision: simulated 0.0% (2k guards), 0.2% (20k), ~1% (100k). Collisions are masked across targets. Real identity only when both harnesses load the same instrumented module.

## Mechanics
- `SharedVirgin.virgin` is allocated at `VIRGIN_DENSE_MAX` (16 MiB, lazily committed): a sharer that grew a private array by rebinding would detach.
- `ShmCoverage.share_virgin` folds existing state in (late join is lossless), rebinds the four containers by reference, sets `_local_ids`.
- Hot path unchanged when unshared (`_local_ids is None`). Shared adds one set difference per slow-path scan.
- `Fuzzer._enable_shared_virgin`: joins `_target_shm_covs` (plus `shm_cov` for targets that fell back to it); ignored with a message if not multi-target or fewer than two tables.
- `reject_phantoms` / `mask_edges` act on the shared store: masked for all targets.

## Tests
`tests/test_shared_virgin.py` (18): unshared control, union, buckets, masking, late join, per-target counts, phantoms, wide ids, diverged join, Fuzzer gate, CLI. Hail-mary gates test updated.

## Not done
- No A/B on a real multi-target campaign; effect for unrelated binaries is ~0 by construction.
- Seed calibration/minimization is still off in multi-target (fuzzer.py ~8554); a shared id space does not fix that, a seed's edge set is per target.
- Resume does not persist the shared store (the per-target virgin maps were not persisted either).
