# Handover: touched-slot bitmap ("pre-table") and bitmap-based stability calibration (2026-10-07)

## What was asked
A bit-per-edge pre-table in front of the edge table, then XOR of the current
bitmap against the last one plus popcount, to detect change.

## Findings (all measured, none assumed)
- **As a write-side filter it cannot help.** A hit still has to bump `count`,
  and different edges collide in a slot, so the entry must be read to compare
  `edge_id`. The table is also a persistent registry (a stale entry for a
  *different* edge is never reclaimed), so a clear bit does not mean the slot
  is claimable.
- **As a per-generation "slot went live" summary it helps the read side.**
  The Python scan walks the whole table (cost scales with map size). Decoding
  a bitmap instead: 0.7-0.8x at 8192 entries (slower), 2.4x/1.2x at 65536,
  8x/3x at 262144, 24x/10.6x at 1M (500 / 1800 live edges). Not wired into
  `_scan`: that would need the bitmap cleared by every reset path
  (`inprocess.reset_bitmap` memsets only the table), and a stale bit there is a
  wrong coverage read, not a missed optimisation.
- **XOR + popcount** over the bitmap: 2.5 / 3.6 / 7.7 / 23 us at 8192 / 65536 /
  262144 / 1M entries. `path_hash` already gives O(1) "did the path change";
  XOR adds the *magnitude* (Hamming distance), which is what stability
  calibration wants.

## What was built
Bitmap region appended after the distance tail (8-byte `TOUCHBM1` magic, then
one bit per slot). Allocated only when `--calibrate-stability` > 0. The shim
sets bits on first claim / stale reclaim (never on a hit) and announces the
region at attach. `_calibrate_seed_stability` uses `OR & ~AND` over per-run
snapshots (`unstable_slots`; XOR for two runs) and gathers the differing
slots' edge ids.

No `__AFL_SHM_LAYOUT` bump: nothing existing moved. A segment without the
region is byte-for-byte the old layout; an older shim never writes the magic
(`touched_supported` False -> edge-set path as before).

## Measured
- 3-run analysis (1800 live + 1200 stale edges): edge-set 1.4 ms vs bitmap
  26 us at 8192 entries; 8.8 ms vs 141 us at 1M. Most of the saving is not
  building Python sets of ~1800 ints per run, not the table scan.
- Shim per 500-edge exec: region absent -1.7% / -1.3% (noise); region present
  +0.9% (8192) / +3.9% (262144).
- End to end against the real shim: bitmap and edge-set verdicts identical on
  a flaky-tail-edge script; a deterministic seed masks nothing.

## Limits / not done
- Saves ~1.4-9 ms per accepted seed, which is a few percent of three target
  executions unless the target is very fast. Opt-in already (`--calibrate-stability`).
- Falls back to edge ids when a later run is under generation tag 0 (table
  wiped, slots may move); vetoes on dropped edges as before.
- Bit sets are plain read-modify-writes. A multithreaded target can lose a bit;
  lost in one run but present in the others, the slot reads as unstable (a
  FALSE positive, and masking is permanent). A bit lost in every run that has
  it only hides a real unstable edge. `-D__AFL_TOUCHED_ATOMIC=1` makes the set
  atomic: +16-24% shim time per exec with the region present (8192 / 262144
  entries) vs +1-4% plain, so it is a build switch, not the default.
- Not run against a real long campaign; no A/B of mask quality on a real target.
- Read-side `_scan` acceleration, Hamming-distance seed diversity and
  near-duplicate detection are possible follow-ups, unmeasured.

## Addendum: read side wired (`--touched-scan`)
- `_scan` now has a bitmap path behind `--touched-scan`. Same arrays, same
  order as the full scan; differential tests cover random tables (sizes 64 to
  65536, stale-generation registry entries, word/byte boundaries, padding bits)
  and the real shim across generations.
- Real shim-filled tables, scan alone (500 / 1800 live): 8192 -> 1.1x / 0.9x,
  65536 -> 2.5x / 1.3x, 262144 -> 7.2x / 3.3x, 1M -> 21x / 9x. Per-exec
  overhead: bit clear 0.9 us (8192) to 3.8 us (1M) plus ~1.2 us of atomic ORs.
  Net positive from 65536 entries; no gain at 8192, so it is opt-in, not auto.
- Contract: bits are cleared where entries go stale (`reset_edge_map`) and set
  only by the shim. A stale bit is filtered by the entry test; a live entry
  without a bit would be invisible, so any new writer of live entries must set
  the bit. The shim OR is atomic by default now for that reason.
- Not done: no A/B on a real long campaign (does the exec rate move end to
  end?), no auto-enable by map size, `_scan_with_positions` still walks the
  table.
