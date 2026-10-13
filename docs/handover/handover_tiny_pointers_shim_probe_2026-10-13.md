# Handover: Tiny Pointers vs the afl_shim linear probe (2026-10-13)

Status: **analysis only. Nothing implemented, nothing measured, nothing committed.**
Repo: `github.com/daedalus/fuzzer`, HEAD `19f663e8` (docs: add afl_shim.c per-edge hot-path attribution).
Paper: Bender, Conway, Farach-Colton, Kuszmaul, Tagliavini, *Tiny Pointers*, arXiv:2111.12800v1 (24 Nov 2021).

## 1. TL;DR

The paper mostly does **not** apply to the shim's linear probe. The shim table is a keyed,
enumerable, effectively insert-only set, not a dereference table. Three narrower ideas from the
paper are still usable, in this order of value:

1. **Spill table for dropped edges** (paper Thm 1, primary/secondary table). Small, testable.
2. **Bucketed two-choice layout** (paper Lemma 2). Larger change, and it likely slows the hot path. Only if drops persist after (1).
3. **Quotienting the stored key** (paper Sec 6.6 filter idea). Speculative; needs an entry-size change to pay off.

Recommendation: measure the current drop rate on ffmpeg first, then prototype (1).

## 2. The code under analysis

All paths under `src/fuzzer_tool/adapters/`.

| Piece | Location | Notes |
|---|---|---|
| Probe window constant | `afl_shim.c:629-650` | `__AFL_PROBE_MAX` default 64, override `-D__AFL_PROBE_MAX=N` |
| Home slot | `afl_shim.c:1365-1370` | `((u32)(edge_id * 0x9E3779B1) * map_size) >> 32`, no divide |
| Probe/insert loop | `afl_shim.c:1392-1450` | `__afl_probe_insert(edge_id, pos, window, gen)` |
| Caller of the loop | `afl_shim.c:1500-1523` | `__afl_map_id_raw` clamps window to `map_size` |
| Drop counter | `afl_shim.c` `__afl_note_drop`, read via `ShmCoverage.read_dropped_edges()` | counts edges whose whole window was occupied |
| Python mirror | `shm.py` `home_slot`, `_scan_with_positions`, `_live_entries` | must stay bit-exact with the shim |
| Layout | `docs/refs/architecture.md` | 32-byte front region, entries of 8 B (4 B `edge_id`, 4 B `count` = 8-bit generation + 24-bit hit count), distance tail, optional touched bitmap; layout version 3 (`__afl_shm_layout_3`) |

Behaviour that matters for the analysis:

- **Empty slot = `edge_id == 0`.** A real id of 0 is remapped to 1 in `__afl_map_loc`.
- **Generations.** `__afl_map_reset` advances the tag. Stale entries for the *same* edge are reclaimed in place. A stale entry for a *different* edge is never freed, because dropping it would lose cumulative coverage. So occupancy is the union of distinct edges ever seen, and slots are never freed.
- **Bound is symmetric.** Insert and lookup use the same window. The shim comment says never bound one without the other: an edge is always within the window of its home slot or absent.
- **Drop is permanent.** An edge whose whole window is occupied is invisible to the fuzzer for good, only counted.

Measured numbers already recorded in the repo (not re-measured by me):

- Drop rate by window, for ffmpeg_read (201,279 distinct edges, 262,144-entry map, load 0.77): window 8 -> 4.43%, 16 -> 1.60%, 32 -> 0.40%, 64 -> 0.04% (comment at `afl_shim.c:629-650`).
- Every other instrumented target sits near 13% load and drops nothing at any of these windows.
- Microbenchmark (`docs/afl_shim_hotpath_analysis.md`, 8192 entries, 2000 distinct): probe 3.32 ns/edge, full `__afl_map_loc` 5.99 ns (ctx=1) and 4.67 ns (ctx=0).
- History: modulo homing clustered ctx variants and caused 51M drops at 28% global load on ffmpeg; the Fibonacci hash fixed it.

## 3. Why the paper mostly does not fit

1. **No owner stores a pointer.** A tiny pointer `p` is small because the *owner* of key `k` keeps it, and `(k, p)` together determine the slot. In the shim the instrumented binary recomputes `edge_id` on every execution and keeps nothing per edge. The fuzzer enumerates by scanning the table, so each slot must carry the full `edge_id`. The shim is a dictionary, not a dereference table.
2. **The hard dynamic case never occurs.** The paper's novelty is analysing delete/reinsert sequences where state depends on the probe sequence. The shim is effectively insert-only (see Sec 2), so classic linear-probing analysis applies.
3. **Probe cost is already low.** Textbook estimates under idealised uniform hashing (not measured on the real Fibonacci hash):
   - successful lookup about 0.5 * (1 + 1/(1-a)): a=0.77 -> about 2.7 probes; a=0.13 -> about 1.07.
   - unsuccessful about 0.5 * (1 + 1/(1-a)^2): a=0.77 -> about 10.
   The hot path is mostly successful lookups of existing edges. The paper's Sec 7.3 result (poly(1/delta) average probes) offers no gain over this.
4. **A constant window cannot reach zero failures.** For linear probing the longest probe run at constant load grows like Theta(log n) (known result; I did not re-derive it). So any fixed `__AFL_PROBE_MAX` leaves a nonzero drop rate by construction. This is consistent with the paper's fixed-pointer lower bound (Thm 4, Omega(log log log n + log 1/delta)). The 64-slot window is a 6-bit "fixed-size tiny pointer", and the 0.04% is the failure probability the paper handles with a secondary table.

## 4. Candidate 1: spill table for dropped edges (recommended first)

**Paper basis.** Thm 1: a primary load-balancing table that is allowed to fail on a small fraction of allocations, with failures caught by a small, much sparser secondary table.

**Design sketch.**

- Keep the primary probe unchanged. On window exhaustion (the `i == window - 1` path that calls `__afl_note_drop`), probe a small spill table instead of dropping.
- Size: on the order of 1% of the primary table (the paper uses delta/2 of n for the secondary; the real size should come from the measured drop count, plus headroom).
- Hash: a different multiplier than `__AFL_HOME_GOLDEN`, so a clustered neighbourhood in the primary does not cluster in the spill.
- Lookup order: primary window, then spill only when the primary window was fully scanned without a match or an empty slot.
- **Invariant to preserve:** an edge lives in the primary window or in the spill, never both. This holds only because slots are never freed in the primary. If anyone adds deletion or eviction, the invariant breaks. Document it next to the "do not bound one without the other" comment.
- `__afl_note_drop` then counts spill failures only. Add a separate "spilled" counter so the primary pressure stays observable.

**Touch points that must change together.**

- Layout version 3 -> 4 and a new `__afl_shm_layout_4` advertised symbol; the spill region goes after the existing tail. Old targets must be detected and warned about the way `__afl_ctx_relative_capable` is.
- `shm.py`: `_live_entries` / `_active_edge_ids` / `_scan` must include the spill region; `home_slot` mirror unchanged for the primary.
- `_scan_with_positions` and the edge-matrix tooling assume `position == home + displacement` with displacement in `[0, PROBE_MAX)`. Spill entries break that; either exclude them or give them positions offset by `map_size` and document it.
- Touched-slot bitmap: spill slots need bits (index `map_size + j`), or the bitmap scan will miss them and the fuzzer will silently lose those edges.
- Write guard: every spill store must go through `__afl_wguard_open()/__afl_wguard_close()`; `tests/test_shim_write_guard.py` runs every path in `mprotect` mode and will fault on a missed site.
- Resize and generation-wrap wipe must clear the spill too.

**Order of work.**

1. Measure first: run ffmpeg_read and read `read_dropped_edges()` to confirm the 0.04% (about 80 edges of 201,279) is real on current HEAD. If it is 0, stop.
2. Prototype in `tools/shim-micro-benchmark.c` (it `-include`s `afl_shim.c`) to check the common path is unchanged: the spill code should only run on window exhaustion.
3. Tests to run and extend: `tests/test_shim_map_slot.py`, `tests/test_shm_distance_channel.py`, `tests/test_shim_write_guard.py`, plus a new test that forces window exhaustion with a tiny `AFL_MAP_SIZE` and checks the spilled edge is visible and counted.

## 5. Candidate 2: bucketed two-choice layout (only if (1) is not enough)

**Paper basis.** Lemma 2: hash a key to two buckets, allocate in the one with more free slots (power of two choices), with bucket-local bitmaps.

**Why it is not the default recommendation.**

- Lookups of existing edges, the dominant operation, would have to check both buckets (two cache lines) where linear probing usually touches one. That likely costs more than the 3.3 ns/edge probe cost today. Unmeasured.
- Cache alignment: entries start at `SHM_TABLE_OFFSET` = 32 bytes after the segment base, so an 8-entry (64 B) bucket straddles two cache lines unless the front region is padded to 64. That is another layout change.
- It also requires the same Python-mirror, bitmap and layout-version work as Candidate 1, for every slot rather than a small side region.

It would allow a higher load factor than linear probing for the same drop rate, which is the only reason to consider it.

## 6. Candidate 3: quotienting the stored key (speculative)

**Paper basis.** Sec 6.6: the adaptive filter does not store the quotient of a fingerprint because the bin already encodes it.

**Observation.** `__AFL_HOME_GOLDEN` (0x9E3779B1) is odd, so `h = id * G` is a bijection on u32. For a power-of-two map of 2^k entries, `home = h >> (32 - k)`, i.e. the top k bits of `h`. A slot could store only the low `32 - k` bits of `h` plus a 6-bit displacement, and the id is recoverable as `h * G^-1 mod 2^32`. For k=18 that is 20 bits instead of 32.

**Caveats.**

- Only exact for power-of-two sizes; `AFL_MAP_SIZE` is currently any entry count.
- The saving is not realised unless the 8-byte entry shrinks (it holds a 4 B id plus an 8-bit generation and 24-bit count). That means redesigning `count`.
- The empty sentinel (`edge_id == 0`) would need a separate occupied bit.
- COMPCOV uses bit 31 of the id as a synthetic tag; the bijection preserves it, but all readers would need to apply the inverse.
- Benefit (cache footprint of the 2 MB ffmpeg table) is unmeasured. Lowest priority.

## 7. Other findings

- **Stale comments (code and docs disagree).** These still say `edge_id % map_size` / `% num_entries`, but the code and `docs/refs/architecture.md` use the Fibonacci hash:
  - `afl_shim.c:1314` (probe comment)
  - `shm.py:580` (`_scan_with_positions` docstring)
  - `edge_tracker.py:2910`
  Cheap, safe docs-only fix.
- **Other linear-probe sites in `afl_shim.c`** (not analysed in detail): around line 2171 (a key `% capacity` table with linear probing) and the cmp-site table around line 2795 (`__AFL_CMP_SITE_SLOTS` 4096, `__AFL_CMP_SITE_PROBES` 8, never grows or evicts, counts refusals). The cmp-site table is the same fail-and-count design and could take the same spill idea, but nothing suggests it has a measured problem.
- **Pre-existing note, not verified by me:** `docs/afl_shim_hotpath_analysis.md` says `__afl_probe_distance` uses `key % size` and `(pos+i) % size` and is O(size) worst case on a dense table, "never terminates" if every slot is filled. Worth checking against current code before relying on it.
- **Unverified lead:** the Python side keeps arrays sized from edge ids (`_virgin`, `_max_counts` grown to `1 << bit_length(eid)`). I did not check whether those are indexed by raw 32-bit id or by slot position. If by raw id, a compact slot index (a pointer-like handle) could shrink them, which is a more natural use of the paper than anything in the shim itself. Needs a read of `shm.py` around lines 683-712 and 1060.

## 8. Not done

- No benchmark or drop-rate measurement was run.
- No code was changed; no commit; no patch zip.
- The probe-length figures in Sec 3 are idealised formulas, not measurements.

## 9. Suggested next prompt

"Measure the drop rate on ffmpeg_read at HEAD, then implement the spill table (Candidate 1) with layout version 4."
