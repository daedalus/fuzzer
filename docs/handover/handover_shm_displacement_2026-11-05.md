# SHM edge-table placement: hashing math, closed forms, displacement diagnostic

**Status (2026-11-05, PR #290):** analysis + one diagnostic shipped
(`Probe displacement` line in `--report`). No shim change, no layout bump.
Verdict on fuzzgoat: placement is not a bottleneck (99.1% of edges in their
home slot). ffmpeg unmeasured.

## Trigger

"How far and how close at most two edges land in the SHM given the hashing
math?" → "What can we optimize?" → "Closed form in M?" → "Build the
displacement histogram, run it on fuzzgoat."

## The math (`afl_shim.c`)

```
edge_id = ctx(8 bits) ^ (prev_loc >> 1) ^ cur_loc            # __afl_edge_hash_as
home    = ((u32)(edge_id * 0x9E3779B1) * M) >> 32            # __afl_home_slot, M = __afl_map_size
slot    = first free/own slot in [home, home+63], wrapping   # __AFL_PROBE_MAX = 64, else drop
byte    = 32 + slot * 8                                      # SHM_TABLE_OFFSET, 8-byte entry
```

- `cur_loc` is a splitmix64 hash of the guard index, masked to
  `ceil(log2 N) + 10` bits, max 24 (`__afl_guard_width`).
- Stale entries (old generation, other edge) block probes. Occupancy = union
  of distinct edges since the last wipe (every 256 resets, `reset_edge_map`).
- Python mirror: `adapters/shm.py:home_slot`, `record_edge`.

## Findings

### F1. Bounds, arbitrary edge pairs

| | value |
|---|---|
| min home distance | 0. `x·C mod 2^32` is a bijection, but a home slot spans 2^32/M hash values: ~2^24/M ids share each (2048 at M=8192) |
| min stored distance | 1 slot (8 B). Same-home edges stack up to 63 slots past home |
| max, linear | M−1 slots, (M−1)·8 B (65,528 B at M=8192) |
| max, circular (probe wraps) | ⌊M/2⌋ |
| stored vs home | ±63 slots |

### F2. Closed form, context variants (ids differing in the low 8 bits)

α = 0x9E3779B1/2^32 ≈ 1/φ (agrees to ~10 decimal places). Homes are
‖d·α‖·M apart (‖·‖ = distance to the nearest integer). By
|F_n/φ − F_{n−1}| = φ^−n:

- min: d = 233 = F₁₃ (largest Fibonacci ≤ 255)
- max: d = 72 (2d = 144 = F₁₂, so d·α is closest to a half)

Exact form, verified by brute force at M ∈ {256 … 2^20}, powers of two and not:

```
min(M) = ⌊ 8241689 · M / 2^32 ⌋                    8241689    = 233·C mod 2^32   (≈ M/521.0)
max(M) = ⌈ min(x, M−x) ⌉, x = 2140813768·M/2^32    2140813768 = 72·C mod 2^32    (≈ 0.49845·M)
```

| M | min | max | ids one apart |
|---|---|---|---|
| 8192 | 15 | 4084 | 3129–3130 |
| 65536 | 125 | 32667 | 25032–25033 |
| 131072 | 251 | 65333 | 50065–50066 |

Floor/ceil come from the home slot rounding down (±1 slot).

General ctx width b, K = 2^b − 1:
- min ≈ M·φ^−n, F_n the largest Fibonacci ≤ K
- max ≈ (M/2)(1 − φ^−n′), F_n′ the largest even Fibonacci ≤ 2K (n′ ≡ 0 mod 3)

Only b = 8 is verified for max.

Consequences:
- Two variants can share a home when M < φ¹³ ≈ 521.
- Variant windows can overlap the 64-slot probe below M ≈ 64·521 ≈ 33K.
- This only matters if those slots are occupied, and at measured loads they aren't.

### F3. Measured displacement, fuzzgoat (ASAN `.so`, direct_lite, 50k execs)

| map | claims | home | max | mean | peak load | theory mean ½(1/(1−α) − 1) |
|---|---|---|---|---|---|---|
| auto (32,768 B) | 86,745 | 99.1% | 1 | 0.01 | 1.63% | 0.008 |
| `--map-size 1024` | 61,113 | 84.9% | 7 | 0.21 | 38.4% | ≤ 0.31 at peak |

Matches Knuth's linear-probing successful-search cost. The aggregate is below
the peak-load prediction because it averages over windows with lower load.

### F4. Already tried or already cheap (do not re-propose)

- **Stepped probe (MSI):** LOSS / NULL, AB_LEDGER #86–92.
- **Per-exec memset:** replaced by generation tags; the table is wiped once per 256 resets.
- **Touched-slot bitmap scan:** shipped (`--touched-scan`).
- **Dropping the Fibonacci multiply:** saves 1 imul/edge. A modulo homed all 256 ctx variants in one block and cost ffmpeg 51M drops at 28% load.
- **More ctx-variant spread:** irrelevant at 0.5–2% load.

## Diagnostic shipped

```
src/fuzzer_tool/adapters/shm.py
  displacements(slots, ids, n)        vectorized (slot - home) mod n
  ShmCoverage._table_displacement()   histogram + load of the current table
  ShmCoverage._fold_displacement()    called from _notify_table_loss (wrap, resize)
  ShmCoverage.displacement_histogram() -> (hist[PROBE_MAX+1], peak_load)
src/fuzzer_tool/services/report.py
  _displacement_lines(cov)            one line in Coverage Analysis
tests/test_shm_displacement.py        10 tests (falsification: empty table; adversarial: foreign entry -> overflow bin)
```

Wiring:

```
reset_edge_map (gen wraps to 0) ─┐
resize ──────────────────────────┴─> _notify_table_loss ─> _fold_displacement ─> _disp_hist += ...
                                                        └─> on_table_loss (edge_tracker.fold_tags), unchanged
--report ─> generate_report ─> _coverage_analysis ─> _displacement_lines ─> displacement_histogram()
```

Cost (min-of-N; fold once per 256 resets):

| table | fold | per exec |
|---|---|---|
| 8,192, sparse | 24 us | 0.09 us |
| 65,536, 2k claimed | 155 us | 0.6 us |
| 262,144, saturated | 5.6 ms | ~22 us |

Vectorization: a u32 multiply + sign fix-up instead of `% n` is 16% faster
saturated and equal on sparse tables. A bincount tail fold and a strided u32
view were each slower on sparse tables: rejected (Hard Rule 58).

## Plans, ranked

### P1. Run on ffmpeg_read (do first)

It's the only target with recorded drops; it wasn't built in this container.

```
tools/vendor_ffmpeg.sh && tools/build_targets.sh --asan
fuzzer-tool fuzz ~/fuzzing/builds/ffmpeg_read_<ver>_asan.so -d ~/fuzzing/corpus/ffmpeg --inprocess-direct \
    --max-execs 20000 --report ~/fuzzing/ffmpeg_disp.txt
```

Decision rule:
- **Home ≥ 95% and max < 32:** placement is closed; stop.
- **Max near 63 or drops > 0:** go to P2.

### P2. Size the map up front (only if P1 shows pressure)

`recommended_map_size()` only reacts after drops. Guard count and ctx bits are
known before the first exec (`elf.detect_ctx_bits`, guard total), so pick
`M = next_pow2(E_est / α_target)` with α_target ≈ 0.5. Check
`elf.estimate_map_size` first; it may already do this.

```python
# services/fuzzer.py, setup (ifs up: once per run)
est = estimate_live_edges(guards, ctx_bits)          # core/elf.py
size = max(size, next_pow2(int(est / LOAD_TARGET)))  # LOAD_TARGET = 0.5
```

Wiring: `services/fuzzer.py` setup → `ShmCoverage(size)` → env
`AFL_MAP_SIZE` → shim. Validate with the displacement line (target max < 16).

### P3. 4-byte entries (only if a hotpath profile shows table writes)

At 8 bytes the default table is 64 KB, more than the 48 KB L1d, and edges land
on random cache lines. Halving the entry needs an `__AFL_SHM_LAYOUT` bump and
a rebuild of every target (Hard Rules 53, 56), and the counts must still fit.
Warm microbenchmarks overstate gains here (17–20% micro vs 0–1% in-run).
Needs an in-run A/B on fuzzgoat plus ffmpeg, with an A/A control, logged in
AB_LEDGER.

### Small follow-ups

- At each wrap, `_fold_displacement` and `entry_tags` both read the full
  table. Share one `ids` view if the wrap cost shows in a profile.
- The histogram is not persisted across `--resume` (use the pickle machinery,
  Hard Rule 15, if needed).

## Environment notes (authoring container)

- `libclang-rt-18-dev` was missing: no ASAN runtime, and the build silently
  skipped ASAN variants. Installed. This also fixed 5 errors in
  `test_shm_distance_channel.py` (missing ubsan runtime).
- `impactguard` 0.1.21 crashes on import without `tree-sitter`
  (`_log` undefined in `languages/lib/shared.py`). Install `tree-sitter`;
  worth reporting upstream.
- AGENTS.md says the pre-commit hook runs pytest; `.pre-commit-config.yaml` doesn't.
- `ruff format src/ tests/` touches 5 unrelated files, and `ruff check` flags
  I001 in `test_os_net_scheduler_wiring.py` and `test_shim_msi_probe.py`.
  Both are pre-existing on master.

## Reproduce F2

```python
C, U = 0x9E3779B1, (1 << 32) - 1
home = lambda i, M: (((i * C) & U) * M) >> 32
def ctx_bounds(M):
    lo, hi = M, 0
    for base in range(1 << 12):
        a = base << 8
        for c in range(1, 256):
            d = abs(home(a ^ c, M) - home(a, M)); d = min(d, M - d)
            lo, hi = min(lo, d), max(hi, d)
    return lo, hi
```
