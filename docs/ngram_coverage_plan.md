# n-gram Edge Coverage — Implementation Plan

> Research basis: analysis of `src/fuzzer_tool/adapters/afl_shim.c` (1668 lines),
> `src/fuzzer_tool/adapters/shm.py`, `src/fuzzer_tool/core/elf.py`,
> `src/fuzzer_tool/services/ptrace_coverage.py`, and
> `src/fuzzer_tool/services/fuzzer.py`.

---

## Background

### Current edge hash (k = 1)

```
edge_id = caller_ctx ^ __afl_prev_loc ^ cur_loc   [__AFL_CTX_SENSITIVE=1, default]
edge_id = __afl_prev_loc ^ cur_loc                [__AFL_CTX_SENSITIVE=0]
```

`__afl_prev_loc` (`afl_shim.c:283`) is a single `uint32_t`.  It holds
`cur_loc >> 1` from the immediately preceding basic block (`afl_shim.c:668`).
This is a **2-node path** (predecessor + current), which is blind to paths
that share the same final hop but differ in their earlier history.

An **n-gram** (k-gram) extends this to a **k-node path**: the hash encodes
the last k−1 predecessors plus the current block, giving much finer path
discrimination at the cost of increased edge-ID cardinality.

---

## Data-Structure Changes (`afl_shim.c`)

### Replace `__afl_prev_loc` with a ring buffer

```c
// Before (afl_shim.c:283)
uint32_t __afl_prev_loc = 0;

// After
#ifndef __AFL_NGRAM_K
#define __AFL_NGRAM_K 2     /* default: retain current 2-node behaviour */
#endif

#if __AFL_NGRAM_K > 1
static uint32_t __afl_prev_locs[__AFL_NGRAM_K - 1];
static uint8_t  __afl_prev_idx  = 0;
#else
uint32_t __afl_prev_loc = 0;   /* k=1: unchanged ABI */
#endif
```

- `__AFL_NGRAM_K = 2` (default) keeps the existing single-word layout so no
  existing binaries need recompilation.
- The ring holds exactly k−1 entries; index wraps modulo k−1.

### Symbol advertisement

```c
/* afl_shim.c, near line 272 (alongside __afl_ctx_bits_N) */
const uint32_t __AFL_CAT(__afl_ngram_k_, __AFL_NGRAM_K) = __AFL_NGRAM_K;
```

Python detects k with the same scan used for context bits (`elf.py:1608`).

---

## Edge-ID Computation (`afl_shim.c:578–668`)

```c
static inline void __afl_map_edge(uint32_t cur_loc) {
#if __AFL_NGRAM_K > 1
    /* FNV-1a mix over the ring + cur_loc */
    uint32_t h = 2166136261u;
    for (int i = 0; i < __AFL_NGRAM_K - 1; i++) {
        uint32_t slot = __afl_prev_locs[(__afl_prev_idx + i) % (__AFL_NGRAM_K - 1)];
        h ^= slot;
        h *= 16777619u;
    }
    h ^= cur_loc;
    h *= 16777619u;
#if __AFL_CTX_SENSITIVE
    uint32_t edge_id = __afl_get_caller_ctx() ^ h;
#else
    uint32_t edge_id = h;
#endif
#else
    /* k=2 / original path — unchanged */
    ...
#endif
    edge_id |= 1;   /* preserve empty-slot sentinel */
    ...
    /* Advance ring */
#if __AFL_NGRAM_K > 1
    __afl_prev_locs[__afl_prev_idx] = cur_loc >> 1;
    __afl_prev_idx = (__afl_prev_idx + 1) % (__AFL_NGRAM_K - 1);
#else
    __afl_prev_loc = cur_loc >> 1;
#endif
}
```

The FNV-1a mix is cheap (2 ops per slot), order-sensitive (path direction
matters), and has good avalanche — better than XOR-chain for k > 2 where
XOR becomes commutative and loses direction information.

---

## Reset Path (`afl_shim.c:848–853`)

```c
/* __afl_map_reset, currently at afl_shim.c:850 */
#if __AFL_NGRAM_K > 1
    memset(__afl_prev_locs, 0, sizeof(__afl_prev_locs));
    __afl_prev_idx = 0;
#else
    __afl_prev_loc = 0;
#endif
```

Must zero all k−1 slots and reset the index. Missing a slot contaminates
the first edge of the next iteration — the exact aliasing bug the
generation scheme was designed to prevent.

The fork-server pre-fork path (comment at `afl_shim.c:1552–1559`) needs
no change here. It doesn't zero `__afl_prev_loc` before forking — it
deliberately leaves it untouched: `__afl_area` is set `NULL`, so
`__afl_map_edge`'s null-check returns before `__afl_prev_loc` (or the
ring, once added) is ever read or written, and every child forks from
identical coverage state by construction. Since the ring is a
zero-initialized static array that this loop never writes to, it stays
zero across all fork iterations with no extra code required.

---

## Python-Side Changes

### `elf.py` — detection and map-size estimation

1. **`detect_ngram_k(target: str) -> int`** (new, mirrors `detect_ctx_bits`
   at `elf.py:1587`): scan the ELF symtab for `__afl_ngram_k_N`; return N.
   Return 2 (current default) when the symbol is absent.

2. **`ngram_inflation_factor(k: int) -> float`**: k-gram cardinality grows
   roughly as `E^(k−1)` for a target with E distinct edges.  A conservative
   cap (matching `ctx_inflation_factor`'s `min(2^(k/2), 16.0)` shape):
   ```python
   def ngram_inflation_factor(k: int) -> float:
       if k <= 2:
           return 1.0
       return min(float(k - 1) ** 2.0, 32.0)
   ```

3. **`_size_from_blocks` (`elf.py:1640`)**: multiply by both
   `ctx_inflation_factor` and `ngram_inflation_factor` when sizing the map.
   Without this correction the default 8192-entry map saturates early
   (drop-rate exceeds 1 % at load > 0.75, see `afl_shim.c:336–339`).

4. **`MapSizeEstimate` (`elf.py:1670`)**: add `ngram_k: int` field alongside
   `ctx_bits: int`.

### `shm.py` — no layout change

The SHM header (24 bytes) and entry format (`{edge_id: u32, count: u32}`)
are unchanged.  k is a compile-time constant; the `__afl_ngram_k_N` symbol
carries it.  No new SHM fields are needed.

`VIRGIN_DENSE_MAX = 1 << 24` (`shm.py` comment): safe up to k ≈ 4 with
8-bit context because all inputs to the hash are small integers (~guard
count).  Re-evaluate if k ≥ 5 or `__AFL_CTX_BITS ≥ 16`.

### `ptrace_coverage.py:430–434`

The ptrace path simulates `(rel ^ self.prev_location) % map_size`.
Extend with a ring:
```python
# prev_locations: collections.deque(maxlen=k-1), initially all-zero
edge_id = rel
for p in self.prev_locations:
    edge_id ^= p          # simple XOR chain is fine for ptrace (no perf path)
bucket = edge_id % self.map_size
self.prev_locations.appendleft(rel >> 1)
```
Note: the ptrace path has no `caller_ctx` and never will; this is an
acknowledged coverage-mode gap (`ptrace_coverage.py:430` comment).

`reset_edge_map` (`ptrace_coverage.py:421`) currently only zeros
`self.prev_location`. It must also clear the new `prev_locations` deque
(e.g. `self.prev_locations.clear()` then re-pad with zeros, or
reassign a fresh `deque(maxlen=k-1)`), or the ring carries stale
entries across iterations — the same aliasing bug the C-side reset
is designed to prevent.

### `fuzzer.py` — ASLR note

`fuzzer.py:773–775` explains why `_seen_edge_ids` comparisons require ASLR
disabled: `caller_ctx` contains a return-address hash.  n-gram adds no
new ASLR exposure because guard IDs (not raw PCs) populate `__afl_prev_locs`
when built via SanitizerCoverage; the ASLR concern is unchanged.

---

## Compatibility Concerns

| Concern | Impact | Mitigation |
|---|---|---|
| **Corpus portability** | edge_ids change when k changes; existing `_seen_edge_ids`, `EdgeTracker`, `state.json` are incompatible | Add `ngram_k` to `state.json`; refuse resume if k mismatches |
| **`__afl_prev_loc` ABI** | Symbol is non-static (`afl_shim.c:283`). Note: `fuzzer.py:308` is `_AFL_SYMS`, an unrelated instrumentation-detection tuple — nothing in the Python codebase currently scans for `__afl_prev_loc` itself, so this row needs re-verification before relying on it | Keep symbol at k=2 default; the ring is introduced only at k > 2, emitting a new symbol `__afl_ngram_k_N` |
| **cmplog/perf shims** | Both use `-include afl_shim.c`; new statics duplicate into each TU | No change — `static` globals are already per-TU; ring follows the same pattern |
| **`direct_lite` mode** | `.so` and fuzzer share address space; ring lives in `.so` BSS | Reset path updated (`__afl_map_reset`) → no new concern |
| **Map-size pressure** | k=3 can triple cardinality, pushing load above 0.9 | `ngram_inflation_factor` + `recommended_map_size` (`edge_tracker.py`) must account for k |
| **`AFL_MAP_SIZE` env** | Python sets this from `MapSizeEstimate` (`runner.py:249`) | `_size_from_blocks` correction (above) is the fix; without it the map saturates silently |

---

## Files to Change (Complete List)

| File | Change |
|---|---|
| `src/fuzzer_tool/adapters/afl_shim.c:283` | `uint32_t __afl_prev_loc` → ring + index (k > 2) |
| `src/fuzzer_tool/adapters/afl_shim.c:272` | Add `__afl_ngram_k_N` symbol advertisement |
| `src/fuzzer_tool/adapters/afl_shim.c:578–668` | n-gram FNV hash in `__afl_map_edge` |
| `src/fuzzer_tool/adapters/afl_shim.c:848–853` | Zero ring in `__afl_map_reset` |
| `src/fuzzer_tool/core/elf.py:1587` | Add `detect_ngram_k`, `ngram_inflation_factor` |
| `src/fuzzer_tool/core/elf.py:1640` | `_size_from_blocks`: apply `ngram_inflation_factor` |
| `src/fuzzer_tool/core/elf.py:1670` | `MapSizeEstimate`: add `ngram_k` field |
| `src/fuzzer_tool/adapters/shm.py` | None (layout unchanged) |
| `src/fuzzer_tool/services/ptrace_coverage.py:421` | `reset_edge_map`: clear `prev_locations` deque alongside `prev_location` |
| `src/fuzzer_tool/services/ptrace_coverage.py:430` | Ring-based prev_location simulation |
| `src/fuzzer_tool/core/edge_tracker.py` | `recommended_map_size` — account for k |
| `state.json` / resume logic | Record and validate `ngram_k` on resume |

No change needed at the fork-server init (`afl_shim.c:1552`): see the
Reset Path section above — the ring is never written there, so it
carries no explicit zeroing step of its own.

---

## Test Plan

- **Falsification**: build target with `__AFL_NGRAM_K=3`; confirm that two
  inputs that share the same final edge but differ in their k−2 predecessors
  produce distinct edge_ids.
- **Adversarial**: fill the map to > 0.8 load with k=3; confirm drop-rate
  stays below the expected 1 % window bound (`afl_shim.c:338`).
- **Regression**: existing k=2 corpus loads and resumes without error when
  `state.json` records `ngram_k=2`.
- **Reset invariant**: after `__afl_guarded_reset`, all ring slots must be
  zero and `__afl_prev_idx` must be 0; verify via a unit test that calls
  the reset and then asserts deterministic output for a known input sequence.
