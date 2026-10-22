# Handover — SBBF + folding vs the repo's flat bloom: evaluated, rejected

**Date:** 2026-10-21. **Verified against:** `9140a67`. **Status:** analysis
only — no code changed. Decision record so this is not re-litigated.

**Question raised:** an uploaded `bloom_folding.py` (educational Split Block
Bloom Filter with folding, from the Pydantic/Logfire Parquet article below)
vs. the repo's `core/bloom.py`. Should we adopt SBBF and/or folding?

**Verdict:** No SBBF configuration beats the flat filter on FPR-per-byte in
this codebase. SBBF's reason to exist — one cache line per op — does not pay
off in interpreted Python, and its FPR floor is strictly worse. The only
defensible slice is a `fold_to_target` on the persisted negative bloom, and
even that competes with correct up-front sizing (which the negative bloom
already has — it knows corpus cardinality at build time). Recommendation:
do not integrate.

---

## 1. Source: the Pydantic/Logfire article

"Faster, smaller Bloom filters in Parquet", Adrian Garcia Badaracco,
2026-04-15. Folding was contributed to Apache Arrow (Rust Parquet).

**SBBF** (Putze/Sanders/Singler): array of `z` **256-bit blocks** (cache-line
pair, SIMD-friendly). One value touches one block → one memory access.

- block index: `i = ((h >> 32) * z) >> 32` (multiply-shift, not modulo).
- the low 32 bits set **8 bits, one per 32-bit word**, via 8 odd salt
  constants. FPP ≈ `fill^8` (one factor per word).

**Sizing dilemma:** right size depends on NDV, unknown up front. `arrow-rs`
used a fixed guess; the alternative is write-several-pick-one.

**Folding** solves it — size *after* seeing the data:

```
block[i] = block[2i] | block[2i + 1]     # halves block count, OR adjacent pairs
```

Correct because halving `z` halves the index: `i = ((h>>32)*z)>>32` becomes
`floor(i/2)` the instant `z` halves, so blocks `2i` and `2i+1` are exactly the
two that collapse into `i`. OR keeps every set bit → **no false negatives**.
"Provably equivalent to having written into the smaller filter from the
start" (and the same property lets two SBBFs merge).

**Their algorithm:** allocate generously large → write all values → measure
real fill across blocks → fold while FPP ≤ target → serialize right-sized.

**Their costs (production):** ~1 MiB RAM per filter at write; one popcount
scan + a log number of OR passes. PR benchmark: a filter built for 1M NDV that
received 100K values → folding added **~27 µs on ~190 µs insert (~14%)** on
that step. Read side is smaller and sharper forever. (General bloom speedups
cited: DuckDB 50×, InfluxData 30× on point lookups.)

The article's premise is **unknown NDV per row group**. That premise does not
hold for our negative bloom, which is built from a known corpus — see §7.

---

## 2. The repo's two designs

- `core/bloom.py` `BloomFilter`: flat `bytearray` of `m` bits (power-of-two
  `m`). One SHA-256 per key supplies all `k` probes, scattered across the
  whole array — sliced digest (`k·log2(m) ≤ 256`) or Kirsch-Mitzenmacher
  double hashing otherwise. `k` is derived from `(capacity, error_rate)`.
- `core/negative_bloom.py`: builds a `BloomFilter` sized `NEG_BLOOM_FACTOR=10`
  × corpus, `error_rate=1e-3`, frozen and saved as gzip'd pickle
  (`to_dict`). Append-only; no reset (a false negative would lose corpus).

SBBF is a different layout, FPP model, and serialization — it would be a new
module, not a patch to `bloom.py`.

---

## 3. Experiment 1 — FPR at matched memory (16 KiB, 512 blocks, 300k probes)

| n | occ | flat FPR | sbbf FPR | sbbf/flat |
|---|-----|----------|----------|-----------|
| 4000 | 22% | 0.00001 | 0.00003 | 5.0× |
| 8000 | 39% | 0.00038 | 0.00120 | 3.2× |
| 16000 | 62% | 0.01905 | 0.02980 | 1.6× |

Zero false negatives both filters, every row. The flat filter adapts `k` to
load (182 probes at low fill down to 6 at high); SBBF is locked at 8 bits →
blocking penalty. The `fill^8` FPP estimate is **optimistic**: observed FPR
runs above it at mid loads (block bits are correlated), converging only at
saturation.

Folding control (512 blocks, OR to 1): **FN=0 at every level**; FPR climbs
predictably with occupancy. Practical floor ≈ 60% occupancy before FPR
explodes.

---

## 4. Experiment 2 — A/B throughput, size, write (matched 256 KiB, n=50k)

Apples-to-apples: SBBF takes its 64 bits from the **same C `sha256`** the flat
filter computes (hash held constant; any gap is structural).

| metric | flat (sha256) | sbbf (sha256) |
|--------|---------------|----------------|
| insert | 0.32 M/s | **0.35** M/s |
| query hit | 0.38 M/s | 0.39 M/s |
| query miss | **1.05** M/s | 0.86 M/s |
| occupancy | 30% | 17% |
| gzip size | 235 KB | **192 KB** |
| write (pickle→gz6→fsync) | **14 ms** | 39 ms |

- **Insert:** SBBF wins (8 contiguous lane-ORs vs 15 double-hash probes).
- **Miss:** flat wins (tight masked early-exit) — and miss is the *dominant*
  op for a negative bloom (most mutants are new). Net: a wash, slightly
  flat-favored for the real access pattern.
- **Size:** SBBF smaller **only because it sets 8 bits/key vs flat's 15** →
  lower occupancy. Same lever that raises its FPR; not a free saving.
- **Write:** SBBF ~2.7× slower *despite smaller output* — deflate works harder
  on the sparser (more compressible) bitmap. Absolute cost trivial (written
  once per session).
- **Rule-16 aside:** an SBBF using in-house pure-Python `xxh64` (what Rule 16
  requires) is ~35% slower across the board and loses the insert edge. The
  hash, not the layout, dominates. C `sha256` ~2.4 M/s vs `xxh64` ~0.37 M/s.

Folding (8192→2048 blocks): 256 KB → 64 KB raw, 192 KB → 64 KB gz, occ
17%→53%, **FN=0**. 1024 blocks (78% occ) is too far. `fold_once` 1.6–6.6 ms.

---

## 5. Experiment 3 — does using 256 bits instead of 64 help? No

The 64-bit scheme reuses lo-32 via salts (8 lane bits from 32 bits of entropy
→ correlated). The 256-bit scheme slices **8 independent 5-bit lane fields**
directly from the full digest, block index still `(h>>224)*z>>32` (fold
stable). Hypothesis: more entropy → less correlation → lower FPR.

**Refuted.** sbbf256 FPR ≈ sbbf64 FPR within noise at every load (1.5× vs 5.0×
at n=4000 is a handful of events on 300k probes; identical by n≥8000). The
salt-multiply already decorrelates adequately. The 1.5–5× penalty vs flat is
**structural** (all 8 bits confined to one block), not an entropy artifact.
More hash material cannot fix a locality penalty. Throughput roughly equal
(256-bit marginally faster on insert — shifts beat the 8 salt multiplies).

---

## 6. Experiment 4 — can more bits-per-key cross the flat curve? No

Sweep `K` bits/key placed anywhere in the 256-bit block (drawn from the full
digest, fold-stable index), matched 16 KiB, 512 blocks, 200k probes:

| load | keys/block | predicted k\* | flat (k, FPR) | SBBF best K | SBBF best FPR | best ratio |
|------|-----------|--------------|---------------|-------------|---------------|-----------|
| n=4000 | 7.8 | 22.7 | 23, 0.00001 | 14 | 0.00002 | 1.5× |
| n=8000 | 15.6 | 11.4 | 11, 0.00038 | 8 | 0.00109 | 2.9× |
| n=16000 | 31.2 | 5.7 | 6, 0.01911 | 6 | 0.02426 | 1.3× |

FN=0 at every K. Two results:

1. **The local k-optimum is real and predictable** — best K tracks
   `k* = (256 / keys_per_block)·ln2` (spot-on at n=16000: predicted 5.7,
   best 6). Past the optimum, overfilling the block tanks FPR (n=16000 K=24 →
   94% occ, 0.298).
2. **Tuned-K SBBF never crosses flat** — best case is 1.3–2.9× above, the
   **block-load-variance floor** (Putze et al.): keys Poisson-scatter across
   blocks, overloaded blocks dominate FPR, no K removes that variance. Note
   flat's own `k` (23, 11, 6) ≈ SBBF's optimal K at each load — same probe
   count, applied globally without the per-block overload. Flat is SBBF's best
   case minus the variance tax.

---

## 7. Where folding would / wouldn't help the fuzzer

The article's use case is **unknown NDV**. Map onto the fuzzer:

- **Negative bloom** — built once from the full corpus at a known 10× factor,
  frozen, gzip'd. Cardinality is known at build → already sized correctly.
  Folding reclaims over-provisioning that isn't there; the gzip space win is
  marginal (sparse flat filters already compress).
- **Corpus dedup filter** — grows unbounded across sessions (append-only).
  Folding shrinks; wrong direction. Bounded case already handled by
  `update_bytes(reset_on_full=True)`.

In C, SBBF's single-block access is the whole point; in interpreted Python
that locality advantage evaporates (both are interpreter loops), leaving only
the FPR penalty.

---

## 8. Bug the control caught (Hard Rule 46)

First `fold_once` reimplementation (and an early sketch) paired array *halves*
(`src[i] | src[i+half]`) instead of adjacent blocks (`2i, 2i+1`). The FN
control flagged it immediately: ~50k false negatives. For a `(hi*N)>>32` index
the correct fold is adjacent pairs — exactly the article's
`block[2i] | block[2i+1]`. The half-overlay corresponds to low-bits indexing,
a different (incompatible) block scheme. The FN guard is the oracle that can
fail; without it the broken fold looks fine on inserts. §11 names both trees:
adjacent pairs = Eytzinger order, half-overlay = bit-reversed order.

---

## 9. If ever integrated (narrow slice only)

A `fold_to_target` on `negative_bloom` is the one defensible use: after
`build()`, measure occupancy and fold while estimated FPP ≤ target, so the
persisted `neg.pkl.gz` is sized to measured fill instead of the fixed 10×.
Caveats: it would be a **new** `core/split_bloom.py` (not a patch to
`bloom.py`); use in-house `xxh64` (Rule 16) or `sha256`; `array('I')` lanes
(not Python big-ints); versioned `to_dict`/`from_dict` matching the pickle
path (Rule 15); plus the mandatory falsification + adversarial tests
(Rule 23). The `fold_once` body must pair adjacent blocks (§8). Even so the
win is a smaller on-disk artifact, not speed, and competes with sizing the
filter right up front — so this was **not** pursued.

---

## 10. Reproducers

Scripts were session-local (scratchpad), not committed. Core SBBF + correct
fold, enough to reproduce §3–§6:

```python
from array import array
import hashlib
BLOCK_BITS, LANES, U32 = 256, 8, 0xFFFFFFFF
SALTS = (0x47B6137B, 0x44974D91, 0x8824AD5B, 0xA2B7289D,
         0x705495C7, 0x2DF1424B, 0x9EFC4947, 0x5C6BFB31)

def locate_salt64(dig, N):          # 64-bit scheme (§4)
    h = int.from_bytes(dig[:8], "little"); hi, lo = (h >> 32) & U32, h & U32
    return (hi * N) >> 32, [((lo * s) & U32) >> 27 for s in SALTS]

def locate_slice256(dig, N):        # full-digest scheme (§5), K=8
    h = int.from_bytes(dig, "big")
    return (h >> 224) * N >> 32, [(h >> (5 * i)) & 31 for i in range(LANES)]

class SBBF:
    def __init__(self, n, loc): self._n, self._loc = n, loc; self._l = array("I", bytes(LANES * n * 4))
    def add(self, k):
        b, bits = self._loc(hashlib.sha256(k).digest(), self._n); base = b * LANES
        for j, bit in enumerate(bits): self._l[base + j] |= 1 << bit
    def query(self, k):
        b, bits = self._loc(hashlib.sha256(k).digest(), self._n); base = b * LANES
        return all(self._l[base + j] & (1 << bit) for j, bit in enumerate(bits))
    def fold_once(self):            # CORRECT pairing: adjacent blocks (§8)
        out = SBBF(self._n // 2, self._loc); s, d = self._l, out._l
        for i in range(out._n * LANES):
            blk, lane = divmod(i, LANES); a = (2 * blk) * LANES + lane
            d[i] = s[a] | s[a + LANES]
        return out
```

Flat reference is `fuzzer_tool.core.bloom.BloomFilter`; match memory by forcing
`m` to `num_blocks * 256`. Control (Rule 46): every FPR comparison asserts
FN=0 on the inserted keys before trusting the FPR.

---

## 11. Folding is the Eytzinger (binary-heap) index map

Stack every fold level in one 1-indexed array, the level with `z` blocks at
`[z, 2z)`, root at 1:

```
index:  1 | 2  3 | 4  5  6  7 | 8 ... 15      z = 1, 2, 4, 8
        root     |  fold of 8..15 |  leaves
rule:   heap[i] = heap[2i] | heap[2i+1]
```

That is the Eytzinger child rule. Verified for every internal node (64
leaves, 300 keys). One key's node per level: 122 → 61 → 30 → 15 → 7 → 3 → 1,
so each fold is a parent step, `>> 1`.

**Why.** `((h >> 32) * z) >> 32` is the top `log2(z)` bits of the hash: a
prefix. Blocks are the leaves of a binary trie over hash prefixes, left to
right. Eytzinger numbers the trie node for prefix `p` at depth `d` as
`2^d + p`. Halving `z` drops the last prefix bit and moves each block to its
parent, so only adjacent pairs can merge.

**The §8 bug in these terms.** The half-overlay fold indexes by low hash bits
(a suffix), not the prefix. Verified: it equals the adjacent-pair fold applied
to the leaves in bit-reversed order. Wrong tree → keys land in blocks they
were never written to.

**Where it stops.** Eytzinger stores sorted keys in BFS order so binary search
is cache-friendly. Folded levels hold OR-combined bits; there is nothing to
search. It is a heap-indexed OR segment tree, not a search tree.

**Stacked pyramid: not useful here.** A "no" at a coarse level means "no" at
every finer level, so top-down early rejection is possible. But coarse levels
fill first (1024 blocks were already 78% full in §4), so that "no" almost
never fires; the finest level is the most selective and must be checked
anyway; and the pyramid costs ~2× memory.

**Repo link.** Open candidate C1 in
`handover_oeis_port_candidates_2026-09-12.md` (Eytzinger layout) is gated on
repeated binary search over a static sorted array. This does not change that
gate: folding has no search.

Reproducer:

```python
import hashlib
Z = 64
keys = [f"k-{i}".encode() for i in range(300)]
hi32 = lambda k: int.from_bytes(hashlib.sha256(k).digest()[:4], "big")

leaves = [0] * Z                                  # block -> bitmask of key ids
for n, k in enumerate(keys):
    leaves[(hi32(k) * Z) >> 32] |= 1 << n

heap, cur = [0] * (2 * Z), leaves
heap[Z:] = leaves
while len(cur) > 1:                               # article's fold rule
    cur = [cur[2 * i] | cur[2 * i + 1] for i in range(len(cur) // 2)]
    heap[len(cur):2 * len(cur)] = cur
assert all(heap[i] == heap[2 * i] | heap[2 * i + 1] for i in range(1, Z))

rev = lambda x, b: int(f"{x:0{b}b}"[::-1], 2)     # §8 bug = bit-reversed tree
half = [leaves[i] | leaves[i + Z // 2] for i in range(Z // 2)]
via_rev = [leaves[rev(2 * i, 6)] | leaves[rev(2 * i + 1, 6)] for i in range(Z // 2)]
assert sorted(half) == sorted(via_rev)
```

---

## 12. Corrections (2026-10-22, independent re-analysis)

Re-derived from the article, the uploaded script and `core/bloom.py` without
reading this file first. Verdict stands: no SBBF. Corrections:

1. **§7, negative bloom "already sized correctly": wrong.** `NEG_BLOOM_FACTOR=10`
   over-provisions 8-16x. Realised FPR at save, target 1e-3:

   | seeds | 10x RAM / FPR | 1x RAM / FPR |
   |-------|---------------|--------------|
   | 500 | 16 KiB / 6e-22 | 1 KiB / 4.4e-4 |
   | 5,000 | 128 KiB / 4e-18 | 16 KiB / 1.5e-5 |
   | 50,000 | 1 MiB / 1e-14 | 128 KiB / 2.0e-5 |

   Keeping 10x is still right, for a different reason: a negative-bloom false
   positive skips a novel mutant unexecuted; 1x loses ~1 in 2,300 at 500 seeds
   to save at most ~1 MiB.
2. **§7, corpus filter "grows unbounded across sessions": wrong.**
   `Fuzzer.bloom` (100k, sliced) is never persisted; `load_corpus` rebuilds it
   every start, and an exact set backs it.
3. **§4, "Rule 16 requires xxh64": misread.** Rule 16 is the PRNG rule. The
   xxh64 rule (`hash_data()`) covers filename/content matching, not bloom keys.
4. **§8/§9 omit the flat fold.** Half-overlay (`bits[:m/2] | bits[m/2:]`) is the
   *correct* fold for `core/bloom.py`, which indexes by low bits (`& mask`).
   Double-hash mode: FN 0/20000. Sliced mode: FN 19998/20000 unless the
   original slice width is kept (then 0). So a fold needs no new module: a
   `BloomFilter.fold()` plus a persisted slice width (`from_dict` recomputes it
   from `m` today).
5. **Uploaded script misses its target.** `estimated_fpp = occupancy**8`
   (Jensen): observed FPR exceeded `target_fpp` in 9/100 configs (n=13000,
   target 0.01: est 0.0081, observed 0.0115). Exact per-block
   `mean(prod(word_fill))` matched observed. `for_capacity` floors (360 bits
   requested -> 256).
6. **Shipped instead:** `_is_repeat` hashed each mutant twice with
   `--negative-bloom`. Now one `BloomFilter.digest_bytes` feeds
   `query_digest` / `update_digest`: 64 B -15%, 16 KiB -43%, default path
   unchanged.

Fold-sized SBBF vs flat, matched FPR target, 5k-500k keys, misses only,
bytearray SBBF with flat-identical bit addressing: query 0.71-0.93x of flat,
wall 1.02-1.54x slower, zip size set by power-of-two rounding (0.85-1.83x).
