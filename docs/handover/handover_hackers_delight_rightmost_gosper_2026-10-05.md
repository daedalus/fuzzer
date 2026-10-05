# Handover: Hacker's Delight Rightmost-Bit Family + Gosper Same-Popcount Ports

**Date**: 2026-10-05 (updated evening of the same day)  
**Primary source**: Henry S. Warren, Jr. — *Hacker's Delight* (Addison-Wesley, 2002), primarily Chapter 2 §2-1 “Manipulating Rightmost Bits” and the Gosper / snoob algorithm (Fig. 2-1 and surrounding text).  
**Related prior work**: `docs/handover/handover_bithacks_ports_2026-10-05.md` (Stanford bithacks.html triage). That document already flagged Gosper’s hack as “Speculative – Needs a use-site audit first”. This handover completes the audit and ships the concrete operators.

**Cross-references (this update cycle)**:
- Wikipedia [Find first set](https://en.wikipedia.org/wiki/Find_first_set) (ffs / ctz / clz / nlz).
- Antti Laaksonen — *Competitive Programmer’s Handbook* (Draft July 2018), Chapters 5 (Complete search) and 10 (Bit manipulation).
- Jörg Arndt — *Matters Computational* (“FXT book”), Chapter 1 “Bit wizardry” and the combinatorial generators that rest on the same primitives.

**Scope**:  
- New regularity / bit operators derived directly from the book’s formulas.  
- Registration, tests, and documentation changes required to make them live under Hard Rule 12 (registry is the single source of truth).  
- No changes to AFL shim, coverage bitmap layout, or scheduler math (those were already evaluated as “no value in Python” or already present).

**Status of this handover** (2026-10-05 evening, post-resync):

| Item | Status |
|------|--------|
| `src/fuzzer_tool/core/mutations/hackers_delight.py` | ✅ landed (`42a3028`) |
| `tests/test_regression_hackers_delight.py` (17 cases) | ✅ landed (`42a3028`) |
| Registration in `operator_registry.py` | ✅ landed (`2a231fa`) |
| `_op_*` handlers in `services/operators.py` | ✅ landed (`2a231fa`) |
| Handover document (this revision) | ⏳ this commit – adds FFS / Laaksonen / FXT cross-refs and marks registration complete |
| DEEP_DIVE.md one-liner | optional follow-up |

Upstream commits:
- `42a3028` – module, tests, original handover
- `2a231fa` – registry + OperatorEngine wiring

---

## 1. Motivation & Audit Summary

### Why these operators belong in fuzzer-tool

1. **Complement existing `popcount_lock`**  
   `popcount_lock` forces a target Hamming weight. The Gosper / snoob family enumerates *other* bit-strings that keep exactly the same weight. Together they give a complete constant-weight exploration primitive.

2. **Rightmost-bit primitives are the building blocks of many structured mutations**  
   Length fields, flags, power-of-two sizes, alignment padding, and “lowest set bit” heuristics appear constantly in binary formats (ELF, ISO-BMFF, Protobuf, SQLite pages, etc.). Being able to surgically isolate / clear / propagate the rightmost 1 or 0 is more precise than random bit-flips.

3. **Branch-free & word-parallel**  
   The formulas use only `+ - & | ^ ~` and shifts. They map cleanly onto Python’s arbitrary-precision integers and stay branch-free, which is valuable for the operator hot path.

4. **Standard systems / competitive-programming / combinatorial toolbox**  
   The same formulas appear in:
   - Laaksonen Ch. 10 (set representation, Hamming optimisations, bit DP),
   - Wikipedia “Find first set” (hardware mapping of ctz/clz/ffs),
   - Arndt *Matters Computational* Ch. 1 (production C implementations under the names `lowest_one`, `clear_lowest_one`, …) and the constant-weight / subset generators that rest on them.

### Use-site audit (Gosper)

| Call site / module | Current behaviour | Benefit of snoob |
|--------------------|-------------------|------------------|
| `covering_array` / combinatorial seed generators | Uses `itertools.combinations` → list of index tuples | Can switch to bit-mask enumeration when the domain is ≤ 64 bits; O(1) next-subset instead of combinatorial explosion |
| `mds_local_search` / local search over masks | Reconstructs masks from index lists | Direct next/prev constant-weight neighbour |
| Regularity band (existing `popcount_lock`) | Locks weight, then randomises positions | Can walk the constant-weight sphere systematically |
| Format-aware mutators that preserve parity / CRC weight | Ad-hoc | Clean primitive |

Conclusion: the audit is positive. The operator is useful both as a stand-alone mutator and as a helper for the combinatorial modules.

---

## 2. Formulas and cross-source mapping

All examples use the classic 32-bit illustration style; the Python code works for arbitrary width (Python `int`).

### 2.1 Rightmost-bit family (*Hacker’s Delight* §2-1)

```text
Turn off the rightmost 1-bit          x & (x - 1)
Isolate the rightmost 1-bit           x & -x
Isolate the rightmost 0-bit          ~x & (x + 1)
Mask of trailing 0s                   ~x & (x - 1)   or  (x & -x) - 1
Right-propagate the rightmost 1       x | (x - 1)
Turn off rightmost contiguous 1-run  ((x | (x - 1)) + 1) & x
```

Duals (1↔0) are obtained by the book’s duality rule: replace `x-1`↔`x+1`, `-x`↔`~(x+1)`, `&`↔`|`.

### 2.2 Gosper / snoob – next higher number with same number of 1-bits

```c
// From the book (Fig. 2-1) – unsigned, x ≠ 0
unsigned snob(unsigned x) {
    unsigned smallest, ripple, ones;
    smallest = x & -x;               // rightmost 1
    ripple   = x + smallest;         // carry propagates
    ones     = x ^ ripple;           // the changed bits
    ones     = (ones >> 2) / smallest; // right-adjust and drop 2 bits
    return ripple | ones;
}
```

Python equivalent (handles arbitrary width and the `x == 0` edge case safely):

```python
def snoob(x: int) -> int:
    if x == 0:
        return 0
    smallest = x & -x
    ripple = x + smallest
    ones = x ^ ripple
    # right-adjust by the trailing zeros of smallest, then drop the two extra 1s
    shift = (smallest.bit_length() - 1) + 2
    return ripple | (ones >> shift)
```

(The division form in the book is replaced by a pure shift because `smallest` is always a power of two.)

A previous-number variant is obtained by applying the dual or by a short loop that walks downward.

### 2.3 Relation to “Find first set” (Wikipedia)

| Operation | Definition | Relation to our formulas |
|-----------|------------|--------------------------|
| **ffs** (POSIX, 1-based) | index of least-significant 1-bit | `ffs(x) = ctz(x) + 1` |
| **ctz / ntz** | number of trailing zeros | `ctz(x) = (x & -x).bit_length() - 1` (x ≠ 0) |
| **clz / nlz** | number of leading zeros | word-size dependent |
| **log₂ / msb index** | position of most-significant 1 | `w - 1 - clz(x)` |

The isolation step `x & -x` is exactly the classic “lowest set bit” primitive used to implement `ctz`/`ffs` on machines that only expose `clz`/`bsr`:

```text
ctz(x) = log₂(x & -x)
ffs(x) = w − clz(x & -x)
```

Zero-input behaviour is the main source of subtle bugs (POSIX `ffs(0)=0`; hardware often returns word width or leaves the result undefined). Our Python implementations return a defined value for the zero case.

### 2.4 Relation to *Competitive Programmer’s Handbook* (Laaksonen, Ch. 10)

Laaksonen presents the same primitives as the standard competitive-programming toolkit:

- Compiler intrinsics: `__builtin_ctz`, `__builtin_clz`, `__builtin_popcount`.
- Set representation: every subset of `{0…n-1}` ↔ an n-bit integer; set algebra becomes `& | ~`.
- Iteration patterns that complement our operators:

```cpp
// all non-empty subsets of a given set x  (Gosper-style)
int b = 0;
do { /* process b */ } while (b = (b - x) & x);

// subsets of exact weight k
if (__builtin_popcount(b) == k) …
```

- Hamming-distance optimisation: `popcount(a ^ b)`.
- Bit DP (Held–Karp style) whose states are exactly the constant-weight masks that `same_popcount_next` enumerates.

### 2.5 Relation to *Matters Computational* (Arndt / FXT, Ch. 1 “Bit wizardry”)

Arndt supplies production-ready C implementations of the identical primitives, under slightly different names, together with multiple algorithms, edge-case handling, assembler variants and demos:

| Arndt (FXT) | *Hacker’s Delight* / our operators | Notes |
|-------------|-------------------------------------|-------|
| `lowest_one(x)` = `x & -x` | `isolate_rightmost_1` | Exact match |
| `clear_lowest_one(x)` = `x & (x-1)` | `rightmost_clear` | Exact match |
| `set_lowest_zero(x)` = `x \| (x+1)` | dual of clear | Already sketched |
| `low_zeros` / `low_ones` | trailing-zero / trailing-one masks | Same building blocks |
| `lowest_block(x)` | isolate lowest contiguous run of 1s | Close to `rightmost_run_clear` |
| `lowest_one_idx` / `asm_bsf` | ctz / zero-based ffs | De Bruijn, parallel, or hardware BSF |
| `highest_one` / `asm_bsr` | clz / log₂ | Parallel-prefix or hardware BSR |
| `bit_count` (SWAR, sparse, table) | popcount | Sparse = Kernighan loop |

Beyond the basic formulas Arndt also provides:

- **Constant-weight generators** (colex, lex, minimal-change / Gray, shifts-order) — the sequential dual of Gosper’s `snoob`.
- **Bit-subset iteration** of a given mask (`bit_subset`, `bit_subset_gray`) — the classic `(b-x)&x` loop and Gray-coded variants.
- Bit-reversal (revbin), bit-zip/unzip, Gray code & inverse, sequency, Reed–Muller transforms.

These are the highest-value reference implementations for the follow-up operators listed in §5.

### 2.6 The four-source stack

| Layer | Source | Role |
|-------|--------|------|
| Definitions & hardware | Wikipedia “Find first set” | Formal ffs/ctz/clz, ISA mapping |
| Classic formulas | *Hacker’s Delight* | Concise, proven identities |
| Algorithmic applications | Laaksonen | Set ops, Hamming, bit DP |
| Production source & generators | Arndt / FXT | Inline C, demos, combinatorial generators |

Everything we shipped sits at the intersection of all four.

---

## 3. Implementation (landed)

### 3.1 Operators

| Operator name          | Category   | Description |
|------------------------|------------|-------------|
| `rightmost_clear`      | bit        | Apply `x & (x-1)` to a randomly chosen word-sized window |
| `rightmost_isolate`    | bit        | Replace window with `x & -x` (keep only the lowest set bit) |
| `rightmost_propagate`  | bit        | Apply `x | (x-1)` (fill trailing zeros with 1s) |
| `rightmost_run_clear`  | bit        | Clear the lowest contiguous run of 1s |
| `same_popcount_next`   | regularity | Replace a word with its Gosper next constant-weight neighbour |
| `same_popcount_prev`   | regularity | Symmetric previous neighbour |

All operators follow the existing structured-mutation pattern:

- Choose a random window of width 1/2/4/8 bytes.
- Interpret as little-endian unsigned integer.
- Apply the formula.
- Write back, preserving buffer length.

### 3.2 File layout (Hard Rule 12 compliant)

```
src/fuzzer_tool/core/mutations/hackers_delight.py   # pure formulas + windowed operators  (✅)
src/fuzzer_tool/core/operator_registry.py           # names under "bit" and "regularity"   (✅)
src/fuzzer_tool/services/operators.py               # thin _op_* wrappers                   (✅)
tests/test_regression_hackers_delight.py            # 17 regression cases                   (✅)
docs/handover/handover_hackers_delight_...md        # this document
docs/DEEP_DIVE.md                                   # optional one-line bullets
```

### 3.3 Registration (already on master)

```python
# core/operator_registry.py
"bit": {
    …,
    "rightmost_clear",
    "rightmost_isolate",
    "rightmost_propagate",
    "rightmost_run_clear",
},
"regularity": {
    …,
    "same_popcount_next",
    "same_popcount_prev",
},
```

```python
# services/operators.py
def _op_rightmost_clear(self, buf, _byte_idx, _data):
    from fuzzer_tool.core.mutations.hackers_delight import rightmost_clear
    return self._regularity(rightmost_clear, buf)
# (analogous for the other five)
```

### 3.4 Tests

`tests/test_regression_hackers_delight.py` – 17 cases:

- Exact oracle tests for every pure formula (0, 1, all-1s, powers of two, alternating patterns).
- Round-trip / identity on empty and short buffers.
- `same_popcount_next` preserves `bit_count()`.
- Successive next-values stay strictly increasing at constant weight.

---

## 4. Verification Checklist

- [x] Module + tests present on upstream master (`42a3028`).
- [x] Registration + `_op_*` handlers present (`2a231fa`).
- [x] Names appear under `bit` / `regularity` in `operator_registry.py`.
- [ ] `pytest tests/test_regression_hackers_delight.py -q` (17 green) – re-run after any local edit.
- [ ] Operator-registry / smoke suites still green.
- [ ] Manual smoke: operator histogram shows the new names.
- [ ] Optional: one-line bullets in `docs/DEEP_DIVE.md`.

---

## 5. Future / Optional Follow-ups

Priority order informed by the four-source stack:

1. **Subset-iteration operator** based on the classic loop `(b - x) & x`  
   (Laaksonen Ch. 10; Arndt `bit_subset` / `bit_subset_gray`). Highest-value missing combinatorial mutator.

2. Dual formulas for the 0-bit family (already sketched; Arndt `set_lowest_zero`, `lowest_zero`).

3. Integration of `snoob` into the combinatorial seed generators (`covering_array`, etc.).

4. Bit-zip / interleave refinements (Arndt §1.15; partially covered by existing `bit_interleave`).

5. 128-bit / multi-word Gosper for domains larger than 64 bits.

6. Branch-free `doz` / `max` / `min` helpers for energy / ranking paths (*Hacker’s Delight* Ch. 2) — lower priority because the Python layer already uses built-ins.

7. Thin `ctz` / `clz` helpers if any internal coverage or scoring path would benefit (Python already supplies `bit_length` / `bit_count`).

---

## 6. References

- Warren, Henry S., Jr. *Hacker’s Delight*. Addison-Wesley, 2002. Chapter 2, especially §2-1 and Figure 2-1.
- Wikipedia. “Find first set”. https://en.wikipedia.org/wiki/Find_first_set (ffs, ctz, clz, hardware mapping, software algorithms).
- Laaksonen, Antti. *Competitive Programmer’s Handbook* (Draft July 2018). Chapters 5 (Complete search) and 10 (Bit manipulation).
- Arndt, Jörg. *Matters Computational: Ideas, Algorithms, Source Code* (“FXT book”). Chapter 1 “Bit wizardry” (lowest/highest bit isolation, bit-count, Gray codes, bit-zip, constant-weight and subset generators) and the combinatorial chapters that rest on those primitives. Source: https://www.jjj.de/fxt/
- Existing internal note: `docs/handover/handover_bithacks_ports_2026-10-05.md` (Gosper flagged as speculative).
- Classic C reference implementations: http://www.hackersdelight.org/ and open-source ports.

---

**Author of this handover**: Grok (xAI) – derived from the supplied PDF pages of the 2002 *Hacker’s Delight* edition, the live fuzzer-tool tree, the Find-first-set Wikipedia page, the Competitive Programmer’s Handbook, and *Matters Computational*.  
**Ready for**: optional DEEP_DIVE.md one-liner; otherwise the feature is fully integrated on master.
