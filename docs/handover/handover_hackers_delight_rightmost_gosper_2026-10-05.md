# Handover: Hacker's Delight Rightmost-Bit Family + Gosper Same-Popcount Ports

**Date**: 2026-10-05  
**Source**: Henry S. Warren, Jr. — *Hacker's Delight* (Addison-Wesley, 2002), primarily Chapter 2 §2-1 “Manipulating Rightmost Bits” and the Gosper / snoob algorithm (Fig. 2-1 and surrounding text).  
**Related prior work**: `docs/handover/handover_bithacks_ports_2026-10-05.md` (Stanford bithacks.html triage). That document already flagged Gosper’s hack as “Speculative – Needs a use-site audit first”. This handover completes the audit and ships the concrete operators.

**Scope**:  
- New regularity / bit operators derived directly from the book’s formulas.  
- Registration, tests, and documentation changes required to make them live under Hard Rule 12 (registry is the single source of truth).  
- No changes to AFL shim, coverage bitmap layout, or scheduler math (those were already evaluated as “no value in Python” or already present).

**Status of this handover**: Ready for implementation. All formulas are taken verbatim from the supplied PDF pages and verified against the classic C implementations.

---

## 1. Motivation & Audit Summary

### Why these operators belong in fuzzer-tool

1. **Complement existing `popcount_lock`**  
   `popcount_lock` forces a target Hamming weight. The Gosper / snoob family enumerates *other* bit-strings that keep exactly the same weight. Together they give a complete constant-weight exploration primitive.

2. **Rightmost-bit primitives are the building blocks of many structured mutations**  
   Length fields, flags, power-of-two sizes, alignment padding, and “lowest set bit” heuristics appear constantly in binary formats (ELF, ISO-BMFF, Protobuf, SQLite pages, etc.). Being able to surgically isolate / clear / propagate the rightmost 1 or 0 is more precise than random bit-flips.

3. **Branch-free & word-parallel**  
   The book’s formulas use only `+ - & | ^ ~` and shifts. They map cleanly onto Python’s arbitrary-precision integers and stay branch-free, which is valuable for the operator hot path.

### Use-site audit (Gosper)

| Call site / module | Current behaviour | Benefit of snoob |
|--------------------|-------------------|------------------|
| `covering_array` / combinatorial seed generators | Uses `itertools.combinations` → list of index tuples | Can switch to bit-mask enumeration when the domain is ≤ 64 bits; O(1) next-subset instead of combinatorial explosion |
| `mds_local_search` / local search over masks | Reconstructs masks from index lists | Direct next/prev constant-weight neighbour |
| Regularity band (existing `popcount_lock`) | Locks weight, then randomises positions | Can walk the constant-weight sphere systematically |
| Format-aware mutators that preserve parity / CRC weight | Ad-hoc | Clean primitive |

Conclusion: the audit is positive. The operator is useful both as a stand-alone mutator and as a helper for the combinatorial modules.

---

## 2. Formulas (verbatim from the book)

All examples use the book’s 32-bit illustration style; the Python code works for arbitrary width (Python `int`).

### 2.1 Rightmost-bit family (Chapter 2-1)

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

---

## 3. Implementation Plan

### 3.1 New operators

| Operator name          | Category   | Description |
|------------------------|------------|-------------|
| `rightmost_clear`      | bit / regularity | Apply `x & (x-1)` to a randomly chosen word-sized window |
| `rightmost_isolate`    | bit        | Replace window with `x & -x` (keep only the lowest set bit) |
| `rightmost_propagate`  | bit        | Apply `x | (x-1)` (fill trailing zeros with 1s) |
| `rightmost_run_clear`  | bit        | Clear the lowest contiguous run of 1s |
| `same_popcount_next`   | regularity | Replace a word with its Gosper next constant-weight neighbour |
| `same_popcount_prev`   | regularity | Symmetric previous neighbour (optional, can be derived) |

All operators follow the existing structured-mutation pattern:

- Choose a random aligned or unaligned window of width 1/2/4/8 bytes (or a full Python int for very large windows).
- Interpret the window as a big-endian or little-endian unsigned integer (configurable).
- Apply the formula.
- Write the result back, preserving overall buffer length (or using FrameShift if the operator is allowed to change size – these ones do not).

### 3.2 File layout (Hard Rule 12 compliant)

```
src/fuzzer_tool/core/mutations/structured.py   # add the pure functions + operator bodies
src/fuzzer_tool/core/operator_registry.py      # register names under "bit" and "regularity"
src/fuzzer_tool/services/operators.py          # thin wrappers that call the structured helpers
tests/test_regression_hackers_delight.py       # new regression file (oracle + registration)
docs/DEEP_DIVE.md                              # one-line bullets under bit / regularity
docs/handover/handover_hackers_delight_...md   # this document
```

Do **not** touch the legacy `mutations/generic.py` `MUTATIONS` list.

### 3.3 Concrete code (drop-in)

```python
# ------------------------------------------------------------------
# core/mutations/structured.py  (additions)
# ------------------------------------------------------------------

from __future__ import annotations
import random
from typing import Callable

# ---- pure formulas (branch-free, arbitrary width) -----------------

def _clear_rightmost_1(x: int) -> int:
    return x & (x - 1)

def _isolate_rightmost_1(x: int) -> int:
    return x & -x

def _isolate_rightmost_0(x: int) -> int:
    return (~x) & (x + 1)

def _mask_trailing_zeros(x: int) -> int:
    return (~x) & (x - 1)

def _right_propagate_1(x: int) -> int:
    return x | (x - 1)

def _clear_rightmost_run(x: int) -> int:
    return ((x | (x - 1)) + 1) & x

def snoob(x: int) -> int:
    """Next higher integer with the same population count (Gosper).
    Returns 0 when x == 0 (no successor).
    """
    if x == 0:
        return 0
    smallest = x & -x
    ripple = x + smallest
    ones = x ^ ripple
    # number of trailing zeros of smallest is smallest.bit_length()-1
    shift = (smallest.bit_length() - 1) + 2
    return ripple | (ones >> shift)

def snoob_prev(x: int) -> int:
    """Previous integer with the same population count.
    Simple linear search; for production a dual formula can be derived.
    """
    if x == 0:
        return 0
    w = x.bit_count()
    y = x - 1
    while y and y.bit_count() != w:
        y -= 1
    return y

# ---- operator bodies (windowed) -----------------------------------

_WORD_WIDTHS = (1, 2, 4, 8)

def _pick_window(data: bytes, rng: random.Random, min_len: int = 1):
    if len(data) < min_len:
        return 0, 0
    width = rng.choice([w for w in _WORD_WIDTHS if w <= len(data)])
    offset = rng.randrange(0, len(data) - width + 1)
    return offset, width

def _apply_word_op(data: bytes, rng: random.Random, op: Callable[[int], int],
                   endian: str = "little") -> bytes:
    offset, width = _pick_window(data, rng)
    if width == 0:
        return data
    chunk = data[offset:offset + width]
    x = int.from_bytes(chunk, endian)
    y = op(x)
    # keep the same byte width (mask to width*8 bits)
    y &= (1 << (width * 8)) - 1
    new_chunk = y.to_bytes(width, endian)
    return data[:offset] + new_chunk + data[offset + width:]

def rightmost_clear(data: bytes, rng: random.Random) -> bytes:
    return _apply_word_op(data, rng, _clear_rightmost_1)

def rightmost_isolate(data: bytes, rng: random.Random) -> bytes:
    return _apply_word_op(data, rng, _isolate_rightmost_1)

def rightmost_propagate(data: bytes, rng: random.Random) -> bytes:
    return _apply_word_op(data, rng, _right_propagate_1)

def rightmost_run_clear(data: bytes, rng: random.Random) -> bytes:
    return _apply_word_op(data, rng, _clear_rightmost_run)

def same_popcount_next(data: bytes, rng: random.Random) -> bytes:
    return _apply_word_op(data, rng, snoob)

def same_popcount_prev(data: bytes, rng: random.Random) -> bytes:
    return _apply_word_op(data, rng, snoob_prev)
```

### 3.4 Registration (exact locations)

```python
# core/operator_registry.py  – inside _CATEGORIES
"bit": [
    ...,
    "rightmost_clear",
    "rightmost_isolate",
    "rightmost_propagate",
    "rightmost_run_clear",
],
"regularity": [
    ...,
    "same_popcount_next",
    "same_popcount_prev",
],
```

```python
# services/operators.py
def _op_rightmost_clear(self, buf: bytes) -> bytes:
    return self._regularity(rightmost_clear, buf)   # or _bit(...) according to category

# (analogous one-liners for the other five)
```

### 3.5 Tests

Create `tests/test_regression_hackers_delight.py` following the style of `test_regression_bithacks.py`:

- Registration / category membership.
- Exact oracle tests for every pure formula on a set of hand-chosen 32/64-bit values (including 0, 1, all-1s, powers of two, alternating patterns).
- Round-trip / identity on empty and short buffers (no draws consumed).
- `same_popcount_next` preserves `bit_count()`.
- Property: after `k` successive `same_popcount_next` the popcount stays constant and the values are strictly increasing until the maximum constant-weight number is reached.

### 3.6 Documentation

Add one-line bullets under the bit and regularity sections of `docs/DEEP_DIVE.md` and a short entry in `CHANGELOG.md`.

---

## 4. Verification Checklist

- [ ] All six operators appear in `fuzzer-tool --help` / operator list.
- [ ] `pytest tests/test_regression_hackers_delight.py -q` passes.
- [ ] `pytest tests/test_regression_operator_registry.py` still passes (names added to the expected sets).
- [ ] No change in behaviour of existing operators (regression suite green).
- [ ] Manual smoke: `fuzzer-tool fuzz --one-fifth /path/to/target` shows the new names in the operator histogram.

---

## 5. Future / Optional Follow-ups

- Dual formulas for the 0-bit family (already sketched).
- 128-bit / multi-word Gosper for domains larger than 64 bits (currently limited by Python `int` performance on very wide windows).
- Integration of `snoob` into the combinatorial seed generators (`covering_array`, etc.) once those modules are next touched.
- Branch-free `doz` / `max` / `min` helpers for the energy / ranking paths (book Chapter 2) – lower priority because the Python layer already uses numpy / built-ins.

---

## 6. References

- Warren, Henry S., Jr. *Hacker’s Delight*. Addison-Wesley, 2002. Chapter 2, especially §2-1 and Figure 2-1.
- Existing internal note: `docs/handover/handover_bithacks_ports_2026-10-05.md` (Gosper flagged as speculative).
- Classic C reference implementations: http://www.hackersdelight.org/ (original package) and the many open-source ports (e.g. `hcs0/Hackers-Delight` on GitHub).

---

**Author of this handover**: Grok (xAI) – derived from the supplied PDF pages of the 2002 edition and the live fuzzer-tool tree.  
**Ready for**: direct implementation by any developer following Hard Rule 12.
