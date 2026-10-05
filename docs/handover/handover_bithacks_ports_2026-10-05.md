# bithacks.html triage and ports

Date: 2026-10-05. Source: <https://graphics.stanford.edu/~seander/bithacks.html>,
read in full and checked against `fuzzer-tool` at `3a71af90`.

Scope: what on that page is worth porting into a Python coverage-guided
fuzzer. One new operator (`swar_lane`) and three bit-exact speedups shipped.
One proposed speedup was **wrong and was not shipped** — see "Refuted". The
remaining ideas are recorded with verdicts, not built.

Everything under "Shipped" is committed and measured. Everything under "Not
built" is a design derived from reading, not from running.

## Summary

| Page item | Verdict | Action |
|---|---|---|
| SWAR lane-boundary inputs | Real gap | **Shipped** — `swar_lane` operator |
| Gray decode is a per-byte bit loop | Real, cheap | **Shipped** — 256-entry translate table |
| Bit-plane gather is a per-bit loop | Real, cheap | **Shipped** — slice+translate, multiply-pack |
| Per-byte bit reversal over a buffer | Real, expensive | **Shipped** — `bytes.translate` table |
| 8x8 nested loops are a bit-matrix transpose | **Refuted** | Not shipped; wrong reading |
| Popcount / parity via SWAR | No value in Python | Skipped |
| De Bruijn log2/ctz tables | No value in Python | Skipped |
| Lowest-set-bit / next-power-of-2 | Already present | Skipped |
| Hit-count bucket branchless min/max | Already present | Skipped |
| Sub-byte sign extension | Real gap, unbuilt | Sketch below |
| Power-of-2 ±1 rounding probes | Small data gap | Noted below |
| `abs(INT_MIN)` stays negative | Data gap | Noted below |
| Masked-merge bit crossover | Real gap, unbuilt | Sketch below |
| Gosper's-hack constant-weight enumeration | Speculative | Needs a use-site audit first |

## Already covered — do not port

The page is C. Most of it is arithmetic the Python layer either already gets
from the interpreter or does not need at all.

- **Popcount / parity.** `int.bit_count()` is used in `core/gf2_common.py`,
  `core/randomness.py` and `structured._popcount_table`. A SWAR popcount is
  6 Python ops against 1 builtin method call; it is strictly slower.
- **log2 / next power of 2.** `bit_length()` is used in `cuckoo`, `count_class`
  and `op_exp3`. The de Bruijn log2/ctz tables exist to turn a shift-and-mask
  chain into one memory load; Python has no such chain to shorten.
- **Lowest set bit.** `(x & -x).bit_length() - 1` is already
  `services/operators.py:494`.
- **Hit-count buckets.** `count_class` already uses a 256-entry table plus
  numpy, so the branchless `min`/`max`/`abs` tricks have nothing to remove.
- **The AFL shim** (`adapters/afl_shim.c`). The 2-gram `% (K-1)` is dead code at
  the default `K=2`; `% 256` on an unsigned value already compiles to an AND;
  slot math already uses Lemire fastmod and the mixers are already splitmix.
  Nothing to port.

## Refuted: the "bit-matrix transpose" reading of `bit_interleave`

This was the largest single item on the original list ("only per-call hot loop
in that list", "3 delta-swaps on one `int.from_bytes`, no inner loops"). **It
does not work, and the reason is structural.**

`bit_interleave`'s forward half is

```python
dst[j * 8 + g] = (grp[g * 8 + j] >> (7 - j)) & 1
```

Each source byte contributes **one bit**, and that bit becomes a whole
destination byte. So 8 source bytes collapse into 8 destination bytes whose
values are only ever `0` or `1`. A 64x64 bit-matrix transpose — including the
3-delta-swap form on the page (`(x ^ (x >> 7)) & 0x00AA00AA00AA00AA` and two
more stages) — is a **permutation**: it preserves the number of set bits. A
permutation can never equal an operation that discards 7 of every 8 bits.

Verified, not assumed: the loop output over random input has `set(dst) ==
{0, 1}`, and the delta-swap routine applied to the same input does not
reproduce it. The claim that the nested loops "are an 8x8 bit-matrix
transpose" is simply false; it is a gather with a stride.

What *is* available is the same win by a different route, and it shipped — see
`_interleave_gather` below. The lesson generalises: before porting a C
bit-twiddling idiom, check whether the operation is a permutation. A
permutation has a closed form in shifts and masks; a gather does not.

The delta-swap routine was not discarded unread, though. It is the right tool
for a genuine transpose, and there is no such operator in the repo today.

## Shipped

### 1. `swar_lane` — new regularity operator

**The gap.** Word-at-a-time byte scanners do several bytes per operation by
carrying per-lane arithmetic inside one machine word. The carries are only
exact for particular lane values:

| Idiom | Exact while | Breaks at |
|---|---|---|
| `haszero(v) = (v - 0x0101..) & ~v & 0x8080..` | always | lane `0x80`: `~v` has bit 7 set there, so a borrow out of the lane below reads as a zero that is not there |
| `hasless(v, n)` (`<= n` per lane) | `n <= 128` | lane `>= 0x80` beside `0x00`/`0x01` |
| `hasmore(v, n)` (`>= n` per lane) | `n <= 128` | same `0x80` boundary |
| `hasbetween(v, m, n)` | `m <= 127, n <= 128` | both thresholds |

A grep of `core/` for `0x80808080`, `0x7f7f7f7f`, `0x01010101`, `haszero`,
`hasless`, `hasmore` returned nothing — no operator, and no constant, reaches
these lanes. That is the definition of a gap rather than a duplicate.

**Why random mutation cannot find it.** The critical values are 7 of 256, and
they have to line up across *neighbouring* lanes as well. Even a large corpus
essentially never puts one there. The construction has to be deliberate.

**The three shapes**, all length-preserving, all writing only critical bytes:

```
uniform   one value repeated across the window
          -> the literal 0x80808080 / 0x7f7f7f7f / 0x01010101 vectors

mixed     an independent draw per lane, with lane[0] forced to 0x00
          and lane[n-1] forced to 0x80
          -> the borrow haszero mistakes for a zero always runs downwards,
             so the phantom only appears with a 0x80 lane ABOVE a 0x00 lane

strided   an 8-byte pattern planted at one byte-offset of a 64-byte window
          -> retried at each of the 8 alignments a scanner might be using;
             untouched lanes stay 0x00, which is itself a critical value
```

**Falsifiable invariant:** no byte the operator writes is ever outside
`{00, 01, 7f, 80, 81, fe, ff}`. That is what
`test_swar_window_is_all_critical_lanes` asserts, and it is the property that
would catch the operator silently degrading into scribbling noise. The seven
values are also pinned against the page by
`test_swar_lanes_match_the_page`, so a future edit to the set cannot quietly
weaken the construction.

**Code sketch** (`core/mutations/structured.py`):

```python
_SWAR_LANES = (0x00, 0x01, 0x7F, 0x80, 0x81, 0xFE, 0xFF)
_SWAR_WIDTHS = (4, 8)            # a scan narrower than 4B sees <2 lanes
_SWAR_MODES  = ("uniform", "mixed", "strided")
_SWAR_SWEEP, _SWAR_ALIGNMENTS = 64, 8

def swar_lane(data, rng):
    if len(data) < _SWAR_WIDTHS[0]:
        return data                       # 0 draws
    mode = rng.choice(_SWAR_MODES)
    if mode == "strided":
        offset, length = _region(len(data), rng, min_len=_SWAR_SWEEP, max_len=_SWAR_SWEEP)
        if length < _SWAR_SWEEP:
            return data
        shift = rng.randint(0, _SWAR_ALIGNMENTS - 1)
        window = bytearray(length)
        window[shift : shift + _SWAR_ALIGNMENTS] = bytes(
            rng.choice(_SWAR_LANES) for _ in range(_SWAR_ALIGNMENTS))
        return _splice(data, offset, bytes(window))
    width = rng.choice(_SWAR_WIDTHS)
    offset, length = _region(len(data), rng, min_len=width)
    if length < width:
        return data
    window = _swar_uniform(rng, length) if mode == "uniform" else _swar_mixed(rng, length)
    return _splice(data, offset, bytes(window))
```

`align` is deliberately left at 1: forcing the window to an aligned boundary
would skip the unaligned reads, which are the common case in real parsers.
Alignment is swept by the `strided` mode instead.

### 2. Full wiring

Hard Rule 12 — the registry is the single source of truth, so this is the whole
of it:

| File | Change |
|---|---|
| `core/mutations/structured.py` | the operator + `_SWAR_*` constants |
| `core/operator_registry.py` | `"swar_lane"` in `_CATEGORIES["regularity"]` |
| `services/operators.py` | `_op_swar_lane` calling `self._regularity(swar_lane, buf)` |
| `tests/test_regression_operator_registry.py` | name added to `REGULARITY_OPS` (asserted by equality, so mandatory) |
| `docs/DEEP_DIVE.md` | regularity bullet |

Notes on what was deliberately *not* touched:

- `_AVAILABLE` — the operator is unconditional, so `available=None`. No
  formatter gating.
- `mutations/generic.py` `MUTATIONS` — the legacy name list. Hard Rule 12
  forbids adding to it, and the two most recent regularity operators
  (`crc_advanced`, `murmurhash3`) are absent from it too.
- `--hail-mary` — it force-enables a flat tuple of *scheduler and diagnostic*
  flags, not operator pools. An unconditional regularity operator is available
  in every run mode already, so there is nothing to switch on.

### 3. Speedups — all three bit-exact, all verified against the old loop

A speedup that changes output is not a speedup, it is a different mutator: it
moves every seeded run's results. So each rewrite keeps the original loop in
`tests/test_regression_bithacks.py` as an oracle and asserts equality over
fixed seeds.

| Rewrite | Before | After | Speedup |
|---|---|---|---|
| `gray_code` decode | per-byte `while k:` shift loop, up to 7 iterations/byte | 256-entry `bytes.translate` table | **1.24x** @256B, **1.30x** @1KiB, **3.70x** @4KiB |
| `bit_interleave` | 128 Python iterations per 64-byte chunk | 8 slice+translate, 8 multiply-pack | **1.02x** @256B, **1.26x** @1KiB, **1.21x** @4KiB |
| `berlekamp_massey` reflected buffer | `bytes(_reverse_byte(b) for b in data)` | `data.translate(_REV8)` | **~330x** on a 2 KiB buffer |

**Gray decode.** `b ^ (b >> 1)` inverted by repeated XOR-shift, per byte. The
map is 256 entries, so it is a table and the decode is one C-level translate.
Note the inverse is a **suffix** XOR (`b_i = XOR of g_j for j >= i`), not a
prefix XOR — the tests derive it from that definition rather than from the
table, so the two can disagree.

**Bit-plane gather** (the substitute for the refuted delta-swaps):

```python
_INTERLEAVE_BITS = tuple(bytes((b >> (7 - j)) & 1 for b in range(256)) for j in range(8))

def _interleave_gather(chunk):                 # 64 iterations -> 8
    dst = bytearray(64)
    for j in range(8):
        dst[j * 8 : j * 8 + 8] = chunk[j::8].translate(_INTERLEAVE_BITS[j])
    return dst
```

The destination run `dst[j*8 : j*8+8]` is contiguous while its source
`chunk[j::8]` is strided, so one strided slice plus one translate per plane
replaces 64 interpreter iterations.

The inverse half packs 8 zero/one bytes into one byte with a single multiply:

```python
_PACK_8_LANES = sum(1 << (70 - 9 * j) for j in range(8))
out[g] = ((int.from_bytes(dst[g * 8 : g * 8 + 8], "little") * _PACK_8_LANES) >> 63) & 0xFF
```

Read little-endian, byte `j`'s only set bit is at position `8j`. Multiplied by
the constant it lands at `70 - j`, i.e. every plane is gathered into the top
byte of the product with no two planes colliding there; the shift isolates that
byte and everything else falls outside the window. Verified over 200k random
lane patterns before it was used.

**A quirk worth knowing about.** The original loop allocates
`out = bytearray(64)` and fills only indices 0..7, so every 64-byte chunk
loses its 56-byte tail to zeros. That is long-standing operator behaviour; the
rewrite preserves it deliberately, and
`test_preserves_the_56_byte_zero_tail` pins it. A rewrite that "helpfully"
wrote just the 8 packed bytes would keep the length and change every output —
no exception, no warning.

**Why `bit_interleave` only got ~1.2x.** Measured per chunk on the new code:
`rng.shuffle` 2.4us, gather 2.4us, pack 1.5us. The shuffle is now 38% of the
cost, and `_region`/`_splice` are fixed overhead on top. The bit work itself
went 8.8us -> 3.9us (2.25x); the operator total is diluted by everything
around it. Reporting 2.25x as the operator speedup would have been wrong.

An earlier version of this benchmark constructed a fresh `RandPool(seed=1)`
inside the timed lambda on both sides and reported 1.16x for a change that is
really ~1.8x at the chunk level. Fixed overhead on both sides of an A/B
cancels into the ratio. Hoist per-run setup out of the timing.

### 4. Oracle controls (Hard Rule 46)

Each oracle is paired with a sensitivity control — the same comparison run
against a deliberately wrong reference, asserted to FAIL. A control that
passes means the equality assertion above it is vacuous.

The `bit_interleave` control earned its keep. It pins the transpose-vs-gather
confusion from the "Refuted" section as the specific wrong answer the oracle
must reject, plus `test_gather_really_collapses_bits` asserting the gather's
output is in `{0, 1}`. The oracle caught that mistake during development; a
control that only compared the reference against a second run of itself would
have called the two implementations identical.

## Not built

### Sub-byte sign extension — real gap, next one to build

`type_promote` (`structured.py:1930`) simulates sign/zero-extension bugs but
only on whole-byte fields (1/2/4 bytes). Nothing covers **b-bit fields for
2 <= b <= 7**, where bit `b-1` is set and the bits above it in the byte are
garbage — the `(x ^ m) - m` family. That is exactly the layout of packed
headers and codec fields (4/5/6/7-bit palette indices, 5/6-bit GIF/LZW code
widths, sub-byte audio sample widths).

Sketch, in the `type_promote` mould:

```python
_SIGN_EXT_WIDTHS = (2, 3, 4, 5, 6, 7)          # bits, not bytes

def subbyte_sign_extend(data, rng):
    # pick a byte-aligned field of b bits, set bit b-1, fill above it with
    # arbitrary bits -- the value a reader that sign-extends correctly and a
    # reader that masks differ on, by construction
    ...
```

Wiring: `_CATEGORIES["regularity"]`, `_op_subbyte_sign_extend` via
`_regularity`, `REGULARITY_OPS` in the registry test. Same four files as
`swar_lane`; nothing new.

### Masked-merge bit crossover — real gap

There is **no** bit-level crossover anywhere in the codebase. `crossover`
(`generic.py:1054`) and `splice_common_prefix` (`generic.py:462`) are both
byte-granular. The only `(x ^ y) & mask` sites are unrelated (a ZigZag encode,
a Redqueen single-byte XOR, `invariant_break` masking against a corpus-fixed
value).

`a ^ ((a ^ b) & mask)` is a bit-level splice of a donor under a random mask —
a different shape from byte crossover, and the shape that matters for formats
where a single flag or exponent bit is the whole branch condition.

Caveat before building it: the `regularity` band is entirely single-parent
(`_regularity(fn, buf)` takes no donor), so a two-parent operator introduces a
new pattern — it needs the corpus, and therefore an `_AVAILABLE` gate on
`len(corpus) >= 2` like `crossover` has. Worth deciding whether it belongs in
`regularity` or in `structural` first.

### Power-of-2 ±1 rounding probes — small data gap

The page's "round up to next power of 2" returns 0 for `v > 2^31` and for
`v == 0`. `INTERESTING_32` (`generic.py:199`) already carries `2^15`, `2^15+1`,
`2^16`, `2^16+1`, `2^31+1` and both signed extremes — but only for `k` in
{15, 16, 31}. `2^k±1` at every `k` in the small widths is not covered, and
`size_field_overflow` / `length_miscalculate` do not fill the gap. Small,
cheap, additive; `tests/test_mutations.py:38` asserts `len(INTERESTING_32) >= 9`
so additions are safe.

### `abs(INT_MIN)` stays negative — data gap

`INTERESTING_32` has `-2147483648` and `2147483647` but never unsigned
`0x80000000`; `INTERESTING_UNSIGNED_32` jumps from `0x7FFFFFFF` to
`0xFFFFFFFE` and skips it too. So `neg`/`abs` pairs that overflow (`-INT_MIN`,
`INT_MIN / -1`) are only reachable by accident. Confirmed by reading both
tables.

### Gosper's-hack constant-weight enumeration — speculative

The lexicographic next-bit-permutation would replace `itertools.combinations`
where subsets are really bitmasks (`covering_array`, `mds_local_search`).
**Not audited**: those call sites were not read, so it is unknown whether they
want masks or tuples. Do that audit before estimating anything.

## Test coverage added

`tests/test_regression_bithacks.py`, 27 tests:

- `swar_lane` — registration and category placement; three scripted-draw exact
  byte assertions; the critical-lane falsification test; short-input identity
  (asserting *zero* draws consumed); exactly-one-word mutability; no-raise over
  lengths 0..39.
- `gray_code` — exhaustive 256-case decode equivalence; suffix-XOR derivation;
  encode-inversion round trip; whole-operator equality across 25 seeds; oracle
  control.
- `bit_interleave` — whole-operator equality across 25 seeds and across all 8
  chunk rotations; the 56-byte-zero-tail regression; transpose-vs-gather
  control; gather-collapses-bits property.
- `berlekamp_massey` — `_REV8` against a binary-string oracle (independent of
  any bit loop); `_reverse_bits` at 8/16/32/64; adversarial "bits above width
  are discarded, not reversed"; control.

## Known pre-existing failures, not caused by this work

Confirmed identical on a clean tree at `3a71af90` with the changes stashed:

- `test_operator_smoke.py::test_all_ops_fire` —
  `grimoire_extend: AttributeError: 'NoneType' object has no attribute 'book'`
- `test_regression_no_op_mutations.py::test_every_selectable_operator_is_reachable`
  — reports `{'grimoire_extend', 'grimoire_recurse', 'grimoire_string', 'ltl_prefix'}`

`swar_lane` is not in the unreachable set, so it is offered by the sweep.

## Also noted, untouched

`structured.py` has a duplicated unreachable `return` immediately before
`_LZ_WINDOW` (the line pair ending `restored[:length])`). Dead code from
copy-adapting an operator. Out of scope here; worth deleting on its own.

## Doc debt found while working

`docs/DEEP_DIVE.md` describes the regularity band as "fourteen constructive
inverses" — it is now 50. `crc_advanced` and `murmurhash3` have no bullet at
all. Fixed the stale count and added `swar_lane`; the two missing bullets are
still missing and are not this change's business.
