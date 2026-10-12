# Handover — operator gaps vs AFL++ / libFuzzer / honggfuzz / Radamsa

**Date:** 2026-10-11
**Base:** `19943bc8` (`Add orbit dedup: --orbit-dedup swap/endian canonicalization before exec filter (G1)`)
**Status:** analysis only. Nothing implemented. **[verified]** = read in the tree at the
base commit. **[unverified]** = grep-only or design assumption; confirm before coding.

## Scope

282 operators registered in `core/operator_registry.py` (`REGISTRY.names()`).
Categories: bit 20, byte 21, block 18, dict 8, structural 41, radamsa 11, format 57,
regularity 57, adaptive 27. Compared by name and by grep against the mutator sets of
AFL++, libFuzzer, honggfuzz and Radamsa. The base set is wide; the gaps below are narrow.

## Corrections to the first-pass chat summary

| Claim | Reality |
|---|---|
| `bitcast_float` lacks NaN/Inf handling | Wrong. It writes 0.0, +inf, -inf, one NaN, or value*1e3 into a random 4/8-byte slot, both endiannesses **[verified]** `core/mutations/structured.py:1406-1452`. Real gap is narrower: see O2. |
| `leb128_encode` pads | Wrong in the useful sense. It re-encodes the value minimally and appends `\x00` bytes after it **[verified]** `core/mutations/generic.py:2907-2913`. No continuation-bit padding, so no overlong encoding: O1 stands. |

## Item index

| ID | Operator | Category | Value | Cost | Status |
|---|---|---|---|---|---|
| **O1** | `leb128_overlong` | byte | high | S | gap **[verified]** |
| **O2** | `float_special` | regularity | med | S | gap **[verified]** |
| **O3** | `case_map_grow` | radamsa | med | S | gap **[verified]** |
| **O4** | `unicode_ctrl` | radamsa | low-med | S | gap **[unverified]** |
| **O5** | `int_str_boundary` | structural | low | S | may overlap **[unverified]** |
| **O6** | `hash_len_ext` | format | low | M | needs digest oracle |
| **O7** | sequence-level mutation | structural | high | L | **[unverified]** whether absent |

Recommended order: O1, O2, then O3.

## O1 — `leb128_overlong`

Non-canonical varints: pad with continuation bytes (`0x80 0x80 0x00` for 0,
`0xff 0x80 0x00` for 127). Also signed (`0xff 0x7f` for -1 padded) and vu64.
Targets parsers that assume canonical form or cap the byte count (protobuf, DWARF,
WASM, DEX, LLVM bitcode).

- Existing: `leb128_encode` (`generic.py:2878`), `sleb128_encode`, `vu64_encode`
  (`mutations/vu64.py`), handlers `_op_leb128_encode` (`services/operators.py:3349`).
- Design: new function next to `leb128_encode`; reuse `encode_uleb128`. Pick a 1-5 byte
  candidate integer as `leb128_encode` does, re-encode with N extra continuation bytes
  (N up to 10, past the 5/10-byte limits to hit length caps). Cover signed via sign
  extension bytes.
- Tests: falsification (decode of the output equals the original integer, so the op is
  value-preserving); adversarial (`max_len` boundary, empty input, value 0, 10-byte
  input, input already overlong).

## O2 — `float_special`

What `bitcast_float` does not emit: -0.0, signaling NaN, NaN with nonzero payload,
smallest subnormal, largest subnormal, FLT_MIN/DBL_MIN, FLT_MAX/DBL_MAX, 1 ulp either
side of 0/1/2^24 (integer-exactness boundary).

- Caveat **[unverified]**: `struct.pack("<f", float("nan"))` may canonicalize the
  payload. Build the pattern as integer bit patterns and write bytes directly.
- Design: table of u32/u64 bit patterns in a module constant, overwrite one aligned
  slot; either endianness. Alternative: extend `bitcast_float` strategies. Prefer a new
  operator to keep `bitcast_float`'s arm statistics unchanged.
- Tests: falsification (written pattern decodes to the intended class via
  `math.isnan`/`copysign`); adversarial (len 3, len 4, len 8, buffers where the slot
  straddles the end).

## O3 — `case_map_grow`

Characters whose case mapping or normalization changes UTF-8 length: U+0130 (İ),
U+FB01 (ﬁ), U+1E9E (ẞ), U+00DF (ß), U+0149 (ŉ), U+1F80..U+1FAF. Bug class: output
buffer sized as input length, then `toupper`/`lower`/NFKC grows it.

- Existing: `core/mutations/utf8.py` has `mode_truncate/orphan/widen/surrogate/
  boundary/cont_flip` **[verified]** (`utf8.py:163-233`). No case-mapping set
  **[verified]** (no `case`/`upper`/`lower` hit in the file).
- Design: new mode or sibling operator in `utf8.py`; insert or replace a code point
  with one from the table, encoded via `encode_width`.
- Tests: falsification (output is valid UTF-8 and `len(s.upper()) > len(s)` for the
  inserted code point); adversarial (non-UTF-8 input, input with only ASCII, at
  `max_len`).

## O4 — `unicode_ctrl`

Bidi overrides (U+202A-202E, U+2066-2069), zero-width (U+200B-200D, U+FEFF), embedded
NUL, CR/LF, U+0085, U+2028/2029 inside text fields. No hit for these in the grep
**[unverified]**: `utf8_insert` may already draw from a code-point table. Read
`_op_utf8_insert` (`services/operators.py:1920`) first; if it covers these, drop O4.

## O5 — `int_str_boundary`

Decimal/hex strings at width boundaries: `2147483647`, `2147483648`, `4294967296`,
`-0`, `0x7fffffffffffffff`, leading zeros, trailing space. Likely overlaps
`ascii_num_replace` (go-fuzz case 15, `operators.py:3595`), `ascii_num`, `magic_values`.
Diff the value tables before adding anything.

## O6 — `hash_len_ext`

Merkle-Damgard length extension for `H(key||msg)` with MD5/SHA-1/SHA-256: append
`padding || suffix` and recompute the digest from the known state. Same family as
`poly1305_fix` and `rs_fix`. Needs a known-digest field location and message length,
passed by flags like `--poly1305-key`. Narrow target set; build only if a target
exists.

## O7 — sequence-level mutation

Reorder, duplicate, drop or splice whole messages for stateful protocols. Grep for
`sequence|multi_packet|stateful|session` in `core/mutations` and `services/operators.py`
gave only unrelated hits **[unverified]**. Not a single-operator change: needs a message
boundary model (`format_lock`, `tlv_nest_mutate`, the `record_stride` seed meta used by
`chunk_shuffle` may already provide one). Scope separately.

## Checked and covered

- `arithmetic` already uses widths 1/2/4/8 on aligned offsets (`operators.py:2179`).
- `line_mutate` implements all six Radamsa `sed-line-*` ops (`operators.py:1957`).
- Format-string and path-traversal strings exist in `generic.py`.
- honggfuzz memswap equivalent: `swap_regions`, `span_relocate`. Not added.

## Implementation rules (AGENTS.md)

- One entry in `core/operator_registry.py` `REGISTRY` plus its `_op_<name>` handler on
  `OperatorEngine` (`services/operators.py`). Add the name to one `_CATEGORIES` set. Do
  not edit `OPERATOR_CATEGORIES` or the legacy lists (rule 12).
- Randomness only from `core/rand_pool.py` (rule 16).
- One falsification and one adversarial test per operator (rule 23); clean up temp
  files (rule 21).
- Vectorize after correctness, keep the faster version (rule 14).
- Document in `docs/DEEP_DIVE.md` (rule 11).
- Operators arm automatically through `_register_arms`; check
  `tests/test_regression_scheduler_operator_reach.py` still passes (operator count
  grows).
- Keep new operators out of `--hail-mary` until measured, as with `--pos-harmonic`
  before promotion.

## Open

- No operator measured on a real target. A/B needs a target that parses varints or
  floats (e.g. a protobuf or WASM reader built with clang + ASAN).
- Tests were not run: this commit is documentation only.
