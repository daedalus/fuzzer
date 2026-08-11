# kmer_saturate_bits: byte-aligned de Bruijn saturation has a blind spot at every non-8-bit phase

**Date:** 2026-08-10
**Context:** `src/fuzzer_tool/core/mutations/structured.py` + `services/operators.py` +
`core/operator_registry.py`, TODO item "Bit-granularity de Bruijn"

## Problem

`kmer_saturate` (`de_bruijn_fill`) builds a de Bruijn sequence B(k, n) and tiles
it across the buffer so an OPSO/OQSO/DNA/bitstream-style occupancy count sees
zero missing k-mers. For a byte-at-a-time lexer or DFA that's the right shape.
For a *bit*-at-a-time reader it isn't: `de_bruijn_bytes` renders each symbol as
one byte scaled by `256 // k` (`de_bruijn_bytes`, structured.py:296-297), so a
`k=2` sequence only ever writes `0x00`/`0x80` — the top bit carries the symbol,
the low 7 bits are constant zero for the entire stream. The exhaustive-window
guarantee therefore only holds for windows that start at a byte boundary.
Anything that pulls bits from an arbitrary offset — Exp-Golomb fields inside an
H.264 RBSP, a protobuf varint's continuation bit, a hand-packed bitfield struct
— samples mostly-zero windows at 7 out of every 8 phases, i.e. it never sees
the saturation at all.

## Approach

Added a parallel construction rather than modifying `de_bruijn_fill`:

- `_pack_bits_msb`: packs a list of 0/1 symbols one-per-bit, MSB-first.
- `de_bruijn_bits(n)`: binary (`k=2`) FKM sequence via the existing
  `_de_bruijn_symbols`, packed bit-tight and `lru_cache`d like `de_bruijn_bytes`.
- `kmer_saturate_bits`: same tile-the-whole-buffer shape as `de_bruijn_fill`
  (same `len(data) < 16` guard, same cyclic truncate-to-length), but selects a
  bit-window width from `_DE_BRUIJN_BIT_WORDS` and calls `de_bruijn_bits`.
- Wired through the existing dispatch convention: `_op_kmer_saturate_bits` in
  `operators.py` (mirrors `_op_kmer_saturate`), `"kmer_saturate_bits"` added to
  the `regularity` category in `operator_registry.py`.

Kept it as a separate function instead of an `align_to_bit` flag on
`de_bruijn_fill` because the two constructions don't share a code path once
packing changes granularity — `de_bruijn_bytes`'s `step` scaling has no
equivalent at 1 bit/symbol, and forcing one function to do both would mean an
`if bit_packed:` branch down the middle of an otherwise five-line body.

## Key insight

`de_bruijn_bytes`'s wide alphabets (k=16, k=256) don't have this blind spot —
they already vary every bit of every byte, so the byte-aligned guarantee and
the bit-level guarantee coincide. The blind spot is specific to *small*
alphabets rendered byte-per-symbol, which is exactly the `k=2` shape that maps
onto `diehard_bitstream` in the first place. So the byte-granularity version
was already right for three of its four target tests (opso/oqso/dna use wider
alphabets) and wrong for the one test named directly in its own docstring
(`diehard_bitstream`) — the bug was narrow enough to hide behind three
passing round trips.

## Verification

- `TestDeBruijnBits.test_is_a_binary_de_bruijn_sequence_at_every_bit_offset`:
  rebuilds the window set from the packed output at n ∈ {3,4,8,10} and checks
  all `2**n` cyclic windows appear — this is the property `de_bruijn_bytes`
  cannot offer for the `k=2` case.
- `TestKmerSaturationBits.test_covers_windows_the_byte_aligned_variant_misses`:
  measures window coverage separately at each of the 8 bit phases; all 8 must
  reach >50% of the `2**n` space, which is precisely what fails for
  `de_bruijn_fill(k=2, ...)` at phases 1-7.
- `test_denser_than_the_byte_aligned_variant` / `test_shorter_period_than_byte_aligned_fill_at_equal_n`:
  confirms the 8x density claim (`de_bruijn_bits(n)` is `de_bruijn_bytes(2, n)`
  divided by 8) rather than just asserting it in a docstring.
- Round trip against the existing detector: `kmer_occupancy` (already bit-level
  internally, `randomness.py:333` calls `_bits(data)` before windowing) reads
  < 0.01 occupancy over 20 seeded draws, same threshold as `de_bruijn_fill`'s
  existing test.
- Full suite: 3961 passed, 154 skipped (unchanged skip count; no regressions).

## Generalizes to

- A statistic being "bit-level" in its detector doesn't mean a byte-level
  construction moves it correctly at every phase — check the detector's own
  windowing code (`_bits(data)` + `reshape(n, tuple_bits)`, non-overlapping)
  against what the construction actually varies per byte, not just whether the
  aggregate p-value crosses a threshold at phase 0.
- When a "saturate every k-mer" construction is built by scaling symbols into
  a wider range (`step = 256 // k`), the unscaled low bits are a fixed
  constant for every symbol below the top of the alphabet — worth checking
  whether anything downstream reads at a granularity finer than the scaling
  preserves.
