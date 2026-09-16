# Handover — FFmpeg algorithm port candidates

**Date:** 2026-09-16
**Base (fuzzer):** current working tree at time of analysis
**Base (vendored FFmpeg):** 9.0.1 at `~/fuzzing/vendoring/ffmpeg/`
**Scope:** algorithms in the vendored FFmpeg tree that could be ported as op
mutators. Survey only — nothing implemented.

---

## 0. What was already covered

The fuzzer already has two operators sourced from the same algorithm families
FFmpeg uses internally. These are **not** FFmpeg ports — they are independent
re-implementations — so there is no duplication concern, but the existing
operators define the bar for what a FFmpeg port would need to displace.

| Existing operator | Location | Algorithm family | Note |
|---|---|---|---|
| `golomb` | `core/mutations/structured.py:1614` | Golomb/Rice coding | Generic Rice parameter k=4; does not use FFmpeg's `get_ueg_31`/`get_ue`/`golomb_rice_decode` tables |
| `rle` | `core/mutations/structured.py:1211` | Run-length encode/decode | Generic RLE roundtrip; does not use FFmpeg's `ff_rle_encode`/`ff_rle_count_pixels` parameter tables |

**Conclusion:** a FFmpeg port of either would need to demonstrate a mutation
*shape* the generic version cannot produce (e.g. CABAC-aware Golomb parameters,
RLE tables with codec-specific bpp flags), not just re-implement the same
math with a different source cite.

---

## 1. Survey methodology

Five explore subagents were dispatched in parallel across FFmpeg subsystems.
Two (libavcodec, libswscale/libswresample) timed out due to the large file
counts and were partially covered manually. Findings below are cross-referenced
against the fuzzer's operator taxonomy in `operator_registry.py` and tagged
with where they would land if ported.

**Target constraint (Hard Rule 12):** any port lands in
`_CATEGORIES` → one band, with a `_op_<name>` handler on `OperatorEngine`
(or a `MutatorBase` subclass via `register_mutator()`) plus an `_AVAILABLE`
predicate. The function itself lives in `core/mutations/` (band-matched).

---

## 2. Candidates by subsystem

### 2.1 libavutil — Primitives

Pure utility code with no codec-specific logic. These are the lowest-effort
ports (fewest internal dependencies) and map cleanly onto existing mutator
patterns.

| # | Algorithm | File:line | Band if ported | Mutator value |
|---|---|---|---|---|
| U1 | `av_murmur3_init_seeded` / `av_murmur3_update` / `av_murmur3_final` | `libavutil/murmur3.c` | regularity | Overwrite region with bytes derived from a Murmur3 hash of the input. Produces avalanche-distributed bytes that look structured but aren't — exercises parsers that assume hash output uniformity. Different mixing constants from the fuzzer's existing CRC-only checksum logic. |
| U2 | `ff_sfc64_get` / `ff_sfc64_init` | `libavutil/sfc64.h` | adaptive | A PRNG with fundamentally different statistical structure than `RandPool`. Could feed a mutator that generates values biased toward SFC64's output distribution rather than RandPool's uniform/weighted draws. **Note:** Hard Rule 16 says "Always use the prng in rand_pool.py" — this would *feed* a mutator, not replace RandPool. |
| U3 | `av_crc` (configurable polynomial) | `libavutil/crc.c` | regularity | The fuzzer's CRC mutator (`_op_crc_learn`) models CRCs as GF(2) polynomial recovery. FFmpeg's `av_crc` ships with 13 predefined polynomials (CRC-8, CRC-16-CCITT, CRC-32, CRC-32C, etc.). A mutator that overwrites checksums using a *wrong but plausible* polynomial exercises validation paths that assume CRC-32 specifically. |
| U4 | `av_md5_sum` | `libavutil/md5.c` | regularity | 64 rounds of mixing per 512-bit block. A mutator that produces MD5-diffusion-shaped output rather than random bytes could hit parsers with cryptographic assumptions. |
| U5 | `av_bswap64` / `AV_RB16` / `AV_WL32` | `libavutil/bswap.h`, `libavutil/intreadwrite.h` | bit | Cross-platform endianness macros with fixed width/endianness combos. Partial overlap with existing `endian_convert` but the explicit LE/BE per-width tables could produce patterns the current operator's uniform-width approach misses. |

### 2.2 libavcodec — Codec Core

Algorithms with real codec-specific logic. Stronger mutation value but
higher integration cost.

| # | Algorithm | File:line | Band | Mutator value |
|---|---|---|---|---|
| C1 | `ff_h264_decode_mb_cavlc` — Exp-Golomb coding | `libavcodec/h264_cavlc.c:665` | regularity | H.264 CAVLC uses context-adaptive Exp-Golomb with Rice parameters derived from neighboring coefficients. Distinct from the fuzzer's generic `golomb` operator, which uses fixed k=4. A H.264-aware Golomb mutator could produce codeword patterns that exercise VLC decoder branches. |
| C2 | `ff_init_cabac_decoder` + CABAC renorm | `libavcodec/cabac.h:49`, `cabac.c` | structural | Arithmetic coding with adaptive probability states. Mutates the *probability model* itself (state transitions) rather than just the bitstream — unique mutation shape that targets the decoder's state-machine branches. |
| C3 | `ff_ref_fdct` / `ff_ref_idct` | `libavcodec/dctref.c:59,95` | regularity | Reference DCT/IDCT with fixed 8×8 block structure. Could produce coefficient patterns (e.g. all-DC, max-AC, zigzag order) that exercise IDCT saturation/clamping. Distinct from `spectral_peak` which uses synthetic patterns. |
| C4 | `ff_rle_encode` / `ff_rle_count_pixels` | `libavcodec/rle.c:53,28` | regularity | Codec-specific RLE with bpp parameter (1/4/8/16/32) and a `same` flag. The fuzzer's generic `rle` operator doesn't model these parameters — a port would produce RLE streams that match specific codec expectations. |
| C5 | `ff_er_frame_start` / `ff_er_frame_end` | `libavcodec/error_resilience.c:812,910` | structural | Error concealment state machine. A mutator that produces inputs designed to *trigger* error concealment (missing blocks, misaligned motion vectors) would exercise recovery paths. |
| C6 | Quantization tables | `libavcodec/h264_slice.c:174` | regularity | `ff_h264_dequant4_coeff_init` and `ff_h264_quant_div6` — quantization parameter → coefficient scaling maps. A mutator that perturbs quantization tables could exercise QP boundary code paths. |

### 2.3 libavfilter — Signal Processing

| # | Algorithm | File:line | Band | Mutator value |
|---|---|---|---|---|
| F1 | Motion estimation (SAD + sub-pixel) | `libavfilter/vf_mestimate.c:80-329` | structural | Block-based motion estimation uses Sum of Absolute Differences to find matching blocks. A mutator that generates structured spatial offset patterns (block-aligned, half-pixel offsets) would exercise motion-compensated code paths — relevant for video codec targets. |
| F2 | Color space conversion LUTs | `libavfilter/vf_colorspace.c:105-913` | regularity | RGB↔YUV↔HSV matrix coefficients and gamma ramps. A mutator that overwrites a region with values from a generated color-space LUT (e.g. gamma-corrected ramp) produces structured byte values that exercise color processing fast paths. |
| F3 | FFT denoise | `libavfilter/af_arnndn.c:409-450` | regularity | Forward/inverse transform pairs for frequency-domain denoising. Could produce coefficient patterns that exercise transform-coded format parsers. |

### 2.4 libavformat — Container Parsing

| # | Algorithm | File:line | Band | Mutator value |
|---|---|---|---|---|
| M1 | WAV RIFF chunk parser | `libavformat/wavdec.c:189` | format | `wav_parse_fmt_tag` — RIFF/fmt/data/fact chunk parsing with endianness conversion. A WAV-specific chunk mutator (reorder chunks, corrupt fmt fields, inject metadata) would target WAV demuxer code paths. **Requires a WAV target** (not currently in `targets/`). |
| M2 | Index entry management | `libavformat/seek.c:64` | structural | `ff_add_index_entry` — timestamp/position indexing. A mutator that produces structured timestamp/offset pairs (monotonic, wrapped, sparse) would exercise seek/index validation in demuxers. |
| M3 | Probe detection | `libavformat/wavdec.c:161` | structural | `wav_probe` — signature-based format detection. Mutator that produces near-miss magic bytes to probe detection thresholds. |

### 2.5 libswscale / libswresample

| # | Algorithm | File:line | Band | Mutator value |
|---|---|---|---|---|
| S1 | Sw_scale (interpolation) | `libswscale/swscale.c` | regularity | Pixel interpolation algorithms (nearest, bilinear, bicubic, Spline36). Could produce resampling-kernel-shaped byte patterns. |
| S2 | Audio resampling | `libswresample/resample.c` | regularity | Sinc/linear interpolation resamplers. Coefficient patterns for audio format parsers. |

---

## 3. Priority-ranked port list

Ranked by (implementation effort × fuzzer impact). All require a format target
to fully validate — see gating questions in §5.

| Rank | Candidate | Est. effort | Why now |
|---|---|---|---|
| 1 | **U3: configurable CRC-32** (`av_crc`) | Low | Fuzzer's CRC recovery assumes CRC-32 by spec (PNG, ZIP) via `crc32.py`. FFmpeg ships 13 polynomials; a mutator using CRC-32C or CRC-8 produces checksums that pass *a* CRC check but not the one the target expects — targeting false-positive acceptance paths. Builds on existing CRC learner infrastructure. |
| 2 | **C1: H.264-aware Golomb** (CAVLC Exp-Golomb) | Medium | Extends the generic `golomb` op with codec-specific Rice parameters. Requires an H.264 target to validate against. The generic `golomb` op is marked unmeasured by the harness (TODO.md:33). |
| 3 | **U1: MurmurHash3-based** (`av_murmur3`) | Low | Pure byte transformation — no codec dependency. Could seed a `regularity` op that overwrites a region with hash-avalanche-distributed bytes, catching parsers that reject inputs based on entropy assumptions. |
| 4 | **C4: RLE with codec params** (`ff_rle_encode`) | Low-Medium | Extends generic `rle` op with bpp/same-flag tables. Requires a BMP/PIC target (both use RLE). |
| 5 | **F1: motion estimation patterns** (`vf_mestimate`) | Medium-High | Structured spatial offsets. Requires a video target (H.264/VP9). |
| 6 | **C2: CABAC probability mutator** (`cabac`) | High | Operates on a bit-level probability model — unique mutation shape. Requires a CABAC-decoding target. |
| 7 | **F2: color space LUT mutator** (`vf_colorspace`) | Medium | Structured LUT values. Requires a color/image target. |
| 8 | **U2: SFC64-based mutator** (`sfc64`) | Medium | Different PRNG distribution than RandPool. **Needs explicit approval** — Hard Rule 16 says always use RandPool; this would feed a mutator that draws from RandPool but uses SFC64 *internally* for a specific distributional shape. |
| 9 | **C3: DCT coefficient patterns** (`dctref`) | Medium | Extends `spectral_peak` with actual transform values. Requires a DCT-based codec target. |
| 10 | **M1/M2: WAV/index mutators** (`wavdec`, `seek`) | Medium | Container-specific. Requires WAV target (not in `targets/`). |

---

## 4. How to wire up (Hard Rule 12 checklist)

1. **Register** in `src/fuzzer_tool/core/operator_registry.py`:
   - Add operator name to the right band in `_CATEGORIES`
   - Add availability predicate in `_AVAILABLE` (or mark `None` for unconditional)
2. **Implement** the function in `core/mutations/<band>.py` (for regularity)
   or `core/mutations/generic.py` (for structural/bit/byte)
3. **Add handler** `_op_<name>` in `services/operators.py` OR create a
   `MutatorBase` subclass and call `REGISTRY.register_mutator()`
4. **Add format sniffer** in `_FORMAT_SNIFFERS` if the operator is target-format-specific
5. **Add regression + adversarial tests** (Hard Rule 23) in `tests/`
6. **Wire into `tools/build_targets.sh`** if a new target is needed for validation
7. **Update `docs/DEEP_DIVE.md`** with the new operator (Hard Rule 10)
8. **Update `docs/architecture.png`** if adding a new subsystem (Hard Rule 44)

---

## 5. Gating questions (must answer before implementing)

| Candidate | Gating question |
|---|---|
| C1 (H.264 Golomb) | Does the fuzzer have or plan to have an H.264 target? Without one, `cavlc`-shaped input has no parser to exercise. |
| C2 (CABAC) | Same — CABAC is H.264-specific. No H.264 target → no mutation value. |
| C4 (RLE) | Does the fuzzer have a BMP or PIC target that uses FFmpeg's RLE mode? |
| F1 (motion estimation) | Does the fuzzer have a video codec target (H.264, VP9, etc.)? |
| U2 (SFC64) | Hard Rule 16 explicitly forbids replacing RandPool. This candidate requires proof that SFC64 *seeds* distribution shape without bypassing RandPool reproducibility. **Approvals needed.** |
| M1 (WAV) | No WAV target exists in `targets/`. Would need `tools/vendor_*` + `targets/` entry first. |

---

## 6. What was rejected

- **U5: bswap/intreadwrite macros** — pure macros, not algorithms. The existing
  `endian_convert` op already covers cross-endianness field swaps. No added
  mutation shape.
- **M3: probe detection** — overlaps with existing magic-byte probing in format
  mutators. No new surface.
- **S1/S2: swscale/swr resample** — coefficient-based interpolation that
  produces smooth ramps. Already approximated by existing `monotone_fill` and
  `spectral_peak` operators. Low marginal value.

---

## 7. Measurement baseline needed

Before any port, capture:
- `pytest tests/test_operator_smoke.py` — current op count and coverage
- `pytest tests/test_new_operators.py` — pass/fail per operator
- A 10k-exec baseline run on a representative target to measure any EPS impact
  (Hard Rule 41)
