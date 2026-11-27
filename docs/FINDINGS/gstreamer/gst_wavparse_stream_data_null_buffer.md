# SIGSEGV in `gst_wavparse_stream_data` (GStreamer WAV Demuxer)

**File**: `subprojects/gst-plugins-good/gst/wavparse/gstwavparse.c:2212` (1.28.9); `:2184` on 1.28.0 — same bug, different line drift
**Severity**: Medium — crafted RIFF/WAVE input crashes any GStreamer audio pipeline with SIGSEGV
**Root cause**: an inflated `blockalign` on a non-PCM fmt chunk rounds `max_buf_size` down to 0, so `gst_adapter_take_buffer (adapter, 0)` returns NULL (`gstadapter.c:1083`), and the buffer-offset assignment dereferences that NULL unconditionally
**Discovered by**: fuzzer-tool via `targets/gstreamer_read.c` — no-ASAN build (`_noasan.so`) with AFL edge coverage
**ASAN replay**: confirmed SIGSEGV on ASAN build (`gstreamer_read_1.28.9_asan.so`) — exit 139

---

## Target

[GStreamer](https://gstreamer.freedesktop.org/) 1.28.9 — gst-plugins-good `wavparse` + core `gstpad`.

**Library versions** (vendored at `vendoring/gstreamer-1.28.9/`):

| Library / Component | Version |
|---------------------|---------|
| gst-plugins-good | 1.28.9 |
| gst (core) | 1.28.9 |
| gst-plugins-bad/ugly/libmfx | 1.28.9 (wavparse only) |

**Affected versions** — the unguarded round-down
`wav->max_buf_size -= (wav->max_buf_size % wav->blockalign)` in `gst_wavparse_stream_headers` is
present in every tree vendored for this analysis, released and dev alike:

| Version | Round-down at | Verdict |
|---------|---------------|---------|
| 1.28.0 | `gstwavparse.c:1786` | SEGV at `:2184` (verified, ASAN harness) |
| 1.28.4 | `gstwavparse.c:1811` | same code, unguarded (source check) |
| 1.28.8 | `gstwavparse.c:1814` | same code, unguarded (source check) |
| 1.28.9 | `gstwavparse.c:1814` | SEGV at `:2212` (verified, ASAN harness) |
| 1.29.2 dev (`1.28.9-88-g7bbbda8b30`) | `gstwavparse.c:1854` | same code, unguarded (source check) |

Not fixed in any released 1.28.x or the current dev tree; the fix exists only as this repo's
`patches/gstwavparse_null_max_buf_size.patch` (auto-applied by `tools/vendor_gstreamer.sh`).

---

## Fuzz Target

`targets/gstreamer_read.c` — feeds arbitrary bytes into the RIFF/WAVE demuxer via `GstElement` chain:
`sinkpad (probe/decode) → gst_pad_push → gst_wavparse_chain → gst_wavparse_stream_data → srcpad push`.

---

## Fuzzing Configuration

- **Engine**: fuzzer-tool (AFL-compatible SHM edge coverage via `afl_shim.c`)
- **Coverage**: AFL edge coverage (`__afl_map_edge`)
- **Inputs**: incremental `revbin_permute` mutations on corpus WAV/Riff seeds

---

## Results

| Metric | Value |
|--------|-------|
| Total crashes discovered | 1 (this report) |
| Crash input size | 173 bytes (original find) → 44 bytes (`tmin` minimization) → 60 bytes (deterministic builder, see [Regression Test](#regression-test)) |
| Crash input type | RIFF/WAVE stream — non-PCM fmt chunk (tag `0x0070` Voxware AC8) with inflated `blockalign` + a `data` chunk declaring more than `blockalign` bytes |
| Trigger | non-PCM fmt skips the PCM-only `blockalign` sanity → `blockalign` 8192 > `max_buf_size` 7065 rounds it down to 0 → `desired` = 0 → `gst_adapter_take_buffer (0)` returns NULL → unchecked `GST_BUFFER_OFFSET` store |
| Crash type | `SIGSEGV` (signal 11, returncode -11) |
| Severity | **Medium** |
| Exploitability | Low (local DoS via crafted file) |

---

## Bug

`gst_wavparse_stream_data` writes the buffer offset/end-offset immediately after making the buffer
writable, without checking the result:

```c
  obtained = gst_buffer_get_size (buf);

  /* our positions in bytes */
  pos = wav->offset - wav->datastart;
  nextpos = pos + obtained;

  /* update offsets, does not overflow. */
  buf = gst_buffer_make_writable (buf);
  GST_BUFFER_OFFSET (buf) = pos / wav->bytes_per_sample;
  GST_BUFFER_OFFSET_END (buf) = nextpos / wav->bytes_per_sample;
```

`GST_BUFFER_OFFSET` stores into the `GstBuffer` structure (field slot at offset `0x60` in the
release build); the disassembly at the crash PC confirms the faulting instruction is exactly
`mov %rax,0x60(%r14)` — a write whose effective address was `0x60`, i.e. base register value `0x0`.

### Root cause

`gst_buffer_make_writable (buf)` returned `NULL` — but only because `buf` was **already NULL** when it
reached that call. The NULL originates three frames of state earlier, in `gst_wavparse_stream_headers`:

1. The fmt tag `0x0070` (Voxware AC8) is not `WAVE_FORMAT_PCM`, so the only `blockalign` sanity check —
   `blockalign <= channels * ceil(bits/8)` at `gstwavparse.c:1258-1260` — does not apply; the `default:`
   fmt branch (`:1269-1274`) only checks `av_bps <= blockalign * rate`, which an inflated `blockalign`
   trivially satisfies.
2. `gst_wavparse_stream_headers` then computes the 40 ms byte budget and floors it at 4096:
   `max_buf_size = MAX(0.04 * bps, MAX_BUFFER_SIZE)` = 7065 for `av_bps` 176640 (`:1812`), and rounds it
   down to a whole number of blocks: `max_buf_size -= max_buf_size % blockalign` (`:1813-1814`).
   With `blockalign` 8192 > 7065, `7065 % 8192 = 7065`, so `max_buf_size` becomes **0**. There is no
   guard keeping it `>= blockalign`.
3. In `gst_wavparse_stream_data`, `desired = MIN(dataleft, max_buf_size * |segment.rate|)` =
   `MIN(~1.26 GB, 0)` = 0 (`:2126-2128`). The `desired >= blockalign` guard on the `desired` round-down
   (`:2130-2131`) is false, so `desired` stays 0 — and 0 is not less than the adapter's available bytes,
   so execution falls straight to `buf = gst_adapter_take_buffer (adapter, desired)` (`:2180`).
4. `gst_adapter_take_buffer (adapter, 0)` hits `g_return_val_if_fail (nsize > 0, NULL)`
   (`gstadapter.c:1083`), returning NULL; the GLib critical warning is not fatal, so parsing continues.
5. `gst_buffer_get_size (NULL)` and `gst_buffer_make_writable (NULL)` both return 0/NULL without
   crashing, and the next statement, `GST_BUFFER_OFFSET (buf) = ...`, executes the store
   `mov %rax,0x60(%r14)` with `%r14 = 0` — the SIGSEGV at fault address `0x60` (`:2212`).

---

## Trigger Chain

1. **Fuzz driver** (`fuzz_gstreamer` at `targets/gstreamer_read.c:163`) pushes mutated buffer chunks
   through the demuxer via `gst_pad_push` → `gst_pad_push_data` → `gst_pad_chain_data_unchecked`.
2. **WAV demuxer** (`gst_wavparse_chain` at `gstwavparse.c:2475`) → `gst_wavparse_stream_headers`
   parses the fmt chunk: tag `0x0070` takes the `default` branch, so `blockalign` 8192 is accepted.
3. **Round-down to zero**: `stream_headers` computes `max_buf_size = MAX(0.04*bps, 4096) = 7065` and
   applies `max_buf_size -= max_buf_size % blockalign` → 0 (`:1813-1814`).
4. **Zero-byte request**: `gst_wavparse_stream_data` derives `desired = MIN(dataleft, 0 * rate) = 0`
   (`:2126-2128`) and calls `gst_adapter_take_buffer (adapter, 0)` (`:2180`).
5. **NULL propagation**: the adapter's `g_return_val_if_fail` returns NULL (`gstadapter.c:1083`);
   `gst_buffer_make_writable (NULL)` stays NULL (`:2211`).
6. **Unchecked deref**: `GST_BUFFER_OFFSET (buf) = pos / wav->bytes_per_sample;` writes to the NULL
   buffer (`:2212`) → SIGSEGV (signal 11). On 1.28.0 the same chain lands at `:2184`.

---

## Crash Input

The crash exists in three equivalent forms: the original **173-byte** find (base64 reproducer in the
crash `*.sh`), the **44-byte** `tmin` minimization (envelope below), and a **60-byte** deterministic
builder (`gst_wav_blockalign_zero_take` in `tools/gen_ffmpeg_gst_seeds.py`). All three produce
identical verdicts on every harness (see [Cross-Library Verdict](#cross-library-verdict)).

```
00000000  52 49 46 46 a4 00 00 00 57 41 56 45 66 6d 74 20   |RIFF....WAVEfmt |
00000010  10 00 00 00 70 00 02 00 44 01 01 00 00 b2 02 00   |....p...D.......|
00000020  00 20 08 00 64 61 74 61 80 00 00 04 f8 01 7e 5e   |. ..data......~^|
```

**RIFF/WAVE envelope breakdown (minimized 44-byte input):**

| Offset | Bytes | Field | Value |
|--------|-------|-------|-------|
| 0 | RIFF + a4 00 00 00 | RIFF size | 164 — stale value inherited from the original 173-byte find; wavparse tolerates the mismatch (actual minimized file: 44 bytes) |
| 8 | WAVE | form type | |
| 12 | fmt + 10 00 00 00 | fmt chunk size | 16 |
| 20 | 70 00 | format tag | 0x0070 |
| 22 | 02 00 | channels | 2 |
| 24 | 44 01 01 00 | sample rate | 65860 |
| 28 | 00 b2 02 00 | byte rate | 176640 (2 × 1 byte × 65860 should be 131720) |
| 32 | 00 20 | block align | 8192 (should be 2) |
| 34 | 08 00 | bits per sample | 8 |
| 36 | data + 80 00 2b 4b | data size | 0x4B2B0080 ≈ 1.26 GB, 0 bytes after it |

`fields (riff; baseline: parent 54e73eab8615a0e1)` — `fmt[0].data @0x14 +16` value differs from baseline:
`700002004401010000b2020000200800` (was `700002004401010000b2020008000020`).

---

## Crash Evidence

```
Signal: SIGSEGV (11)
RIP:    0x7ffff6ffdb7b
Fault:  0x60
Error:  Segmentation fault.

#0  0x00007ffff6ffdb7b in gst_wavparse_stream_data (wav=wav@entry=0xc0eca0, flushing=flushing@entry=0) at ../../vendoring/gstreamer-1.28.9/subprojects/gst-plugins-good/gst/wavparse/gstwavparse.c:2212
#1  0x00007ffff7000f95 in gst_wavparse_chain (pad=<optimized out>, parent=parent@entry=0xc0eca0, buf=buf@entry=0xbf8dc0) at .../gstwavparse.c:2475
#2  0x00007ffff708f48c in gst_pad_chain_data_unchecked (pad=pad@entry=0xc8b4f0, type=type@entry=4112, data=data@entry=0xbf8dc0) at ../../vendoring/gstreamer/gst/gstpad.c:4605
#3  0x00007ffff70908f8 in gst_pad_push_data (pad=pad@entry=0xc8d440, type=type@entry=4112, data=data@entry=0xbf8dc0) at ../../vendoring/gstreamer/gst/gstpad.c:4898
#4  0x00007ffff70905f8 in gst_pad_push (pad=pad@entry=0xc8d440, buffer=0xbf8dc0) at ../../vendoring/gstreamer/gst/gstpad.c:5017
#5  0x00007ffff6e77d42 in fuzz_gstreamer (buf=buf@entry=0x7ffff7bbaef0 "RIFF\244", size=size@entry=173) at /fuzz/src/targets/gstreamer_read.c:163
```

Shim symbolization resolved via module DWARF → `gst_wavparse_stream_data` @ `gstwavparse.c:2212`.


## Crash Metadata

| Field | Value |
|-------|-------|
| Signal | `SIGSEGV` (returncode −11) |
| Fault address / RIP | `0x7f4db21fdb7b` (RIP), fault `0x60` |
| RSP | `0x7fffffffcad0` |
| Execs to find | 196,677 |
| Corpus at find | 5,141 entries |
| Elapsed | 00:03:13 |
| Parent seed | `54e73eab8615a0e1` |
| Mutation ops | `revbin_permute` (site 11) |
| Target SHA256 | `ff8b9229a79cb5d1` |
| Exploitability | LOW |

---

## Exploitability Assessment

| Factor | Assessment |
|--------|-----------|
| Crash determinism | **Deterministic** — SIGSEGV reproduced on the no-ASAN build and the ASAN build replay; identical verdict on 1.28.0 (`:2184`) and 1.28.9 (`:2212`) |
| Trigger depth | **Shallow** — any non-PCM fmt tag + `blockalign` > 4096 (and > 0.04·av_bps) + a `data` chunk declaring ≥ `blockalign` bytes; the 44-byte minimization is a complete RIFF/WAVE header with zero data payload |
| Preconditions | None — input fully self-contained |
| Signal type | SIGSEGV (write to NULL + buffer field offset) |
| Memory safety | NULL-deref via unchecked `GST_BUFFER_OFFSET` write |
| Reach | Any application feeding GStreamer a crafted WAV via `gst_element` API |
| Severity | **Medium** — reliable DoS on audio playback of crafted files |

---

## Seed Minimization

Ran `fuzzer-tool tmin` with `--inprocess --snapshot-restore globals` on the original 173-byte seed:

```
fuzzer-tool tmin -t 10 -O /tmp/minimized_crash.bin \
  /home/dclavijo/fuzzing/builds/gstreamer_read_1.28.9_noasan.so \
  /home/dclavijo/fuzzing/gstreamer_read_1.28.9_noasan.so/crashes/crash_1795708263_f443105b_sig_signal11.bin \
  --inprocess --snapshot-restore globals
```

| Metric | Value |
|--------|-------|
| Original size | 173 bytes |
| Minimized size | **44 bytes** |
| Reduction | 75% |
| Post-minimization crash | Confirmed (exit 139 / SIGSEGV) |

The minimized 44-byte input still triggers the same SIGSEGV at `gstwavparse.c:2212` — the NULL-buffer dereference in `gst_wavparse_stream_data`. Note what the minimization proves: the trigger is **shallow**, not fragile. Nothing but the 44-byte header needs to exist — no data payload, no chunk-ordering tricks. Three header fields do all the work (fmt tag `0x0070`, `blockalign` 8192, and a `data` size >= `blockalign`); a field-by-field bisect against both GStreamer versions confirmed each field's role independently.

---

## Suggested Fix

Upstream fix, applied as `patches/gstwavparse_null_max_buf_size.patch` via
`tools/vendor_gstreamer.sh`. The patch adds three defensive guards (line numbers below are
pre-patch `gstwavparse.c` 1.28.9; other releases drift — 1.28.0 rounds down at `:1786`):

**1. Keep `max_buf_size` >= `blockalign` after rounding** (`gstwavparse.c:1813-1814`).

```c
if (wav->blockalign > 0) {
    wav->max_buf_size -= (wav->max_buf_size % wav->blockalign);
    /* Make sure we can fit at least one complete block. */
    if (wav->max_buf_size == 0)
        wav->max_buf_size = wav->blockalign;
}
```
If `blockalign` (e.g. 8192) is larger than `max_buf_size` (e.g. 7065),
`max_buf_size` rounds down to 0, then `desired = MIN(dataleft, 0 * rate) = 0`.
This drives `gst_adapter_take_buffer(adapter, 0)` to return NULL (the `g_return_val_if_fail`
at `gstadapter.c:1083`), verified by hand-stepping `gstwavparse.c` and by the ASAN harnesses
on 1.28.0 and 1.28.9. Guard prevents this by clamping to `blockalign`. This is the guard that
matters: it fixes the state corruption at its origin, in `gst_wavparse_stream_headers`.

**2. Never request a zero-sized buffer** (after the `desired` round-down, `gstwavparse.c:2130-2131`).

```c
/* Never request a zero-sized buffer. */
if (G_UNLIKELY (desired == 0)) {
    GST_WARNING_OBJECT (wav, "Cannot fetch zero bytes of audio data");
    return GST_FLOW_EOS;
}
```
Bails early with `GST_FLOW_EOS` before calling `gst_adapter_take_buffer`.

**3. NULL-check every `gst_buffer_make_writable()` result** (the two call sites at
`gstwavparse.c:2194` short-buffer resize and `:2211` offset update — the latter directly
preceding the `:2212` crash store).

Both call sites (short-buffer resize path and main offset-update path)
now guard against NULL:

```c
buf = gst_buffer_make_writable (buf);
if (G_UNLIKELY (buf == NULL))
    goto buffer_error;
```

with a single error label:

```c
buffer_error:
{
    GST_WARNING_OBJECT (wav, "Failed to make audio buffer writable");
    return GST_FLOW_ERROR;
}
```

This matches the existing validation in `vpk_read_header`
(`ch_layout.nb_channels <= 0` rejected).

---

## Cross-Library Verdict

The same bytes are parsed cleanly by both FFmpeg releases vendored for this analysis, which makes
this the cross-library pack's first **live** (unfixed) divergence — see
`tools/gen_ffmpeg_gst_seeds.py` (`gst_wav_blockalign_zero_take`) and
[DEEP_DIVE.md](../../DEEP_DIVE.md):

| Input | FFmpeg 9.0.2 | FFmpeg 9.1 | GStreamer 1.28.0 | GStreamer 1.28.9 |
|-------|--------------|------------|------------------|------------------|
| 173-byte original find | clean | clean | SEGV `:2184` | SEGV `:2212` |
| 44-byte minimized seed | clean | clean | SEGV `:2184` | SEGV `:2212` |
| 60-byte pack builder | clean | clean | SEGV `:2184` | SEGV `:2212` |

FFmpeg rounds the same field the **other** way: `wav_read_packet` sizes the read **up** to
`block_align` and clamps with `FFMIN(size, left)` (`libavformat/wavdec.c:803-808`), so an inflated
`blockalign` only shortens the stream — no zero-size request exists on that side. GStreamer rounds
**down**, and the subtraction is what zeroes the request.

---

## Regression Test

The 44-byte minimized seed as a C array (for in-tree C harnesses):

```c
/* Trigger: crafted RIFF/WAVE input → SIGSEGV at gst_wavparse_stream_data (gstwavparse.c:2212) */
static const unsigned char gs_wav_crash[] = {
    0x52, 0x49, 0x46, 0x46, 0xa4, 0x00, 0x00, 0x00, 0x57, 0x41, 0x56, 0x45, 0x66, 0x6d, 0x74, 0x20,
    0x10, 0x00, 0x00, 0x00, 0x70, 0x00, 0x02, 0x00, 0x44, 0x01, 0x01, 0x00, 0x00, 0xb2, 0x02, 0x00,
    0x00, 0x20, 0x08, 0x00, 0x64, 0x61, 0x74, 0x61, 0x80, 0x00, 0x2b, 0x4b,
};
```

The regression also lives in the cross-library pack:

- `tools/gen_ffmpeg_gst_seeds.py` — deterministic 60-byte builder `gst_wav_blockalign_zero_take`
  (same verdicts: FFmpeg 9.0.2/9.1 clean, GStreamer 1.28.0 `:2184` / 1.28.9 `:2212` SEGV);
- `tests/test_regression_ffmpeg_gst_seeds.py` — `TestWavBlockalignSeed` asserts the builder's fmt
  fields exceed both the 40 ms budget and the 4096 floor, and re-derives the crash preconditions
  from the bytes.

---

## Upstream Status

Not reported upstream at the time of discovery. The next step would be a confidential security report through GStreamer's security process. See [GStreamer security](https://gitlab.freedesktop.org/gstreamer/gstreamer/-/work_items/5485) for the work item associated with this class of bug (wavparse NULL buffer dereference). The round-down remains unguarded in every release checked for this report — 1.28.0 through 1.28.9 and the 1.29 dev tree — so upstream users of those versions are exposed; the three-guard fix ships only in this repo as `patches/gstwavparse_null_max_buf_size.patch`.

---

## See Also

- **[Integer divide-by-zero in `vpk_read_packet`](../ffmpeg/vpk_divide_by_zero.md)** — SIGFPE via zeroed `ch_layout` copied back by `avformat_find_stream_info`; same fuzzer, same class of corrupt-state propagation bug.
- **[Cross-library FFmpeg vs GStreamer seed pack](../../DEEP_DIVE.md)** — this crash as the pack's `gs_wav_blockalign_zero_take` entry, the only live (unfixed in 1.28.x) divergence of the set.
- **[`crash-analysis` skill](../../skills/crash-analysis/SKILL.md)** — project crash analysis methodology.
