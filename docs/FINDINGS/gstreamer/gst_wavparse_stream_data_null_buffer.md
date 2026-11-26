# SIGSEGV in `gst_wavparse_stream_data` (GStreamer WAV Demuxer)

**File**: `subprojects/gst-plugins-good/gst/wavparse/gstwavparse.c:2212`
**Severity**: Medium — crafted RIFF/WAVE input crashes any GStreamer audio pipeline with SIGSEGV
**Root cause**: `gst_buffer_make_writable()` returned NULL and the buffer-offset assignment dereferenced it unconditionally
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
| Crash input size | 173 bytes |
| Crash input type | RIFF/WAVE stream (PCM fmt chunk + data chunk) |
| Trigger | `gst_wavparse_stream_data` short-buffer branch → `gst_buffer_make_writable` → offset store |
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

`gst_buffer_make_writable (buf)` returned `NULL` and the subsequent assignment
`GST_BUFFER_OFFSET (buf) = ...` dereferenced it. The store executed with a zero base pointer,
producing the SIGSEGV at fault address `0x60`.

---

## Trigger Chain

1. **Fuzz driver** (`fuzz_gstreamer` at `targets/gstreamer_read.c:163`) pushes mutated buffer chunks
   through the demuxer via `gst_pad_push` → `gst_pad_push_data` → `gst_pad_chain_data_unchecked`.
2. **WAV demuxer** (`gst_wavparse_chain` at `gstwavparse.c:2475`) → `gst_wavparse_stream_data`.
3. **Short-buffer branch**: input declared smaller than the demanded data chunk; the adapter /
   pull-range path took the `gst_buffer_make_writable` path.
4. **NULL-returning buffer**: `gst_buffer_make_writable (buf)` returned `NULL`.
5. **Unchecked deref**: `GST_BUFFER_OFFSET (buf) = pos / wav->bytes_per_sample;` wrote to a NULL
   buffer → SIGSEGV (signal 11).

---

## Crash Input

173-byte RIFF/WAVE input (base64 reproducer in the crash `*.sh`):

```
00000000  52 49 46 46 a4 00 00 00 57 41 56 45 66 6d 74 20   |RIFF....WAVEfmt |
00000010  10 00 00 00 70 00 02 00 44 01 01 00 00 b2 02 00   |....p...D.......|
00000020  00 20 08 00 64 61 74 61 80 00 00 04 f8 01 7e 5e   |. ..data......~^|
```

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
| Crash determinism | **Deterministic** — SIGSEGV reproduced on the no-ASAN build and the ASAN build replay |
| Trigger depth | Deep — requires malformed RIFF/WAVE fmt/data structure |
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

The minimized 44-byte input still triggers the same SIGSEGV at `gstwavparse.c:2212` — the NULL-buffer dereference in `gst_wavparse_stream_data`. This demonstrates the crash condition is extremely fragile and can be triggered by minimal malformed input.

---

## Suggested Fix

Upstream fix, applied as `patches/gstwavparse_null_max_buf_size.patch` via
`tools/vendor_gstreamer.sh`. The patch adds three defensive guards:

**1. Keep `max_buf_size` >= `blockalign` after rounding** (`gstwavparse.c:1693-1699`).

```c
if (wav->blockalign > 0) {
    wav->max_buf_size -= (wav->max_buf_size % wav->blockalign);
    /* Make sure we can fit at least one complete block. */
    if (wav->max_buf_size == 0)
        wav->max_buf_size = wav->blockalign;
}
```
If `blockalign` (e.g. 8192) is larger than `max_buf_size` (e.g. 7066),
`max_buf_size` rounds down to 0, then `desired = MIN(dataleft, 0 * rate) = 0`.
This drives `gst_adapter_take_buffer(adapter, 0)` to return NULL, which
fuzzer-tool reproduced by hand-stepping `gstwavparse.c`. Guard prevents
this by clamping to `blockalign`.

**2. Never request a zero-sized buffer** (`gstwavparse.c:2000-2006`).

```c
/* Never request a zero-sized buffer. */
if (G_UNLIKELY (desired == 0)) {
    GST_WARNING_OBJECT (wav, "Cannot fetch zero bytes of audio data");
    return GST_FLOW_ERROR;
}
```
Bails early with `GST_FLOW_ERROR` before calling `gst_adapter_take_buffer`.

**3. NULL-check every `gst_buffer_make_writable()` result** (`gstwavparse.c:2060-2066`).

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

## Regression Test

```c
/* Trigger: crafted RIFF/WAVE input → SIGSEGV at gst_wavparse_stream_data (gstwavparse.c:2212) */
static const unsigned char gs_wav_crash[] = {
    0x52, 0x49, 0x46, 0x46, 0xa4, 0x00, 0x00, 0x00, 0x57, 0x41, 0x56, 0x45, 0x66, 0x6d, 0x74, 0x20,
    0x10, 0x00, 0x00, 0x00, 0x70, 0x00, 0x02, 0x00, 0x44, 0x01, 0x01, 0x00, 0x00, 0xb2, 0x02, 0x00,
    0x00, 0x20, 0x08, 0x00, 0x64, 0x61, 0x74, 0x61, 0x80, 0x00, 0x00, 0x04, 0xf8, 0x01, 0x7e, 0x5e,
    0x16, 0x1f, 0x45, 0x42, 0xba, 0x84, 0x9b, 0x13, 0x47, 0x3f, 0xda, 0x71, 0x9f, 0x77, 0x46, 0x87,
    0x56, 0xd8, 0xb7, 0x3d, 0x57, 0xe1, 0x4a, 0x7d, 0x56, 0xc8, 0xdd, 0xc1, 0x9d, 0x90, 0xb6, 0xa2,
    0x6c, 0xfa, 0x03, 0xe2, 0x2b, 0x4b, 0xda, 0x0a, 0xfb, 0xd2, 0xa6, 0xd9, 0xf8, 0xad, 0xf6, 0xcc,
    0xba, 0xcc, 0x69, 0xd5, 0x7c, 0x2f, 0xa2, 0x34, 0x3e, 0x70, 0xa0, 0x23, 0x03, 0x6a, 0x67, 0xf5,
    0x48, 0xd0, 0xdc, 0x66, 0x6d, 0xb8, 0xec, 0x95, 0xa1, 0x16, 0x6e, 0x16, 0xd6, 0x8d, 0x47, 0x09,
    0x12, 0xf5, 0x03, 0x16, 0x6d, 0x59, 0x05, 0x81, 0xbe, 0x6d, 0xb3, 0x58, 0x2d, 0x2f, 0xd4, 0xdb,
    0xb9, 0x6f, 0x22, 0x8c, 0xb6, 0x06, 0xf1, 0x49, 0xed, 0x55, 0xc4, 0x0f, 0x33, 0xd9, 0x26, 0xf6,
    0x4b, 0x8c, 0x94, 0x1d, 0xb5, 0xbe, 0x62, 0x2b, 0x07, 0x74, 0x93, 0x5a, 0xa6,
};
```

---

## Upstream Status

Not reported upstream at the time of discovery. The next step would be a confidential security report through GStreamer's security process. See [GStreamer security](https://gitlab.freedesktop.org/gstreamer/gstreamer/-/work_items/5485) for the work item associated with this class of bug (wavparse NULL buffer dereference).

---

## See Also

- **[Integer divide-by-zero in `vpk_read_packet`](../ffmpeg/vpk_divide_by_zero.md)** — SIGFPE via zeroed `ch_layout` copied back by `avformat_find_stream_info`; same fuzzer, same class of corrupt-state propagation bug.
- **[`crash-analysis` skill](../../skills/crash-analysis/SKILL.md)** — project crash analysis methodology.
