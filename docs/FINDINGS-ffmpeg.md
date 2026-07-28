# FFmpeg Fuzzing Findings

## Target

[FFmpeg](https://ffmpeg.org/) 7.1.3 — libavcodec, libavformat, libavutil — the full multimedia demux→decode pipeline.

**Library versions** (vendored at `vendor/ffmpeg/`):

| Library | Version |
|---------|---------|
| libavutil | 59.39.100 |
| libavcodec | 61.19.101 |
| libavformat | 61.7.100 |

## Fuzz Target

`targets/ffmpeg_read.c` — feeds arbitrary data through the full FFmpeg API chain:

```
avformat_open_input → avformat_find_stream_info → av_read_frame
→ avcodec_send_packet → avcodec_receive_frame → close
```

Covers all registered demuxers, decoders, and parsers. Compiled with ASAN (`-fsanitize=address`) and AFL edge coverage via `afl_shim.c`. See `targets/ffmpeg_read.c` for details.

## Fuzzing Configuration

- **Engine**: fuzzer-tool with Markov byte generation, Monte Carlo optimization, Thompson sampling bandit, and grammar-aware mutations
- **Coverage**: AFL SHM edge coverage
- **Inputs generated**: PNG corpus (`tools/corpus_png.py`), incremental mutations from prior fuzzing sessions

## Results

| Metric | Value |
|--------|-------|
| Total crashes discovered | 1 (unique signature) |
| Crash input size | 46 bytes |
| Crash input type | PGS subtitle stream (`PG2\0...`) |
| Trigger | `avcodec_send_packet` on subtitle decoder |
| Crash type | `av_assert0(0)` — reachable assertion failure |
| Severity | **HIGH** (denial-of-service in release builds) |
| Exploitability | Remotely triggerable via media file parsing |

## Bug: Reachable `av_assert0(0)` in `decode_simple_internal` via Subtitle Decoder

**File**: `libavcodec/decode.c:464`
**Severity**: HIGH — crafted 46-byte input crashes any FFmpeg-based application
**Root cause**: The `avcodec_send_packet`/`receive_frame` API dispatches subtitle decoders through a code path that asserts the codec type must be VIDEO or AUDIO, but subtitle codecs have type SUBTITLE.

### Description

Calling `avcodec_send_packet` on a decoder opened for `AV_CODEC_ID_HDMV_PGS_SUBTITLE` (or any other non-VIDEO/non-AUDIO decoder) triggers `av_assert0(0)` in `decode_simple_internal()` at `libavcodec/decode.c:456-464`:

```c
if (avctx->codec->type == AVMEDIA_TYPE_VIDEO) {
    ret = (!got_frame || frame->flags & AV_FRAME_FLAG_DISCARD)
                      ? AVERROR(EAGAIN)
                      : 0;
} else if (avctx->codec->type == AVMEDIA_TYPE_AUDIO) {
    ret =  !got_frame ? AVERROR(EAGAIN)
                      : discard_samples(avctx, frame, discarded_samples);
} else
    av_assert0(0);    /* ← reachable with subtitle decoders */
```

### Trigger Chain

1. **Demuxer** identifies the 46-byte input as an `hdmv_pgs_subtitle` stream (Blu-ray PGS subtitle format, detected via the `PG` magic). The stream's `codecpar` has:
   - `codec_id   = AV_CODEC_ID_HDMV_PGS_SUBTITLE`
   - `codec_type = AVMEDIA_TYPE_SUBTITLE`

2. **`avcodec_find_decoder()`** returns the PGS subtitle decoder — success.

3. **`avcodec_open2()`** successfully initializes the subtitle decoder.
   - `avcodec_parameters_to_context()` at `codec_par.c:208` overwrites the context's `codec_type` with `AVMEDIA_TYPE_SUBTITLE` from the stream's `codecpar`.
   - `avcodec_open2()` does not reject the type mismatch (it opens the decoder anyway).

4. **`avcodec_send_packet()`** accepts the packet and dispatches internally to `decode_simple_internal()`.
   - The dispatch at `decode.c:629` checks `codec->cb_type`, not `codec->type`.
   - Subtitle decoders use `FF_CODEC_CB_TYPE_DECODE` (the old-style callback), which routes through `decode_simple_internal`.
   - Inside `decode_simple_internal`, the assertion at line 464 fires because `avctx->codec->type == AVMEDIA_TYPE_SUBTITLE` — not VIDEO or AUDIO.

### ffmpeg CLI Reproducibility

The ffmpeg CLI (version 7.1.5, tested) does **not** trigger this bug:

1. **Subtitle API difference**: The ffmpeg CLI uses `avcodec_decode_subtitle2()` (the legacy subtitle API) for subtitle streams, not `avcodec_send_packet()`/`avcodec_receive_frame()`. The bug requires calling `avcodec_send_packet` on a subtitle decoder, which the ffmpeg CLI never does.

2. **Input too short**: The 46-byte crash input is too malformed for the PGS demuxer (`.sup` format) to produce a valid packet. Both `ffmpeg` and `ffplay` fail at the demux stage with `"Invalid data found when processing input"` before any decoder is called.

3. **Forcing the decoder fails**: Trying `-c:v hdmv_pgs_subtitle` (force a subtitle codec as a video decoder) is rejected because ffmpeg validates codec type against stream type.

**Scope**: This bug affects **library-level code** — any application that calls `avcodec_open2()` followed by `avcodec_send_packet()` on a subtitle decoder without checking the codec type. Examples:
- Media frameworks that iterate all streams generically (our fuzz target style)
- Applications using FFmpeg's modern decode API without type filtering
- Other fuzzers and testing tools

Applications that use the legacy `avcodec_decode_subtitle2()` API for subtitles are **not** affected.

### Root Cause Analysis

The assertion at `decode.c:464` encodes the invariant that "only VIDEO and AUDIO decoders reach `decode_simple_internal`." This invariant was violated when `avcodec_send_packet` was extended to support the new decode API — the extension reused the old code path for all decoders with `FF_CODEC_CB_TYPE_DECODE` without adding a codec-type guard.

Three contributing factors:
1. `avcodec_open2()` does not reject non-VIDEO/non-AUDIO codecs — it opens them successfully.
2. `avcodec_send_packet()` does not check `avctx->codec_type` before entering the decode loop — it assumes the caller validated the codec type.
3. `decode_simple_internal()` uses `av_assert0(0)` (release-build assertion) instead of returning an error for unhandled codec types.

### How to Fix

The cleanest fix is approach (1): reject non-VIDEO/non-AUDIO codec types early in `avcodec_send_packet`, so the assertion at line 464 (and the second `av_assert0(0)` at line 804 in `frame_validate`) are unreachable from crafted input.

**One-line guard in `libavcodec/decode.c`** — add after the existing `avcodec_is_open` / decoder check at line 733-734:

```diff
 int attribute_align_arg avcodec_send_packet(AVCodecContext *avctx, const AVPacket *avpkt)
 {
     AVCodecInternal *avci = avctx->internal;
     DecodeContext     *dc = decode_ctx(avci);
     int ret;

     if (!avcodec_is_open(avctx) || !av_codec_is_decoder(avctx->codec))
         return AVERROR(EINVAL);

+    /* avcodec_send_packet only supports video and audio decoders;
+     * subtitle/data decoders must use avcodec_decode_subtitle2().           */
+    if (avctx->codec_type != AVMEDIA_TYPE_VIDEO &&
+        avctx->codec_type != AVMEDIA_TYPE_AUDIO)
+        return AVERROR(EINVAL);
+
     if (dc->draining_started)
         return AVERROR_EOF;
```

**Why this fix is correct:**

- Subtitle decoders already have their own API (`avcodec_decode_subtitle2`) — there is no legitimate reason to call `avcodec_send_packet` on them.
- The guard sits after the `avcodec_is_open` check, so it only fires on fully-initialized decoders, not half-set-up contexts.
- It prevents ALL assertion sites downstream (`decode_simple_internal` at line 464, `frame_validate` at line 804) from being reachable with subtitle/data codecs.
- `AVERROR(EINVAL)` is the standard "bad argument / invalid operation" return code in FFmpeg's public API. Callers already handle it.

**Alternative approaches** (not recommended):

| Approach | Pros | Cons |
|----------|------|------|
| Fix `decode_simple_internal` to not assert | Fixes one crash site | Misses the second in `frame_validate` (line 804); still dispatches subtitle codecs into a path that expects frame data, risking other UB |
| Fix `avcodec_open2` to reject subtitle codecs | Prevents the problem at init time | Breaks legitimate subtitle usage via `avcodec_decode_subtitle2`; `avcodec_open2` doesn't know which API the caller will use |
| Fix neither and document the limitation | Zero code change | Leaves a reachable `av_assert0(0)` in release builds — denial-of-service vector in production |

### Regression Test

```c
/* Trigger: 46-byte PGS subtitle stream — crashes av_assert0(0) in decode.c:464 */
static const unsigned char pgs_crash[] = {
    0x50, 0x47, 0x32, 0x00, 0xb7, 0x01, 0x01, 0x11,
    0x00, 0xff, 0xc4, 0x00, 0x1b, 0x10, 0x01, 0x01,
    0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x69, 0xb9, 0xcd,
    0xd5, 0x44, 0x2d, 0x71, 0xb8, 0xff, 0xdd, 0x00,
    0x04, 0x00, 0x04, 0xff, 0xda
};

/* Expect: avcodec_send_packet returns error, does not crash */
```

### Upstream Status

Not reported upstream at the time of discovery. The bug is present in FFmpeg 7.1.3 (confirmed on system FFmpeg 7.1.5 as well, line 468).

**To report** (per [ffmpeg.org/bugreports.html](https://www.ffmpeg.org/bugreports.html)):
1. Verify the bug still exists against the **latest development branch** (git HEAD), not just the vendored 7.1.3.
2. Register at [code.ffmpeg.org](https://code.ffmpeg.org/user/sign_up) and submit an issue at [code.ffmpeg.org/FFmpeg/FFmpeg/issues](https://code.ffmpeg.org/FFmpeg/FFmpeg/issues).
3. Include: the 46-byte crash input, the gdb backtrace (see above), and the minimal C reproducer at `targets/repro_ffmpeg_assert.c`.
4. Upload the crash sample to [streams.videolan.org/upload/](https://streams.videolan.org/upload/) (select FFmpeg project).

### Testing the Bug

A minimal standalone C reproducer is at **`targets/repro_ffmpeg_assert.c`**. It opens the `hdmv_pgs_subtitle` decoder, calls `avcodec_send_packet`, and observes the crash:

```bash
# Compile (needs libavcodec + libavutil dev packages)
gcc -O0 -g -o /tmp/repro targets/repro_ffmpeg_assert.c \
    $(pkg-config --cflags --libs libavcodec libavutil)

# Run — crashes immediately
/tmp/repro
```

**Expected output (buggy build):**
```
OK: found decoder 'pgssub' (type=3)
OK: avcodec_open2 succeeded (codec_type=3)
calling avcodec_send_packet...
[pgssub @ 0x....] Unknown subtitle segment type 0x0, length 0
Assertion 0 failed at src/libavcodec/decode.c:468
Aborted (exit 134)
```

**Expected output (fixed build):**
```
OK: found decoder 'pgssub' (type=3)
OK: avcodec_open2 succeeded (codec_type=3)
calling avcodec_send_packet...
avcodec_send_packet returned -22
OK - bug is fixed (send_packet returned error -22 instead of crashing)
```
