# Integer Divide-by-Zero in `vpk_read_packet` (VPK Demuxer)

**File**: `libavformat/vpk.c:89`
**Severity**: Medium — crafted 21-byte input crashes any FFmpeg-based application that opens a malicious `.vpk` file or stream
**Root cause**: `vpk_read_packet` divides `vpk->last_block_size` by `par->ch_layout.nb_channels` without checking whether `nb_channels` is zero. The header cannot set it to zero — `vpk_read_header` rejects `<= 0` — but a failed `avcodec_open2()` inside `avformat_find_stream_info()` zeroes `ch_layout`, and the empty layout is copied back over the container's channel count, so the division is reached with a zero divisor. (Corrected 2026-08-30; the original report blamed a probe/packet data divergence in the custom-AVIO path. See [Upstream Status](#upstream-status).)
**Discovered by**: fuzzer-tool with ASAN via `targets/ffmpeg_read.c`

---

## Target

[FFmpeg](https://ffmpeg.org/) 7.1.3 — libavcodec, libavformat, libavutil — the full multimedia demux→decode pipeline.

**Library versions** (vendored at `vendor/ffmpeg/`):

| Library | Version |
|---------|---------|
| libavutil | 59.39.100 |
| libavcodec | 61.19.101 |
| libavformat | 61.7.100 |

---

## Fuzz Target

`targets/ffmpeg_read.c` — feeds arbitrary data through the full FFmpeg API chain:

```
avformat_open_input → avformat_find_stream_info → av_read_frame
→ avcodec_send_packet → avcodec_receive_frame → close
```

Covers all registered demuxers, decoders, and parsers. Compiled with ASAN (`-fsanitize=address`) and AFL edge coverage via `afl_shim.c`. See `targets/ffmpeg_read.c` for details.

---

## Fuzzing Configuration

- **Engine**: fuzzer-tool with Markov byte generation, Monte Carlo optimization, Thompson sampling bandit, and grammar-aware mutations
- **Coverage**: AFL SHM edge coverage
- **Inputs generated**: PNG corpus (`tools/corpus_png.py`), incremental mutations from prior fuzzing sessions

---

## Results

| Metric | Value |
|--------|-------|
| Total crashes discovered | 1 (unique signature) |
| Crash input size | 21 bytes |
| Crash input type | VPK container header |
| Trigger | `vpk_read_packet` final-block branch |
| Crash type | `SIGFPE` — integer divide-by-zero |
| Severity | **Medium** |
| Exploitability | Remotely triggerable via media file parsing |

---

## Description

The Sony PS2 VPK demuxer (`libavformat/vpk.c`) reads audio blocks from a custom container format. In `vpk_read_packet`, the last block of the stream is handled specially:

```c
if (vpk->current_block == vpk->block_count) {
    unsigned size = vpk->last_block_size / par->ch_layout.nb_channels;
    unsigned skip = (par->block_align - vpk->last_block_size)
                    / par->ch_layout.nb_channels;
    ...
}
```

Both `size` and `skip` divide by `par->ch_layout.nb_channels`. When `nb_channels` is zero, the CPU raises `SIGFPE` (integer divide-by-zero exception).

## Trigger Chain

1. **Demuxer probe** (`vpk_probe`) matches the `VPK ` big-endian magic and assigns the input to the VPK demuxer.
2. **`vpk_read_header`** parses the 24-byte header and *does* validate the channel count (`nb_channels <= 0` is rejected), so the demuxer sets a positive value — 80 for this input. Nothing is wrong yet.
3. **`avformat_find_stream_info`** probes the stream, fails to open the `adpcm_psx` decoder, and copies codec parameters back from the decoder context anyway. A failed `avcodec_open2()` has already zeroed the context's `ch_layout` via `ff_codec_close()` → `av_opt_free()` (`ch_layout` is an `AV_OPT_TYPE_CHLAYOUT` option), so the empty layout is written over the container's 80 channels on `codecpar`. `adpcm_psx` has no parser and does not set `AVSTREAM_PARSE_*`, so the demuxer header was the only source of the channel count — unlike mp3, where the decoder is expected to fill the layout.
4. **`vpk_read_packet`** reaches the final-block branch with a live-but-zero divisor and divides by zero on both `size` and `skip`.

> **Correction (2026-08-30).** Steps 2–3 above replace the original explanation, which claimed the probe data and the packet-read data diverge in the fuzzer's custom-AVIO path. That was a hypothesis, never checked against `libavformat`, and it is not what upstream found: the zeroing happens in `avformat_find_stream_info` and does not depend on custom AVIO. The wrong version is quoted verbatim in [issue #24290](https://code.ffmpeg.org/FFmpeg/FFmpeg/issues/24290) and in third-party coverage, so it is corrected here rather than silently dropped. Mechanism per Jun Zhao's analysis in [PR #24297](https://code.ffmpeg.org/FFmpeg/FFmpeg/pulls/24297).

## ffmpeg CLI Reproducibility

An independent attempt with the ffmpeg CLI (6.1.1, Ubuntu) did **not** crash — see [External Coverage](#external-coverage):

```sh
printf '\x20\x4b\x50\x56\x56\x50\x00\xf8\x04\x00\x3b\x03\x61\x39\x56\x32\x36\x36\x30\x38\x50' > vpk_crash.bin
ffmpeg -i vpk_crash.bin -c:a copy -f null -
```

FFmpeg detected the VPK container, reported a 942,683,702 Hz / 80-channel audio stream, failed to open the ADPCM decoder, and exited with a demuxing error.

The failed decoder open — the first half of the upstream root cause — *does* happen on the CLI path, and the 80 channels it prints is the container value upstream's fix exists to preserve. So the custom-AVIO explanation cannot be why the CLI escapes: that explanation is wrong (see the correction above). What the difference actually is has not been established here. The plausible candidate is the read loop — `targets/ffmpeg_read.c:548` drives `av_read_frame()` until it fails, while the CLI stops at the first demux error — so the harness may simply be the only one that reaches the final-block branch. Unverified; do not repeat it as fact.

## Crash Input

Hex dump of the 21-byte crash input (`crash_1787378545_34bc062c_sig_signal8.bin`):

```
00000000  20 4b 50 56 56 50 00 f8 04 00 3b 03 61 39 56 32  | KPVVP....;.a9V2|
00000010  36 36 30 38 50                                    |6608P|
```

Decoded against `vpk_read_header`, which reads six little-endian 32-bit fields:

| Offset | Bytes | Field | Value |
|---|---|---|---|
| `0x00` | `20 4b 50 56` | magic | `AV_RL32` = `MKBETAG('V','P','K',' ')`, i.e. the tag stored big-endian — what `vpk_probe` requires |
| `0x04` | `56 50 00 f8` | duration source | `0xf8005056` |
| `0x08` | `04 00 3b 03` | `offset` | `0x033b0004` |
| `0x0c` | `61 39 56 32` | `block_align` | `0x32563961` = 844,511,585 |
| `0x10` | `36 36 30 38` | `sample_rate` | `0x38303636` = 942,683,702 |
| `0x14` | `50` + EOF | `nb_channels` | `0x50` = **80** — the field is truncated by the 21-byte input, so `avio_rl32` pads with zeros |

Two of these are confirmed against an independent ffmpeg CLI run, which reports exactly 942,683,702 Hz and 80 channels for this input (see [ffmpeg CLI Reproducibility](#ffmpeg-cli-reproducibility)); upstream's FATE regression test asserts the same 80.

> **Correction (2026-08-30).** This section previously claimed `00 00 00 00` at `0x0e`–`0x11` was `nb_channels = 0` and "the crash trigger". That is wrong on every count: the bytes at that offset are `56 32 36 36`, no field starts at `0x0e`, and the channel count this input parses to is 80, not 0. Nothing in the 21 bytes sets a zero channel count — the zero is manufactured later by `libavformat` itself.

## GDB Backtrace

```
Program received signal SIGFPE, Arithmetic exception.
0x00005555557a9877 in vpk_read_packet (s=0x555557fed700, pkt=0x555557fed300)
    at libavformat/vpk.c:89
89   unsigned size = vpk->last_block_size / par->ch_layout.nb_channels;

#0  vpk_read_packet
#1  ff_read_packet
#2  read_frame_internal
#3  av_read_frame
#4  fuzz_ffmpeg
#5  main
```

## Crash Metadata

| Field | Value |
|---|---|
| Signal | `SIGFPE` (returncode −8) |
| Fault address / RIP | `0x7ffff48a66d7` (instruction itself) |
| RSP | `0x7fffffffcce0` |
| Execs to find | 495,211 |
| Corpus at find | 13,188 entries |
| Elapsed | 10 h 43 m |
| Parent seed | `36e65f4009ba0cab` |
| Target SHA256 | `d704c2a52b21bd33` |

## Exploitability Assessment

| Factor | Assessment |
|---|---|
| Crash determinism | **Deterministic** — 21 bytes, single demuxer code path |
| Trigger depth | Shallow — `avformat_open_input` auto-detects format from magic |
| Preconditions | None — input is self-contained, no network, no heap setup |
| Signal type | SIGFPE (integer divide-by-zero), not memory corruption |
| Memory safety | No OOB read/write, no use-after-free, no NULL dereference |
| Reach | Any application that calls `avformat_open_input` + `av_read_frame` on untrusted data |
| Severity | **Medium** — reliable DoS; not an immediate code-execution primitive |

The divide-by-zero is a **denial-of-service** primitive. There is no controlled write or arbitrary read adjacent to the faulting instruction. The input can be embedded in a `.vpk` file or a container that identifies itself as VPK to trigger the crash in any FFmpeg-linked application.

## Suggested Fix

Add a guard at the top of `vpk_read_packet` to reject zero-channel streams cleanly:

```c
static int vpk_read_packet(AVFormatContext *s, AVPacket *pkt)
{
    AVCodecParameters *par = s->streams[0]->codecpar;
    VPKDemuxContext *vpk = s->priv_data;
    int ret, i;

    if (par->ch_layout.nb_channels == 0)
        return AVERROR_INVALIDDATA;

    vpk->current_block++;
    ...
}
```

This is consistent with the existing validation in `vpk_read_header` (`if (st->codecpar->ch_layout.nb_channels <= 0) return AVERROR_INVALIDDATA;`) and returns a clean error instead of `SIGFPE`.

This is what we proposed upstream, and it is only half of what landed in review: it stops the `SIGFPE` but leaves the channel count silently clobbered. See [Upstream Status](#upstream-status) for the `avformat_find_stream_info()` change that fixes the cause rather than the symptom.

## Regression Test

```c
/* Trigger: 21-byte VPK stream with nb_channels=0 — SIGFPE in vpk.c:89 */
static const unsigned char vpk_crash[] = {
    0x20, 0x4b, 0x50, 0x56, 0x56, 0x50, 0x00, 0xf8,
    0x04, 0x00, 0x3b, 0x03, 0x61, 0x39, 0x56, 0x32,
    0x36, 0x36, 0x30, 0x38, 0x50
};

/* Expect: av_read_frame returns -22 (AVERROR_INVALIDDATA), does not crash */
```

## Upstream Status

Reported upstream on 2026-08-27 as **[FFmpeg/FFmpeg#24290 — Integer Divide-by-Zero in `vpk_read_packet` (VPK Demuxer)](https://code.ffmpeg.org/FFmpeg/FFmpeg/issues/24290)**. Status as of 2026-08-30: **open**, fix pending review.

| | |
|---|---|
| Issue | [#24290](https://code.ffmpeg.org/FFmpeg/FFmpeg/issues/24290) — open, filed 2026-08-27 |
| Fix | [PR #24297 `fix/24290-vpk-div0`](https://code.ffmpeg.org/FFmpeg/FFmpeg/pulls/24297) by Jun Zhao — open, 3 commits, +32 −3, all FATE checks green, 0/1 approvals |
| Prior art | [ffmpeg-devel, November 2024](https://ffmpeg.org/pipermail/ffmpeg-devel/2024-November/335598.html) — flagged by Jun Zhao as the same issue; the guard was proposed then and never landed |
| Credit | `Reported-by: Darío Clavijo`; the vpk guard carries `Original-patch-by: Kacper Michajłow` from that 2024 thread |

The upstream fix is broader than the guard suggested above. Its three commits:

1. **`avformat: Restore the container channel layout after a failed decoder open`** — the actual root cause. `avformat_find_stream_info()` writes codec parameters back from the decoder context even when `avcodec_open2()` failed, and a failed open zeroes `ch_layout` through `ff_codec_close()` → `av_opt_free()`. The commit restores the container layout when it was specified and the decoder result is unspecified, matching the existing restore of color metadata in `parameters_from_context()`.
2. **`avformat/vpk: Check the channel count before dividing in the last block`** — the defence-in-depth guard, the same shape as the one suggested above.
3. **`tests/fate: Add a VPK regression test for a zeroed channel layout`** — generates the 21-byte sample from the issue and asserts the dump still reports 80 channels. `ffprobe -show_entries` is not usable because the decoder cannot be opened at all.

Commit 1 is the part our report missed: we proposed only the divisor guard, which stops the `SIGFPE` but leaves the channel count silently clobbered for every codec in that class. Worth carrying into future reports — a guard at the faulting instruction is a symptom fix, and the maintainer will look for where the bad value came from.

The commit trailers also read `Found-by: OSS-Fuzz` and reference `issues.oss-fuzz.com/issues/42536474`, so the same defect was in the OSS-Fuzz queue independently.

## External Coverage

- **[21 Bytes Can Crash FFmpeg: Inside the Vibecoded Fuzzer That Found What Years of Audits Missed](https://dev.to/jamilxt/21-bytes-can-crash-ffmpeg-inside-the-vibecoded-fuzzer-that-found-what-years-of-audits-missed-fpe)** — dev.to, jamilxt, 2026-08-29. Independent writeup: clones this repo, walks `targets/ffmpeg_read.c` and `AGENTS.md`, and attempts a reproduction against a system ffmpeg — which does not crash. Its output is the source of [ffmpeg CLI Reproducibility](#ffmpeg-cli-reproducibility) above. It repeats this document's original custom-AVIO explanation, which the correction above supersedes.
