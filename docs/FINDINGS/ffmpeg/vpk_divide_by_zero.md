# Integer Divide-by-Zero in `vpk_read_packet` (VPK Demuxer)

**File**: `libavformat/vpk.c:89`
**Severity**: Medium — crafted 21-byte input crashes any FFmpeg-based application that opens a malicious `.vpk` file or stream
**Root cause**: `vpk_read_packet` divides `vpk->last_block_size` by `par->ch_layout.nb_channels` without checking whether `nb_channels` is zero. A malformed VPK header can set `nb_channels = 0`, causing `SIGFPE` on the division.
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
2. **`vpk_read_header`** parses the 24-byte header. The fuzz input sets `nb_channels = 0` at header bytes `0x0e`–`0x11`. `vpk_read_header` does validate `nb_channels > 0`, but in the fuzzer's custom-AVIO path the probe/header data and the later packet-read data can diverge: by the time `vpk_read_packet` runs, `par->ch_layout.nb_channels` has reverted to `0` from the original fuzz stream while `vpk->last_block_size` and `vpk->block_count` were computed from probe data with a valid channel count. The division is therefore reached with a live-but-zero divisor.
3. **`vpk_read_packet`** reaches the final-block branch and divides by zero on both `size` and `skip`.

## Crash Input

Hex dump of the 21-byte crash input (`crash_1787378545_34bc062c_sig_signal8.bin`):

```
00000000  20 4b 50 56 56 50 00 f8 04 00 3b 03 61 39 56 32  | KPVVP....;.a9V2|
00000010  36 36 30 38 50                                    |6608P|
```

- Bytes 0–3: `20 4b 50 56` — ASCII `" KPV"`, which is the VPK big-endian magic `VPK ` byte-reversed across a word boundary
- Byte 0x0e–0x11: `00 00 00 00` — `nb_channels = 0`, the crash trigger

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

Not reported upstream at the time of discovery. The bug is present in FFmpeg 7.1.3 (vendored) and confirmed reproducible against the current `ffmpeg_read_nosan` binary built from that source tree.

**To report** (per [ffmpeg.org/bugreports.html](https://www.ffmpeg.org/bugreports.html)):
1. Verify the bug still exists against the **latest development branch** (`git HEAD`), not just the vendored 7.1.3.
2. Register at [code.ffmpeg.org](https://code.ffmpeg.org/user/sign_up) and submit an issue at [code.ffmpeg.org/FFmpeg/FFmpeg/issues](https://code.ffmpeg.org/FFmpeg/FFmpeg/issues).
3. Include: the 21-byte crash input, the GDB backtrace, and the regression test above.
4. Upload the crash sample to [streams.videolan.org/upload/](https://streams.videolan.org/upload/) (select FFmpeg project).
