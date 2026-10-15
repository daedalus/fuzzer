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
| Total crashes discovered | 2 (1 valid) |
| Crash input size | 46 bytes / 21 bytes |
| Crash input type | PGS subtitle stream / VPK container header |
| Trigger | `avcodec_send_packet` on subtitle decoder / `vpk_read_packet` final-block branch |
| Crash type | `av_assert0(0)` / `SIGFPE` |
| Severity | none (invalid, harness misuse) / **Medium** |
| Exploitability | n/a / remotely triggerable via media file parsing |

## Bugs

- **[vpk_divide_by_zero](vpk_divide_by_zero.md)** — Integer divide-by-zero in `vpk_read_packet` (VPK demuxer). A 21-byte VPK stream whose channel count is zeroed by a failed decoder open causes `SIGFPE`. Severity: Medium. Reported as [FFmpeg#24290](https://code.ffmpeg.org/FFmpeg/FFmpeg/issues/24290); fix pending in [PR #24297](https://code.ffmpeg.org/FFmpeg/FFmpeg/pulls/24297).

## Invalid

- **[av_assert0_subtitle_decoder](av_assert0_subtitle_decoder.md)** — harness called `avcodec_send_packet` on a subtitle decoder. Closed upstream as misuse ([FFmpeg#23945](https://code.ffmpeg.org/FFmpeg/FFmpeg/issues/23945)). Harness fixed; contract enforced by `tests/test_regression_ffmpeg_api_contract.py`.

## Triage before reporting

Rule out harness misuse first:

1. Minimize the input. If any payload crashes once a given decoder/parser is opened, the harness is the cause.
2. An abort in API dispatch (`av_assert0` in `decode.c`/`encode.c`), not in a parser, means a broken API contract until proven otherwise.
3. Reproduce through the library's own entry point (the ffmpeg CLI or an upstream `tools/target_*_fuzzer.c`). If only our harness crashes, fix the harness.
