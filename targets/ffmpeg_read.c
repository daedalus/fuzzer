/* Fuzz target for FFmpeg — exercises the full demux→decode pipeline.
 *
 * Feeds arbitrary data through:
 *   avformat_open_input → avformat_find_stream_info → av_read_frame
 *   → avcodec_send_packet → avcodec_receive_frame → close
 *
 * Covers all registered demuxers, decoders, parsers, and BSFs.
 * Core library paths (buffer mgmt, packet lifecycle, error recovery)
 * account for ~80% of executable code; per-format leaf functions ~20%.
 *
 * Compile standalone:
 *   gcc -O2 -g -fsanitize=address -include src/fuzzer_tool/adapters/afl_shim.c \
 *       -o targets/ffmpeg_read targets/ffmpeg_read.c -lavformat -lavcodec -lavutil -lswresample -lm
 *
 * Compile shared library (for inprocess modes):
 *   gcc -O2 -g -shared -fPIC -fsanitize-coverage=trace-cmp \
 *       -include src/fuzzer_tool/adapters/afl_shim.c \
 *       src/fuzzer_tool/adapters/cmplog_shim.c -ldl \
 *       -o targets/ffmpeg_read.so targets/ffmpeg_read.c \
 *       -lavformat -lavcodec -lavutil -lswresample -lm -Wl,--export-dynamic
 */
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/mem.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* AFL edge coverage — provided by afl_shim.c */
extern void __afl_map_edge(unsigned int cur_loc);

/* I/O buffer size for AVIOContext (256KB) */
#ifndef FFMPEG_IO_BUF_SIZE
#define FFMPEG_IO_BUF_SIZE (256 * 1024)
#endif

/* ── In-memory I/O for avformat ──────────────────────────────────── */

typedef struct {
    const unsigned char *data;
    size_t size;
    size_t offset;
} FuzzIOState;

static int fuzz_read_packet(void *opaque, unsigned char *buf, int buf_size) {
    FuzzIOState *st = (FuzzIOState *)opaque;
    int avail = (int)(st->size - st->offset);
    if (avail <= 0) return AVERROR_EOF;
    if (buf_size > avail) buf_size = avail;
    memcpy(buf, st->data + st->offset, buf_size);
    st->offset += buf_size;
    return buf_size;
}

/* ── Core fuzz function ──────────────────────────────────────────── */

__attribute__((visibility("default")))
int fuzz_ffmpeg(const unsigned char *buf, size_t size) {
    __afl_map_edge(0x1000);
    if (size < 16) { __afl_map_edge(0x1001); return 0; }

    /* Initialize I/O state */
    FuzzIOState io_state = { buf, size, 0 };
    __afl_map_edge(0x1002);

    /* Allocate format context */
    AVFormatContext *fmt_ctx = avformat_alloc_context();
    if (!fmt_ctx) { __afl_map_edge(0x1003); return 0; }
    __afl_map_edge(0x1004);

    /* Create AVIOContext for reading from memory */
    unsigned char *io_buffer = av_malloc(FFMPEG_IO_BUF_SIZE);
    if (!io_buffer) { avformat_free_context(fmt_ctx); return 0; }

    AVIOContext *avio_ctx = avio_alloc_context(
        io_buffer, FFMPEG_IO_BUF_SIZE, 0, &io_state, fuzz_read_packet, NULL, NULL);
    if (!avio_ctx) {
        av_free(io_buffer);
        avformat_free_context(fmt_ctx);
        return 0;
    }
    fmt_ctx->pb = avio_ctx;
    __afl_map_edge(0x1005);

    /* Open input — probes all registered demuxers */
    int ret = avformat_open_input(&fmt_ctx, NULL, NULL, NULL);
    if (ret < 0) {
        __afl_map_edge(0x1006);
        avio_context_free(&avio_ctx);
        return 0;
    }
    __afl_map_edge(0x1010);

    /* Find stream info — runs codec probing for each stream */
    ret = avformat_find_stream_info(fmt_ctx, NULL);
    if (ret < 0) {
        __afl_map_edge(0x1011);
        avformat_close_input(&fmt_ctx);
        return 0;
    }
    __afl_map_edge(0x1020);

    /* Iterate packets and decode each stream */
    unsigned nb_streams = fmt_ctx->nb_streams;
    if (nb_streams > 128) nb_streams = 128;  /* bound to prevent OOM */

    AVPacket *pkt = av_packet_alloc();
    AVFrame *frame = av_frame_alloc();
    if (!pkt || !frame) {
        av_packet_free(&pkt);
        av_frame_free(&frame);
        avformat_close_input(&fmt_ctx);
        return 0;
    }

    unsigned total_packets = 0;
    unsigned total_frames = 0;

    /* Per-stream decoder contexts */
    AVCodecContext *dec_ctxs[128];
    memset(dec_ctxs, 0, sizeof(dec_ctxs));

    for (unsigned i = 0; i < nb_streams; i++) {
        const AVCodecParameters *par = fmt_ctx->streams[i]->codecpar;
        const AVCodec *codec = avcodec_find_decoder(par->codec_id);
        if (!codec) { dec_ctxs[i] = NULL; continue; }

        AVCodecContext *dec_ctx = avcodec_alloc_context3(codec);
        if (!dec_ctx) { dec_ctxs[i] = NULL; continue; }

        avcodec_parameters_to_context(dec_ctx, par);
        ret = avcodec_open2(dec_ctx, codec, NULL);
        if (ret < 0) {
            avcodec_free_context(&dec_ctx);
            dec_ctxs[i] = NULL;
            continue;
        }
        dec_ctxs[i] = dec_ctx;
        __afl_map_edge(0x1100 + (i & 0xFF));
    }

    /* Read and decode frames */
    while (av_read_frame(fmt_ctx, pkt) >= 0) {
        total_packets++;
        __afl_map_edge(0x1200 + (pkt->stream_index & 0x1F));

        int si = pkt->stream_index;
        if (si >= 0 && si < (int)nb_streams && dec_ctxs[si]) {
            ret = avcodec_send_packet(dec_ctxs[si], pkt);
            if (ret >= 0) {
                while (avcodec_receive_frame(dec_ctxs[si], frame) >= 0) {
                    total_frames++;
                    __afl_map_edge(0x1400 + (si & 0xFF));
                }
            }
        }

        av_packet_unref(pkt);

        /* Guard against pathological inputs */
        if (total_packets > 10000 || total_frames > 5000) break;
    }

    /* Cleanup */
    for (unsigned i = 0; i < nb_streams; i++) {
        if (dec_ctxs[i]) avcodec_free_context(&dec_ctxs[i]);
    }
    av_frame_free(&frame);
    av_packet_free(&pkt);
    avformat_close_input(&fmt_ctx);  /* frees avio_ctx + io_buffer */

    __afl_map_edge(0x1500);
    return 0;
}

#ifdef __AFL_HAVE_MANUAL_CONTROL
int main(void) {
    __AFL_INIT();
    unsigned char *buf = __AFL_FUZZ_TEST_CASE_BUF;
    while (__AFL_LOOP(1000)) {
        int len = __AFL_FUZZ_TEST_CASE_LEN;
        fuzz_ffmpeg(buf, len);
    }
    return 0;
}
#else
int main(int argc, char **argv) {
    if (argc == 2) {
        FILE *f = fopen(argv[1], "rb");
        if (!f) return 1;
        fseek(f, 0, SEEK_END);
        long size = ftell(f);
        rewind(f);
        unsigned char *buf = malloc(size);
        if (buf) {
            fread(buf, 1, size, f);
            int rc = fuzz_ffmpeg(buf, size);
            free(buf);
            fclose(f);
            return rc;
        }
        fclose(f);
        return 1;
    } else {
        unsigned char buf[65536];
        size_t n = fread(buf, 1, sizeof(buf), stdin);
        if (n > 0) return fuzz_ffmpeg(buf, n);
    }
    return 0;
}
#endif

/* Standard in-process entry point for fuzzer-tool .so mode */
__attribute__((visibility("default")))
int fuzz_shm_run(const unsigned char *buf, size_t size) {
    return fuzz_ffmpeg(buf, size);
}
