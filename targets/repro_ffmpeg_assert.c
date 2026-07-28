/* Minimal reproducer for the reachable av_assert0(0) in decode.c:464.
 *
 * Opens a subtitle decoder (hdmv_pgs_subtitle) and calls avcodec_send_packet
 * on it — this hits av_assert0(0) in decode_simple_internal() because the
 * function only handles AVMEDIA_TYPE_VIDEO and AVMEDIA_TYPE_AUDIO.
 *
 * Compile:
 *   gcc -O0 -g -o /tmp/repro_ffmpeg_assert repro_ffmpeg_assert.c \
 *       $(pkg-config --cflags --libs libavcodec libavutil)
 *
 * Expected (buggy):  abort / exit 134 / "Assertion av_assert0(0) failed"
 * Expected (fixed):  avcodec_send_packet returned -22 (AVERROR(EINVAL))
 */
#include <libavcodec/avcodec.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    const AVCodec *codec;
    AVCodecContext *ctx;
    AVPacket *pkt;
    int ret;

    /* 1. Find the PGS subtitle decoder */
    codec = avcodec_find_decoder(AV_CODEC_ID_HDMV_PGS_SUBTITLE);
    if (!codec) {
        fprintf(stderr, "FAIL: avcodec_find_decoder(AV_CODEC_ID_HDMV_PGS_SUBTITLE) returned NULL\n");
        return 1;
    }
    fprintf(stderr, "OK: found decoder '%s' (type=%d)\n", codec->name, codec->type);

    /* 2. Allocate and open the decoder */
    ctx = avcodec_alloc_context3(codec);
    if (!ctx) {
        fprintf(stderr, "FAIL: avcodec_alloc_context3 returned NULL\n");
        return 1;
    }
    ret = avcodec_open2(ctx, codec, NULL);
    if (ret < 0) {
        fprintf(stderr, "FAIL: avcodec_open2 returned %d\n", ret);
        return 1;
    }
    fprintf(stderr, "OK: avcodec_open2 succeeded (codec_type=%d)\n", ctx->codec_type);

    /* 3. Allocate a dummy packet */
    pkt = av_packet_alloc();
    if (!pkt) {
        fprintf(stderr, "FAIL: av_packet_alloc returned NULL\n");
        return 1;
    }
    ret = av_new_packet(pkt, 4);
    if (ret < 0) {
        fprintf(stderr, "FAIL: av_new_packet returned %d\n", ret);
        return 1;
    }
    memcpy(pkt->data, "\x00\x00\x00\x01", 4);

    /* 4. Call avcodec_send_packet — this triggers the assert on buggy builds */
    fprintf(stderr, "calling avcodec_send_packet...\n");
    ret = avcodec_send_packet(ctx, pkt);
    fprintf(stderr, "avcodec_send_packet returned %d\n", ret);

    /* Cleanup */
    av_packet_free(&pkt);
    avcodec_free_context(&ctx);

    if (ret < 0) {
        printf("OK - bug is fixed (send_packet returned error %d instead of crashing)\n", ret);
        return 0;
    } else {
        printf("UNEXPECTED - send_packet returned %d >= 0 (no crash, but no assert either)\n", ret);
        return 0;
    }
}
