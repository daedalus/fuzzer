/* Fuzz target for LZ4 (vendored, v1.10.0) — exercises both the frame
 * decoder (LZ4F_*, the format with a real header/checksum layer) and the
 * raw block decoder (LZ4_decompress_safe).
 *
 * Modeled on png_read.c: same __afl_map_edge landmark scheme, same
 * fuzz_shm_run entry point for direct_lite mode, same "bound the output
 * size so bad input can't OOM" discipline.
 *
 * Input layout:
 *   byte 0: mode selector
 *     even -> LZ4 frame decode (LZ4F_decompress streaming loop)
 *     odd  -> raw block decode (LZ4_decompress_safe)
 *   bytes 1.. : payload fed to the chosen decoder
 *
 * The frame path is the interesting one for a format-aware fuzzer: it has
 * a magic number (0x184D2204), FLG/BD descriptor bytes, an xxhash header
 * checksum, block headers, and an optional content checksum — so PNG-style
 * structural mutators and cmplog both have real work to do.
 *
 * Neither decoder is supposed to read out of bounds on ANY input; that is
 * precisely the safety contract of LZ4_decompress_safe / LZ4F_decompress.
 * A SIGSEGV here is a real bug, not a malformed-input artifact. Returning
 * an error code is the expected behaviour and is NOT treated as a crash.
 *
 * Vendor the LZ4 sources first (extracts to vendor/lz4/):
 *   tools/vendor_lz4.sh
 *
 * Then build via the normal path, which handles ASAN/cmplog variants:
 *   tools/build_targets.sh
 *
 * Manual build (what build_targets.sh does under the hood) — note the
 * library objects are compiled SEPARATELY, without the shim:
 *   for f in lz4 lz4frame lz4hc xxhash; do
 *     clang -O2 -g -fPIC -I vendor/lz4/lib -c vendor/lz4/lib/$f.c -o /tmp/$f.o
 *   done
 *   clang -O2 -g -shared -fPIC -I vendor/lz4/lib \
 *       -include src/fuzzer_tool/adapters/afl_shim.c \
 *       -o targets/lz4_read.so targets/lz4_read.c \
 *       /tmp/lz4.o /tmp/lz4frame.o /tmp/lz4hc.o /tmp/xxhash.o \
 *       -Wl,--export-dynamic
 *
 * NOTE: `-include afl_shim.c` applies to EVERY .c on the command line, so
 * the vendored sources must not be passed alongside the wrapper — doing so
 * emits __afl_map_shm / __afl_area / __afl_guarded_call into all five
 * objects and the link fails with multiple-definition errors. Likewise, do
 * not add src/fuzzer_tool/adapters/cmplog_shim.c here; build_so_target()
 * compiles it separately and links the object.
 */
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "lz4.h"
#include "lz4frame.h"

/* AFL edge coverage — provided by afl_shim.c */
extern void __afl_map_edge(unsigned int cur_loc);

/* Cap decompressed output so a decompression bomb can't OOM the fuzzer
 * process (direct_lite runs in-process — an OOM kills the whole run). */
#define LZ4_FUZZ_MAX_OUT (16u * 1024u * 1024u)
#define LZ4_FUZZ_CHUNK   (64u * 1024u)

/* ── Frame path: LZ4F_decompress streaming loop ─────────────────────── */
static int fuzz_lz4_frame(const unsigned char *buf, size_t size) {
    __afl_map_edge(0x2100);

    LZ4F_dctx *dctx = NULL;
    LZ4F_errorCode_t err = LZ4F_createDecompressionContext(&dctx, LZ4F_VERSION);
    if (LZ4F_isError(err) || dctx == NULL) { __afl_map_edge(0x2101); return 0; }
    __afl_map_edge(0x2102);

    unsigned char *out = malloc(LZ4_FUZZ_CHUNK);
    if (!out) { LZ4F_freeDecompressionContext(dctx); return 0; }

    /* Peek at the frame header first — this is the branchy, checksum-gated
     * part of the format and worth its own coverage landmark. */
    LZ4F_frameInfo_t info;
    size_t hdr_consumed = size;
    size_t hint = LZ4F_getFrameInfo(dctx, &info, buf, &hdr_consumed);
    if (LZ4F_isError(hint)) {
        __afl_map_edge(0x2200);
        free(out);
        LZ4F_freeDecompressionContext(dctx);
        return 0;
    }
    __afl_map_edge(0x2201);

    /* Header parsed. Landmark a few frame-descriptor fields so the fuzzer
     * gets distinct coverage for structurally different valid frames. */
    __afl_map_edge(0x2300 + (unsigned)(info.blockSizeID & 0xF));
    __afl_map_edge(0x2310 + (unsigned)(info.blockMode & 0x1));
    __afl_map_edge(0x2320 + (unsigned)(info.contentChecksumFlag & 0x1));
    __afl_map_edge(0x2330 + (unsigned)(info.blockChecksumFlag & 0x1));

    size_t total_out = 0;
    size_t src_pos = hdr_consumed;
    unsigned rounds = 0;

    while (src_pos < size && hint != 0) {
        size_t src_left = size - src_pos;
        size_t dst_cap = LZ4_FUZZ_CHUNK;

        size_t ret = LZ4F_decompress(dctx, out, &dst_cap, buf + src_pos, &src_left, NULL);
        if (LZ4F_isError(ret)) { __afl_map_edge(0x2400); break; }

        __afl_map_edge(0x2500 + (rounds & 0xFF));

        total_out += dst_cap;
        if (total_out > LZ4_FUZZ_MAX_OUT) { __afl_map_edge(0x2401); break; }

        /* No forward progress on either side — stop rather than spin. */
        if (src_left == 0 && dst_cap == 0) { __afl_map_edge(0x2402); break; }

        src_pos += src_left;
        hint = ret;
        if (++rounds > 4096) { __afl_map_edge(0x2403); break; }
    }
    __afl_map_edge(0x2600);

    free(out);
    LZ4F_freeDecompressionContext(dctx);
    return 0;
}

/* ── Raw block path: LZ4_decompress_safe ────────────────────────────── */
static int fuzz_lz4_block(const unsigned char *buf, size_t size) {
    __afl_map_edge(0x2700);
    if (size == 0 || size > (size_t)INT32_MAX) { __afl_map_edge(0x2701); return 0; }

    /* Try a few output capacities: LZ4_decompress_safe's bounds handling
     * differs between "plenty of room" and "exactly/barely enough", and
     * the truncation path is where historical bugs have lived. */
    static const unsigned mults[] = { 1, 4, 255 };
    for (unsigned i = 0; i < sizeof(mults) / sizeof(mults[0]); i++) {
        size_t cap = size * mults[i];
        if (cap > LZ4_FUZZ_MAX_OUT) cap = LZ4_FUZZ_MAX_OUT;
        if (cap == 0) continue;

        char *out = malloc(cap);
        if (!out) continue;

        int rc = LZ4_decompress_safe((const char *)buf, out, (int)size, (int)cap);
        __afl_map_edge(0x2800 + i * 2 + (rc >= 0 ? 1u : 0u));
        free(out);
    }
    __afl_map_edge(0x2900);
    return 0;
}

__attribute__((visibility("default")))
int fuzz_lz4(const unsigned char *buf, size_t size) {
    __afl_map_edge(0x2000);
    if (size < 2) { __afl_map_edge(0x2001); return 0; }

    unsigned char mode = buf[0];
    const unsigned char *payload = buf + 1;
    size_t plen = size - 1;

    if ((mode & 1u) == 0u) {
        __afl_map_edge(0x2002);
        return fuzz_lz4_frame(payload, plen);
    }
    __afl_map_edge(0x2003);
    return fuzz_lz4_block(payload, plen);
}

/* Standard in-process entry point for fuzzer-tool .so mode */
__attribute__((visibility("default")))
int fuzz_shm_run(const unsigned char *buf, size_t size) {
    return fuzz_lz4(buf, size);
}
