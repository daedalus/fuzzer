/* Fuzz target for zlib — decompress gzip or raw deflate data.
 *
 * Supports two modes:
 *   1. Gzip format (magic 0x1F 0x8B) — inflate with automatic header detection
 *   2. Raw deflate (no header) — inflate with -MAX_WBITS
 *
 * Returns 1 on errors that suggest a zlib bug (not just invalid input).
 *
 * Compile shared library (for inprocess modes):
 *   gcc -O2 -g -shared -fPIC -include src/fuzzer_tool/adapters/afl_shim.c \
 *       -o zlib_read.so zlib_read.c -lz -Wl,--export-dynamic
 *
 * Compile standalone:
 *   gcc -O2 -g -o targets/zlib_read targets/zlib_read.c -lz
 */
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <zlib.h>

/* AFL edge coverage — provided by afl_shim.c */
extern void __afl_map_edge(unsigned int cur_loc);

__attribute__((visibility("default")))
int fuzz_zlib(const unsigned char *buf, size_t size) {
    __afl_map_edge(0x1000);
    if (size < 2) { __afl_map_edge(0x1001); return 0; }

    z_stream strm;
    memset(&strm, 0, sizeof(strm));
    strm.next_in = (Bytef *)buf;
    strm.avail_in = size;

    int ret;

    /* Detect format: gzip, zlib, or raw deflate */
    if (buf[0] == 0x1F && buf[1] == 0x8B) {
        /* Gzip format — auto-detect gzip/zlib */
        __afl_map_edge(0x1002);
        ret = inflateInit2(&strm, 15 + 32);
    } else if (size >= 2 && (buf[0] & 0x0F) == 8 && (buf[0] >> 4) <= 7) {
        /* Zlib format — CMF byte: CM=8 (deflate), CINFO<=7 */
        __afl_map_edge(0x1003);
        ret = inflateInit2(&strm, 15);
    } else {
        /* Raw deflate — no header */
        __afl_map_edge(0x1004);
        ret = inflateInit2(&strm, -MAX_WBITS);
    }

    if (ret != Z_OK) {
        __afl_map_edge(0x1005);
        return 0;
    }
    __afl_map_edge(0x1100);

    /* Decompress in chunks */
    unsigned char outbuf[65536];
    int chunk_count = 0;

    do {
        strm.next_out = outbuf;
        strm.avail_out = sizeof(outbuf);

        ret = inflate(&strm, Z_NO_FLUSH);
        chunk_count++;
        __afl_map_edge(0x1200 + (chunk_count & 0xFF));

        if (ret != Z_OK && ret != Z_STREAM_END) {
            __afl_map_edge(0x1300);
            inflateEnd(&strm);
            return 1;  /* zlib error — possible bug */
        }

        /* Guard against unreasonable output */
        if (chunk_count > 1000) {
            __afl_map_edge(0x1301);
            inflateEnd(&strm);
            return 0;
        }
    } while (ret != Z_STREAM_END);

    __afl_map_edge(0x1400);

    ret = inflateEnd(&strm);
    if (ret != Z_OK) {
        __afl_map_edge(0x1401);
        return 1;  /* inflateEnd error */
    }

    __afl_map_edge(0x1500);
    return 0;
}

#ifdef __AFL_HAVE_MANUAL_CONTROL
int main(void) {
    __AFL_INIT();
    unsigned char *buf = __AFL_FUZZ_TEST_CASE_BUF;
    while (__AFL_LOOP(1000)) {
        int len = __AFL_FUZZ_TEST_CASE_LEN;
        fuzz_zlib(buf, len);
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
            int rc = fuzz_zlib(buf, size);
            free(buf);
            fclose(f);
            return rc;
        }
        fclose(f);
        return 1;
    } else {
        unsigned char buf[65536];
        size_t n = fread(buf, 1, sizeof(buf), stdin);
        if (n > 0) return fuzz_zlib(buf, n);
    }
    return 0;
}
#endif

/* Standard in-process entry point for fuzzer-tool .so mode */
__attribute__((visibility("default")))
int fuzz_shm_run(const unsigned char *buf, size_t size) {
    return fuzz_zlib(buf, size);
}
