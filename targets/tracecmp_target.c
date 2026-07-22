/* tracecmp_target.c — Test target for compiler-IR comparison tracing.
 *
 * Exercises comparisons that GCC -O2 inlines into integer compares,
 * making them invisible to symbol-based cmplog interposition.
 * When compiled with clang -fsanitize-coverage=trace-cmp, every
 * comparison below emits a __sanitizer_cov_trace_cmp callback.
 *
 * Input format: raw bytes from stdin or fuzz_shm_run()
 * Returns: exit code indicating which comparison path was taken.
 *
 * Compile with trace-cmp:
 *   clang -O2 -g -fsanitize-coverage=trace-cmp,trace-pc-guard \
 *       -include src/fuzzer_tool/adapters/afl_shim.c \
 *       -o targets/tracecmp_target targets/tracecmp_target.c
 */
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* AFL edge coverage — provided by afl_shim.c */
extern void __afl_map_edge(unsigned int cur_loc);

__attribute__((visibility("default")))
int fuzz_tracecmp(const unsigned char *buf, size_t size) {
    __afl_map_edge(0x2000);
    if (size < 16) { __afl_map_edge(0x2001); return 0; }

    /* ── memcmp against compile-time constants ──────────────────────
     * These are the exact patterns that GCC -O2 folds into single
     * integer compares, making them invisible to cmplog_shim.c.
     * trace-cmp instrumentation WILL catch these. */

    /* 8-byte PNG signature — inlined to cmp rax,[rdi] at -O2 */
    if (memcmp(buf, "\x89PNG\r\n\x1a\n", 8) != 0) {
        __afl_map_edge(0x2100);
        return 1;
    }
    __afl_map_edge(0x2101);

    /* 8-byte IHDR chunk type + length */
    if (memcmp(buf + 8, "\x00\x00\x00\x0dIHDR", 8) != 0) {
        __afl_map_edge(0x2200);
        return 2;
    }
    __afl_map_edge(0x2201);

    /* ── Byte-level comparisons — individual byte checks ──────────── */
    if (buf[0] != 0x89) { __afl_map_edge(0x2300); return 3; }
    if (buf[1] != 'P')  { __afl_map_edge(0x2301); return 4; }
    if (buf[2] != 'N')  { __afl_map_edge(0x2302); return 5; }
    if (buf[3] != 'G')  { __afl_map_edge(0x2303); return 6; }
    __afl_map_edge(0x2304);

    /* ── Wider integer comparisons ────────────────────────────────── */
    uint16_t val16 = (uint16_t)buf[4] | ((uint16_t)buf[5] << 8);
    if (val16 == 0x1a0d) {  /* \r\n in little-endian */
        __afl_map_edge(0x2400);
        return 7;
    }

    uint32_t val32 = (uint32_t)buf[8] | ((uint32_t)buf[9] << 8) |
                     ((uint32_t)buf[10] << 16) | ((uint32_t)buf[11] << 24);
    if (val32 == 0x0d000000) {  /* IHDR length in LE */
        __afl_map_edge(0x2500);
        return 8;
    }
    __afl_map_edge(0x2501);

    /* ── Switch statement — trace_switch captures these ───────────── */
    __afl_map_edge(0x2600);
    switch (buf[12]) {
        case 0x00: __afl_map_edge(0x2601); return 10;
        case 0x01: __afl_map_edge(0x2602); return 11;
        case 0x49: __afl_map_edge(0x2603); return 12;  /* 'I' */
        case 0x50: __afl_map_edge(0x2604); return 13;  /* 'P' */
        case 0x53: __afl_map_edge(0x2605); return 14;  /* 'S' */
        case 0x74: __afl_map_edge(0x2606); return 15;  /* 't' */
        default:   __afl_map_edge(0x2607); return 16;
    }
}

__attribute__((visibility("default")))
int fuzz_shm_run(const unsigned char *buf, size_t size) {
    return fuzz_tracecmp(buf, size);
}

#ifdef __AFL_HAVE_MANUAL_CONTROL
int main(void) {
    __AFL_INIT();
    unsigned char *buf = __AFL_FUZZ_TEST_CASE_BUF;
    while (__AFL_LOOP(1000)) {
        int len = __AFL_FUZZ_TEST_CASE_LEN;
        fuzz_tracecmp(buf, len);
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
            int rc = fuzz_tracecmp(buf, (size_t)size);
            free(buf);
            fclose(f);
            return rc;
        }
        fclose(f);
        return 1;
    } else {
        unsigned char buf[65536];
        size_t n = fread(buf, 1, sizeof(buf), stdin);
        if (n > 0) return fuzz_tracecmp(buf, n);
    }
    return 0;
}
#endif
