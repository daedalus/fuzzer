/* Fuzz target for libjpeg-turbo — reads JPEG from stdin or file.
 *
 * Uses setjmp/longjmp for libjpeg error handling.
 * On error, longjmp returns to setjmp which exits with code 1.
 *
 * Includes AFL-style edge coverage via afl_shim.c. Compile with:
 *   gcc -O2 -g -shared -fPIC -include src/fuzzer_tool/adapters/afl_shim.c \
 *       -o jpeg_read.so jpeg_read.c -ljpeg -Wl,--export-dynamic
 */
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <jpeglib.h>
#include <setjmp.h>

/* AFL edge coverage — provided by afl_shim.c */
extern void __afl_map_edge(unsigned int cur_loc);

static jmp_buf jpeg_jmpbuf;

struct my_error_mgr {
    struct jpeg_error_mgr pub;
};

void my_error_exit(j_common_ptr cinfo) {
    __afl_map_edge(0x2000);
    longjmp(jpeg_jmpbuf, 1);
}

__attribute__((visibility("default")))
int fuzz_jpeg(const unsigned char *buf, size_t size) {
    __afl_map_edge(0x1000);
    if (size < 2) { __afl_map_edge(0x1001); return 0; }

    /* Verify JPEG SOI marker */
    if (buf[0] != 0xFF || buf[1] != 0xD8) { __afl_map_edge(0x1002); return 0; }
    __afl_map_edge(0x1003);

    struct jpeg_decompress_struct cinfo;
    struct my_error_mgr jerr;

    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = my_error_exit;

    if (setjmp(jpeg_jmpbuf)) {
        __afl_map_edge(0x1100);
        jpeg_destroy_decompress(&cinfo);
        return 1;
    }
    __afl_map_edge(0x1101);

    jpeg_create_decompress(&cinfo);
    __afl_map_edge(0x1102);

    jpeg_mem_src(&cinfo, buf, size);
    __afl_map_edge(0x1103);

    int rc = jpeg_read_header(&cinfo, TRUE);
    __afl_map_edge(0x1104);

    if (rc != JPEG_HEADER_OK) {
        __afl_map_edge(0x1105);
        jpeg_destroy_decompress(&cinfo);
        return 0;
    }
    __afl_map_edge(0x1200);

    /* Check dimensions */
    if (cinfo.image_width > 16384 || cinfo.image_height > 16384) {
        __afl_map_edge(0x1201);
        jpeg_destroy_decompress(&cinfo);
        return 0;
    }
    __afl_map_edge(0x1202);

    jpeg_start_decompress(&cinfo);
    __afl_map_edge(0x1300);

    while (cinfo.output_scanline < cinfo.output_height) {
        unsigned char *row[1];
        row[0] = malloc(cinfo.output_width * cinfo.output_components);
        jpeg_read_scanlines(&cinfo, row, 1);
        __afl_map_edge(0x1400 + (cinfo.output_scanline & 0xFF));
        free(row[0]);
    }
    __afl_map_edge(0x1500);

    jpeg_finish_decompress(&cinfo);
    __afl_map_edge(0x1501);

    jpeg_destroy_decompress(&cinfo);
    __afl_map_edge(0x1502);

    return 0;
}

#ifdef __AFL_HAVE_MANUAL_CONTROL
int main(void) {
    __AFL_INIT();
    unsigned char *buf = __AFL_FUZZ_TEST_CASE_BUF;
    while (__AFL_LOOP(1000)) {
        int len = __AFL_FUZZ_TEST_CASE_LEN;
        fuzz_jpeg(buf, len);
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
            int rc = fuzz_jpeg(buf, size);
            free(buf);
            fclose(f);
            return rc;
        }
        fclose(f);
        return 1;
    } else {
        unsigned char buf[65536];
        size_t n = fread(buf, 1, sizeof(buf), stdin);
        if (n > 0) return fuzz_jpeg(buf, n);
    }
    return 0;
}
#endif

/* Standard in-process entry point for fuzzer-tool .so mode */
__attribute__((visibility("default")))
int fuzz_shm_run(const unsigned char *buf, size_t size) {
    return fuzz_jpeg(buf, size);
}
