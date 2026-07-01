/* Fuzz target for libpng — reads PNG from stdin or file.
 *
 * Uses setjmp/longjmp for libpng error handling (the intended mechanism).
 * On error, longjmp returns to setjmp which exits with code 1 — the
 * fuzzer detects this as a crash (non-zero return) while libpng cleanup
 * happens properly via png_destroy_read_struct.
 *
 * Includes AFL-style edge coverage via afl_shim.c. Compile with:
 *   gcc -O2 -g -shared -fPIC -include src/fuzzer_tool/adapters/afl_shim.c \
 *       -o png_read.so png_read.c -lpng -lz -Wl,--export-dynamic
 */
#include <png.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* AFL edge coverage — provided by afl_shim.c */
extern void __afl_map_edge(unsigned int cur_loc);

static jmp_buf png_jmpbuf;

static jmp_buf png_jmpbuf;

/* Custom error handler: longjmp back to setjmp — proper libpng error flow */
static void png_error_handler(png_structp png_ptr, png_const_charp msg) {
    (void)png_ptr;
    (void)msg;
    longjmp(png_jmpbuf, 1);
}

/* Custom warning handler: no-op */
static void png_warning_handler(png_structp png_ptr, png_const_charp msg) {
    (void)png_ptr;
    (void)msg;
}

static void user_read_data(png_structp png_ptr, png_bytep data, png_size_t length) {
    FILE *f = (FILE *)png_get_io_ptr(png_ptr);
    if (fread(data, 1, length, f) != length)
        png_error(png_ptr, "read error");
}

__attribute__((visibility("default")))
int fuzz_png(const unsigned char *buf, size_t size) {
    __afl_map_edge(0x1000);
    if (size < 8) { __afl_map_edge(0x1001); return 0; }

    /* Verify PNG signature */
    if (png_sig_cmp(buf, 0, 8)) { __afl_map_edge(0x1002); return 0; }
    __afl_map_edge(0x1003);

    png_structp png_ptr = png_create_read_struct(
        PNG_LIBPNG_VER_STRING, NULL, png_error_handler, png_warning_handler);
    if (!png_ptr) { __afl_map_edge(0x1004); return 0; }
    __afl_map_edge(0x1005);

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) { __afl_map_edge(0x1006); png_destroy_read_struct(&png_ptr, NULL, NULL); return 0; }
    __afl_map_edge(0x1007);

    /* Write buf to temp file — create before setjmp so it's in scope */
    FILE *f = tmpfile();
    if (!f) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        return 0;
    }
    fwrite(buf, 1, size, f);
    rewind(f);

    /* setjmp: on error, longjmp returns here with val=1 */
    if (setjmp(png_jmpbuf)) {
        /* Error occurred — libpng called our handler which longjmp'd.
         * Clean up and return non-zero so the fuzzer detects the error. */
        __afl_map_edge(0x1100);
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(f);
        return 1;
    }
    __afl_map_edge(0x1101);

    png_set_read_fn(png_ptr, f, user_read_data);
    __afl_map_edge(0x1102);
    png_read_info(png_ptr, info_ptr);
    __afl_map_edge(0x1103);

    /* Force transforms */
    png_set_expand(png_ptr);
    png_set_strip_16(png_ptr);
    png_set_gray_to_rgb(png_ptr);
    png_set_add_alpha(png_ptr, 0xFF, PNG_FILLER_AFTER);
    png_set_interlace_handling(png_ptr);

    png_uint_32 width, height;
    int bit_depth, color_type;
    png_get_IHDR(png_ptr, info_ptr, &width, &height, &bit_depth, &color_type, NULL, NULL, NULL);

    /* Bounds to prevent OOM */
    if (width > 16384 || height > 16384) {
        __afl_map_edge(0x1200);
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(f);
        return 0;
    }
    __afl_map_edge(0x1201);

    png_read_update_info(png_ptr, info_ptr);
    __afl_map_edge(0x1300);

    size_t rowbytes = png_get_rowbytes(png_ptr, info_ptr);
    unsigned char *row = malloc(rowbytes);
    if (row) {
        for (png_uint_32 y = 0; y < height; y++) {
            png_read_row(png_ptr, row, NULL);
            __afl_map_edge(0x1400 + (y & 0xFF));
        }
        free(row);
    }
    __afl_map_edge(0x1500);

    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    fclose(f);
}

#ifdef __AFL_HAVE_MANUAL_CONTROL
int main(void) {
    __AFL_INIT();
    unsigned char *buf = __AFL_FUZZ_TEST_CASE_BUF;
    while (__AFL_LOOP(1000)) {
        int len = __AFL_FUZZ_TEST_CASE_LEN;
        fuzz_png(buf, len);
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
            fuzz_png(buf, size);
            free(buf);
        }
        fclose(f);
    } else {
        unsigned char buf[65536];
        size_t n = fread(buf, 1, sizeof(buf), stdin);
        if (n > 0) fuzz_png(buf, n);
    }
    return 0;
}
#endif
