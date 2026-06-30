/* Fuzz target for libpng — reads PNG from stdin or file.
 *
 * No setjmp error handling: libpng errors abort the process,
 * allowing the fuzzer to detect crashes and explore deeper paths.
 */
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Custom error handler: abort so fuzzer detects the crash as SIGABRT */
static void png_abort_handler(png_structp png_ptr, png_const_charp msg) {
    (void)png_ptr;
    (void)msg;
    abort();
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

static void fuzz_png(const unsigned char *buf, size_t size) {
    if (size < 8) return;

    /* Verify PNG signature */
    if (png_sig_cmp(buf, 0, 8)) return;

    /* Custom error/warning handlers — no setjmp needed */
    png_structp png_ptr = png_create_read_struct(
        PNG_LIBPNG_VER_STRING, NULL, png_abort_handler, png_warning_handler);
    if (!png_ptr) return;

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        return;
    }

    /* Write buf to temp file and read from it */
    FILE *f = tmpfile();
    if (!f) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        return;
    }
    fwrite(buf, 1, size, f);
    rewind(f);

    png_set_read_fn(png_ptr, f, user_read_data);
    png_read_info(png_ptr, info_ptr);

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
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(f);
        return;
    }

    png_read_update_info(png_ptr, info_ptr);

    size_t rowbytes = png_get_rowbytes(png_ptr, info_ptr);
    unsigned char *row = malloc(rowbytes);
    if (row) {
        for (png_uint_32 y = 0; y < height; y++)
            png_read_row(png_ptr, row, NULL);
        free(row);
    }

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
