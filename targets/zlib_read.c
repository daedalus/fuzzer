/* Fuzz target for zlib — decompress data from stdin or file.
 *
 * Uses zlib's inflate function to decompress input data.
 * On error, returns the zlib error code.
 *
 * Compile with:
 *   gcc -O2 -g -o targets/zlib_read targets/zlib_read.c -lz
 */
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <zlib.h>

__attribute__((visibility("default")))
int fuzz_zlib(const unsigned char *buf, size_t size) {
    /* Check minimum gzip header size */
    if (size < 10) {
        return 0;
    }

    /* Verify gzip magic number */
    if (buf[0] != 0x1F || buf[1] != 0x8B) {
        return 0;
    }

    /* Check compression method (deflate = 8) */
    if (buf[2] != 8) {
        return 0;
    }

    /* Initialize zlib stream */
    z_stream strm;
    memset(&strm, 0, sizeof(strm));
    strm.next_in = (Bytef *)buf;
    strm.avail_in = size;

    int ret = inflateInit2(&strm, -MAX_WBITS);  /* raw deflate, skip header */
    if (ret != Z_OK) {
        return 0;
    }

    /* Decompress in chunks */
    unsigned char outbuf[65536];
    int chunk_count = 0;

    do {
        strm.next_out = outbuf;
        strm.avail_out = sizeof(outbuf);

        ret = inflate(&strm, Z_NO_FLUSH);
        chunk_count++;

        if (ret != Z_OK && ret != Z_STREAM_END) {
            inflateEnd(&strm);
            return 0;
        }

        /* Check for unreasonable output size */
        if (chunk_count > 1000) {
            inflateEnd(&strm);
            return 0;
        }
    } while (ret != Z_STREAM_END);

    ret = inflateEnd(&strm);
    if (ret != Z_OK) {
        return 0;
    }

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
