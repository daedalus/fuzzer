/* Minimal ASAN test target — triggers heap-buffer-overflow,
 * use-after-free, or stack-buffer-overflow based on input.
 *
 * Compile standalone:
 *   gcc -g -fsanitize=address -o targets/asan_target targets/asan_target.c
 *
 * Compile shared library (for inprocess modes):
 *   gcc -g -fsanitize=address -shared -fPIC -o targets/asan_target.so targets/asan_target.c
 */
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Fuzz function for ctypes / inprocess modes */
__attribute__((visibility("default")))
int fuzz(const unsigned char *buf, size_t len) {
    if (len >= 5 && memcmp(buf, "BUG!", 4) == 0) {
        if (buf[4] == 'H') {
            /* heap-buffer-overflow: write past heap allocation */
            char *p = malloc(8);
            memset(p + 8, 'X', 1);  /* off-by-one OOB write */
            free(p);
        } else if (buf[4] == 'U') {
            /* heap-use-after-free: use freed memory */
            char *p = malloc(8);
            memset(p, 'A', 8);
            free(p);
            volatile char c = p[0];  /* use after free */
            (void)c;
        } else if (buf[4] == 'S') {
            /* stack-buffer-overflow: write past stack buffer */
            char small[4];
            memset(small, 'Y', 8);  /* 8 bytes into 4-byte buffer */
        }
    }
    return 0;
}

/* Main for standalone execution (reads from stdin) */
int main(void) {
    char buf[256];
    ssize_t n = read(0, buf, sizeof(buf) - 1);
    if (n <= 0) return 0;
    return fuzz((unsigned char *)buf, (size_t)n);
}
