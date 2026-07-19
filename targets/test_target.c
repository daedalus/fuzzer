/* Minimal crash target for integration tests.
 * Crashes on input starting with "CRASH" + specific byte. */
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Fuzz function for ctypes / inprocess modes */
__attribute__((visibility("default")))
int fuzz_test(const unsigned char *buf, size_t len) {
    if (len >= 6 && memcmp(buf, "CRASH", 5) == 0) {
        if (buf[5] == 'S') {
            ((void(*)())0)();
        }
        if (buf[5] == 'A') {
            abort();
        }
    }
    return 0;
}

/* Standard in-process entry point for fuzzer-tool .so mode */
__attribute__((visibility("default")))
int fuzz_shm_run(const unsigned char *buf, size_t size) {
    return fuzz_test(buf, size);
}

int main(void) {
    char buf[256];
    ssize_t n = read(0, buf, sizeof(buf) - 1);
    if (n <= 0) return 0;
    buf[n] = '\0';

    return fuzz_test((unsigned char *)buf, (size_t)n);
}
