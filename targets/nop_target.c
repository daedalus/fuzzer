/* No-op fuzz target for EPS benchmarking — always returns 0. */
#include <stddef.h>

__attribute__((visibility("default")))
int fuzz_test(const unsigned char *buf, size_t len) {
    (void)buf;
    (void)len;
    return 0;
}

__attribute__((visibility("default")))
int fuzz_shm_run(const unsigned char *buf, size_t size) {
    return fuzz_test(buf, size);
}
