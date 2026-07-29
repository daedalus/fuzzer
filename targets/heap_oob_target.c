/* Fuzz target with ASAN-detectable heap-buffer-overflow.
 *
 * Triggered by input starting with "OOB" followed by at least 1 byte:
 *   OOB<X>  — writes 1 byte past the end of a heap buffer
 *
 * Compile:
 *   clang -g -fsanitize=address -shared -fPIC -include src/fuzzer_tool/adapters/afl_shim.c \
 *         -o targets/heap_oob_target.so targets/heap_oob_target.c
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

__attribute__((visibility("default")))
int fuzz_shm_run(const uint8_t *data, size_t size) {
    if (size < 4 || data[0] != 'O' || data[1] != 'O' || data[2] != 'B')
        return 0;

    /* Heap-buffer-overflow: allocate 4 bytes, write past end */
    char *buf = malloc(4);
    if (!buf) return 0;
    memset(buf + 4, data[3], 1);  /* OOB write at offset 4 */
    free(buf);
    return 0;
}
