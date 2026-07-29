/* Fuzz target with UBSAN-detectable undefined behaviour.
 *
 * Triggered by input starting with "OFL" (signed integer overflow):
 *   OFL<X>  — multiplies INT_MAX by 2
 *
 * Triggered by input starting with "ZER" (division by zero):
 *   ZER     — divides 1 by 0
 *
 * Triggered by input starting with "SFT" (shift overflow):
 *   SFT     — shifts 1 by 99999
 *
 * Compile:
 *   clang -g -fsanitize=undefined -shared -fPIC -include src/fuzzer_tool/adapters/afl_shim.c \
 *         -o targets/ubsan_target.so targets/ubsan_target.c
 */
#include <stdint.h>
#include <stdlib.h>
#include <limits.h>

__attribute__((visibility("default")))
int fuzz_shm_run(const uint8_t *data, size_t size) {
    if (size < 3) return 0;

    if (data[0] == 'O' && data[1] == 'F' && data[2] == 'L') {
        /* Signed integer overflow */
        int x = INT_MAX;
        return x * 2;  /* UBSAN: signed-integer-overflow */
    }

    if (data[0] == 'Z' && data[1] == 'E' && data[2] == 'R') {
        /* Division by zero */
        int x = 1;
        return x / 0;  /* UBSAN: integer-divide-by-zero */
    }

    if (data[0] == 'S' && data[1] == 'F' && data[2] == 'T') {
        /* Oversized shift */
        int x = 1;
        return x << 99999;  /* UBSAN: shift-exponent */
    }

    return 0;
}
