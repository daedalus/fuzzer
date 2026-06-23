/* Minimal crash target for integration tests.
 * Crashes on input starting with "CRASH" + specific byte. */
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(void) {
    char buf[256];
    ssize_t n = read(0, buf, sizeof(buf) - 1);
    if (n <= 0) return 0;
    buf[n] = '\0';

    if (n >= 6 && memcmp(buf, "CRASH", 5) == 0) {
        if (buf[5] == 'S') {
            /* SIGSEGV */
            ((void(*)())0)();
        }
        if (buf[5] == 'A') {
            /* SIGABRT */
            abort();
        }
    }
    return 0;
}
