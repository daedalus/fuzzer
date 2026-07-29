/* sigguard.c — Signal-jump guard for in-process crash isolation.
 *
 * Installs SIGSEGV/SIGABRT handlers that use siglongjmp to escape the
 * signal handler before glibc's abort() can reset the handler to SIG_DFL
 * and re-raise.  This is the only reliable way to survive abort() calls
 * in-process (the AFL shim's #define abort() macro covers the target's
 * own code, but pre-compiled libraries like libasan call the real abort()).
 *
 * Two entry points:
 *   sigguard_call(func, arg) — generic: calls func(arg), returns 0 or signal
 *   sigguard_call_fuzz(func, data, size) — fuzz-specific: calls func(data, size)
 *
 * Compile with: gcc -shared -fPIC -O2 -o sigguard.so sigguard.c
 */

#include <setjmp.h>
#include <signal.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

typedef int (*fuzz_func_t)(const uint8_t *, size_t);
typedef void (*void_func_t)(void *);

static struct sigaction _old_segv;
static struct sigaction _old_abrt;
static sigjmp_buf _jmp_buf;
static volatile int _crashed_sig;

static void _handler(int sig) {
    _crashed_sig = sig;
    siglongjmp(_jmp_buf, 1);
}

/* Generic: call func(arg), crash-safe */
int sigguard_call(void_func_t func, void *arg) {
    struct sigaction sa;
    _crashed_sig = 0;
    sa.sa_handler = _handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGSEGV, &sa, &_old_segv);
    sigaction(SIGABRT, &sa, &_old_abrt);
    if (sigsetjmp(_jmp_buf, 1) == 0)
        func(arg);
    sigaction(SIGSEGV, &_old_segv, NULL);
    sigaction(SIGABRT, &_old_abrt, NULL);
    return _crashed_sig;
}

/* Fuzz-specific: call func(data, size), crash-safe */
int sigguard_call_fuzz(fuzz_func_t func, const uint8_t *data, size_t size) {
    struct sigaction sa;
    _crashed_sig = 0;
    sa.sa_handler = _handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGSEGV, &sa, &_old_segv);
    sigaction(SIGABRT, &sa, &_old_abrt);
    if (sigsetjmp(_jmp_buf, 1) == 0)
        func(data, size);
    sigaction(SIGSEGV, &_old_segv, NULL);
    sigaction(SIGABRT, &_old_abrt, NULL);
    return _crashed_sig;
}
