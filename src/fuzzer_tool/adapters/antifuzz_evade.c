/* antifuzz_evade.c — LD_PRELOAD countermeasure for AntiFuzz-hardened targets.
 *
 * AntiFuzz (Güler et al., USENIX Security '19) hardens a binary against
 * fuzzing with, among others, two tricks this fuzzer's binary-only modes
 * (ptrace coverage, Intel-PT, --no-shm) cannot see through on their own:
 *
 *   §4.2  self-ptrace anti-debug: the target calls ptrace(PTRACE_TRACEME).
 *         If it fails, something is already tracing it, so the target exits
 *         before any input is processed — blinding the whole campaign.
 *
 *   §4.3  delay-on-malformed-input: the target sleeps (sleep/usleep/
 *         nanosleep) on its own error paths, which is almost every fuzzer
 *         input, cutting exec/s by orders of magnitude.
 *
 * This preload neutralises both from outside the target, with no source
 * access, so the fuzzer's own instrumentation keeps working:
 *
 *   ptrace(PTRACE_TRACEME, ...) ─▶ return 0 (pretend "nobody is tracing")
 *   sleep/usleep/nanosleep/     ─▶ return success immediately (no wait)
 *   clock_nanosleep
 *
 * Every other ptrace request is forwarded to the real libc call unchanged,
 * so the fuzzer's ptrace-based coverage (the actual tracer) is unaffected.
 *
 * Build:  cc -shared -fPIC -O2 -o antifuzz_evade.so antifuzz_evade.c -ldl
 * Use:    LD_PRELOAD=.../antifuzz_evade.so  (injected per-target by the
 *         fuzzer when --antifuzz-evade is set).
 *
 * Opt-out per behaviour via env, so a target that legitimately needs one
 * can keep it while the campaign disables the other:
 *   ANTIFUZZ_EVADE_PTRACE=0   leave ptrace() alone
 *   ANTIFUZZ_EVADE_SLEEP=0    leave the sleep family alone
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ptrace.h>
#include <time.h>
#include <unistd.h>

/* An env var is "off" only when explicitly set to 0/false/no. */
static int evade_disabled(const char *name) {
    const char *v = getenv(name);
    if (!v) return 0;

    return strcmp(v, "0") == 0 || strcasecmp(v, "false") == 0 ||
           strcasecmp(v, "no") == 0;
}

/* ── §4.2  self-ptrace anti-debug ──────────────────────────────────────
 * Only PTRACE_TRACEME is faked: that is the self-check. Every other
 * request (the fuzzer's own tracer uses ATTACH/CONT/GETREGS/...) is
 * forwarded verbatim, so coverage collection is untouched.             */
long ptrace(enum __ptrace_request request, ...) {
    static long (*real)(enum __ptrace_request, ...) = NULL;
    if (!real)
        real = (long (*)(enum __ptrace_request, ...))dlsym(RTLD_NEXT, "ptrace");

    /* ptrace is variadic in the libc wrapper; the 2nd..4th args are only
     * meaningful for forwarding, which we only do for non-TRACEME calls. */
    va_list ap;
    va_start(ap, request);
    void *pid = va_arg(ap, void *);
    void *addr = va_arg(ap, void *);
    void *data = va_arg(ap, void *);
    va_end(ap);

    if (request == PTRACE_TRACEME && !evade_disabled("ANTIFUZZ_EVADE_PTRACE"))
        return 0;  /* "success": the target believes it is untraced */

    return real(request, pid, addr, data);
}

/* ── §4.3  delay on malformed input ────────────────────────────────────
 * The delay is pure wall-clock waiting with no observable side effect, so
 * returning "slept fine, zero remaining" is behaviourally identical for
 * the target and removes the fuzzing tax entirely.                      */
unsigned int sleep(unsigned int seconds) {
    if (evade_disabled("ANTIFUZZ_EVADE_SLEEP")) {
        static unsigned int (*real)(unsigned int) = NULL;
        if (!real) real = (unsigned int (*)(unsigned int))dlsym(RTLD_NEXT, "sleep");
        return real(seconds);
    }

    return 0;  /* zero seconds left unslept */
}

int usleep(useconds_t usec) {
    if (evade_disabled("ANTIFUZZ_EVADE_SLEEP")) {
        static int (*real)(useconds_t) = NULL;
        if (!real) real = (int (*)(useconds_t))dlsym(RTLD_NEXT, "usleep");
        return real(usec);
    }

    return 0;
}

int nanosleep(const struct timespec *req, struct timespec *rem) {
    if (evade_disabled("ANTIFUZZ_EVADE_SLEEP")) {
        static int (*real)(const struct timespec *, struct timespec *) = NULL;
        if (!real)
            real = (int (*)(const struct timespec *, struct timespec *))dlsym(
                RTLD_NEXT, "nanosleep");
        return real(req, rem);
    }

    if (rem) {
        rem->tv_sec = 0;
        rem->tv_nsec = 0;
    }

    return 0;  /* slept the full request, nothing remaining */
}

int clock_nanosleep(clockid_t clock_id, int flags, const struct timespec *req,
                    struct timespec *rem) {
    if (evade_disabled("ANTIFUZZ_EVADE_SLEEP")) {
        static int (*real)(clockid_t, int, const struct timespec *,
                           struct timespec *) = NULL;
        if (!real)
            real = (int (*)(clockid_t, int, const struct timespec *,
                            struct timespec *))dlsym(RTLD_NEXT, "clock_nanosleep");
        return real(clock_id, flags, req, rem);
    }

    if (rem && flags != TIMER_ABSTIME) {
        rem->tv_sec = 0;
        rem->tv_nsec = 0;
    }

    return 0;
}
