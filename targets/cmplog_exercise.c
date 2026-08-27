/* Cmplog exercise target — uses all cmplog-interceptable functions
 * (memcmp, strcmp, strncmp, memchr). Bounds-checked everywhere,
 * never crashes. Designed to test the LD_PRELOAD cmplog shim. */
#include <stddef.h>
#include <string.h>
#include <unistd.h>

/* ── Compare-then-return: each path adds to the return value so the
 *    fuzzer sees different coverage / exit codes based on which
 *    comparisons were solved.
 *    Using memcpy for safe null-terminated copy (not intercepted by
 *    cmplog, so no infinite regress). ────────────────────────────── */

__attribute__((visibility("default")))
int fuzz_test(const unsigned char *buf, size_t len) {
    int score = 0;

    /* ── memcmp: magic byte sequences ──────────────────────────── */
    /* Three successive magic blocks:
     *   [0..3]  "CMPl"
     *   [4..6]  "OG!"
     *   [7..10] "fuzz"
     * Each match independently adds to score. */
    if (len >= 4 && memcmp(buf, "CMPl", 4) == 0) score += 10;
    if (len >= 8 && memcmp(buf + 4, "OG!", 3) == 0) score += 10;
    if (len >= 12 && memcmp(buf + 7, "fuzz", 4) == 0) score += 10;

    /* Full 11-byte header match gives bonus */
    if (len >= 11 && memcmp(buf, "CMPLOGfuzz", 10) == 0) score += 30;

    /* ── strcmp / strncmp: null-terminated comparisons ─────────── */
    /* Create a safe null-terminated copy (clamped to 255 to avoid
     * large stack copies). */
    size_t copy_len = len < 255 ? len : 255;
    char copy[256];
    memcpy(copy, buf, copy_len);
    copy[copy_len] = '\0';

    if (strcmp(copy, "CMPLOG_ACTIVE") == 0) score += 20;
    if (strcmp(copy, "CMPLOG_DISABLED") == 0) score += 5;

    /* Prefix matching via strncmp */
    if (strncmp(copy, "TEST_", 5) == 0) score += 15;
    if (strncmp(copy, "FUZZ_", 5) == 0) score += 5;
    if (strncmp(copy, "BENCH_", 6) == 0) score += 3;

    /* ── memchr: find sentinel bytes ───────────────────────────── */
    if (memchr(copy, 'X', copy_len) != NULL) score += 10;
    if (memchr(copy, 'Y', copy_len) != NULL) score += 3;
    if (memchr(copy, 'Z', copy_len) != NULL) score += 1;

    /* Sentinel byte that must NOT be present */
    if (memchr(copy, '\xff', copy_len) == NULL) score += 5;

    /* ── strcmp with multiple valid test patterns ──────────────── */
    if (strcmp(copy, "COMPARE_A") == 0) score += 8;
    if (strcmp(copy, "COMPARE_B") == 0) score += 6;
    if (strcmp(copy, "COMPARE_C") == 0) score += 4;

    return score;
}

/* ── New cmplog interceptors: bcmp, wide-char, set-scan, memrchr ──── */
static int new_cmplog_test(const unsigned char *buf, size_t len) {
    int score = 0;
    if (len < 16) return 0;

    /* bcmp: byte-block comparison */
    if (bcmp(buf, buf + 8, 8) == 0) score += 10;
    if (bcmp(buf, buf + 8, 4) != 0) score += 3;

    /* memrchr: find last occurrence of byte */
    if (memrchr(buf, 'X', len) != NULL) score += 7;
    if (memrchr(buf, '\0', len) != NULL) score += 5;
    if (memrchr(buf, 0xff, len) == NULL) score += 2;

    /* strpbrk: find first char from accept set */
    if (strpbrk((char *)buf, "XYZ") != NULL) score += 6;
    if (strpbrk((char *)buf, "ABC") != NULL) score += 3;

    /* strspn: length of initial segment from accept set */
    if (strspn((char *)buf, "ABCDEFGHIJKLMNOPQRSTUVWXYZ") > 4) score += 5;
    if (strspn((char *)buf, "abcdefghijklmnopqrstuvwxyz") > 0) score += 3;

    /* strcspn: length of initial segment from reject set */
    if (strcspn((char *)buf, "0123456789") > 0) score += 5;
    if (strcspn((char *)buf, "!@#$%") > 0) score += 3;

    /* wide-char comparisons */
    wchar_t wa[64], wb[64];
    size_t wlen = len < 60 ? len : 60;
    for (size_t i = 0; i < wlen; i++) wa[i] = (wchar_t)buf[i];
    for (size_t i = 0; i < wlen; i++) wb[i] = (wchar_t)buf[wlen - 1 - i];
    wa[wlen] = L'\0';
    wb[wlen] = L'\0';

    /* wmemcmp */
    if (wmemcmp(wa, wb, wlen) == 0) score += 8;
    if (wmemcmp(wa, wb, 4) != 0) score += 4;

    /* wcscmp */
    if (wcscmp(wa, wb) == 0) score += 6;
    if (wcscmp(wa, wa) == 0) score += 3;

    /* wcsncmp */
    if (wcsncmp(wa, wb, 4) == 0) score += 5;
    if (wcsncmp(wa, wa, wlen) == 0) score += 3;

    /* wcscasecmp */
    if (wcscasecmp(wa, wb) == 0) score += 6;
    if (wcscasecmp(wa, wa) == 0) score += 3;

    return score;
}

__attribute__((visibility("default")))
int fuzz_shm_run(const unsigned char *buf, size_t size) {
    return fuzz_test(buf, size) + new_cmplog_test(buf, size);
}

int main(void) {
    char buf[1024];
    ssize_t n = read(0, buf, sizeof(buf) - 1);
    if (n <= 0) return 0;
    buf[n] = '\0';
    return fuzz_test((unsigned char *)buf, (size_t)n) + new_cmplog_test((unsigned char *)buf, (size_t)n);
}
