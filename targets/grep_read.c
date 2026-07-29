/* Fuzz target for GNU Grep — exec-based wrapper around /usr/bin/grep.
 *
 * Tests the system-installed GNU Grep binary across multiple search modes:
 *   1. Basic regex (grep -G)
 *   2. Extended regex (grep -E)
 *   3. Fixed string (grep -F)
 *   4. PCRE2 (grep -P, if available)
 *   5. Case-insensitive (grep -i)
 *   6. Whole-word (grep -w)
 *   7. Whole-line (grep -x)
 *   8. Invert match (grep -v)
 *
 * Input format: one byte mode | one byte pattern length | pattern | text
 *   byte 0:    mode (0-8)
 *   byte 1:    pattern length (clamped to 255)
 *   bytes 2..2+plen-1: pattern string
 *   bytes 2+plen..: text to search
 *
 * The target writes text to a temp file and exec's /usr/bin/grep.
 * This is a file-mode target; use with fuzzer-tool's -F flag for
 * full file-per-test execution.
 *
 * Returns 1 on crashes (signal exit from grep), 0 on clean execution.
 *
 * AFL edge coverage tracks the wrapper's own code paths. For code
 * coverage of grep itself, use ptrace mode (--no-shm --deep-coverage)
 * or hook up the fuzzer's ptrace coverage.
 *
 * Compile standalone:
 *   gcc -O2 -g -o targets/grep_read targets/grep_read.c
 *
 * Compile shared library:
 *   gcc -O2 -g -shared -fPIC -include src/fuzzer_tool/adapters/afl_shim.c \
 *       -o targets/grep_read.so targets/grep_read.c -Wl,--export-dynamic
 */
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <signal.h>

/* AFL edge coverage — provided by afl_shim.c */
extern void __afl_map_edge(unsigned int cur_loc);

/* GNU Grep binary path */
#define GREP_BINARY "/usr/bin/grep"

/* The grep exec wrapper modes */
enum grep_mode {
    MODE_BASIC_REGEX  = 0,  /* grep -G (default) */
    MODE_EXT_REGEX    = 1,  /* grep -E */
    MODE_FIXED        = 2,  /* grep -F */
    MODE_PCRE         = 3,  /* grep -P */
    MODE_ICASE        = 4,  /* grep -Fi */
    MODE_WORD         = 5,  /* grep -w */
    MODE_LINE         = 6,  /* grep -x */
    MODE_INVERT       = 7,  /* grep -v */
    MODE_ALL          = 8,  /* run grep in raw mode to exercise arg parsing */
    MODE_MAX          = 9,
};

/* Run grep with given flags, pattern, and file path.
 * Returns 0 on clean exit, negative signal number on crash. */
static int run_grep(const char *flag, const char *pattern, const char *filepath) {
    pid_t pid = fork();
    if (pid == -1) {
        __afl_map_edge(0x1010);
        return -1;
    }

    if (pid == 0) {
        /* Child: exec grep */
        /* Redirect stdout/stderr to /dev/null to suppress output */
        FILE *null = fopen("/dev/null", "w");
        if (null) {
            dup2(fileno(null), STDOUT_FILENO);
            dup2(fileno(null), STDERR_FILENO);
            if (fileno(null) > STDERR_FILENO)
                fclose(null);
        }
        execlp(GREP_BINARY, "grep", flag, pattern, filepath, (char *)NULL);
        /* If execlp fails, exit silently */
        _exit(127);
    }

    /* Parent: wait for child */
    int status;
    pid_t rc;
    do {
        rc = waitpid(pid, &status, 0);
    } while (rc == -1 && errno == EINTR);

    if (rc == -1) {
        __afl_map_edge(0x1020);
        return -1;
    }

    if (WIFEXITED(status)) {
        return 0;  /* clean exit (match or no match, both valid) */
    }

    if (WIFSIGNALED(status)) {
        int sig = WTERMSIG(status);
        return -sig;  /* crash */
    }

    return -1;
}

__attribute__((visibility("default")))
int fuzz_grep(const unsigned char *buf, size_t size) {
    __afl_map_edge(0x1000);
    if (size < 3) { __afl_map_edge(0x1001); return 0; }

    unsigned char mode = buf[0];
    size_t plen = buf[1];
    if (plen > 255) plen = 255;
    if (2 + plen >= size) { __afl_map_edge(0x1002); return 0; }

    /* Extract null-terminated pattern */
    char pattern[256];
    memcpy(pattern, buf + 2, plen);
    pattern[plen] = '\0';

    /* Write the text portion to a temp file */
    const char *text = (const char *)buf + 2 + plen;
    size_t text_len = size - 2 - plen;

    char tmpname[] = "/tmp/grep_fuzz_XXXXXX";
    int fd = mkstemp(tmpname);
    if (fd == -1) { __afl_map_edge(0x1011); return 0; }
    __afl_map_edge(0x1003);

    ssize_t written = write(fd, text, text_len);
    close(fd);
    if (written < 0 || (size_t)written != text_len) {
        __afl_map_edge(0x1012);
        unlink(tmpname);
        return 0;
    }
    __afl_map_edge(0x1004);

    /* Ensure the file is readable */
    if (text_len == 0) {
        /* Write at least one byte so grep doesn't EOF-immediately */
        fd = open(tmpname, O_WRONLY | O_APPEND);
        if (fd != -1) {
            write(fd, "\n", 1);
            close(fd);
        }
    }

    int result = 0;
    int crash_sig = 0;

    __afl_map_edge(0x1100 + mode);

    switch (mode % MODE_MAX) {
        case MODE_BASIC_REGEX:
            crash_sig = run_grep("-G", pattern, tmpname);
            break;
        case MODE_EXT_REGEX:
            crash_sig = run_grep("-E", pattern, tmpname);
            break;
        case MODE_FIXED:
            crash_sig = run_grep("-F", pattern, tmpname);
            break;
        case MODE_PCRE:
            crash_sig = run_grep("-P", pattern, tmpname);
            break;
        case MODE_ICASE:
            crash_sig = run_grep("-Fi", pattern, tmpname);
            break;
        case MODE_WORD:
            crash_sig = run_grep("-w", pattern, tmpname);
            break;
        case MODE_LINE:
            crash_sig = run_grep("-x", pattern, tmpname);
            break;
        case MODE_INVERT:
            crash_sig = run_grep("-v", pattern, tmpname);
            break;
        case MODE_ALL:
            /* Exercise combined flags: match whole words, case-insensitive */
            crash_sig = run_grep("-wi", pattern, tmpname);
            break;
        default:
            crash_sig = run_grep("-G", pattern, tmpname);
            break;
    }

    if (crash_sig < 0) {
        __afl_map_edge(0x1200);
        result = 1;  /* crash detected */
    } else {
        __afl_map_edge(0x1201);
    }

    /* Clean up temp file */
    unlink(tmpname);
    __afl_map_edge(0x1005);

    return result;
}

/* Standard in-process entry point for fuzzer-tool .so mode */
__attribute__((visibility("default")))
int fuzz_shm_run(const unsigned char *buf, size_t size) {
    return fuzz_grep(buf, size);
}

#ifdef __AFL_HAVE_MANUAL_CONTROL
int main(void) {
    __AFL_INIT();
    unsigned char *buf = __AFL_FUZZ_TEST_CASE_BUF;
    while (__AFL_LOOP(1000)) {
        int len = __AFL_FUZZ_TEST_CASE_LEN;
        fuzz_grep(buf, len);
    }
    return 0;
}
#else
int main(int argc, char **argv) {
    if (argc == 2) {
        FILE *f = fopen(argv[1], "rb");
        if (!f) return 1;
        fseek(f, 0, SEEK_END);
        long s = ftell(f);
        rewind(f);
        unsigned char *buf = malloc(s);
        if (buf) {
            fread(buf, 1, s, f);
            int rc = fuzz_grep(buf, s);
            free(buf);
            fclose(f);
            return rc;
        }
        fclose(f);
        return 1;
    } else {
        unsigned char buf[65536];
        size_t n = fread(buf, 1, sizeof(buf), stdin);
        if (n > 0) return fuzz_grep(buf, n);
    }
    return 0;
}
#endif
