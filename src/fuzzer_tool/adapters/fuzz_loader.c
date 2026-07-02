/* fuzz_loader.c — Minimal C loader for coverage-guided fuzzing.

Compiles once, reuses across all iterations. Loads a target (shared library
via dlopen or standalone executable via fork+exec), calls
LLVMFuzzerTestOneInput, and returns the coverage bitmap.

Protocol (stdin/stdout binary):
  Init:   "INIT <target> <func> <bitmap_out> <timeout>\n"
          NOTE: target path must not contain whitespace (%s sscanf).
  Run:    "RUN <len>\n<data>"
          len is capped at PIPE_BUF_LIMIT (56KB) to avoid pipe deadlock.
  Quit:   "QUIT\n"
  Reply:  "RC <rc> <bmp_len>\n<bmp>"

Timeout enforcement:
  .so targets: sigsetjmp/siglongjmp via SIGALRM — no fork overhead.
  executables: fork+exec with SIGALRM-interrupted waitpid — child is
               SIGKILL'd if waitpid returns EINTR.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <dlfcn.h>
#include <sys/wait.h>
#include <unistd.h>
#include <signal.h>
#include <setjmp.h>

/* Pipe buffer on Linux is 65536 (PIPE_BUF). Cap data well below that
   to avoid deadlock: parent write() blocks when buffer is full, but
   the child may not have started reading yet. */
#define MAX_DATA 57344  /* 56KB — safely under 64KB pipe buffer */
#define MAX_BMP  65536

typedef int (*fuzz_fn)(const uint8_t *, size_t);

static fuzz_fn target_fn = NULL;
static char bitmap_out[256] = {0};
static int is_executable = 0;

static sigjmp_buf timeout_jmp;
static volatile sig_atomic_t timed_out = 0;

/* Read a line from stdin (up to newline, not including it) */
static int read_line(char *buf, int maxlen) {
    int i = 0;
    while (i < maxlen - 1) {
        int c = fgetc(stdin);
        if (c == EOF || c == '\n') break;
        buf[i++] = c;
    }
    buf[i] = '\0';
    return i;
}

/* Read exactly n bytes from stdin */
static void read_bytes(uint8_t *buf, size_t n) {
    size_t got = 0;
    while (got < n) {
        size_t r = fread(buf + got, 1, n - got, stdin);
        if (r == 0) break;
        got += r;
    }
}

/* Write exactly n bytes to stdout */
static void write_bytes(const uint8_t *buf, size_t n) {
    fwrite(buf, 1, n, stdout);
    fflush(stdout);
}

/* Read coverage bitmap from file */
static int read_bitmap_file(uint8_t *buf, int maxlen) {
    if (bitmap_out[0] == '\0') return 0;
    FILE *f = fopen(bitmap_out, "rb");
    if (!f) return 0;
    int n = fread(buf, 1, maxlen, f);
    fclose(f);
    return n;
}

static char target_path_global[256] = {0};
static int timeout_seconds = 5;

/* SIGALRM handler for direct .so timeout — longjmps back to caller */
static void timeout_handler(int sig) {
    (void)sig;
    timed_out = 1;
    siglongjmp(timeout_jmp, 1);
}

/* SIGALRM handler for fork/exec path — kills the child, then waitpid reaps */
static pid_t exec_child_pid = -1;

static void exec_alarm_handler(int sig) {
    (void)sig;
    if (exec_child_pid > 0) {
        kill(exec_child_pid, SIGKILL);
    }
}

static int run_executable(const uint8_t *data, size_t len, uint8_t *bmp, int *bmp_len) {
    int pipefd[2];
    if (pipe(pipefd) < 0) return -2;

    pid_t pid = fork();
    if (pid < 0) { close(pipefd[0]); close(pipefd[1]); return -2; }

    if (pid == 0) {
        close(pipefd[1]);
        dup2(pipefd[0], STDIN_FILENO);
        close(pipefd[0]);
        signal(SIGCHLD, SIG_DFL);
        signal(SIGALRM, SIG_DFL);
        setenv("_COV_BITMAP_OUT", bitmap_out, 1);
        execl(target_path_global, target_path_global, (char *)NULL);
        _exit(127);
    }

    close(pipefd[0]);
    size_t written = 0;
    while (written < len) {
        ssize_t w = write(pipefd[1], data + written, len - written);
        if (w <= 0) break;
        written += w;
    }
    close(pipefd[1]);

    /* Set up alarm: handler kills child directly, then waitpid reaps it */
    exec_child_pid = pid;
    struct sigaction sa;
    sa.sa_handler = exec_alarm_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;  /* NO SA_RESTART — we want waitpid interrupted */
    sigaction(SIGALRM, &sa, NULL);
    alarm(timeout_seconds);

    int status = 0;
    waitpid(pid, &status, 0);

    alarm(0);
    signal(SIGALRM, SIG_DFL);
    exec_child_pid = -1;

    int rc = -2;
    if (WIFEXITED(status)) {
        rc = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
        rc = -WTERMSIG(status);
    }

    *bmp_len = read_bitmap_file(bmp, MAX_BMP);
    return rc;
}

int main(void) {
    char line[512];
    uint8_t data[MAX_DATA];
    uint8_t bmp[MAX_BMP];

    if (!read_line(line, sizeof(line))) return 1;
    char func_name[256];
    char timeout_str[16] = "5";
    sscanf(line, "INIT %255s %255s %255s %15s", target_path_global, func_name, bitmap_out, timeout_str);
    timeout_seconds = atoi(timeout_str);
    if (timeout_seconds <= 0) timeout_seconds = 5;

    is_executable = (access(target_path_global, X_OK) == 0);

    if (!is_executable) {
        void *handle = dlopen(target_path_global, RTLD_NOW);
        if (!handle) {
            fprintf(stderr, "dlopen failed: %s\n", dlerror());
            return 1;
        }
        target_fn = (fuzz_fn)dlsym(handle, func_name);
        if (!target_fn) {
            fprintf(stderr, "dlsym failed: %s\n", dlerror());
            return 1;
        }
    }

    printf("READY\n");
    fflush(stdout);

    while (1) {
        if (!read_line(line, sizeof(line))) break;
        if (strcmp(line, "QUIT") == 0) break;
        if (strncmp(line, "RUN ", 4) != 0) continue;

        int data_len = atoi(line + 4);
        if (data_len <= 0 || data_len > MAX_DATA) continue;
        read_bytes(data, data_len);

        int rc = -2;
        int bmp_len = 0;

        if (is_executable) {
            rc = run_executable(data, data_len, bmp, &bmp_len);
        } else if (target_fn) {
            /* Direct call with sigsetjmp timeout — no fork overhead */
            struct sigaction sa_new, sa_old;
            sa_new.sa_handler = timeout_handler;
            sigemptyset(&sa_new.sa_mask);
            sa_new.sa_flags = 0;
            sigaction(SIGALRM, &sa_new, &sa_old);

            timed_out = 0;
            if (sigsetjmp(timeout_jmp, 1) == 0) {
                alarm(timeout_seconds);
                rc = target_fn(data, data_len);
                alarm(0);
            } else {
                /* Longjmp from timeout_handler */
                rc = -1;
            }

            sigaction(SIGALRM, &sa_old, NULL);

            bmp_len = read_bitmap_file(bmp, MAX_BMP);
        }

        printf("RC %d %d\n", rc, bmp_len);
        fflush(stdout);
        if (bmp_len > 0) write_bytes(bmp, bmp_len);
    }

    return 0;
}
