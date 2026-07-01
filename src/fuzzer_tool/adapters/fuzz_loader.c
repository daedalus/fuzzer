/* fuzz_loader.c — Minimal C loader for coverage-guided fuzzing.

Compiles once, reuses across all iterations. Loads a target (shared library
via dlopen or standalone executable via fork+exec), calls
LLVMFuzzerTestOneInput, and returns the coverage bitmap.

Protocol (stdin/stdout binary):
  Init:   "INIT <target> <func> <bitmap_out>\n"
  Run:    "RUN <len>\n<data>"
  Quit:   "QUIT\n"
  Reply:  "RC <rc> <bmp_len>\n<bmp>"
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <dlfcn.h>
#include <sys/wait.h>
#include <unistd.h>
#include <signal.h>

#define MAX_DATA 65535
#define MAX_BMP  65536

typedef int (*fuzz_fn)(const uint8_t *, size_t);

static fuzz_fn target_fn = NULL;
static char bitmap_out[256] = {0};
static int is_executable = 0;

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

static void alarm_handler(int sig) {
    (void)sig;
    /* alarm fired — child will be reaped below */
}

/* Run standalone executable: fork, exec with stdin pipe, read bitmap file */
static int run_executable(const uint8_t *data, size_t len, uint8_t *bmp, int *bmp_len) {
    int pipefd[2];
    if (pipe(pipefd) < 0) return -2;

    pid_t pid = fork();
    if (pid < 0) { close(pipefd[0]); close(pipefd[1]); return -2; }

    if (pid == 0) {
        /* Child */
        close(pipefd[1]);
        dup2(pipefd[0], STDIN_FILENO);
        close(pipefd[0]);
        signal(SIGCHLD, SIG_DFL);
        signal(SIGALRM, SIG_DFL);
        /* Set _COV_BITMAP_OUT for the target */
        setenv("_COV_BITMAP_OUT", bitmap_out, 1);
        execl(target_path_global, target_path_global, (char *)NULL);
        _exit(127);
    }

    /* Parent: write data to child's stdin */
    close(pipefd[0]);
    size_t written = 0;
    while (written < len) {
        ssize_t w = write(pipefd[1], data + written, len - written);
        if (w <= 0) break;
        written += w;
    }
    close(pipefd[1]);

    /* Set alarm for child timeout */
    struct sigaction sa;
    sa.sa_handler = alarm_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGALRM, &sa, NULL);
    alarm(timeout_seconds);

    /* Wait for child */
    int status = 0;
    pid_t waited = waitpid(pid, &status, 0);

    alarm(0); /* cancel alarm */
    signal(SIGALRM, SIG_DFL);

    int rc = -2;
    if (waited < 0) {
        /* waitpid interrupted (by SIGALRM) — child timed out, kill it */
        if (kill(pid, SIGKILL) == 0)
            waitpid(pid, NULL, 0);
        rc = -1; /* timeout */
    } else if (WIFEXITED(status)) {
        rc = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
        rc = -WTERMSIG(status);
    }

    /* Read bitmap from file */
    *bmp_len = read_bitmap_file(bmp, MAX_BMP);
    return rc;
}

int main(void) {
    char line[512];
    uint8_t data[MAX_DATA];
    uint8_t bmp[MAX_BMP];

    /* Read init */
    if (!read_line(line, sizeof(line))) return 1;
    /* INIT <target> <func> <bitmap_out> */
    char func_name[256];
    char timeout_str[16] = "5";
    sscanf(line, "INIT %255s %255s %255s %15s", target_path_global, func_name, bitmap_out, timeout_str);
    timeout_seconds = atoi(timeout_str);
    if (timeout_seconds <= 0) timeout_seconds = 5;

    /* Check if target is executable */
    is_executable = (access(target_path_global, X_OK) == 0);

    if (!is_executable) {
        /* Load shared library */
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

    /* Signal ready */
    printf("READY\n");
    fflush(stdout);

    /* Main loop */
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
            /* Fork child to enforce timeout on direct .so calls */
            int result_pipe[2];
            if (pipe(result_pipe) < 0) {
                rc = -2;
            } else {
                pid_t child = fork();
                if (child < 0) {
                    close(result_pipe[0]); close(result_pipe[1]);
                    rc = -2;
                } else if (child == 0) {
                    /* Child: run target_fn, write result to pipe */
                    close(result_pipe[0]);
                    struct sigaction sa_ch;
                    sa_ch.sa_handler = alarm_handler;
                    sigemptyset(&sa_ch.sa_mask);
                    sa_ch.sa_flags = 0;
                    sigaction(SIGALRM, &sa_ch, NULL);
                    alarm(timeout_seconds);
                    int child_rc = target_fn(data, data_len);
                    alarm(0);
                    /* Atomic write of 4-byte int */
                    write(result_pipe[1], &child_rc, sizeof(child_rc));
                    close(result_pipe[1]);
                    _exit(0);
                }
                /* Parent: wait for child with alarm */
                close(result_pipe[1]);
                struct sigaction sa;
                sa.sa_handler = alarm_handler;
                sigemptyset(&sa.sa_mask);
                sa.sa_flags = 0;
                sigaction(SIGALRM, &sa, NULL);
                alarm(timeout_seconds);
                int status = 0;
                pid_t waited = waitpid(child, &status, 0);
                alarm(0);
                signal(SIGALRM, SIG_DFL);
                if (waited < 0) {
                    /* Interrupted by SIGALRM — child timed out */
                    kill(child, SIGKILL);
                    waitpid(child, NULL, 0);
                    close(result_pipe[0]);
                    rc = -1;
                } else {
                    int child_rc = -2;
                    ssize_t n = read(result_pipe[0], &child_rc, sizeof(child_rc));
                    close(result_pipe[0]);
                    rc = (n == sizeof(child_rc)) ? child_rc : -2;
                }
            }
            bmp_len = read_bitmap_file(bmp, MAX_BMP);
        }

        /* Reply: RC <rc> <bmp_len>\n<bmp> */
        printf("RC %d %d\n", rc, bmp_len);
        fflush(stdout);
        if (bmp_len > 0) write_bytes(bmp, bmp_len);
    }

    return 0;
}
