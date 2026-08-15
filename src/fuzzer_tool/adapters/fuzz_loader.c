/* fuzz_loader.c — Minimal C loader for coverage-guided fuzzing.

Compiles once, reuses across all iterations. Loads a target (shared library
via dlopen or standalone executable via fork+exec) and reports the child's
exit status and stderr.

Coverage is NOT carried over this protocol. The loader is started with
__AFL_SHM_ID / AFL_MAP_SIZE in its environment; every exec'd child inherits
them and afl_shim.c's constructor attaches to that segment on its own, so
the target writes edges straight into the parent fuzzer's SHM. The parent
resets the map before each RUN and reads it after — exactly as it does on
the direct_lite path. An earlier version round-tripped a bitmap through a
file, which was disconnected from the SHM the target actually wrote to.

Protocol (stdin/stdout binary):
  Init:   "INIT <target> <func> <input_file> <timeout>\n"
          NOTE: paths must not contain whitespace (%s sscanf).
  Run:    "RUN <len>\n<data>"
  Quit:   "QUIT\n"
  Reply:  "RC <rc> <err_len>\n<err>"

<input_file> mirrors run_target_fast()'s temp file: the loader writes each
input there and execs `<target> <input_file>` with stdin redirected from it,
so targets that take argv[1] and targets that read stdin both see the data.

Execution modes (executable targets):
  forkserver — preferred. The target is exec'd ONCE; afl_shim.c's
               constructor then serves fork requests over fds 198/199, so
               each input costs a fork() from a fully initialised process
               rather than a fresh ELF load + linker + libc + ASAN init.
  fork+exec  — fallback for targets built without a forkserver-capable
               shim. One exec per input, i.e. no better than posix_spawn.

Return codes: >=0 exit status, -<signum> fatal signal, -1 timeout,
-2 loader-side failure.

Timeout enforcement:
  .so targets: sigsetjmp/siglongjmp via SIGALRM — no fork overhead.
  executables: fork+exec with SIGALRM — the handler SIGKILLs the child, and
               the timeout is reported as -1 rather than the -SIGKILL the
               status would otherwise show (which reads as a crash).
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
#include <fcntl.h>
#include <errno.h>
#include <sys/time.h>

/* Inputs arrive over the stdin pipe and are consumed immediately by
   read_bytes(), so there is no pipe-buffer deadlock and no need for the
   old 56KB cap. The ceiling only bounds the loader's own allocation; an
   oversized RUN is still drained in full so the protocol stays in sync. */
#define MAX_DATA (16 * 1024 * 1024)
/* Matches the 65536 that run_target_fast() reads back from its stderr pipe. */
#define MAX_ERR 65536

typedef int (*fuzz_fn)(const uint8_t *, size_t);

static fuzz_fn target_fn = NULL;
static int is_executable = 0;

static sigjmp_buf timeout_jmp;
static volatile sig_atomic_t timed_out = 0;

/* Read a line from stdin (up to newline, not including it).
   Uses raw read() syscall to avoid stdio buffering issues across fork(). */
static int read_line(char *buf, int maxlen) {
    int i = 0;
    while (i < maxlen - 1) {
        char c;
        ssize_t r = read(0, &c, 1);
        if (r <= 0 || c == '\n') break;
        buf[i++] = c;
    }
    buf[i] = '\0';
    return i;
}

/* Read exactly n bytes from stdin into buf (or discard them if buf is NULL).
   Discarding still consumes the bytes: leaving them in the pipe would
   desynchronise every subsequent command. */
static int read_bytes(uint8_t *buf, size_t n) {
    uint8_t sink[4096];
    size_t got = 0;
    while (got < n) {
        size_t want = n - got;
        ssize_t r;
        if (buf) {
            r = read(0, buf + got, want);
        } else {
            r = read(0, sink, want < sizeof(sink) ? want : sizeof(sink));
        }
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) return 0;
        got += (size_t)r;
    }
    return 1;
}

/* Write exactly n bytes to stdout */
static void write_bytes(const uint8_t *buf, size_t n) {
    size_t written = 0;
    while (written < n) {
        ssize_t w = write(1, buf + written, n - written);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) break;
        written += w;
    }
}

static char target_path_global[256] = {0};
static char input_path[256] = {0};
static int input_fd = -1;
static double timeout_seconds = 5.0;

/* alarm() takes whole seconds, so it cannot express the sub-second timeouts a
 * fast in-process target wants (a p99 exec of ~29ms suggests ~0.04s). Every
 * alarm() site below goes through this instead. Values are clamped to a 1ms
 * floor: setitimer with an all-zero it_value disarms the timer rather than
 * firing immediately, which would silently mean "no timeout at all". */
static void arm_timeout(double seconds) {
    struct itimerval tv;
    if (!(seconds > 0.0)) seconds = 5.0;
    if (seconds < 0.001) seconds = 0.001;
    tv.it_value.tv_sec = (time_t)seconds;
    tv.it_value.tv_usec = (suseconds_t)((seconds - (double)(time_t)seconds) * 1e6);
    tv.it_interval.tv_sec = 0;
    tv.it_interval.tv_usec = 0;
    setitimer(ITIMER_REAL, &tv, NULL);
}

static void disarm_timeout(void) {
    struct itimerval tv;
    tv.it_value.tv_sec = 0;
    tv.it_value.tv_usec = 0;
    tv.it_interval.tv_sec = 0;
    tv.it_interval.tv_usec = 0;
    setitimer(ITIMER_REAL, &tv, NULL);
}

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
    timed_out = 1;
    if (exec_child_pid > 0) {
        kill(exec_child_pid, SIGKILL);
    }
}

/* Stage *data* into the shared input file, rewound and truncated so the
   child sees exactly this input. Returns 0 on failure. */
/* Point the calling process's stdout at /dev/null.  Called in every child
   between fork() and exec().

   stdout is the loader's half of the RUN/RC protocol.  A target that prints
   anything at all inherits that fd and its output lands in the reply stream,
   where the adapter parses the target's own text as an "RC <rc> <err_len>"
   header, fails to match, and returns -2 for a run that in fact succeeded.
   The desync is silent and permanent for the rest of the session.

   /dev/null rather than the stderr pipe: ExecutionRunner.is_crash scans
   stderr for sanitizer reports, and folding stdout in would let ordinary
   target chatter be read as a crash. */
static void redirect_stdout_to_null(void) {
    int devnull = open("/dev/null", O_WRONLY);
    if (devnull < 0) return;
    dup2(devnull, STDOUT_FILENO);
    if (devnull != STDOUT_FILENO) close(devnull);
}

static int stage_input(const uint8_t *data, size_t len) {
    if (input_fd < 0) return 0;
    if (lseek(input_fd, 0, SEEK_SET) < 0) return 0;
    size_t written = 0;
    while (written < len) {
        ssize_t w = write(input_fd, data + written, len - written);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) return 0;
        written += (size_t)w;
    }
    if (ftruncate(input_fd, (off_t)len) < 0) return 0;
    return lseek(input_fd, 0, SEEK_SET) == 0;
}

/* Drain *fd* to EOF, keeping at most maxlen bytes. Reading to EOF before
   waitpid() matters: a child that writes more than the pipe buffer holds
   (an ASAN report runs to several KB) would otherwise block on write()
   while we block on waitpid(), and only the timeout would break the tie. */
static int drain_stderr(int fd, uint8_t *buf, int maxlen) {
    uint8_t sink[4096];
    int got = 0;
    while (1) {
        ssize_t r;
        if (got < maxlen) {
            r = read(fd, buf + got, (size_t)(maxlen - got));
        } else {
            r = read(fd, sink, sizeof(sink));
        }
        if (r < 0 && errno == EINTR) continue;  /* SIGALRM — keep draining */
        if (r <= 0) break;
        if (got < maxlen) got += (int)r;
    }
    return got;
}

/* ── Forkserver client ────────────────────────────────────────────────
 *
 * Mirrors afl_shim.c's __afl_start_forkserver(). The target is exec'd once
 * with the control pipes on AFL's fd numbers; from then on each input is a
 * 4-byte request and two 4-byte replies, with the fork happening inside the
 * already-initialised target.                                              */

#define AFL_FORKSRV_FD 198

static pid_t forksrv_pid = -1;
static int fsrv_ctl_fd = -1;   /* loader -> target: run one */
static int fsrv_st_fd = -1;    /* target -> loader: pid, then status */
static int fsrv_err_fd = -1;   /* target's stderr, held open for the session */
static int use_forksrv = 0;

/* Read exactly n bytes from a fd; 0 on short read/EOF. */
static int read_full(int fd, void *buf, size_t n) {
    size_t got = 0;
    while (got < n) {
        ssize_t r = read(fd, (uint8_t *)buf + got, n - got);
        if (r < 0 && errno == EINTR) {
            if (timed_out) return 0;
            continue;
        }
        if (r <= 0) return 0;
        got += (size_t)r;
    }
    return 1;
}

/* Drain whatever the target has written to its stderr pipe since the last
   call. The pipe outlives every child, so EOF never arrives and the
   read-to-EOF trick used by the fork+exec path does not apply; the child is
   already reaped by the time we get here, so everything it wrote is sitting
   in the buffer and a non-blocking drain is exact. */
static int drain_stderr_nonblock(uint8_t *buf, int maxlen) {
    uint8_t sink[4096];
    int got = 0;
    while (1) {
        ssize_t r;
        if (got < maxlen) {
            r = read(fsrv_err_fd, buf + got, (size_t)(maxlen - got));
        } else {
            r = read(fsrv_err_fd, sink, sizeof(sink));
        }
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) break;   /* EAGAIN on an empty non-blocking pipe */
        if (got < maxlen) got += (int)r;
    }
    return got;
}

/* Exec the target once and complete the forkserver handshake.
   Returns 1 if the target speaks the protocol, 0 to fall back. */
static int start_forkserver(void) {
    int ctl[2], st[2], errp[2];
    if (pipe(ctl) < 0) return 0;
    if (pipe(st) < 0) { close(ctl[0]); close(ctl[1]); return 0; }
    if (pipe(errp) < 0) {
        close(ctl[0]); close(ctl[1]); close(st[0]); close(st[1]);
        return 0;
    }

    pid_t pid = fork();
    if (pid < 0) {
        close(ctl[0]); close(ctl[1]); close(st[0]); close(st[1]);
        close(errp[0]); close(errp[1]);
        return 0;
    }

    if (pid == 0) {
        /* Opt in to the shim's forkserver loop.  This is set *here*, in the
           child, and nowhere else: afl_shim.c gates __afl_start_forkserver()
           on this variable, and every other place the shim gets loaded must
           not see it.  Putting it in the loader's own environment would make
           the dlopen() path below enter the forkserver loop and hang the
           loader, and would do the same to every run_executable() child. */
        setenv("__AFL_FORKSRV", "1", 1);
        dup2(ctl[0], AFL_FORKSRV_FD);
        dup2(st[1], AFL_FORKSRV_FD + 1);
        close(ctl[0]); close(ctl[1]);
        close(st[0]); close(st[1]);
        close(errp[0]);
        dup2(input_fd, STDIN_FILENO);
        dup2(errp[1], STDERR_FILENO);
        close(errp[1]);
        redirect_stdout_to_null();
        signal(SIGCHLD, SIG_DFL);
        signal(SIGALRM, SIG_DFL);
        execl(target_path_global, target_path_global, input_path, (char *)NULL);
        _exit(127);
    }

    close(ctl[0]); close(st[1]); close(errp[1]);

    /* Bound the handshake: a target without the forkserver runs to
       completion instead of answering, and we must not hang on it. */
    struct sigaction sa;
    sa.sa_handler = exec_alarm_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGALRM, &sa, NULL);
    timed_out = 0;
    exec_child_pid = -1;   /* do not let the handler kill the target here */
    arm_timeout(timeout_seconds);

    char hello[4];
    int ok = read_full(st[0], hello, 4);

    disarm_timeout();
    signal(SIGALRM, SIG_DFL);

    if (!ok) {
        kill(pid, SIGKILL);
        waitpid(pid, NULL, 0);
        close(ctl[1]); close(st[0]); close(errp[0]);
        return 0;
    }

    forksrv_pid = pid;
    fsrv_ctl_fd = ctl[1];
    fsrv_st_fd = st[0];
    fsrv_err_fd = errp[0];
    fcntl(fsrv_err_fd, F_SETFL, O_NONBLOCK);
    return 1;
}

static int run_forkserver(const uint8_t *data, size_t len, uint8_t *err, int *err_len) {
    *err_len = 0;
    if (!stage_input(data, len)) return -2;

    char cmd[4] = {0, 0, 0, 0};
    if (write(fsrv_ctl_fd, cmd, 4) != 4) return -2;

    pid_t child = -1;
    if (!read_full(fsrv_st_fd, &child, 4)) return -2;

    struct sigaction sa;
    sa.sa_handler = exec_alarm_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGALRM, &sa, NULL);
    timed_out = 0;
    exec_child_pid = child;   /* handler SIGKILLs this run's child only */
    arm_timeout(timeout_seconds);

    int status = 0;
    int ok = read_full(fsrv_st_fd, &status, 4);

    if (timed_out) {
        /* The handler killed the child; the forkserver still owes us the
           status of the corpse, and skipping it would desync the pipe. */
        if (!ok) ok = read_full(fsrv_st_fd, &status, 4);
    }

    disarm_timeout();
    signal(SIGALRM, SIG_DFL);
    exec_child_pid = -1;

    *err_len = drain_stderr_nonblock(err, MAX_ERR);

    if (timed_out) return -1;
    if (!ok) return -2;
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return -WTERMSIG(status);
    return -2;
}

static int run_executable(const uint8_t *data, size_t len, uint8_t *err, int *err_len) {
    *err_len = 0;
    if (!stage_input(data, len)) return -2;

    int errfd[2];
    if (pipe(errfd) < 0) return -2;

    pid_t pid = fork();
    if (pid < 0) { close(errfd[0]); close(errfd[1]); return -2; }

    if (pid == 0) {
        close(errfd[0]);
        dup2(input_fd, STDIN_FILENO);
        dup2(errfd[1], STDERR_FILENO);
        close(errfd[1]);
        redirect_stdout_to_null();
        signal(SIGCHLD, SIG_DFL);
        signal(SIGALRM, SIG_DFL);
        /* argv[1] = input path, stdin = same file: matches run_target_fast so
           targets keep whichever of the two they already read. */
        execl(target_path_global, target_path_global, input_path, (char *)NULL);
        _exit(127);
    }

    close(errfd[1]);

    exec_child_pid = pid;
    timed_out = 0;
    struct sigaction sa;
    sa.sa_handler = exec_alarm_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;  /* NO SA_RESTART — we want read()/waitpid() interrupted */
    sigaction(SIGALRM, &sa, NULL);
    arm_timeout(timeout_seconds);

    *err_len = drain_stderr(errfd[0], err, MAX_ERR);
    close(errfd[0]);

    int status = 0;
    int waited;
    do {
        waited = waitpid(pid, &status, 0);
    } while (waited < 0 && errno == EINTR && !timed_out);

    disarm_timeout();
    signal(SIGALRM, SIG_DFL);
    exec_child_pid = -1;

    if (timed_out) {
        /* The handler SIGKILL'd the child; reap it and report the timeout as
           -1. Reporting -SIGKILL instead would land in the crash codes and
           file every slow input as a fatal signal. */
        kill(pid, SIGKILL);
        waitpid(pid, NULL, 0);
        return -1;
    }
    if (waited < 0) return -2;
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return -WTERMSIG(status);
    return -2;
}

int main(void) {
    char line[512];
    uint8_t *data = NULL;
    size_t data_cap = 0;
    uint8_t err[MAX_ERR];

    /* Disable stdio buffering on stdin — fork() inherits the buffer and
       _exit() in the child discards the copy, losing read-ahead data. */
    setvbuf(stdin, NULL, _IONBF, 0);

    if (!read_line(line, sizeof(line))) return 1;
    char func_name[256];
    char timeout_str[16] = "5";
    sscanf(line, "INIT %255s %255s %255s %15s", target_path_global, func_name, input_path,
           timeout_str);
    /* strtod, not atoi: the sender may pass a fractional timeout, and atoi
     * truncated anything under one second to 0 -- which this fallback then
     * turned into 5s. A requested 0.04s silently became 5s, 125x larger and
     * the opposite of what tightening the timeout is for. */
    timeout_seconds = strtod(timeout_str, NULL);
    if (!(timeout_seconds > 0.0)) timeout_seconds = 5.0;

    /* Detect shared libraries by extension, not execute permission.
       .so/.dylib files often have +x set but must be loaded via dlopen,
       not exec. Only truly standalone ELF binaries use the exec path. */
    {
        size_t tlen = strlen(target_path_global);
        is_executable = 1;  /* default: assume executable */
        if (tlen >= 3 && strcasecmp(target_path_global + tlen - 3, ".so") == 0)
            is_executable = 0;
        else if (tlen >= 6 && strcasecmp(target_path_global + tlen - 6, ".dylib") == 0)
            is_executable = 0;
        else if (tlen >= 4 && strcasecmp(target_path_global + tlen - 4, ".dll") == 0)
            is_executable = 0;
    }

    if (is_executable) {
        input_fd = open(input_path, O_RDWR | O_CREAT | O_TRUNC, 0600);
        if (input_fd < 0) {
            fprintf(stderr, "open input file failed: %s\n", input_path);
            return 1;
        }
        use_forksrv = start_forkserver();
        if (!use_forksrv)
            fprintf(stderr, "forkserver handshake failed, using fork+exec\n");
    } else {
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

    /* The mode is reported because it is otherwise externally invisible:
       forkserver and fork+exec produce byte-identical results over this
       protocol, so a silently-failing handshake degrades throughput with
       nothing observable to assert on.  The suffix is optional by protocol
       -- older callers compare the line against "READY" and still match on
       the first token. */
    printf("READY %s\n", is_executable ? (use_forksrv ? "forkserver" : "exec") : "dlopen");
    fflush(stdout);

    while (1) {
        if (!read_line(line, sizeof(line))) break;
        if (strcmp(line, "QUIT") == 0) break;
        if (strncmp(line, "RUN ", 4) != 0) continue;

        long data_len = atol(line + 4);
        /* A zero-length input is a legitimate test case, and every RUN must
           produce exactly one reply regardless: `continue` here left the
           caller blocking on a reply that never came, and it tore the
           loader down and restarted it on the join timeout. */
        if (data_len < 0) {
            printf("RC -2 0\n");
            fflush(stdout);
            continue;
        }
        if (data_len > MAX_DATA) {
            /* Drain the payload anyway, then answer: skipping it silently
               would leave the body in the pipe and desync every later RUN. */
            read_bytes(NULL, (size_t)data_len);
            printf("RC -2 0\n");
            fflush(stdout);
            continue;
        }
        if (data_len > 0 && (size_t)data_len > data_cap) {
            uint8_t *grown = realloc(data, (size_t)data_len);
            if (!grown) {
                read_bytes(NULL, (size_t)data_len);
                printf("RC -2 0\n");
                fflush(stdout);
                continue;
            }
            data = grown;
            data_cap = (size_t)data_len;
        }
        if (data_len > 0 && !read_bytes(data, (size_t)data_len)) break;

        int rc = -2;
        int err_len = 0;

        if (is_executable) {
            rc = use_forksrv ? run_forkserver(data, (size_t)data_len, err, &err_len)
                             : run_executable(data, (size_t)data_len, err, &err_len);
        } else if (target_fn) {
            /* Direct call with sigsetjmp timeout — no fork overhead.
               NOTE: SIGSEGV in the target kills fuzz_loader. For crash
               isolation on .so targets, use the persistent_loader adapter
               (which forks per-call) or compile with ASAN. The target's
               stderr is the loader's own stderr here, which the adapter
               drains separately — nothing to report back inline. */
            struct sigaction sa_new, sa_old;
            sa_new.sa_handler = timeout_handler;
            sigemptyset(&sa_new.sa_mask);
            sa_new.sa_flags = 0;
            sigaction(SIGALRM, &sa_new, &sa_old);

            timed_out = 0;
            if (sigsetjmp(timeout_jmp, 1) == 0) {
                arm_timeout(timeout_seconds);
                rc = target_fn(data, (size_t)data_len);
                disarm_timeout();
            } else {
                /* Longjmp from timeout_handler */
                rc = -1;
            }

            sigaction(SIGALRM, &sa_old, NULL);
        }

        printf("RC %d %d\n", rc, err_len);
        fflush(stdout);
        if (err_len > 0) write_bytes(err, (size_t)err_len);
    }

    free(data);
    if (use_forksrv) {
        /* Closing the control pipe makes the forkserver's read() return 0
           and the target exit on its own. */
        if (fsrv_ctl_fd >= 0) close(fsrv_ctl_fd);
        if (forksrv_pid > 0) {
            kill(forksrv_pid, SIGKILL);
            waitpid(forksrv_pid, NULL, 0);
        }
        if (fsrv_st_fd >= 0) close(fsrv_st_fd);
        if (fsrv_err_fd >= 0) close(fsrv_err_fd);
    }
    if (input_fd >= 0) close(input_fd);
    return 0;
}
