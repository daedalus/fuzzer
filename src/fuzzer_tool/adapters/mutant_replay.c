/* Fork replay driver for mutant-kill scoring (adapters/mutant_build.py).
 *
 * Linked into the original and every mutant in place of afl_shim.c: these
 * are oracle binaries, not fuzz targets, so no SHM, no edges.
 *
 *   argv: PACK MASK|- TIMEOUT_MS
 *   PACK: (u32 little-endian length, bytes)*
 *   MASK: one byte per record; 1 = eligible (original ran CLEAN there)
 *   stdout: one status byte per record run: 0 CLEAN, 1 CRASH, 2 HANG
 *
 * With a mask, the driver stops after the first CRASH on an eligible
 * record: one kill is enough, the rest of the pack is wasted forks.
 *
 *   parent ──fork──> child: stdio -> /dev/null, itimer, fuzz_shm_run, _exit(0)
 *      └── waitpid: SIGALRM -> HANG, other signal / nonzero exit -> CRASH
 *
 * Exit codes: 0 done, 2 bad arguments or a malformed pack.
 */
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>

enum { ST_CLEAN = 0, ST_CRASH = 1, ST_HANG = 2 };
enum { EXIT_BAD_INPUT = 2 };

/* Mirrors MAX_INPUT_BYTES in mutant_build.py: write_pack never exceeds it. */
#define MAX_RECORD (1u << 16)
#define MS_PER_S 1000

int fuzz_shm_run(const unsigned char *buf, size_t size);

/* The shim's edge hook: wrappers call it; oracles ignore it. */
void __afl_map_edge(unsigned int cur_loc) { (void)cur_loc; }

static void arm_timer(unsigned timeout_ms)
{
    struct itimerval it = {0};
    it.it_value.tv_sec = timeout_ms / MS_PER_S;
    it.it_value.tv_usec = (timeout_ms % MS_PER_S) * MS_PER_S;
    setitimer(ITIMER_REAL, &it, NULL);
}

/* Silence the child: target output must not mix into the status stream. */
static void mute_stdio(void)
{
    int fd = open("/dev/null", O_WRONLY);
    if (fd < 0)
        _exit(EXIT_BAD_INPUT);
    dup2(fd, STDOUT_FILENO);
    dup2(fd, STDERR_FILENO);
    close(fd);
}

static int run_one(const unsigned char *buf, size_t n, unsigned timeout_ms)
{
    pid_t pid = fork();
    if (pid < 0)
        return -1;
    if (pid == 0) {
        mute_stdio();
        arm_timer(timeout_ms);
        fuzz_shm_run(buf, n);
        _exit(0);
    }

    int st;
    while (waitpid(pid, &st, 0) < 0)
        if (errno != EINTR)
            return -1;

    if (WIFSIGNALED(st))
        return WTERMSIG(st) == SIGALRM ? ST_HANG : ST_CRASH;
    return WEXITSTATUS(st) ? ST_CRASH : ST_CLEAN; /* ASAN exits 1 */
}

/* 1 = record read, 0 = clean end of pack, -1 = malformed. */
static int read_record(FILE *f, unsigned char *buf, uint32_t *len)
{
    unsigned char hdr[4];
    size_t got = fread(hdr, 1, sizeof hdr, f);
    if (got == 0)
        return 0;
    if (got != sizeof hdr)
        return -1;

    *len = (uint32_t)hdr[0] | (uint32_t)hdr[1] << 8 | (uint32_t)hdr[2] << 16 | (uint32_t)hdr[3] << 24;
    if (*len > MAX_RECORD)
        return -1;
    return fread(buf, 1, *len, f) == *len ? 1 : -1;
}

static unsigned char *read_mask(const char *path, size_t *n)
{
    *n = 0;
    if (strcmp(path, "-") == 0)
        return NULL;

    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    rewind(f);

    unsigned char *mask = size > 0 ? malloc((size_t)size) : NULL;
    if (mask && fread(mask, 1, (size_t)size, f) == (size_t)size)
        *n = (size_t)size;
    fclose(f);
    return mask;
}

int main(int argc, char **argv)
{
    if (argc != 4) {
        fprintf(stderr, "usage: %s PACK MASK|- TIMEOUT_MS\n", argv[0]);
        return EXIT_BAD_INPUT;
    }

    FILE *pack = fopen(argv[1], "rb");
    if (!pack) {
        fprintf(stderr, "cannot open pack %s\n", argv[1]);
        return EXIT_BAD_INPUT;
    }

    size_t mask_n;
    unsigned char *mask = read_mask(argv[2], &mask_n);
    if (strcmp(argv[2], "-") != 0 && !mask) {
        fprintf(stderr, "cannot read mask %s\n", argv[2]);
        return EXIT_BAD_INPUT;
    }

    unsigned timeout_ms = (unsigned)strtoul(argv[3], NULL, 10);
    static unsigned char buf[MAX_RECORD];
    uint32_t len;
    size_t i = 0;
    int rc;

    while ((rc = read_record(pack, buf, &len)) == 1) {
        int st = run_one(buf, len, timeout_ms);
        if (st < 0) {
            fprintf(stderr, "fork/wait failed at record %zu\n", i);
            return EXIT_BAD_INPUT;
        }
        putchar(st);

        int eligible = mask && i < mask_n && mask[i];
        i++;
        if (eligible && st == ST_CRASH)
            break;
    }

    fflush(stdout);
    free(mask);
    fclose(pack);
    if (rc < 0) {
        fprintf(stderr, "malformed pack at record %zu\n", i);
        return EXIT_BAD_INPUT;
    }
    return 0;
}
