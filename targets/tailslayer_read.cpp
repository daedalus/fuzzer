/* Fuzz target for tailslayer — hedged reader library.
 *
 * Exercises: page allocation (regular), replica address computation,
 * value insertion, and hedged reads with immediate-return signal.
 *
 * Input format:
 *   byte 0:     flags
 *     bit 0:    use_hugepage (1=enabled, may fail without hugepage support)
 *     bits 1-2: reserved
 *   byte 1-2:   channel_offset (16-bit LE, clamped to 256..65536, aligned to 256)
 *   byte 3:     channel_bit (clamped 1..63)
 *   byte 4-5:   num_values (16-bit LE, max 200)
 *   byte 6..N:  values (uint8_t each, N = num_values)
 *   last byte:  read_index target (modulo MAX(num_values, 1))
 *
 * Compile shared library (for inprocess modes):
 *   g++ -O2 -g -shared -fPIC -std=c++17 \
 *       -include src/fuzzer_tool/adapters/afl_shim.c \
 *       -I$HOME/code/tailslayer/include \
 *       -o tailslayer_read.so tailslayer_read.cpp \
 *       -lpthread -Wl,--export-dynamic
 *
 * Compile standalone:
 *   g++ -O2 -g -std=c++17 \
 *       -I$HOME/code/tailslayer/include \
 *       -o tailslayer_read tailslayer_read.cpp -lpthread
 */
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <tailslayer/hedged_reader.hpp>

/* AFL edge coverage — provided by afl_shim.c */
extern void __afl_map_edge(unsigned int cur_loc);

/* ── Fuzz control globals ─────────────────────────────────────────────── */

/* Read index set from fuzz input before each iteration. The signal function
 * (fuzz_signal) returns this immediately — no busy-loop. */
static size_t g_fuzz_read_index = 0;

/* Immediate-return signal: the fuzz harness controls when and which index
 * to read, not a wall-clock delay. */
[[gnu::always_inline]] inline std::size_t fuzz_signal() {
    return g_fuzz_read_index;
}

/* Consume the read value so the compiler cannot elide the load. */
template <typename T>
[[gnu::always_inline]] inline void fuzz_final_work(T val) {
    asm volatile("" :: "r"(val) : "memory");
}

/* ── Helpers ──────────────────────────────────────────────────────────── */

/* Clamp a value to [lo, hi] — all inputs are attacker-controlled. */
template <typename T>
static inline T clamp(T v, T lo, T hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

/* ── Fuzz entry point ─────────────────────────────────────────────────── */

extern "C" {

__attribute__((visibility("default")))
int fuzz_tailslayer(const unsigned char *buf, size_t size) {
    __afl_map_edge(0x1000);

    if (size < 6) { __afl_map_edge(0x1001); return 0; }

    /* ── Parse config ─────────────────────────────────────────────── */

    int use_hugepage  = (buf[0] & 1);  /* parsed but ignored — see below */

    int channel_offset = (int)buf[1] | ((int)buf[2] << 8);
    /* Clamp: must be >= 256, aligned to 256, otherwise the address
     * calculation breaks stride assumptions. */
    if (channel_offset < 256)       channel_offset = 256;
    if (channel_offset > 65536)     channel_offset = 65536;
    channel_offset = (channel_offset / 256) * 256;  /* align */

    int channel_bit = (int)buf[3];
    channel_bit = clamp(channel_bit, 1, 63);

    size_t num_values = (size_t)buf[4] | ((size_t)buf[5] << 8);
    if (num_values > 200) num_values = 200;
    /* Need at least 1 value byte + 1 read-index byte after the 6-byte header */
    if (6 + num_values > size) { __afl_map_edge(0x1002); return 0; }

    /* Signal function reads this index after the workers start. */
    size_t read_idx = buf[size - 1];
    g_fuzz_read_index = read_idx % (num_values > 1 ? num_values : 1);

    __afl_map_edge(0x1003);

    /* ── Configure tailslayer ─────────────────────────────────────── */

    /* Always use regular mmap — hugepages are a deployment concern, not
     * something the fuzzer should spend cycles rediscovering.  The library's
     * destructor always unmaps SUPERPAGE_SIZE bytes, so match it to the
     * non-hugepage allocation of 256 KB (see HedgedReader::setup_memory). */
    tailslayer::set_use_hugepage(false);
    tailslayer::SUPERPAGE_SIZE = 256 * 1024;

    /* Use a small page to avoid exhausting memory in constrained envs. */
    tailslayer::hugepage_size_mb = 2;

    /* ── Create reader ────────────────────────────────────────────── */
    using T = uint8_t;

    __afl_map_edge(0x1004);

    /* Build the reader.  The constructor allocates memory (mmap) and
     * computes replica addresses.  Allocation may fail under hugepage
     * mode if the kernel has no free hugepages — that path is exercised
     * intentionally (the library handles it silently, returning a no-op
     * object whose insert/workers crash harmlessly on nullptr). */
    tailslayer::HedgedReader<T, fuzz_signal, fuzz_final_work<T>> reader{
        channel_offset,
        channel_bit,
        2,  /* num_channels — default, only 2 guaranteed to work */
    };

    __afl_map_edge(0x1005);

    /* ── Insert values ────────────────────────────────────────────── */
    size_t capacity = reader.capacity();
    size_t to_insert = num_values < capacity ? num_values : capacity;

    for (size_t i = 0; i < to_insert; i++) {
        __afl_map_edge(0x1100 + (i & 0xFF));
        reader.insert(static_cast<T>(buf[6 + i]));
    }

    __afl_map_edge(0x1200);

    /* ── Start hedged workers ─────────────────────────────────────── */
    /* Each worker:
     *   1. Pins to its replica core (may fail if core absent — harmless)
     *   2. Calls fuzz_signal() → returns g_fuzz_read_index immediately
     *   3. Reads the value at that index from its replica
     *   4. Calls fuzz_final_work(val) — consumes the value
     */
    reader.start_workers();              /* spawns threads, sleeps 10ms */

    __afl_map_edge(0x1300);

    /* Workers complete on their own (signal returns immediately).
     * Destructor joins threads + munmaps memory. */

    __afl_map_edge(0x1FFF);
    return 0;
}

/* ── Entry points ─────────────────────────────────────────────────────── */

#ifdef __AFL_HAVE_MANUAL_CONTROL
int main(void) {
    __AFL_INIT();
    unsigned char *afl_buf = __AFL_FUZZ_TEST_CASE_BUF;
    while (__AFL_LOOP(1000)) {
        int len = __AFL_FUZZ_TEST_CASE_LEN;
        fuzz_tailslayer(afl_buf, (size_t)len);
    }
    return 0;
}
#else
int main(int argc, char **argv) {
    if (argc == 2) {
        FILE *f = fopen(argv[1], "rb");
        if (!f) return 1;
        fseek(f, 0, SEEK_END);
        long fsize = ftell(f);
        rewind(f);
        unsigned char *fbuf = (unsigned char *)malloc((size_t)fsize);
        if (fbuf) {
            size_t n = fread(fbuf, 1, (size_t)fsize, f);
            int rc = 0;
            if (n > 0) rc = fuzz_tailslayer(fbuf, n);
            free(fbuf);
            fclose(f);
            return rc;
        }
        fclose(f);
        return 1;
    } else {
        unsigned char stdin_buf[65536];
        size_t n = fread(stdin_buf, 1, sizeof(stdin_buf), stdin);
        if (n > 0) return fuzz_tailslayer(stdin_buf, n);
    }
    return 0;
}
#endif

/* Standard in-process entry point for fuzzer-tool .so mode */
__attribute__((visibility("default")))
int fuzz_shm_run(const unsigned char *buf, size_t size) {
    return fuzz_tailslayer(buf, size);
}

} /* extern "C" */
