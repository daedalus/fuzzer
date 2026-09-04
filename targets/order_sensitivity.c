/* Order-sensitivity target with per-region ground truth.
 *
 * Companion to tests/test_region_order_attribution.py. Built with gcc and
 * manual guards for the same reason tools/gen_synthetic_target.py is: gcc has
 * no -fsanitize-coverage=trace-pc-guard, so the blocks call the shim's
 * callback themselves. Keep -O1 or lower or the block bodies merge.
 *
 *   gcc -O1 -D__AFL_CTX_SENSITIVE=0 \
 *       -include src/fuzzer_tool/adapters/afl_shim.c \
 *       -o order_sensitivity targets/order_sensitivity.c
 * 16384 bytes = 32 records x 512 B; the region profiler cuts 4x4096, so 8 records/region.
 *   region 0 [0,4096)      HOT-LIVE   : pairwise compares of adjacent records
 *   region 1 [4096,8192)   ORDER-DEAD : commutative sum
 *   region 2 [8192,12288)  ORDER-DEAD : commutative xor
 *   region 3 [12288,16384) COLD-LIVE  : fires a distinct edge only when the record
 *                                       with the max first byte sits at local slot 0.
 *                                       The parent puts it at slot 3, so this is
 *                                       unreachable in fewer than 3 adjacent swaps.
 */
#include <stdint.h>
#include <stdio.h>
#define REC 512
#define N_BLOCKS 128
static volatile uint64_t sink;
extern void __sanitizer_cov_trace_pc_guard(uint32_t *guard);
extern void __sanitizer_cov_trace_pc_guard_init(uint32_t *start, uint32_t *stop);
static uint32_t g[N_BLOCKS + 1];
__attribute__((constructor)) static void gi(void){ __sanitizer_cov_trace_pc_guard_init(g, g+N_BLOCKS+1); }
#define MARK(i) __sanitizer_cov_trace_pc_guard(&g[(i) % N_BLOCKS])
#define B(r) (d[(size_t)(r) * REC])

static void run(const uint8_t *d, size_t n) {
    if (n < (size_t)REC * 32) return;
    for (int i = 0; i < 7; i++) {                       /* region 0: hot-live */
        if (B(i) < B(i+1)) { MARK(2*i);   sink += B(i); }
        else               { MARK(2*i+1); sink += B(i+1); }
    }
    uint32_t s = 0;                                      /* region 1: order-dead */
    for (int i = 8;  i < 16; i++) s += B(i);
    MARK(40 + (s % 16));
    uint32_t x = 0;                                      /* region 2: order-dead */
    for (int i = 16; i < 24; i++) x ^= B(i);
    MARK(70 + (x % 16));
    int am = 0; uint8_t mx = B(24);                      /* region 3: cold-live */
    for (int i = 1; i < 8; i++) { uint8_t v = B(24+i); if (v > mx) { mx = v; am = i; } }
    if (am == 0) MARK(120); else MARK(121);
}
int main(int argc, char **argv){
    if (argc < 2) return 1;
    FILE *f = fopen(argv[1], "rb"); if(!f) return 1;
    static uint8_t buf[1<<20]; size_t n = fread(buf,1,sizeof buf,f); fclose(f);
    run(buf, n); return 0;
}
