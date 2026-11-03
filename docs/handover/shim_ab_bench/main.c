/* Guard-path A/B driver. Built with: cc -include <afl_shim.c> main.c (run.sh).
 * Hand-attaches a fake coverage segment, calls __afl_fast_refresh() so the
 * leaf guard fast path is live (without it every guard takes
 * __afl_guard_slow), then times run() x iters. Prints ns per run(). */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
void run(void);
int main(int argc, char **argv) {
  int iters = atoi(argv[1]);
  uint32_t map = 4096;
  uint8_t *g = calloc(1, 32 + (size_t)map * 8 + 16);
  __afl_area = (struct __afl_entry *)(g + 32); __afl_map_size = map;
  __afl_stack_depth = (uint32_t *)g; __afl_gen_word = (uint32_t *)(g + 4);
  __afl_path_hash = (uint64_t *)(g + 8);
  __afl_edge_count = (uint64_t *)(g + 16); __afl_dropped = (uint64_t *)(g + 24);
  __afl_fast_refresh();
  if (getenv("SHOWFAST")) fprintf(stderr, "fast=%d\n", __afl_fast);
  run();
  struct timespec a, b; clock_gettime(CLOCK_MONOTONIC, &a);
  for (int i = 0; i < iters; i++) run();
  clock_gettime(CLOCK_MONOTONIC, &b);
  printf("%.3f\n", ((b.tv_sec - a.tv_sec) * 1e9 + (b.tv_nsec - a.tv_nsec)) / iters);
  return 0;
}
