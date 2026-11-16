"""Toy target shared by the mutant-kill tests (adapters/mutant_build, services/mutant_kill)."""

# Toy library: the mutated source. 'C' crashes, 'H' hangs, 'P' prints to
# stdout; the malloc(4) boundary is the kill target; `dead` is a dead store
# (equivalent once -O2 runs).
TOY_LIB = """\
#include <stdio.h>
#include <stdlib.h>
int toy_run(const unsigned char *b, size_t n)
{
    int dead = (int)n + 7;
    char *p = malloc(4);
    if (!p) return 0;
    if (n < 4) p[n] = 1; else p[0] = 2;
    int r = p[0];  /* p observable: -O2 must not drop the allocation */
    free(p);
    if (n && b[0] == 'C') { volatile int *z = 0; *z = 1; }
    if (n && b[0] == 'H') { volatile int spin = 1; while (spin) {} }
    if (n && b[0] == 'P') puts("noise noise noise");
    return r == 9;
}
"""

# Wrapper in the targets/*_read.c shape: fuzz_shm_run + its own main + a
# shim call that the driver's no-op stub must satisfy.
TOY_WRAP = """\
#include <stddef.h>
extern void __afl_map_edge(unsigned int cur_loc);
int toy_run(const unsigned char *b, size_t n);
int fuzz_shm_run(const unsigned char *buf, size_t size)
{
    __afl_map_edge(1);
    return toy_run(buf, size);
}
int main(void) { return 0; }
"""
