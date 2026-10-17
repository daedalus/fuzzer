/* Link stub for --frontier coverage builds (tools/frontier.py).
 *
 * Replaces afl_shim.c: no SHM, no edges. Targets that crash by design
 * (fuzzgoat) would lose their profile on a fatal signal, so dump it first. */
#include <signal.h>
#include <unistd.h>

int __llvm_profile_write_file(void);

void __afl_map_edge(unsigned int cur_loc) { (void)cur_loc; }

static void dump_profile(int sig)
{
    __llvm_profile_write_file();
    _exit(128 + sig);
}

__attribute__((constructor)) static void arm_dump(void)
{
    static const int sigs[] = {SIGSEGV, SIGABRT, SIGBUS, SIGFPE, SIGILL};
    for (unsigned i = 0; i < sizeof sigs / sizeof *sigs; i++)
        signal(sigs[i], dump_profile);
}
