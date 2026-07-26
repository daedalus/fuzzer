/*
 * perf_shim.c — Lightweight perf_event_open wrapper for fuzzer-tool.
 *
 * Provides C functions that can be called via ctypes from Python to open,
 * read, and close hardware performance counters on target PIDs. This is
 * useful for subprocess mode where the Python process needs to attach
 * perf counters to a child process that's already running.
 *
 * Usage via ctypes:
 *   perf_open(pid, config, exclude_kernel, inherit) -> fd
 *   perf_read(fd) -> uint64 value
 *   perf_close(fd) -> 0
 *
 * Compile:
 *   gcc -O2 -shared -fPIC -o perf_shim.so perf_shim.c
 *
 * Based on honggfuzz linux/perf.c perf_event_open usage.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <linux/perf_event.h>

/* perf_event_open syscall wrapper */
static long do_perf_event_open(
    struct perf_event_attr *hw_event, pid_t pid,
    int cpu, int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, hw_event, (uintptr_t)pid,
                   cpu, group_fd, flags);
}

/*
 * Open a hardware performance counter.
 *
 * Args:
 *   pid             — target PID (use -1 for current process)
 *   hw_config       — perf_hw_config value (e.g. PERF_COUNT_HW_INSTRUCTIONS=1)
 *   exclude_kernel  — 1 to exclude kernel events, 0 to include
 *   inherit         — 1 to inherit to child threads/processes
 *
 * Returns: file descriptor (>= 0) on success, -1 on failure.
 */
int perf_open(int pid, int hw_config, int exclude_kernel, int inherit) {
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(pe));
    pe.size = sizeof(pe);
    pe.type = PERF_TYPE_HARDWARE;
    pe.config = hw_config;
    pe.disabled = 1;
    pe.enable_on_exec = 1;
    pe.exclude_hv = 1;
    if (exclude_kernel) pe.exclude_kernel = 1;
    if (inherit)         pe.inherit = 1;

    int fd = do_perf_event_open(&pe, pid, -1, -1, 0 /* PERF_FLAG_FD_CLOEXEC */);
    return fd;
}

/*
 * Read the current counter value.
 *
 * Args:
 *   fd — file descriptor from perf_open()
 *
 * Returns: 64-bit counter value, or 0 on error.
 */
uint64_t perf_read(int fd) {
    uint64_t val = 0;
    if (fd < 0) return 0;
    ssize_t n = read(fd, &val, sizeof(val));
    if (n != sizeof(val)) return 0;
    return val;
}

/*
 * Reset the counter to zero via ioctl PERF_EVENT_IOC_RESET.
 *
 * Args:
 *   fd — file descriptor from perf_open()
 *
 * Returns: 0 on success, -1 on error.
 */
int perf_reset(int fd) {
    if (fd < 0) return -1;
    return ioctl(fd, PERF_EVENT_IOC_RESET, 0);
}

/*
 * Close a perf counter file descriptor.
 *
 * Args:
 *   fd — file descriptor from perf_open()
 */
void perf_close(int fd) {
    if (fd >= 0) close(fd);
}
