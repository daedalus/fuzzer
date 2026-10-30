# Handover: coverage SHM on hugepages, A/B

**Date:** 2026-10-28
**Repo state:** `master` @ `749f859`. No code changed.
**Verdict:** no measurable gain. Not shipped.

## 1. Question

Does backing the coverage SHM with 2 MiB hugepages (`SHM_HUGETLB`) speed up fuzzing?

## 2. Setup

- Host: 4-core Xeon VM @ 2.8 GHz, clang 18, root.
- Target: `fuzzgoat_read_asan.so` (clang, ASAN, `afl_shim.c`), in-process `direct_lite`, auto-sized map 32,768 entries (256 KiB).
- Hugepages: `echo 64 > /proc/sys/vm/nr_hugepages`. `/proc/<pid>/smaps` showed `KernelPageSize: 2048 kB` on every `SYSV` mapping. Restored to 0 after.
- Arm B hook (out of tree, `PYTHONPATH` `sitecustomize.py`, gated on `HP_AB=1`): replaces `adapters/libc_shm.shmget` with `shmget(key, size, IPC_CREAT | SHM_HUGETLB | perms)` (`SHM_HUGETLB = 0o4000`). Every segment rounds up to 2 MiB.
- End-to-end: 10 pairs, ABBA order, `taskset -c 2`, `-s <rep> --clock virtual --max-execs 5000 --no-save-state`, fresh copy of fuzzgoat `in/` per run. Metric: wall time.
- Isolated: C driver `dlopen`s `fuzzgoat_read_noasan.so`, attaches its own segment (with/without `SHM_HUGETLB`), replays 251 corpus inputs x 200 reps. Per exec: bump generation (Python reset), `fuzz_shm_run` (timed), scalar scan of the id column (timed). 6 interleaved reps per arm and map size.

## 3. Results

| Test | 4K | 2M | B - A (95% CI) |
|---|---:|---:|---|
| End-to-end wall, 5k execs | 27.85 s | 27.87 s | +0.08% ±1.6%; B faster 4/10 |
| Target/exec, 32k map | 1360 ns | 1374 ns | +1.0% ±4.6% |
| Scan/exec, 32k map | 11.1 us | 11.2 us | +0.8% ±5.5% |
| Target/exec, 262k map (2 MiB) | 3156 ns | 2956 ns | -6.3% ±5.4% |
| Scan/exec, 262k map | 102.6 us | 101.3 us | -1.3% ±2.5% |

Control (A vs A, Hard Rule 46): -1.0% to -3.7% mean drift, same size as every effect. 9/10 end-to-end pairs replayed identically (same corpus and edge counts); pair 7 diverged.

## 4. Why

- Target time is ~0.15 ms of a 1.5-5 ms cycle (`tgt` vs `cyc` in the status line). Python fuzzer-side work dominates.
- 256 KiB = 64 pages; the dTLB/STLB covers it.
- The only borderline effect (-200 ns target-side at a 2 MiB map) is < 0.1% of a cycle.

## 5. Costs if shipped

- 2 MiB per segment (~10 segments per run, ~20 MiB pinned).
- Needs a hugepage reservation (`vm.nr_hugepages`, root to provision) or `shmget` fails: fallback path required. Non-root use of a reserved pool needs `vm.hugetlb_shm_group` (or `CAP_IPC_LOCK`).
- Shmem THP (`madvise`) cannot back segments < 2 MiB.

## 6. Not done

- Fork-server targets: each child may fault ~64 4K pages on first touch vs 1 hugepage. Not measured.
- FFmpeg-sized maps (2^19 entries, 4 MiB) end-to-end.
