# Handover: madvise in afl_shim.c, A/B

**Date:** 2026-11-16
**Repo state:** `dd14ab8`. No code changed.
**Verdict:** no flag measurably helps. Not shipped.

## 1. Question

Can `madvise` advice on shim memory speed up fuzzing?

## 2. Flags ruled out without a run

| Flag | Why not |
|---|---|
| `MADV_RANDOM` / `MADV_SEQUENTIAL` | Steer readahead on page-in from file or swap. Coverage SHM is resident shmem: no page-in to steer. |
| `MADV_HUGEPAGE` on coverage SHM | Shmem THP needs segments >= 2 MiB; `SHM_HUGETLB` already null (`handover_shm_hugepages_ab_2026-10-28.md`). |
| `MADV_REMOVE` for the 256-reset table wipe | Table at offset 32 is not page-aligned; refault costs about the memset; saving <= 0.34 us/exec. |
| `MADV_FREE` on freed private-heap slots | Slots are reused LIFO, so RSS is already bounded by peak. Adds a syscall per free and a refault per reuse: cost, no speed. |
| `MADV_DONTDUMP` on the private heap | Core dumps are off here (`ulimit -c 0`; ASAN `disable_coredump=1`). Never-faulted pages are not dumped anyway. |

## 3. Setup

- Host: 4 cores, Linux 6.18, clang 18. THP `enabled=madvise`, `shmem_enabled=never`.
- Arms built from one shim with env-gated experiment lines (`XP_MADV_POP`, `XP_MADV_HUGE`), so A and B run the same binary. Reverted after build.
- End-to-end: 10 pairs, ABBA, `taskset -c 2`, `-s <rep> --clock virtual --no-save-state`, fresh copy of fuzzgoat `in/`. Metric: wall. Exact sign-flip test on per-pair B/A − 1. A/A: same script, both arms without the env var.

## 4. Results

### 4.1 `MADV_POPULATE_WRITE` on the coverage segment in each forkserver child

Mechanism: fork does not copy PTEs of a shared mapping, so each child minor-faults every table page it touches (~1 us/page here). One populate call maps all pages at ~0.25 us/page.

Synthetic (fork + touch E random entries, min of 7, ns/fork):

| Table | E=30 | E=300 | E=3000 |
|---|---|---|---|
| 64 KiB | 87.7k → 80.3k | 88.2k → 84.7k | 90.2k → 84.7k |
| 256 KiB | 95.3k → 94.1k | 128.6k → 93.9k | 135.7k → 96.5k |
| 1 MiB | 102.5k → **136.8k** | 254.5k → 145.0k | 333.1k → 169.9k |

Wins on dense tables, loses on sparse large ones (early-rejected inputs on big targets).

In-run:

| Test | Result | A/A |
|---|---|---|
| fuzzgoat_read_asan, 10 × 3k, wall | B faster 3/10, med +3.3%, p=0.33 | 5/10, med +1.5%, p=0.22 |
| Forkserver only, persistent (fuzzgoat default), 21 rounds × 104 seeds, min | 8K map −2.1..+4.5%; 128K map −2.1% | +2.1..+4.7% |
| Forkserver only, non-persistent | 8K map +1.7%; 128K map −2.1% | −1.7..+5.0% |

Why null: fuzzgoat runs persistent (`__AFL_PERSISTENT=1`), so a child forks once and populate runs once. Non-persistent ASAN forks cost ~5.3 ms; a 35 us saving is < 1%.

### 4.2 `MADV_HUGEPAGE` on the private heap reservation

`fuzzgoat_read_noasan.so`, direct_lite, 10 × 20k. `AnonHugePages` 38 MB with the flag.

| Metric | B vs A | A/A |
|---|---|---|
| Wall | B faster 5/10, med −0.4%, p=0.75 | 5/10, med −0.0%, p=0.51 |

Why null: target time is 0.04 ms of a 0.5 ms cycle. Cost: +38 MB RSS.

### 4.3 FFmpeg 9.0.2, clang + ASAN

`ffmpeg_read_9.0.2_asan` (exe), forkserver, auto-sized map 524,288 entries (4 MiB, 1,024 pages), ~10-14K live edges per exec. Seeds: 11 FATE samples, 14 crafted 8.0.3 seeds, 3 generated (`build_targets.sh`).

| Test | B vs A | A/A |
|---|---|---|
| Forkserver only, persistent, 11 rounds × 28 seeds, min | +6.4% | −3.4% |
| Forkserver only, non-persistent, 31 rounds, paired | B faster 13/31 and 11/31; med +1.5%, +7.9% | 24/31 (p=0.003) and 14/31; med −4.6%, +2.8% |
| Child faults, non-persistent, 5 × 28 execs per arm | 12,001 minflt/exec, sys 13.1 ms | A: 11,906 minflt/exec, sys 13.9 ms |
| End-to-end, 10 pairs × 1k, wall | B faster 6/10, med −3.0%, p=0.60 | 9/10, med −4.3%, p=0.004 |

Why null: a non-persistent ASAN child takes ~12K minor faults per exec; the coverage map is ~1K of them. The rest are copy-on-write faults on the forkserver's heap and ASAN shadow. Populate batches the 1K, saving at most ~0.7 ms of ~13 ms system time per exec. In the default persistent mode it runs once per child.

Both A/A controls fail (p=0.003 forkserver-only, p=0.004 end-to-end): the two-runner harness is biased, and FFmpeg runs diverge within a pair (live-edge counts differ at the same seed). No WIN/LOSS can be read from these designs.

Private-heap `MADV_HUGEPAGE`: not applicable under ASAN. The shim refuses `__AFL_PRIVATE_HEAP` in sanitizer builds, so ASAN's allocator owns the heap.

### 4.4 Per-region flags and their combination, FFmpeg noasan

`ffmpeg_read_9.0.2_noasan.so` clang 18, direct_lite, 524,288-entry map. One process holds harness, target and coverage table, so regions share the TLB. Flags (env-gated, same binary both arms):

| Region | Flag | Mechanism |
|---|---|---|
| Python harness heap | `GLIBC_TUNABLES=glibc.malloc.hugetlb=1` | glibc madvises THP on its heap |
| Shim coverage table | `SHM_HUGETLB` via `libc_shm.shmget` hook (segments >= 2 MiB) | 4 MiB table on 2 MiB pages; 128 pages reserved |
| Target private heap | `MADV_HUGEPAGE` on the reservation (`XP_MADV_HUGE`) | THP for the target's class regions |

Two designs. Fuzz loop: 10 pairs × 3k execs, ABBA, `taskset -c 2`, rusage of the run. Replay: `edge-dump --inprocess-direct` over a fixed 1,706-input corpus, 20 pairs, ABBA; edge sets identical in all runs (deterministic workload). Exact sign-flip test on per-pair B/A − 1.

Fuzz loop:

| Arm | CPU (user+sys) | sys |
|---|---|---|
| A/A | +0.9%, p=0.72 | −2.3%, p=0.17 |
| harness THP | +0.6%, p=0.38 | −13.1%, 10/10, p=0.002 |
| coverage hugetlb | +3.0%, p=0.74 | +5.9%, p=0.25 |
| private-heap THP | +2.7%, p=0.14 | +5.1%, p=0.13 |
| all three | −8.6%, p=0.53 | −21.4%, 10/10, p=0.002 |

Per-pair CPU spreads ±20% in the A/A: FFmpeg runs diverge (live edges differ within a pair), and cheaper paths cost less. CPU cannot resolve a few percent here. A first `all three` run was cut at 5/10 pairs by a container restart and rerun from scratch.

Replay:

| Arm | CPU | user | sys | wall |
|---|---|---|---|---|
| A/A | +0.3%, p=0.30 | +0.3%, p=0.30 | −1.6%, p=0.79 | +0.5%, p=0.30 |
| harness THP | −0.7%, p=0.75 | +0.1%, p=0.48 | −7.9%, p=0.043 | −0.7%, p=0.78 |
| coverage hugetlb | −1.4%, p=0.95 | −1.1%, p=0.80 | −3.7%, p=0.16 | −1.5%, p=0.96 |
| private-heap THP | **−2.2%, 16/20, p=0.020** | **−2.5%, 17/20, p=0.009** | +0.3%, p=0.47 | −2.1%, p=0.013 |
| all three | −1.1%, p=0.47 | +0.1%, p=0.95 | −8.1%, 19/20, p<0.0001 | −0.5%, p=0.49 |

Holm over the 4 arms: private-heap THP p=0.08 (CPU), 0.037 (user). Peak RSS 120 → 122 MB.

Reading:
- Harness THP cuts page-fault time (sys −8% to −13%) but sys is ~8% of CPU; no CPU or wall change on FFmpeg or fuzzgoat (fuzzgoat: CPU +0.15%, p=0.87; sys −7.6%, p=0.16; A/A CPU +0.3%, p=0.65).
- Coverage table on hugepages: null at 4 MiB too; together with the 256 KiB result, closes it.
- Private-heap THP is the only arm with a user-time gain (fewer dTLB misses in target code). No threshold was frozen and it does not survive Holm on CPU: lean, not a win.
- Combining does not add: `all three` keeps the sys saving and loses the user gain. Regions compete for the same STLB; harness THP may displace the target's huge mappings. Unexplained; one run.

## 5. Not done

- Confirmation run for private-heap THP on FFmpeg noasan replay, threshold frozen beforehand (`docs/AB_PENDING.md`), then an in-run check (Hard Rule 58).
- The ~11K non-coverage COW faults per non-persistent FFmpeg ASAN exec are the real fork cost; persistent mode (the default) already avoids them.

## 6. Confirmation of private-heap THP (threshold frozen 2026-11-17, before the run)

- Arms: A = master shim, B = branch shim (`MADV_HUGEPAGE` on the private-heap reservation, unconditional). Both `ffmpeg_read_9.0.2_noasan.so`, clang 18, same FFmpeg archives.
- Primary: replay CPU (user+sys), `edge-dump --inprocess-direct`, same 1,706-input corpus, 20 pairs ABBA, `taskset -c 2`. Exact sign-flip, two-sided, α 0.05. A/A (A vs A) must give p > 0.05 first.
- Ship if B is lower with p < 0.05 and median Δ < 0, and the in-run check (fuzzgoat_read_noasan.so clang, direct_lite, 10 pairs × 20k, master vs branch alternated) is not slower than its A/A spread (Hard Rule 58).

Result:

| Test | B vs A | A/A |
|---|---|---|
| Replay CPU (primary), 20 pairs | B lower 12/20, med −1.3%, p=0.46 | 7/20, med +0.7%, p=0.19 |
| Replay user | 12/20, med −1.5%, p=0.34 | 9/20, med +0.5%, p=0.27 |
| fuzzgoat in-run wall, 10 × 20k | 4/10, med +1.4%, p=0.39 | 6/10, med −2.6%, p=0.34 |

Fails the threshold: #169's −2.2% (p=0.020) did not replicate. Not shipped; the shim change (17ddb773) is reverted. Each build replays deterministically; edge ids differ between the two builds (code layout), total hits within 0.1%.

Verdict on the whole investigation: no `madvise` flag, alone or per region, measurably speeds up fuzzing here.
