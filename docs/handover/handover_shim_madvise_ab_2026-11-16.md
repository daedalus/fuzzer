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

## 5. Not done

- Populate on a large non-persistent target with dense per-exec coverage (ffmpeg, 2^19-entry map). Synthetic data predicts up to ~160 us/fork saved there, and a loss on early-rejected inputs; it would need a gate (e.g. populate only when the previous exec's touched pages exceed ~1/4 of the table).
- Private-heap THP on a heap-heavy target (ffmpeg noasan direct_lite).
