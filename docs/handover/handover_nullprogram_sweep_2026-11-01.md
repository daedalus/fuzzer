# Handover: what ports from nullprogram.com (Chris Wellons) to daedalus/fuzzer

**Date:** 2026-11-01
**Repo state:** `master` @ `2d5faaaf` (fresh clone, so `git pull` was a no-op).
**Scope:** Analysis only. No production code changed, nothing built, nothing measured.

---

## 1. Question

What can be ported from https://nullprogram.com/index/ and the older posts it
leads to?

## 2. Method and coverage (read this first)

- `/index/` returned only the ten 2026 posts. Older posts were found by web
  search, so coverage is partial and biased toward what search surfaced.
- **Read in full:** Prospero, tugz, ssg, w64devkit, Stars!VM, concurrent MSI
  (all 2026, as index excerpts only, not the full articles); MSI (2022-08-08,
  search excerpts); Fuzzing tips for AFL++ (2025-02-05); The Day I Fell in Love
  with Fuzzing (2019-01-25).
- **Search excerpts only:** Hash Prospector (2018-07-31), LCG with Emacs Calc
  (2019-11-19), Two Sum / MSI (2023-06-26), quick hash tables (2025-01-19),
  freestyle linked lists (2025-12-31), pointer hashing (2016-05-30).
- **Not read:** the checklist posts of 2023-01-08 and 2024-12-20, the ASan
  defaults post (2022-06-26), the arena allocation post (2023-09-27).
- The tugz, Prospero and Stars!VM repositories were **not** fetched or built.

## 3. Findings, ranked

### F1. Targets pass the AFL buffer straight into the code under test (HIGH, cheap)

Source: AFL++ tips, tip (5).

`targets/fuzzgoat_read.c` and `targets/gzip_read.c` call
`fuzz_shm_run(buf, len)` / `fuzz_gzip(buf, len)` on
`__AFL_FUZZ_TEST_CASE_BUF` directly. That buffer is larger than `len`, so
ASan cannot flag reads past the end of the input. Fix: copy into a
`realloc`'d buffer of exactly `len` bytes per iteration.

Not yet checked: the other persistent targets (`ffmpeg_read.c`,
`ffmpeg_canary_read.c`, `fgrep_read.c`, `grep_read.c`, `jpeg_read.c`,
`png_read.c`) use the same `__AFL_LOOP(1000)` shape; the same pattern was
seen only in the first two. Suggested first measurement: `fuzzgoat_read`
under ASan, since its heap overread is already documented.

### F2. ASan options lack fill and use-after-return settings (MEDIUM)

Source: AFL++ tips, tip (1).

A grep of `src/`, `tools/` and `targets/` found no
`max_malloc_fill_size` and no `detect_stack_use_after_return`. The options
the fuzzer does set (`halt_on_error=0`, `abort_on_error=0`,
`suppress_equal_pcs=0`, in `fuzzer.py` near line 7783) exist for the
in-process recover mode (see `docs/ASAN-LIMITATION.md`, Layer 3). Raising the
fill size could interact with that mode, so test before enabling by default.
The default ASan fill covers only the first 4 KiB of an allocation, so a large
uninitialized read can go undetected.

### F3. No state isolation between persistent-mode iterations (MEDIUM, unaudited)

Source: tips (4) and (6).

Targets loop with `__AFL_LOOP(1000)`. No per-iteration global reset or
pattern fill was seen. Which globals each target leaks has **not** been audited.

### F4. tugz as a differential target (HIGH value, unverified)

Source: tugz post (2026-10-13), https://github.com/skeeto/tugz (not fetched).

A week-old deflate implementation claimed on par with libdeflate. Existing
fit: `targets/gzip_read.c`, `targets/zlib_read.c`,
`core/mutations/deflate_struct.py`, `gzip.py`, `zlib.py`,
`covering_array_gzip.py`, and differential targets (`bithacks_diff.c`,
`ec_pubkey_diff.c`). Proposed `targets/tugz_diff.c`: inflate each input with
tugz, zlib and libdeflate and flag any disagreement. Open: does tugz build
with clang sancov; is libdeflate available offline.

### F5. MSI probing versus the shim's linear probe (MEDIUM, needs simulation)

Source: MSI post (2022), concurrent MSI post (2026).

Construction (from the excerpts): `step = (hash >> (64 - exp)) | 1`, index
advances by `step` and is masked; the odd step visits every slot once.

The shim (`__afl_map_edge`) probes linearly from `edge_id % map_size` with a
bounded window. Its own comment gives simulated drop rates on ffmpeg at load
0.77: 4.43% at window 8, 1.60% at 16, 0.40% at 32, 0.04% at 64. A stepped probe
avoids primary clustering and should drop less at equal window.

Costs: `__afl_probe_run` vectorizes over contiguous slots (SSE/AVX), and the
Python side maps position to id assuming the linear layout (`ShmCoverage`,
`_scan_with_positions`). Possible hybrid: one contiguous SIMD run, then a
stepped fallback for overflow. Run the existing drop simulation first.

### F6. Concurrent atomic MSI (LOW)

The shim's design rule is one writer per field. `__afl_slot_claim` was seen but
not read, so whether claims are atomic is **unknown**. Relevant only if the
table ever needs multiple writers per slot.

### F7. Corpus as a regression suite (LOW-MEDIUM)

Source: 2019 post. After `afl-cmin`/`afl-tmin` he split the corpus into
accepted and rejected inputs and checked both in, locking parser behavior
across compilers (it exposed a diet libc bug). Minimization is covered here
(`minimize -c`, `set_cover.py`). **Not found:** an export that writes the
minimized corpus split by accepted/rejected. `core/validity.py` and
`core/reject_stats.py` exist but were not read; check them before building.

### F8. Hash functions (LOW)

`lowbias32` is already in `core/hash_inverse.py`. The shim's
`__afl_guard_mix` is the splitmix64 finalizer, so there is no quality gap. A
later 32-bit variant (shifts 16/15/15, multipliers `0x21f0aaad`, `0x735a2d97`,
from hash-prospector issue 19 as quoted in the Boost docs) could be added to
`hash_inverse.py` if more finalizers should be invertible.

### F9. Other targets by the same author (LOW, unverified)

Stars!VM embeds an 80286 emulator and Win16 bridge (parses NE executables).
Prospero is roughly curl plus `popen()` plus JSON. Neither was fetched.

### F10. Small items from the tips article

Tip (7): `memfd_create` plus `/proc/self/fd/N` for path-only targets. No use of
`memfd_create`, `/proc/self/fd` (for input) or `fdmemopen` exists in
`src/` or `targets/`. Tip (8): shrink fixed buffers in targets; no candidates
identified. Tip (3): unity-build includes; `grep_read` already handles a target
that `exit()`s.

## 4. Not applicable

Emacs retirement, code-signing (aas-sign), dcmake, ssg, w64devkit, pointer
hashing, LCG post (possible minor use in `randomness.py`; what it detects
today was not checked).

## 5. Suggested order

1. F1 on `fuzzgoat_read`, compare crash counts under ASan, then roll out.
2. F4 after confirming tugz builds.
3. F2 behind a flag, tested against the recover-mode path.
4. F5 simulation only; implement only if the hybrid beats the 0.40%/32 baseline.
5. F3 audit, F7 check of `validity.py`.
