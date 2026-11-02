# Handover: what ports from nullprogram.com (Chris Wellons) to daedalus/fuzzer

**Date:** 2026-11-01
**Repo state:** `master` @ `2d5faaaf` (fresh clone, so `git pull` was a no-op).
**Scope:** Analysis. No production code changed; the only code-side change is a static test (see F1, corrected 2026-11-02).

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

### F1. Exact-size input copy: ALREADY COVERED by the shim (CORRECTED 2026-11-02)

Source: AFL++ tips, tip (5).

**The original F1 was wrong for this repo.** It assumed the stock AFL++ buffer,
where ASan cannot see reads past `len`. In this tree the macros come from
`afl_shim.c` (`__afl_fuzz_len`), which calls
`__asan_poison_memory_region(data + len, cap - len)` on every iteration, so a
read past the input is reported without a copy. Checked with gcc
`-fsanitize=address -include afl_shim.c` on a harness:

- read of `buf[len]`: `use-after-poison` (caught);
- read of `buf[-1]`: `heap-buffer-overflow` (caught, the buffer is a malloc'd
  region with a redzone before it); same result with an exact-size `malloc` copy.

So the copy would add cost and no coverage. In-process `.so` mode is also
exact-size: `inprocess.py` builds `(c_uint8 * len(data)).from_buffer_copy(data)`
(lines 94, 178). The dynamic test is `tests/test_shm_testcase.py::
test_read_past_input_is_reported` (skipped in the container used for this
sweep, so the check above was done by hand).

Per-target status (every file in `targets/`):

- **11 persistent targets use the macros** (`ffmpeg_canary_read`, `ffmpeg_read`,
  `fgrep_read`, `fuzzgoat_read`, `grep_read`, `gzip_read`, `jpeg_read`,
  `png_read`, `tracecmp_target`, `unrar_read`, `zlib_read`): covered.
  `tests/test_targets_testcase_delivery.py` (new) fails if one stops using
  `__AFL_FUZZ_TEST_CASE_BUF/LEN` or reads input inside `__AFL_LOOP`
  (negative control: patching `zlib_read.c` makes 2 tests fail).
- **Non-persistent fixtures with their own `main()` and a fixed buffer**
  (`proto_target`, `test_target`, `rust_target`, `order_sensitivity`): not
  covered, and reading `buf[n]` there is not ASan-visible. They are small
  fixtures, not fuzz targets of real code; left unchanged.
- **`#else` fallback `main()`s** in the persistent targets use
  `unsigned char buf[65536]` (`gzip_read`, `fuzz_shm`, and similar). Dead
  code in shim builds (`__AFL_HAVE_MANUAL_CONTROL` is always defined), live
  only in a plain build without the shim.
- **Targets with only `fuzz_shm_run` and no `main`** (`sqlite_read`,
  `secp256k1_read`, `lz4_read`, `nop_target`, and others) are driven through
  `inprocess.py` or a loader that supplies the buffer: exact-size in-process.
  Not individually verified under ASan, which `docs/ASAN-LIMITATION.md` says
  needs the recover-mode builds.
- Not verified: `tailslayer_read.cpp`, `prng_token_read`, `order_sensitivity`
  entry shapes beyond the grep above.

Remaining real gap: none for persistent targets. The only open item is whether
to make the fixed-buffer fixtures exact-size, which has low value.

### F2. ASan fill size and use-after-return: IMPLEMENTED as opt-in (2026-11-02)

Source: AFL++ tips, tip (1).

Corrections to the original finding: `uninit_probe.py` already sets
`max_malloc_fill_size=1<<28` for its own differential runs (my first grep was
cut off by `head`), and gcc 13 libasan already defaults
`detect_stack_use_after_return` to on. What was missing is a **global** setting
for fuzzing runs, in both places that build ASan options:
`cli/ldpreload_wrapper.py` (`_ASAN_DEFAULTS`) and `Fuzzer._setup_asan_env`.

Measured (gcc 13 libasan, malloc+free per call, default fill 4096 vs full fill):
1 KiB 0.88 -> 0.88 us; 64 KiB 85 -> 325 us; 1 MiB 91 -> 2753 us;
16 MiB 756 -> 8425 us. That cost is why this is **opt-in**, not a default:
targets that allocate large buffers per input (ffmpeg) would slow down.

What it buys (reproduced): `malloc(8192)` never initialized, then `strlen`:
default fill gives `strlen=4096` with no report (fresh pages are zero past
4 KiB); with the full fill ASan reports a heap-buffer-overflow.
Stack-use-after-return: returning `&local` prints `42` with
`detect_stack_use_after_return=0` and is reported with `=1`.

Implementation: `FUZZER_ASAN_STRICT=1` (exactly `1`; same style as
`FUZZER_KEEP_ASLR`) adds `max_malloc_fill_size=268435456` and
`detect_stack_use_after_return=1` after the existing recover-mode options
(`halt_on_error=0`, `abort_on_error=0`, ...), which are unchanged. User-set
keys win in both places. `uninit_probe` appends its own fill after, so it
still wins. Tests: `tests/test_asan_strict_options.py` (12, including a
compiled uninitialized-read program that is skipped without a compiler).

Not done / not verified:
- No CLI flag; env var only. A flag would need plumbing through
  `commands.py` into the wrapper's argv scan.
- gcc 13 has no `-fsanitize-address-use-after-return` compile flag; with
  clang the runtime option also needs that flag at build time (not checked,
  no clang in the sweep container).
- Not run against the real in-process recover-mode path
  (`docs/ASAN-LIMITATION.md`, Layer 3) or any ffmpeg target; only a small
  persistent harness with the shim ran cleanly under the strict options.
- No exec/s measurement on a real target. Do that before ever making it a default.

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

1. F1 withdrawn (see F1); the static guard test is the only change.
2. F4 after confirming tugz builds.
3. F2 done as opt-in; remaining: measure exec/s on `ffmpeg_read_asan.so` and run the recover-mode path with `FUZZER_ASAN_STRICT=1`.
4. F5 simulation only; implement only if the hybrid beats the 0.40%/32 baseline.
5. F3 audit, F7 check of `validity.py`.
