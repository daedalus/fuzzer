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

### F5. MSI probing versus the shim's linear probe: IMPLEMENTED AS OPT-IN, NOT ADOPTED (2026-11-03)

Source: MSI post (2022), concurrent MSI post (2026).

Construction: `step = (hash >> (64 - exp)) | 1`; the index advances by `step`,
masked to the table size; the odd step visits every slot once.

Simulation (262144 entries, splitmix64 keys, start `(h32*size)>>32` as in the
shim, window W, 20 seeds, source below). **Control (Hard Rule 46):** the linear
column reproduces the shim's documented ffmpeg figures at load 0.77:
4.32 / 1.58 / 0.39 / 0.051 % against the documented 4.43 / 1.60 / 0.40 / 0.04 %
for W = 8 / 16 / 32 / 64.

| load | W | linear drop % (avg probes) | MSI drop % (avg probes) |
|---|---|---|---|
| 0.77 | 8  | 4.318 (1.82) | 1.305 (1.77) |
| 0.77 | 16 | 1.579 (2.17) | 0.086 (1.89) |
| 0.77 | 32 | 0.390 (2.46) | 0.0005 (1.91) |
| 0.77 | **64 (current)** | **0.051 (2.62)** | **0.000 (1.91)** |
| 0.13 | 8-64 | 0.000 (1.08) | 0.000 (1.07) |
| 0.39 | 64 | 0.000 (1.31) | 0.000 (1.26) |
| 0.90 | 64 | 0.988 (4.00) | 0.0015 (2.56) |

What the simulation says: at the shipped W=64, MSI recovers 0.05 % of edges,
and only on ffmpeg_read (load 0.77); at the load of every other target (0.13)
the two are identical; doubling the map already gives 0 drops with linear
probing, which is what `recommended_map_size()` does. MSI pays off at load
0.90 (0.99 % -> 0.0015 %), which no target reaches.

**Implementation** (`afl_shim.c`, `-D__AFL_PROBE_MSI=1`, default 0): the
second and later probes use `step = ((id * 0x85EBCA6B) >> 16 | 1) & (size - 1)`
instead of +1. Non-power-of-two tables keep step 1. `#error` with a vector
`__AFL_PROBE_ISA` (a step has no contiguous run). The default build is the
old code path. `tests/test_shim_msi_probe.py` (20 tests): an independent
Python model of the stepped probe (placement, counts and drops match on dense
streams at load 0.3/0.8/0.95), the default build equals the linear model
(control), the macro changes placement (not inert), fewer drops at load 0.95,
non-power-of-two fallback, same-home clusters past the window, tiny tables, a
re-fire never duplicates, and the ISA clash is rejected. The shim, SIMD-probe,
map-slot and audit tests still pass (117 in the three files re-run after rebasing on upstream).

**A/B** (clang 18.1.3 installed for this; ledger rows 86-92):

| design | MSI vs linear | A/A (lin vs lin copy) | verdict |
|---|---|---|---|
| microbench `probe` load 0.77 | +11.2 % ns, 1/8, p=0.039 | -0.4 %, p=0.508 | LOSS |
| microbench `probe` load 0.13 | +6.5 %, 4/5, p=1.0 | -0.2 % | NULL |
| microbench `full` load 0.13 | -3.7 %, 6/3, p=0.51 | -9.0 % | NULL |
| microbench `full` load 0.77 | +6.4 %, 3/6, p=0.51 | -7.8 % | NULL |
| fuzzgoat in-run, default map | eps +22.6, edges -5.0, tgt +0.015 ms; p 0.73 / 1.0 / 0.29 | eps -31.2, edges -12.5 | NULL |
| fuzzgoat in-run, `--map-size 1024` (load ~0.7) | eps +11.3, edges +34, tgt 0; p 1.0 / 0.73 | eps -51.8, edges -14.5 | NULL |
| new default vs pre-change HEAD (control) | min ns equal within 1.5 % in all four cells | n/a | NULL |

MSI is slower per edge exactly where it should help (load 0.77, mean probe
length 2.62 -> 1.91 in the simulation but every extra probe is its own cache
line). In the real fuzzer it moves nothing: fuzzgoat's per-exec target time is
~0.1 ms of a ~0.9 ms cycle.

Limits: 8 rounds x 20k execs per design on one shared core (host speed drifted
between the two microbench passes, 22.6 -> 28.6 ns on `full` 0.77); `tgt_ms`
has 0.01 ms resolution; drops were not read in-run (only edges found), so the
"edges recovered" claim rests on the simulation; fuzzgoat is the only target
(no ffmpeg build here); the microbenchmark crashes at exit on a free of its own
buffer (`__afl_write_distance_tail_exit`, unmodified tree, output survives with
`stdbuf -o0`), which is a separate bug. Not changed: `Hard Rule 53` layout
version (header and table layout identical; placement differs only under the
macro).

Simulation source (`gcc -O2 -o sim sim.c && ./sim 0.77`):

```c
// Drop-rate simulation: linear probe vs MSI (stepped) probe, bounded window W.
// Keys are splitmix64-hashed (shim's guard mix is splitmix64), start = (h32*size)>>32 as in the shim.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
static uint64_t sm(uint64_t *s){uint64_t z=(*s+=0x9e3779b97f4a7c15ull);z=(z^(z>>30))*0xbf58476d1ce4e5b9ull;z=(z^(z>>27))*0x94d049bb133111ebull;return z^(z>>31);}
// mode 0 linear, 1 MSI (step from high hash bits, odd, masked), returns dropped count; avg probes of successful inserts via *avg
static uint64_t run(uint32_t size,uint32_t n,uint32_t W,int mode,uint64_t seed,double*avg){
  uint8_t*occ=calloc(size,1); uint64_t s=seed,drops=0,probes=0,ins=0; uint32_t mask=size-1;
  for(uint32_t k=0;k<n;k++){
    uint64_t h=sm(&s); uint32_t start=(uint32_t)(((uint64_t)(uint32_t)h*size)>>32);
    uint32_t step= mode? ((uint32_t)(h>>32)|1u) : 1u;
    uint32_t pos=start; int placed=0;
    for(uint32_t i=0;i<W;i++){ if(!occ[pos]){occ[pos]=1;placed=1;probes+=i+1;ins++;break;} pos=(pos+step)&mask; }
    if(!placed) drops++;
  }
  free(occ); *avg=ins?(double)probes/ins:0; return drops;
}
int main(int argc,char**argv){
  uint32_t size=1u<<18; double load=argc>1?atof(argv[1]):0.77; uint32_t n=(uint32_t)(size*load);
  int seeds=20; uint32_t Ws[]={8,16,32,64};
  printf("size=%u load=%.2f keys=%u seeds=%d\n",size,load,n,seeds);
  printf("%4s | %-22s | %-22s\n","W","linear drop% (avgprobe)","MSI drop% (avgprobe)");
  for(int w=0;w<4;w++){ double d[2]={0,0},a[2]={0,0};
    for(int m=0;m<2;m++){ for(int s=1;s<=seeds;s++){ double av; d[m]+= (double)run(size,n,Ws[w],m,s*7919,&av)/n; a[m]+=av; } d[m]=100*d[m]/seeds; a[m]/=seeds; }
    printf("%4u | %8.4f (%5.2f)      | %8.4f (%5.2f)\n",Ws[w],d[0],a[0],d[1],a[1]); }
  return 0; }
```

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
4. F5 done as opt-in and not adopted (see F5); revisit only for a target above load ~0.85 with the map at its cap.
5. F3 audit, F7 check of `validity.py`.
