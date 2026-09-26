# Handover — FFmpeg vendoring and target build on a clean container

**Date:** 2026-09-03
**Base:** `09b188c` (`fix: reorder FFmpeg_asan build before simple targets so ffmpeg_read links`)
**Status: DIAGNOSTIC ONLY. NOTHING FIXED.** Every line below was observed on a
fresh clone of this tree in a 1-core Ubuntu 24.04 container. No code was
changed; the findings are unwritten fixes.

The task was: pull, vendor FFmpeg, build, report. Vendoring succeeded and the
in-process harness links. `tools/build_targets.sh` with no flags does **not**
complete — it aborts on its first target. What follows is the evidence.

---

## 0. What was actually run

| Step | Command | Result |
|---|---|---|
| 1 | `git clone https://github.com/daedalus/fuzzer.git` | head `09b188c` |
| 2 | `apt-get update && apt-get install -y clang` | clang 18.1.3 |
| 3 | `tools/vendor_ffmpeg.sh --nosan --minimal` | **OK**, ~25 min wall, 1 core |
| 4 | `tools/build_ffmpeg_ready.sh --minimal --no-vendor` | **FAIL** — wrong vendor root (F3) |
| 5 | same, with `vendor/ffmpeg` symlinked to the real tree | **OK**, `ffmpeg_read_nosan.so` |
| 6 | `tools/build_targets.sh` (no flags) | **ABORT** at the first target (F1) |

Container facts worth recording: `gcc` present, `clang` absent until installed;
`zlib.h` and `bzlib.h` present, `lzma.h` and `jpeglib.h` absent. The missing
lzma/jpeg headers no longer matter for the FFmpeg path — `_opt_libs()` in
`build_ffmpeg_ready.sh` and `ffmpeg_extralibs()` in `build_targets.sh` both
probe each `-l` with a trivial link and drop what does not resolve. That is the
`dc9bb95` fix doing its job; the old unconditional `-llzma` failure did not
reproduce.

## 1. What works

**Vendoring.** `tools/vendor_ffmpeg.sh --nosan --minimal` completed cleanly.
Source resolution took step **`[1/4] code.ffmpeg.org git clone (tag n9.0.1)`** —
the canonical git host answered, so neither the ffmpeg.org tarball nor the
codeload mirror was needed. This contradicts the older note in the environment
recipe that ffmpeg.org egress is blocked here; the primary source is now first
in the list and it worked.

Output, from the script's own `[4/4] Verifying` pass:

```
libavformat.a    7.6M      77 trace-cmp call sites
libavcodec.a     7.9M      88 trace-cmp call sites
libavutil.a      4.7M
libswresample.a  708K
```

Tree size 151 MB at `~/fuzzing/vendoring/ffmpeg`. FFmpeg version is now 9.0.1.

**Resumability.** The first attempt was killed after the four archives were
produced but before `[4/4]`. Re-running the same command re-entered at `[3/4]`,
rebuilt only the four `libswresample` objects, and finished. `make` incrementality
survives an interrupted vendoring run — no need to start over.

**The in-process harness.** With the vendor path worked around (F3),
`tools/build_ffmpeg_ready.sh --minimal --no-vendor` produced
`targets/ffmpeg_read_nosan.so` (7.7 MB) and both of its self-checks passed:

```
OK: no undefined sancov syms, fuzz_ffmpeg exported
```

It also generated the three-file WAV seed corpus at `corpus_ffmpeg/`.

---

## 2. Findings

### F1 — `tools/build_targets.sh` with no flags aborts on its first target

Reproduction on a fresh clone:

```
Building simple targets (ASAN)...
  WARN: Source not found: /root/fuzzing/builds/asan_target.c

FAIL: build aborted (exit 1) at tools/build_targets.sh:862
       command: return 1
```

The mechanism is in the path block at `tools/build_targets.sh:75-95`:

- `TARGETS` defaults to `$FUZZ_BUILD_ROOT` (the *output* root).
- `TARGETS_SRC` is assigned **only** inside `if [ "${IN_TREE_TARGETS_SRC:-0}" = "1" ]`.
- Every source reference is `"${TARGETS_SRC:-$TARGETS}/<name>.c"` — 73 call sites.

So without `--in-tree-targets-src`, the `.c` sources are looked for in the build
root, where they have never existed. `build_target` at `:861` does
`warn` + `return 1`, `set -e` is live from `:36`, and the ERR trap at `:48`
reports the abort. The very first target attempted (`asan_target`) is enough to
end the run.

`--in-tree-targets-src` makes it work: a concurrent invocation in the same
container running `tools/build_targets.sh --nosan --in-tree-targets-src` built
19 artifacts into `~/fuzzing/builds/` and proceeded into the FFmpeg_asan sancov
rebuild.

This matters because `AGENTS.md` documents the bare form:

> `tools/build_targets.sh` — Build all fuzz targets (ASAN + cmplog by default…)

Either `TARGETS_SRC` should default to the in-tree `targets/` unconditionally
(the sources are tracked in git and never move), or the flag has to be added to
every documented invocation. The first is almost certainly right: there is no
scenario where the `.c` files live in the build root, so the `:-$TARGETS`
fallback has no valid case to serve.

### F2 — the build-root default disagrees with its own documentation

`tools/build_targets.sh:76`:

```sh
: "${FUZZ_BUILD_ROOT:=$HOME/fuzzing/builds}"
```

The header comment at `:30` and `:60-62` of the same file, and the `AGENTS.md`
table and directory map added by `09b188c`, all say `~/fuzzing/targets`. The
code writes to `~/fuzzing/builds`. Whichever is intended, three documents and
one assignment currently point at two different directories, and anyone
following the docs to find their artifacts will look in an empty tree.

### F3 — `tools/build_ffmpeg_ready.sh` was never migrated to the new layout

`tools/build_ffmpeg_ready.sh:29`:

```sh
VENDOR="$ROOT/vendor"
```

`grep -rn FUZZ_VENDOR_ROOT tools/` matches all eight `vendor_*.sh` scripts and
`build_targets.sh`. It does not match this file at all. The script has no
`--in-tree-vendor` parsing and no `FUZZ_VENDOR_ROOT` reference.

Two failure modes:

- **With autovendor (the default).** `have_tree "$ROOT/vendor/ffmpeg"` is false
  on a fresh checkout, so it shells out to `tools/vendor_ffmpeg.sh`, which
  writes to `~/fuzzing/vendoring/ffmpeg` — a directory this script never
  consults. It then proceeds to the `clang -shared` link against
  `$ROOT/vendor/ffmpeg/libavformat/libavformat.a`, which does not exist. The
  documented one-command path pays a full FFmpeg build and *then* fails at
  link, and every re-run repeats it.
- **With `--no-vendor`.** Immediate, observed verbatim:
  `ERROR: /home/claude/fuzzer/vendor/ffmpeg not built (run vendor_ffmpeg.sh --nosan)`
  — after `vendor_ffmpeg.sh` had already succeeded.

Workarounds that do work today: `IN_TREE_VENDOR=1 tools/build_ffmpeg_ready.sh --minimal`
(the env var is inherited by the child `vendor_ffmpeg.sh`, so both agree on the
in-tree path), or symlink `vendor/ffmpeg` at the real tree, which is what step 5
above did.

Note this script also carries its own `_opt_libs()` probe, a near-duplicate of
`ffmpeg_extralibs()` in `build_targets.sh` — worth folding into one helper when
the path fix lands, since they will drift otherwise.

### F4 — the sancov rebuild is unconditional, and cannot be turned off

`build_vendored_ffmpeg_sancov` (`:947`) does, per call: `rm -rf` the staging
dir, `rsync -a` the whole FFmpeg source, `make clean`, `./configure`, and a
full `make`. It runs twice per `build_targets.sh` invocation (nosan and
`_asan`). On one core that is roughly 50+ minutes on top of whatever
`vendor_ffmpeg.sh` already built — and it discards that work, since it rebuilds
from a copy with its own configure.

The guard that would have avoided this is commented out at `:972`:

```sh
local has_cov=$(nm "$FFMPEG_DIR/libavformat/libavformat.a" 2>/dev/null | grep -c '__sanitizer_cov_trace_pc_guard' || true)
# [ "$has_cov" -gt 10 ] && return 0  # early-return removed: always rebuild to pick up patched sources
```

`has_cov` is now computed and never read — dead as written. The stated reason
(picking up patched sources, i.e. `patches/ffmpeg-vpk-divide-by-zero.patch`) is
legitimate, but "always rebuild" is a heavy answer to it; a stamp file holding
the hash of the applied patch set would give the same correctness at zero cost
on an unchanged tree.

Also: `WITH_FFMPEG_SANCOV=1` at `:225` is a plain assignment, so an environment
variable cannot override it. The only related flag is `--ffmpeg-sancov`, which
sets it to the value it already has — a no-op — and there is no
`--no-ffmpeg-sancov`. So every `build_targets.sh` run, including one that only
wants `png_read`, pays two full FFmpeg builds with no way to decline.

### F5 — the libraries that get linked are not the ones that were vendored

`vendor_ffmpeg.sh --minimal` configures a small audio-focused component set.
`build_vendored_ffmpeg_sancov` then reconfigures its staged copy with a
completely different line (`:1026-1032`):

```
--disable-encoders --disable-muxers --disable-devices --disable-filters
--disable-parsers --disable-bsfs --disable-avdevice
--disable-pthreads --disable-network --disable-hwaccels ...
```

No `--minimal`, no `FFMPEG_COMPONENTS`, no propagation of what the vendoring
step chose. Two consequences:

1. The trace-cmp site counts the vendoring step reports (77 / 88) describe a
   tree that is not what `ffmpeg_read` links against. Any measurement anchored
   to those numbers is anchored to the wrong build.
2. `--disable-parsers` removes the parser layer from a harness whose whole
   point is demux + decode. `docs/handover/handover_ffmpeg_tools_fuzzer_port_2026-08-28.md`
   already flagged `av_parser` as absent from our harness; this is the build
   side of the same gap, and it means the parser code path is not merely
   uncalled but not compiled in.

Whether that is deliberate (a smaller map, faster builds) or accidental, it is
undocumented, and it is invisible because the configure output is discarded —
see F6.

### F6 — the sancov configure and make discard all output

Both are `>/dev/null 2>&1` (`:1032`, `:1034`). The only user-visible artifact of
a failure is:

```
WARN: vendored FFmpeg_asan configure failed
```

with no cause, no config.log path, no command echoed. This is the same defect
class as the old `-llzma` failure that surfaced only as `WARN: failed: ffmpeg_read`
— the one `dc9bb95` was written to prevent. Redirect to `$BUILD_LOG` instead of
`/dev/null`; the log file already exists and is already used elsewhere in the
script.

### F7 — the staging directory is a fixed path with no lock, and is opened with `rm -rf`

`STAGE_DIR="$FUZZ_BUILD_ROOT/ffmpeg${asan_suffix}/src"`, then `rm -rf "$STAGE_DIR"`.
Nothing serialises two invocations. During this session a second
`build_targets.sh` was running concurrently in the same container and was
`rm -rf`-ing `~/fuzzing/builds/ffmpeg_asan` while the first run was configuring
inside it.

**This is the most likely explanation for the `FFmpeg_asan configure failed`
line in F6, and it has not been isolated.** Do not treat that failure as a real
configure defect until it is reproduced on a container with exactly one build
running. Reproducing it is the first thing the next session should do, because
the answer decides whether F6 is hiding a genuine ASAN configure problem or
just reported a collision.

Independently of the cause: a build step that begins by `rm -rf`-ing a fixed
path under a shared root should take a lock or use a per-invocation directory.
The same reasoning already applied to the cmplog artifact sweep, where parallel
workers over one cache directory forced the 24-hour age guard rather than an
unconditional wipe.

### F8 — hardcoded developer paths in the build scripts

`tools/build_targets.sh:51-53`:

```sh
if [ -z "${TMPDIR:-}" ]; then
    mkdir -p /home/dclavijo/tmp
    export TMPDIR=/home/dclavijo/tmp
fi
```

and `:98`: `TAILSLAYER="${TAILSLAYER_DIR:-/home/dclavijo/code/tailslayer}"`.

The first creates `/home/dclavijo/tmp` on any machine where it can, and fails
the build under `set -e` on any machine where it cannot — a CI runner with a
read-only `/home`, for instance. The second is merely cosmetic (it degrades to a
`SKIP` line) but it puts a foreign absolute path in the feature matrix every
run. Same class as the `os.chdir("/home/dclavijo/my_code/fuzzer")` already
recorded for `tools/profile_hotpath.py`.

### F9 — the ERR trap's wording

The trap at `:48` prints `FAIL: build aborted`. In F1 that is accurate — `set -e`
is on and the run did end. Worth noting anyway that it fires on *any* `return 1`,
including from `build_target`'s "optional thing missing" path, which reads as
`WARN` one line above and `FAIL: build aborted` one line below. The trap's own
comment explains it was added because a `return 1` on an absent optional vendor
tree used to abort builds silently; the fix reported the abort but left the
`return 1` in a function whose other callers treat a miss as skippable.

### F10 — six `-Wpointer-bool-conversion` warnings in `afl_shim.c`

Emitted on every harness link:

```
afl_shim.c:1923:33: warning: nonnull parameter 's' will evaluate to 'true' on first encounter
afl_shim.c:1923:38: warning: nonnull parameter 'accept' ...
afl_shim.c:1940:33 / :38   (same shape)
afl_shim.c:1957:33 / :38   (same shape)
```

Three call sites guard on pointers that the libc prototype declares `__nonnull`,
so clang folds each test to true. Either the guards are genuinely dead (drop
them) or the pointers really can be NULL at these call sites (in which case the
declaration is wrong for our use and the arguments need checking before the
call, not inside it). Not investigated.

---

## 3. Suggested order

1. **F1** — one-line default for `TARGETS_SRC`. Nothing else can be tested end
   to end until `build_targets.sh` runs with the flags its own documentation
   uses.
2. **F3** — port `build_ffmpeg_ready.sh` to `FUZZ_VENDOR_ROOT` / `--in-tree-vendor`,
   matching `build_targets.sh` verbatim.
3. **F6** — redirect the sancov configure/make to `$BUILD_LOG`, then **F7**:
   re-run a single clean `build_targets.sh --in-tree-targets-src` and find out
   whether the ASAN configure failure is real.
4. **F2** — reconcile `~/fuzzing/builds` vs `~/fuzzing/targets` across the code,
   the script header, and `AGENTS.md`.
5. **F4 / F5** — patch-hash stamp instead of unconditional rebuild, and decide
   whether the sancov component set is meant to differ from the vendored one.
   These are the expensive ones and they are entangled; do them together.
6. **F8**, **F9**, **F10** — cleanup.

## 4. What was not done

- No fix was written and nothing was committed.
- `build_targets.sh` was never observed to complete, so there is no statement
  here about how many of the 19-plus targets build, whether `ffmpeg_read`
  links under ASAN after `09b188c`'s reordering, or what
  `ffmpeg_extralibs()` derives from the 9.0.1 `ffbuild/config.mak`. That last
  one is worth measuring specifically: the previously recorded answer on the
  older tree was `-lm -latomic -lbz2 -lz -pthread`, and the version bump may
  have changed it.
- The test suite was not run. No baseline failure count exists for this
  container at `09b188c`.

---

# Addendum — second pass, same day, later

**Tree:** `426480b` plus this doc. Four commits landed in the working tree
between the first pass and this one (`bbb2645`, `8204cf3`, `71f2e02`,
`426480b`), all touching `tools/build_targets.sh`. They change the answers
above. This section says which ones, retracts one finding, and adds six.

`tools/build_targets.sh --nosan --in-tree-targets-src` **completed** on this
tree: 62 `OK` lines, 2 failures, 24 targets carrying AFL symbols, 28 `.so`
targets with `fuzz_shm_run` and cmplog.

## A. Retraction — F7 did not cause the configure failure

F7 above guessed that `WARN: vendored FFmpeg_asan configure failed` came from
a concurrent run `rm -rf`-ing the shared stage. **That guess was wrong.** The
cause was `nasm` being absent: FFmpeg's configure aborts with `nasm not found
or too old`, and the message went to `/dev/null`. `426480b` adds
`--disable-x86asm` to the in-script configure, matching what
`vendor_ffmpeg.sh:212` already passed, and the failure is gone —
`OK: vendored FFmpeg_asan → /root/fuzzing/builds/ffmpeg_asan`.

The reasoning error is worth naming: two runs *were* colliding, so a collision
was available as an explanation and I reached for it instead of reading the
one line that would have settled it. The collision was real and the diagnosis
was still wrong. F7's substance — a fixed stage path opened with `rm -rf` and
no lock — stands as a hazard; it just wasn't this failure.

## B. What the four commits fixed

- **F4 is fixed** by `71f2e02`. The `rm -rf "$STAGE_DIR"` is gone, the stage
  persists with its objects, and two stamps (config flags + compiler; vendored
  revision + `patches/`) gate the work. Verified on the third call in one run:
  `OK: vendored FFmpeg (nosan): up to date`, no rebuild.
- **F6 is fixed** by `bbb2645`. `configure`/`make`/every compile now append to
  `$BUILD_LOG`, and `warn_failed` prints the first real error line. The
  jpeg failure below reported itself as
  `targets/jpeg_read.c:15:10: fatal error: 'jpeglib.h' file not found` —
  exactly what the old `/dev/null` would have swallowed.
- **F1, F2, F3, F5, F8, F9, F10 are unchanged.** `TARGETS_SRC` is still
  assigned only under `--in-tree-targets-src` (`:90`), `FUZZ_BUILD_ROOT` still
  defaults to `~/fuzzing/builds` against three documents saying `targets`
  (`:76`), `build_ffmpeg_ready.sh:29` still hardcodes `VENDOR="$ROOT/vendor"`,
  and the `/home/dclavijo` paths are still at `:51-53` and `:98`.

## C. New findings

### N1 — the source stamp records filenames, not contents

`build_targets.sh:1153`:

```sh
src_stamp=$( { (cd "$SRC_DIR" && git rev-parse HEAD 2>/dev/null) || echo "no-git"
               find "$SRC_DIR" -name '*.c' -newer "$SRC_DIR/configure" 2>/dev/null | sort
               cat patches/*.patch 2>/dev/null; } | md5sum | cut -d' ' -f1)
```

The `find` contributes a **list of paths**. Once a file is newer than
`configure` it is in the list, and further edits to it do not change the list.
`git rev-parse HEAD` does not move for a dirty tree. So the second and every
later in-place edit to a vendored source produces an identical stamp and the
whole rebuild is skipped.

Demonstrated with three different file contents:

```
after first edit : f045549bf3e1e2abe6b7089314fae1d7
after second edit: f045549bf3e1e2abe6b7089314fae1d7
after third edit : f045549bf3e1e2abe6b7089314fae1d7
```

The realistic way to hit this is the obvious one: edit a demuxer in
`~/fuzzing/vendoring/ffmpeg`, build, test, edit again, build — and from the
second iteration onward you are testing the first edit. It is the exact
failure the removed early-return was guarding against, reintroduced in a
narrower form, and it is silent. Hashing the contents of the listed files
rather than their names costs one `xargs md5sum` and closes it.

Two smaller notes on the same expression: `cat patches/*.patch` is
CWD-relative where everything around it is absolute, so it contributes nothing
when the script is run from elsewhere (the rest of the script assumes the repo
root anyway, so this is a consistency point, not a live bug); and a *deleted*
source is invisible to `-newer` entirely.

### N2 — `ffmpeg_extralibs()` never reads `config.mak`, so its fallback is the only path

The function looks for `"$root/ffbuild/config.mak"` where `$root` is the
**promoted** directory. The promotion loop copies `libav*/libav*.a` and
headers and nothing else — `ffbuild/` is never promoted:

```
$ ls /root/fuzzing/builds/ffmpeg/
libavcodec  libavformat  libavutil  libswresample  libswscale  src
$ ls /root/fuzzing/builds/ffmpeg/ffbuild
ls: cannot access ...: No such file or directory
```

So `[ -f "$mak" ]` is false on every build, `libs` takes the hardcoded
historical fallback, and the config.mak derivation — the entire point of the
change — is dead code on the primary path. The observed link line is the
fallback verbatim:

```
-lm -lz -lbz2 -lpthread -ldl -ldl
```

The real `config.mak` is one directory down, in the stage at
`$root/src/ffbuild/config.mak`, and it says:

```
EXTRALIBS-avformat=-lm -lbz2 -lz -latomic
EXTRALIBS-avutil=-lm -lz -latomic -lX11
```

which is where the `-latomic` recorded on the older tree came from, and where
the `-lX11` that the probe loop exists to drop actually lives. The derivation
works only when `ffmpeg_root` falls back to the legacy `$VENDOR/ffmpeg`, which
does keep its `ffbuild/`. Fix is one line: promote `ffbuild/config.mak` beside
the archives, or read from `$root/src`.

Cosmetic consequence of the fallback: the caller prepends `-lm` and the
fallback list also contains `-lm`, and `-ldl` appears twice. Harmless, but it
is the tell that nothing was derived.

### N3 — `--minimal` does not reach the libraries that get linked

Quantifying F5 on this build. Demuxers enabled, `CONFIG_*_DEMUXER=yes`:

| tree | demuxers |
|---|---|
| `~/fuzzing/vendoring/ffmpeg` (what `vendor_ffmpeg.sh --minimal` built) | 7 |
| `~/fuzzing/builds/ffmpeg/src` (what `ffmpeg_read` links) | 355 |

The sancov configure passes none of `--disable-everything`, the `--enable-demuxer`
list, or `--enable-decoder`. So `--minimal` governs a tree that is copied and
then reconfigured out of existence: you pay the minimal build, then pay a
355-demuxer build, then fuzz the second one. Either propagate the component
set into `build_vendored_ffmpeg_sancov` or stop advertising `--minimal` as
affecting anything downstream of vendoring.

`--disable-parsers` remains in that configure, so the parser layer is still
not compiled into the target.

### N4 — a failed FFmpeg rebuild silently falls back to stale archives

`build_vendored_ffmpeg_sancov` writes into `$OUT_DIR` only on success, which
is right. But on failure it returns after `warn_failed`, and
`build_simple_targets` then finds the *previous* run's archives still sitting
in `$OUT_DIR` and links against them. The console says
`WARN: failed: vendored FFmpeg build` followed by `OK: ffmpeg_read`, and the
target that reports OK contains the old code. Nothing marks the archives as
stale. A sentinel written next to `.sancov_stamp` on entry and cleared on
success would let the link step refuse.

### N5 — the script exits 0 with failures

`:2449-2452`:

```sh
if [ "$BUILD_FAILURES" -gt 0 ]; then
    warn "$BUILD_FAILURES build failure(s) — full output in $BUILD_LOG"
fi
echo "=== Done ==="
```

`warn_failed` deliberately `return 0`s so a failed optional target does not
trip `set -e` — correct. But nothing converts the accumulated count into an
exit status, and `echo` is the last statement, so the script exits 0. This run
ended `WARN: 2 build failure(s)` / `=== Done ===` / `$? = 0`. Any wrapper or CI
step that checks the exit code sees a clean build. This is the same shape as
the defect `bbb2645` set out to fix — a real failure that does not reach the
person who needs it — one level up: the cause is now in the log, but the
signal still is not in the status. `exit 1` when `BUILD_FAILURES > 0`, or a
flag to opt out for local runs.

### N6 — `--disable-x86asm` is now hardcoded in both scripts

`vendor_ffmpeg.sh:212` and `build_targets.sh:1250`. The trigger was a real
failure (no `nasm` here), and hardcoding it makes the build work everywhere.
It also means the vendored FFmpeg contains **no hand-written assembly on any
machine**, including machines that have `nasm`. For a decoder fuzz target that
is not a neutral choice: the SIMD paths are a large, heavily-optimised slice of
`libavcodec`, they are where a good deal of the historical memory-safety
history lives, and the C fallbacks that replace them are a different code path
with different bounds behaviour. Coverage and bug surface both shrink, quietly.

Better shape: probe for `nasm` once, pass `--disable-x86asm` only when it is
absent, warn when doing so, and add `nasm` to the documented prerequisites
alongside `clang`.

## D. Build results, `--nosan --in-tree-targets-src`

Both failures were the same missing header:

```
WARN: failed: jpeg_read       targets/jpeg_read.c:15:10: fatal error: 'jpeglib.h' file not found
WARN: failed: jpeg_read_asan.so
```

`libjpeg-dev` was absent. Installing it mid-run gave a natural experiment: the
later jpeg targets in the same run — `jpeg_read.so`, `jpeg_read_nosan`,
`jpeg_read_ubsan.so` — all built and were reported as new binaries. So the two
failures are environmental, not a defect in the script, and `libjpeg-dev`
belongs in the prerequisites next to `nasm`.

Every FFmpeg target linked, which answers the question the first pass left
open — `09b188c`'s reordering does what it claims:

```
OK: vendored FFmpeg_asan → /root/fuzzing/builds/ffmpeg_asan
OK: ffmpeg_read
OK: ffmpeg_read_asan.so
OK: ffmpeg_read_ubsan.so
OK: ffmpeg_read_nosan
OK: ffmpeg_read.so
```

Verify pass: 24 targets with AFL symbols, 28 `.so` with `fuzz_shm_run`, 28
with cmplog, 33 unchanged / 4 new / 4 changed by checksum.

## E. Revised order

1. **N2** — one line; until it lands, the config.mak derivation is not running
   at all and nobody would know.
2. **F1** — still the reason the documented invocation does not work.
3. **N5** — one line; a build that fails should say so in `$?`.
4. **N1** — hash contents, not names.
5. **F3** — port `build_ffmpeg_ready.sh` to the new roots.
6. **N4**, **N3 / F5**, **N6**, then **F2**, **F8**, **F9**, **F10**.

Still not done: the test suite has not been run, so there is no baseline
failure count for this container on this tree.
