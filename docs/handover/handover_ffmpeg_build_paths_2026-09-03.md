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
