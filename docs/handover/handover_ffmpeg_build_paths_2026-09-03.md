# Handover — FFmpeg vendoring and target build on a clean container

Original 2026-09-03 (base `09b188c`, addendum `426480b`).
Pruned 2026-09-26 to open items; full original: `git show 1c689e8a^:docs/handover/handover_ffmpeg_build_paths_2026-09-03.md`.
Verified against `4c021daa`. Finding IDs (F*/N*) kept from the original.

## Open

### F2 — build-root name disagrees in `AGENTS.md`
`tools/build_targets.sh:77` defaults `FUZZ_BUILD_ROOT` to `~/fuzzing/builds`;
script header agrees. `AGENTS.md` Layout tree under `~/fuzzing/` still shows `targets/` as
`$FUZZ_BUILD_ROOT`. Rename to `builds/`.

### F3 (residual) — duplicate link-lib probe
`tools/build_ffmpeg_ready.sh:_opt_libs` hand-probes `-lz -llzma -lbz2`;
`tools/build_targets.sh:ffmpeg_extralibs` derives from `ffbuild/config.mak`
and probes. Two helpers, will drift. Fold into one shared helper (e.g.
`tools/lib/`), used by both scripts.

### F4 (residual) — FFmpeg sancov rebuild cannot be declined
`tools/build_targets.sh:336` `WITH_FFMPEG_SANCOV=1` is a plain assignment:
no env override, `--ffmpeg-sancov` is a no-op, no `--no-ffmpeg-sancov`. Stamps
(`00f482c4`) make an unchanged tree cheap, but a first build for `png_read` alone
still pays two full FFmpeg builds (nosan + asan). Add `: "${WITH_FFMPEG_SANCOV:=1}"`
and a `--no-ffmpeg-sancov` flag.

### F5 / N3 — sancov build ignores the vendored component set
`build_vendored_ffmpeg_sancov` configure (`tools/build_targets.sh:~1324`) passes
its own flags: no `FFMPEG_COMPONENTS`, no `--minimal` set, and
`--disable-parsers` (parser layer not compiled into a demux+decode harness).
Measured: vendored `--minimal` tree 7 demuxers, linked tree 355. `vendor_ffmpeg.sh
--minimal` therefore affects nothing downstream. `handover_FINDINGS.md` §15 records
"declined", but no rationale exists in code. Open question: propagate the
component set (and drop `--disable-parsers`), or document at the configure call
that the linked set is intentionally independent.

### F7 — shared stage dir, no lock
`$FUZZ_BUILD_ROOT/ffmpeg${suffix}/src` is updated in place (`rm -rf` gone), but
two concurrent `build_targets.sh` runs still `make clean`/configure/make the same
tree. Take a `flock` on `$BUILD_DIR` in `build_vendored_ffmpeg_sancov`.

### F8 (residual) — hardcoded developer paths
`build_targets.sh` fixed (`12433c92`, `6ee98805`). `tools/patcher:PATCH_TARGETS`
still names `/home/dclavijo/my_code/fgrep/...` and legacy `vendor/ffmpeg`/
`vendor/ffmpeg_asan` paths, ignoring `$FUZZ_VENDOR_ROOT`. Resolve via the
vendor root like `build_ffmpeg_ready.sh` (`d54f94b3`).

### F9 — missing source aborts the whole build
`tools/build_targets.sh:build_target` does `warn "Source not found"` + `return 1`;
under `set -e` the ERR trap prints `FAIL: build aborted`. After F1 only a
genuinely missing tracked source hits it. Decide: abort (keep, fix `WARN`
wording to `FAIL`) or skip via `warn_failed`.

### N6 (residual) — prerequisites undocumented
`nasm`/`yasm` is now probed (`c30926d6`) but absent from install docs; without
it FFmpeg builds with no x86 SIMD (different decoder code paths). `libjpeg-dev`
needed by `jpeg_read`. Add both to README prerequisites.
