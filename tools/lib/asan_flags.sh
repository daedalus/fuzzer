# Shared ASAN flags, sourced by tools/build_targets.sh and
# tools/build_rust_target.sh.
#
# Recover mode is required by in-process (direct_lite): the wrapper runs it
# with halt_on_error=0, which only a -fsanitize-recover=address check site
# honours. A plain site runs ASAN's Die() inside the fuzzer --
# fuzzgoat_read_asan.so killed it on the first injected bug. It is per call
# site, so a library built without it stays fatal under a recover wrapper.
# Executables are unaffected: their runtime default is still halt_on_error=1.
ASAN_CFLAGS="-fsanitize=address -fsanitize-recover=address"

# rustc's equivalent, for crates instrumented with -Z sanitizer=address.
RUST_ASAN_FLAGS="-Z sanitizer=address -Z sanitizer-recover=address"
