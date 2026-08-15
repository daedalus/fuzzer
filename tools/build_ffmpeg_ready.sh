#!/bin/bash
# Build ready-to-fuzz FFmpeg harness binaries from the vendored libraries.
#
# Produces:
#   targets/ffmpeg_read_nosan.so   in-process (ctypes) target, clang coverage,
#                                  NO sanitizer  -> fuzz with --inprocess-direct
#   targets/ffmpeg_read_asan       (with --asan) standalone ASAN executable
#                                  for crash triage / reproduction
#
# It links the merged AFL/cmplog shim (afl_shim.c) into the harness so the
# trace-pc-guard / trace-cmp callbacks emitted by the vendored libav* objects
# resolve at load time. The two defines below are load-bearing:
#   -D__AFL_CMPLOG=1        compiles the comparison-logging layer (ex cmplog_shim.c)
#                           that defines __sanitizer_cov_trace_cmp*; without it a
#                           coverage-instrumented libav* leaves those undefined and
#                           ctypes.CDLL fails: "undefined symbol __sanitizer_cov_trace_cmp1"
#   -D__AFL_CTX_SENSITIVE=1 caller-context edge hashing (needs -fno-omit-frame-pointer)
#
# Usage:
#   tools/build_ffmpeg_ready.sh                # nosan .so (auto-vendors if needed)
#   tools/build_ffmpeg_ready.sh --minimal      # + minimal audio component set (fast)
#   tools/build_ffmpeg_ready.sh --asan         # also build the ASAN repro executable
#   tools/build_ffmpeg_ready.sh --no-vendor    # fail instead of auto-vendoring

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SHIM="$ROOT/src/fuzzer_tool/adapters/afl_shim.c"
TARGETS="$ROOT/targets"
VENDOR="$ROOT/vendor"
HARNESS="$TARGETS/ffmpeg_read.c"

MINIMAL_ARG=""; DO_ASAN=0; AUTOVENDOR=1
for a in "$@"; do
    case "$a" in
        --minimal)   MINIMAL_ARG="--minimal" ;;
        --asan)      DO_ASAN=1 ;;
        --no-vendor) AUTOVENDOR=0 ;;
        *) echo "unknown arg: $a" >&2; exit 2 ;;
    esac
done

command -v clang &>/dev/null || { echo "ERROR: clang required" >&2; exit 1; }

# External deps depend on which FFmpeg components were enabled. A full build
# pulls in zlib/lzma/bz2; a --minimal (--disable-autodetect) build needs none.
# Probe each and include only those the linker can actually find.
_opt_libs() {
    local out=""
    for l in z lzma bz2; do
        if echo 'int main(void){return 0;}' | clang -x c - "-l$l" -o /dev/null 2>/dev/null; then
            out="$out -l$l"
        fi
    done
    echo "$out"
}
OPT_LIBS="$(_opt_libs)"
ffmpeg_libs() {  # $1 = ffmpeg tree root
    echo "$1/libavformat/libavformat.a $1/libavcodec/libavcodec.a \
          $1/libavutil/libavutil.a $1/libswresample/libswresample.a \
          -lm $OPT_LIBS -lpthread -ldl"
}
have_tree() { [ -f "$1/libavformat/libavformat.a" ]; }

# ── nosan .so ────────────────────────────────────────────────────
FF_NOSAN="$VENDOR/ffmpeg"
if ! have_tree "$FF_NOSAN"; then
    [ "$AUTOVENDOR" -eq 1 ] || { echo "ERROR: $FF_NOSAN not built (run vendor_ffmpeg.sh --nosan)" >&2; exit 1; }
    echo ">> Vendoring nosan FFmpeg..."
    "$ROOT/tools/vendor_ffmpeg.sh" --nosan $MINIMAL_ARG
fi

echo ">> Linking $TARGETS/ffmpeg_read_nosan.so"
# shellcheck disable=SC2046
clang -O1 -g -shared -fPIC -fno-omit-frame-pointer \
    -D__AFL_CTX_SENSITIVE=1 -D__AFL_CMPLOG=1 \
    -Wl,-Bsymbolic \
    -include "$SHIM" -I "$FF_NOSAN" \
    -o "$TARGETS/ffmpeg_read_nosan.so" "$HARNESS" \
    $(ffmpeg_libs "$FF_NOSAN") \
    -Wl,--export-dynamic

# ── verify the .so is loadable for --inprocess-direct ────────────
und=$(nm -D --undefined-only "$TARGETS/ffmpeg_read_nosan.so" | grep -c 'sanitizer_cov' || true)
if [ "$und" -ne 0 ]; then
    echo "ERROR: .so has $und undefined sancov symbols — ctypes load would fail." >&2
    nm -D --undefined-only "$TARGETS/ffmpeg_read_nosan.so" | grep sanitizer_cov >&2
    exit 1
fi
nm -D "$TARGETS/ffmpeg_read_nosan.so" | grep -q ' T fuzz_ffmpeg' \
    || { echo "ERROR: fuzz_ffmpeg entrypoint not exported" >&2; exit 1; }
echo "   OK: no undefined sancov syms, fuzz_ffmpeg exported"

# ── optional ASAN repro executable ───────────────────────────────
if [ "$DO_ASAN" -eq 1 ]; then
    FF_ASAN="$VENDOR/ffmpeg_asan"
    if ! have_tree "$FF_ASAN"; then
        [ "$AUTOVENDOR" -eq 1 ] || { echo "ERROR: $FF_ASAN not built" >&2; exit 1; }
        echo ">> Vendoring ASAN FFmpeg..."
        "$ROOT/tools/vendor_ffmpeg.sh" --asan $MINIMAL_ARG
    fi
    echo ">> Linking $TARGETS/ffmpeg_read_asan (executable)"
    clang -O1 -g -fno-omit-frame-pointer -fsanitize=address \
        -D__AFL_CTX_SENSITIVE=1 -D__AFL_CMPLOG=1 \
        -include "$SHIM" -I "$FF_ASAN" \
        -o "$TARGETS/ffmpeg_read_asan" "$HARNESS" \
        $(ffmpeg_libs "$FF_ASAN") -Wl,--export-dynamic
    echo "   OK: ffmpeg_read_asan"
fi

# ── seed corpus (only if absent) ─────────────────────────────────
CORPUS="$ROOT/corpus_ffmpeg"
if [ ! -d "$CORPUS" ] || [ -z "$(ls -A "$CORPUS" 2>/dev/null)" ]; then
    echo ">> Generating seed corpus at $CORPUS"
    mkdir -p "$CORPUS"
    python3 - "$CORPUS" <<'PY'
import struct, sys, os
d = sys.argv[1]
def wav(p, n):
    s = b''.join(struct.pack('<hh',(i*137)%3000-1500,(i*91)%3000-1500) for i in range(n))
    body = b'WAVE'+b'fmt '+struct.pack('<I',16)+struct.pack('<HHIIHH',1,2,8000,32000,4,16)+b'data'+struct.pack('<I',len(s))+s
    open(p,'wb').write(b'RIFF'+struct.pack('<I',len(body))+body)
wav(os.path.join(d,'seed_stereo.wav'),64); wav(os.path.join(d,'seed_small.wav'),16)
b=open(os.path.join(d,'seed_stereo.wav'),'rb').read(); open(os.path.join(d,'seed_trunc.wav'),'wb').write(b[:len(b)//2])
PY
fi

cat <<EOF

=== ready-to-fuzz ===
  target : $TARGETS/ffmpeg_read_nosan.so
  seeds  : $CORPUS

Run:
  fuzzer-tool fuzz $TARGETS/ffmpeg_read_nosan.so \\
    --inprocess-direct --inprocess-func fuzz_ffmpeg \\
    -d $CORPUS -o crashes_ffmpeg \\
    -c --elo all --lineage-backtrack --report report_ffmpeg.md
EOF
