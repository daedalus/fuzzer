#!/bin/bash
# Build shim microbenchmark with canonical flags
# Usage: tools/build-shim-bench.sh [--noctx] [--nodist] [--plain]

set -euo pipefail

SHIM_PATH="src/fuzzer_tool/adapters/afl_shim.c"
BENCH_SRC="tools/shim-micro-benchmark.c"

# Default flags
DEF_FLAGS="-O2 -g -fno-omit-frame-pointer -I$(pwd)/src/fuzzer_tool/adapters -include $(pwd)/$SHIM_PATH"
OUT="shim-micro-benchmark"

# Parse options
NOCTX=0
NODIST=0
PLAIN=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --noctx) NOCTX=1; shift ;;
        --nodist) NODIST=1; shift ;;
        --plain) PLAIN=1; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Apply flags
if [[ $NOCTX -eq 1 ]]; then
    DEF_FLAGS="$DEF_FLAGS -D__AFL_CTX_SENSITIVE=0"
    OUT="${OUT}-noctx"
fi
if [[ $NODIST -eq 1 ]]; then
    DEF_FLAGS="$DEF_FLAGS -D__AFL_DISTANCE_MODE=0"
    OUT="${OUT}-nodist"
fi
if [[ $PLAIN -eq 1 ]]; then
    DEF_FLAGS="$DEF_FLAGS -D__AFL_CTX_SENSITIVE=0 -D__AFL_DISTANCE_MODE=0"
    OUT="${OUT}-plain"
fi

clang $DEF_FLAGS "$BENCH_SRC" -o "$OUT"
echo "Built: $OUT"
