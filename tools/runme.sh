#!/bin/bash
# Unified fuzzer runner — usage: tools/runme.sh <target> [extra args...]
#
# Examples:
#   tools/runme.sh png           # run png_read with defaults
#   tools/runme.sh gzip -n 50000 # gzip_read, 50k iterations
#   tools/runme.sh zlib --elo    # zlib_read with Elo scheduling
set -euo pipefail

TARGET="${1:?Usage: $0 <target> [extra args...]}"
shift

EXE="targets/${TARGET}_read.elf"
if [ ! -f "$EXE" ]; then
    echo "Error: $EXE not found. Compile first: make targets" >&2
    exit 1
fi

DICT="dictionaries/${TARGET}.dict"
GRAM="dictionaries/${TARGET}.gram"
DICT_FLAG=""
GRAM_FLAG=""
[ -f "$DICT" ] && DICT_FLAG="-D $DICT"
[ -f "$GRAM" ] && GRAM_FLAG="-g $GRAM"

set -x
fuzzer-tool fuzz "$EXE" \
    -d "/tmp/fuzz_${TARGET}" \
    -c \
    --markov --markov-gen --markov-order 0,1,2,3 \
    $DICT_FLAG \
    $GRAM_FLAG \
    --calibrate 2000 --stall 1000 --max-collision-risk 25 \
    -n 105000 --elo \
    --report "/tmp/${TARGET}_fuzz_report.txt" \
    "$@"
