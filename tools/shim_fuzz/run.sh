#!/bin/bash
# AFL++ campaign on afl_shim.c's cmplog interposers/callbacks (see README / handover).
#   run.sh text|rec [seconds] [shim.c]
# text: _CMPLOG_OUT text log.  rec: records-only segment (_CMPLOG_RECORDS_SHM), header scribbling on.
set -e
cd "$(dirname "$0")"; W=${WORK:-/tmp/shim_fuzz}; mkdir -p "$W"
MODE=${1:-text}; SECS=${2:-300}; SHIM=${3:-../../src/fuzzer_tool/adapters/afl_shim.c}
# the shim excludes itself from coverage; strip that so AFL++ instruments it too
sed -e '/pragma clang attribute push/d' -e '/pragma clang attribute pop/d' "$SHIM" > "$W/shim_instr.c"
afl-clang-fast -O1 -g -fPIC -shared -fno-omit-frame-pointer -D__AFL_PRELOAD_ONLY -D__AFL_NO_COV= -D__AFL_GCC_NO_COV= -w "$W/shim_instr.c" -o "$W/shim_instr.so" -ldl -lpthread
afl-clang-fast -O1 -g -fno-builtin -w harness.c -o "$W/harness" -ldl
export AFL_PRELOAD="$W/shim_instr.so" AFL_SKIP_CPUFREQ=1 AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1 AFL_NO_AFFINITY=1 AFL_AUTORESUME=0
if [ "$MODE" = rec ]; then
  gcc -O1 -w mkrec.c -o "$W/mkrec"; ID=$("$W/mkrec"); export _CMPLOG_RECORDS_SHM=$ID; trap 'ipcrm -m $ID' EXIT
else export _CMPLOG_OUT="$W/cmp.out"; fi
rm -rf "$W/out_$MODE"
afl-fuzz -i seeds -o "$W/out_$MODE" -V "$SECS" -t 2000 -- "$W/harness"
