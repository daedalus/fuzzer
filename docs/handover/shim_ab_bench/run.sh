#!/bin/bash
# run.sh SHIM.c [SHIM.c ...]   (needs clang, valgrind, python3)
# Env: OUT=/tmp/shim_ab  ROUNDS=40
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); out=${OUT:-/tmp/shim_ab}; rounds=${ROUNDS:-40}
mkdir -p "$out"; python3 "$here/gen_targets.py" "$out"
for t in healthy const junk; do
  case $t in healthy) fp=-fno-omit-frame-pointer;; *) fp=-fomit-frame-pointer;; esac
  for k in t h; do
    src=${k}_plain.c; [ "$t" = junk ] && src=${k}_junk.c
    clang -O2 $fp -fsanitize-coverage=trace-pc-guard -w -c "$out/$src" -o "$out/$k-$t.o"
  done
done
names=()
for s in "$@"; do
  n=$(basename "$s" .c); names+=("$n")
  clang -O2 -fno-omit-frame-pointer -w -include "$(realpath "$s")" -c "$here/main.c" -o "$out/m-$n.o"
  for k in t h; do for t in healthy const junk; do
    clang "$out/m-$n.o" "$out/$k-$t.o" -o "$out/p-$n-$k-$t" -ldl -lpthread -lrt
  done; done
  SHOWFAST=1 "$out/p-$n-t-healthy" 1 2>&1 >/dev/null | grep -q 'fast=1' || { echo "$n: __afl_fast=0, bench invalid" >&2; exit 1; }
done
ir(){ valgrind --tool=cachegrind --cache-sim=no --cachegrind-out-file=/dev/null "$1" "$2" 2>&1 | awk '/I *refs/{gsub(",","",$NF); print $NF}'; }
echo "Ir/run = instructions per run(), 150 guards (lower is better)"
printf "%-14s" case; for n in "${names[@]}"; do printf "%14s" "$n"; done; echo
for k in t h; do for t in healthy const junk; do
  printf "%-14s" "$([ $k = t ] && echo chain || echo hit)-$t"
  for n in "${names[@]}"; do
    a=$(ir "$out/p-$n-$k-$t" 0 2>/dev/null); b=$(ir "$out/p-$n-$k-$t" 2000 2>/dev/null)
    printf "%14d" $(( (b - a) / 2000 ))
  done; echo
done; done
echo; echo "wall: min/median ns per run() (lower is better), Mguards/s from min (higher is better)"
python3 "$here/wall.py" "$out" "$rounds" "${names[@]}"
