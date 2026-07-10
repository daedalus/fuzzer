#!/usr/bin/env bash
# Benchmark baseline vs enhanced fuzzer configurations on a target.
#
# Usage:
#   tools/bench.sh [target] [iterations]
#
# Defaults: targets/png_read, 5000 iterations

set -euo pipefail

TARGET="${1:-targets/png_read}"
ITERS="${2:-5000}"
BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BASELINE_DIR="/tmp/fuzz_bench_baseline"
ENHANCED_DIR="/tmp/fuzz_bench_enhanced"

cd "$BASE_DIR"

# Clean previous runs
rm -rf "$BASELINE_DIR" "$ENHANCED_DIR"
mkdir -p "$BASELINE_DIR" "$ENHANCED_DIR"

echo "============================================================"
echo " Benchmark: baseline vs enhanced"
echo " Target:    $TARGET"
echo " Iterations: $ITERS"
echo "============================================================"
echo ""

# Run baseline (no features)
echo "[*] Running baseline (no features)..."
python -m fuzzer_tool fuzz "$TARGET" -d "$BASELINE_DIR" -c -n "$ITERS" 2>&1 | tee /tmp/fuzz_bench_baseline.log
echo ""

# Run enhanced (all features)
echo "[*] Running enhanced (elo + meta-elo + bandit + mopt)..."
python -m fuzzer_tool fuzz "$TARGET" -d "$ENHANCED_DIR" -c -n "$ITERS" --elo --meta-elo --mc-bandit --mopt 2>&1 | tee /tmp/fuzz_bench_enhanced.log
echo ""

# Extract metrics from run summaries
echo "============================================================"
echo " COMPARISON"
echo "============================================================"
echo ""

extract() {
    grep -oP "$1" "$2" | tail -1
}

b_edges=$(extract "Edges discovered:\s+\K[0-9]+" /tmp/fuzz_bench_baseline.log)
e_edges=$(extract "Edges discovered:\s+\K[0-9]+" /tmp/fuzz_bench_enhanced.log)
b_corpus=$(extract "Corpus:\s+\K[0-9]+" /tmp/fuzz_bench_baseline.log)
e_corpus=$(extract "Corpus:\s+\K[0-9]+" /tmp/fuzz_bench_enhanced.log)
b_eps=$(extract "Avg eps:\s+\K[0-9.]+" /tmp/fuzz_bench_baseline.log)
e_eps=$(extract "Avg eps:\s+\K[0-9.]+" /tmp/fuzz_bench_enhanced.log)
b_dur=$(extract "Duration:\s+\K[0-9s]+" /tmp/fuzz_bench_baseline.log)
e_dur=$(extract "Duration:\s+\K[0-9s]+" /tmp/fuzz_bench_enhanced.log)
b_time=$(extract "Exec time p50:\s+\K[0-9.]+ms" /tmp/fuzz_bench_baseline.log)
e_time=$(extract "Exec time p50:\s+\K[0-9.]+ms" /tmp/fuzz_bench_enhanced.log)
b_collision=$(extract "Collision risk:\s+\K[0-9.]+" /tmp/fuzz_bench_baseline.log)
e_collision=$(extract "Collision risk:\s+\K[0-9.]+" /tmp/fuzz_bench_enhanced.log)

printf "%-25s %12s %12s %10s\n" "Metric" "Baseline" "Enhanced" "Delta"
printf "%-25s %12s %12s %10s\n" "-------------------------" "------------" "------------" "----------"
printf "%-25s %12s %12s" "Edges discovered" "${b_edges:-?}" "${e_edges:-?}"
if [[ -n "$b_edges" && -n "$e_edges" && "$b_edges" -gt 0 ]]; then
    pct=$(python3 -c "print(f'{($e_edges - $b_edges) / $b_edges * 100:+.1f}%')")
    printf " %10s\n" "$pct"
else
    printf " %10s\n" "?"
fi

printf "%-25s %12s %12s" "Corpus entries" "${b_corpus:-?}" "${e_corpus:-?}"
if [[ -n "$b_corpus" && -n "$e_corpus" && "$b_corpus" -gt 0 ]]; then
    pct=$(python3 -c "print(f'{($e_corpus - $b_corpus) / $b_corpus * 100:+.1f}%')")
    printf " %10s\n" "$pct"
else
    printf " %10s\n" "?"
fi

printf "%-25s %12s %12s" "Avg eps" "${b_eps:-?}" "${e_eps:-?}"
if [[ -n "$b_eps" && -n "$e_eps" ]]; then
    pct=$(python3 -c "print(f'{($e_eps - $b_eps) / $b_eps * 100:+.1f}%')")
    printf " %10s\n" "$pct"
else
    printf " %10s\n" "?"
fi

printf "%-25s %12s %12s %10s\n" "Duration" "${b_dur:-?}" "${e_dur:-?}" ""
printf "%-25s %12s %12s %10s\n" "Exec time p50" "${b_time:-?}" "${e_time:-?}" ""
printf "%-25s %12s %12s %10s\n" "Collision risk" "${b_collision:-?}%" "${e_collision:-?}%" ""

echo ""
echo "Full logs: /tmp/fuzz_bench_baseline.log, /tmp/fuzz_bench_enhanced.log"
