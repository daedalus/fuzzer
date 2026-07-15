#!/usr/bin/env bash
# Benchmark baseline vs enhanced vs enhanced+ vs optimal fuzzer configurations on a target.
#
# Usage:
#   tools/bench.sh [target] [iterations] [extra_enhanced_flags]
#
# Defaults: targets/png_read, 5000 iterations
# Example:  tools/bench.sh targets/png_read 3000 "--sensitivity"
#
# Configurations:
#   baseline:  no features
#   enhanced:  elo + bandit + mopt
#   enhanced+: elo + bandit + mopt + markov + replicator + shapley
#              + renyi + transfer-entropy + grammar
#   optimal:   elo + mopt + replicator + markov (ensemble 0,1,2,3) + markov-gen
#              Best edges at -n 1k (74 vs 61 baseline) and -n 10k (184 vs 167 baseline)
#
# For a broad sweep of individual feature/combination effects instead of
# these four named configurations, use tools/bench_sweep.sh.

set -euo pipefail

TARGET="${1:-targets/png_read}"
ITERS="${2:-5000}"
EXTRA_FLAGS="${3:-}"
BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BASELINE_DIR="/tmp/fuzz_bench_baseline"
ENHANCED_DIR="/tmp/fuzz_bench_enhanced"
ENHANCEDP_DIR="/tmp/fuzz_bench_enhanced+"
OPTIMAL_DIR="/tmp/fuzz_bench_optimal"
REPORT_FLAG="${BENCH_REPORT:-}"  # set BENCH_REPORT=--report to generate full reports

cd "$BASE_DIR"
# shellcheck source=lib/bench_common.sh
source "$BASE_DIR/tools/lib/bench_common.sh"

# ── Main ──────────────────────────────────────────────────────────────

# Clean previous runs and orphaned SHM
rm -rf "$BASELINE_DIR" "$ENHANCED_DIR" "$ENHANCEDP_DIR" "$OPTIMAL_DIR"
mkdir -p "$BASELINE_DIR" "$ENHANCED_DIR" "$ENHANCEDP_DIR" "$OPTIMAL_DIR"
cleanup_shm

echo "============================================================"
echo " Benchmark: baseline vs enhanced vs enhanced+ vs optimal"
echo " Target:    $TARGET"
echo " Iterations: $ITERS"
echo " Extra flags: ${EXTRA_FLAGS:-none}"
echo "============================================================"
echo ""

# Run baseline
echo "[*] Running baseline (no features)..."
run_with_retry /tmp/fuzz_bench_baseline.log \
    fuzz "$TARGET" -d "$BASELINE_DIR" -c -n "$ITERS" $EXTRA_FLAGS $REPORT_FLAG
echo ""

# Clean SHM between runs to prevent stale segment interference
cleanup_shm
sleep 1

# Run enhanced
echo "[*] Running enhanced (elo + bandit + mopt${EXTRA_FLAGS:+$EXTRA_FLAGS})..."
run_with_retry /tmp/fuzz_bench_enhanced.log \
    fuzz "$TARGET" -d "$ENHANCED_DIR" -c -n "$ITERS" --elo --mc-bandit --mopt $EXTRA_FLAGS $REPORT_FLAG
echo ""

# Clean SHM between runs
cleanup_shm
sleep 1

# Run enhanced+ (markov + replicator + shapley + renyi + transfer-entropy + grammar)
echo "[*] Running enhanced+ (all enhanced + markov + replicator + shapley + renyi + transfer-entropy + grammar)..."
run_with_retry /tmp/fuzz_bench_enhanced+.log \
    fuzz "$TARGET" -d "$ENHANCEDP_DIR" -c -n "$ITERS" \
    --elo --mc-bandit --mopt \
    --markov --markov-gen --markov-order 0,1,2,3 \
    --replicator --shapley --renyi-weight --transfer-entropy \
    -g dictionaries/png.gram \
    $EXTRA_FLAGS $REPORT_FLAG
echo ""

# Clean SHM between runs
cleanup_shm
sleep 1

# Run optimal (elo + mopt + replicator + markov ensemble + markov-gen)
# Sweep-validated: 74 edges at -n 1k, 184 edges at -n 10k (vs 167 baseline, 172 enhanced+).
echo "[*] Running optimal (elo + mopt + replicator + markov ensemble)..."
run_with_retry /tmp/fuzz_bench_optimal.log \
    fuzz "$TARGET" -d "$OPTIMAL_DIR" -c -n "$ITERS" \
    --elo --mopt --replicator \
    --markov --markov-gen --markov-order 0,1,2,3 \
    $EXTRA_FLAGS $REPORT_FLAG
echo ""

# ── Extract metrics ───────────────────────────────────────────────────
echo "============================================================"
echo " COMPARISON"
echo "============================================================"
echo ""

b_edges=$(extract "Edges discovered:\s+\K[0-9]+" /tmp/fuzz_bench_baseline.log)
e_edges=$(extract "Edges discovered:\s+\K[0-9]+" /tmp/fuzz_bench_enhanced.log)
p_edges=$(extract "Edges discovered:\s+\K[0-9]+" /tmp/fuzz_bench_enhanced+.log)
o_edges=$(extract "Edges discovered:\s+\K[0-9]+" /tmp/fuzz_bench_optimal.log)
b_corpus=$(extract "Corpus:\s+\K[0-9]+" /tmp/fuzz_bench_baseline.log)
e_corpus=$(extract "Corpus:\s+\K[0-9]+" /tmp/fuzz_bench_enhanced.log)
p_corpus=$(extract "Corpus:\s+\K[0-9]+" /tmp/fuzz_bench_enhanced+.log)
o_corpus=$(extract "Corpus:\s+\K[0-9]+" /tmp/fuzz_bench_optimal.log)
b_eps=$(extract "Avg eps:\s+\K[0-9.]+" /tmp/fuzz_bench_baseline.log)
e_eps=$(extract "Avg eps:\s+\K[0-9.]+" /tmp/fuzz_bench_enhanced.log)
p_eps=$(extract "Avg eps:\s+\K[0-9.]+" /tmp/fuzz_bench_enhanced+.log)
o_eps=$(extract "Avg eps:\s+\K[0-9.]+" /tmp/fuzz_bench_optimal.log)
b_dur=$(extract "Duration:\s+\K[0-9s]+" /tmp/fuzz_bench_baseline.log)
e_dur=$(extract "Duration:\s+\K[0-9s]+" /tmp/fuzz_bench_enhanced.log)
p_dur=$(extract "Duration:\s+\K[0-9s]+" /tmp/fuzz_bench_enhanced+.log)
o_dur=$(extract "Duration:\s+\K[0-9s]+" /tmp/fuzz_bench_optimal.log)
b_time=$(extract "Exec time p50:\s+\K[0-9.]+ms" /tmp/fuzz_bench_baseline.log)
e_time=$(extract "Exec time p50:\s+\K[0-9.]+ms" /tmp/fuzz_bench_enhanced.log)
p_time=$(extract "Exec time p50:\s+\K[0-9.]+ms" /tmp/fuzz_bench_enhanced+.log)
o_time=$(extract "Exec time p50:\s+\K[0-9.]+ms" /tmp/fuzz_bench_optimal.log)
b_collision=$(extract "Collision risk:\s+\K[0-9.]+" /tmp/fuzz_bench_baseline.log)
e_collision=$(extract "Collision risk:\s+\K[0-9.]+" /tmp/fuzz_bench_enhanced.log)
p_collision=$(extract "Collision risk:\s+\K[0-9.]+" /tmp/fuzz_bench_enhanced+.log)
o_collision=$(extract "Collision risk:\s+\K[0-9.]+" /tmp/fuzz_bench_optimal.log)

# Extract CI for crash rates (space-delimited for direct display in the table below)
b_crash_ci=$(extract_ci /tmp/fuzz_bench_baseline.log "Crash rate:" " ")
e_crash_ci=$(extract_ci /tmp/fuzz_bench_enhanced.log "Crash rate:" " ")
p_crash_ci=$(extract_ci /tmp/fuzz_bench_enhanced+.log "Crash rate:" " ")
o_crash_ci=$(extract_ci /tmp/fuzz_bench_optimal.log "Crash rate:" " ")

printf "%-25s %12s %12s %12s %12s\n" "Metric" "Baseline" "Enhanced" "Enhanced+" "Optimal"
printf "%-25s %12s %12s %12s %12s\n" "-------------------------" "------------" "------------" "------------" "------------"
printf "%-25s %12s %12s %12s %12s\n" "Edges discovered" "${b_edges:-?}" "${e_edges:-?}" "${p_edges:-?}" "${o_edges:-?}"
printf "%-25s %12s %12s %12s %12s\n" "Corpus entries" "${b_corpus:-?}" "${e_corpus:-?}" "${p_corpus:-?}" "${o_corpus:-?}"
printf "%-25s %12s %12s %12s %12s\n" "Avg eps" "${b_eps:-?}" "${e_eps:-?}" "${p_eps:-?}" "${o_eps:-?}"
printf "%-25s %12s %12s %12s %12s\n" "Duration" "${b_dur:-?}" "${e_dur:-?}" "${p_dur:-?}" "${o_dur:-?}"
printf "%-25s %12s %12s %12s %12s\n" "Exec time p50" "${b_time:-?}" "${e_time:-?}" "${p_time:-?}" "${o_time:-?}"
printf "%-25s %12s %12s %12s %12s\n" "Collision risk" "${b_collision:-0}%" "${e_collision:-0}%" "${p_collision:-0}%" "${o_collision:-0}%"

echo ""
echo "Crash rate CI (±1σ ±2σ ±3σ):"
printf "  %-25s %s\n" "Baseline:" "${b_crash_ci:-  -  -}"
printf "  %-25s %s\n" "Enhanced:" "${e_crash_ci:-  -  -}"
printf "  %-25s %s\n" "Enhanced+:" "${p_crash_ci:-  -  -}"
printf "  %-25s %s\n" "Optimal:" "${o_crash_ci:-  -  -}"

echo ""
echo "Full logs: /tmp/fuzz_bench_baseline.log, /tmp/fuzz_bench_enhanced.log, /tmp/fuzz_bench_enhanced+.log, /tmp/fuzz_bench_optimal.log"
if [[ -n "$REPORT_FLAG" ]]; then
    echo "Full reports: /tmp/fuzz_bench_baseline_report.txt, etc."
fi
