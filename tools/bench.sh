#!/usr/bin/env bash
# Benchmark baseline vs enhanced fuzzer configurations on a target.
#
# Usage:
#   tools/bench.sh [target] [iterations] [extra_enhanced_flags]
#
# Defaults: targets/png_read, 5000 iterations
# Example:  tools/bench.sh targets/png_read 3000 "--sensitivity"

set -euo pipefail

TARGET="${1:-targets/png_read}"
ITERS="${2:-5000}"
EXTRA_FLAGS="${3:-}"
BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BASELINE_DIR="/tmp/fuzz_bench_baseline"
ENHANCED_DIR="/tmp/fuzz_bench_enhanced"

cd "$BASE_DIR"

# ── Coverage-attachment sanity check ──────────────────────────────────
# Verify that the SHM bitmap actually receives data before trusting
# any "edges discovered" numbers. Without this, a silently-broken
# SHM attachment produces 0 edges that look like legitimate "no coverage
# found" rather than "coverage not attached."
check_coverage() {
    local log="$1"
    local label="$2"

    # Check for explicit SHM failure messages
    if grep -qi "SHM not attached\|AFL shim area is NULL\|shmat.*failed\|Coverage data will be empty" "$log"; then
        echo "FAIL: $label — SHM coverage did not attach (coverage-blind run)"
        echo "  This run produced meaningless edge counts. Re-run or investigate SHM setup."
        return 1
    fi

    # Check for shm-edges = 0 when the run completed (not just initial seed replay)
    local edges
    edges=$(grep -oP "Edges discovered:\s+\K[0-9]+" "$log" | tail -1)
    local execs
    execs=$(grep -oP "Executions:\s+\K[0-9]+" "$log" | tail -1)

    if [[ -n "$execs" && "$execs" -gt 500 && "${edges:-0}" -eq 0 ]]; then
        echo "WARN: $label — 0 edges after ${execs} executions. Coverage may not be attached."
        echo "  SHM bitmap received no writes. This likely means instrumentation failed."
        return 1
    fi

    return 0
}

# ── Run with retry ────────────────────────────────────────────────────
# Run fuzzer and retry up to MAX_RETRIES times if coverage fails to attach.
# Intermittent SHM failures happen under resource pressure.
MAX_RETRIES=3

run_with_retry() {
    local log="$1"
    shift
    local attempt=1

    while [[ $attempt -le $MAX_RETRIES ]]; do
        echo "[*] Attempt $attempt/$MAX_RETRIES..."
        python -m fuzzer_tool "$@" 2>&1 | tee "$log"

        if check_coverage "$log" "attempt $attempt"; then
            return 0
        fi

        echo "[*] Coverage did not attach. Retrying (attempt $((attempt+1))/$MAX_RETRIES)..."
        attempt=$((attempt + 1))
        # Brief pause to let IPC resources settle
        sleep 1
    done

    echo "FAIL: Coverage failed to attach after $MAX_RETRIES attempts."
    echo "  Last log: $log"
    return 1
}

# Clean previous runs
rm -rf "$BASELINE_DIR" "$ENHANCED_DIR"
mkdir -p "$BASELINE_DIR" "$ENHANCED_DIR"

echo "============================================================"
echo " Benchmark: baseline vs enhanced"
echo " Target:    $TARGET"
echo " Iterations: $ITERS"
echo " Extra flags: ${EXTRA_FLAGS:-none}"
echo "============================================================"
echo ""

# Run baseline (no features) with coverage-attachment check
echo "[*] Running baseline (no features)..."
run_with_retry /tmp/fuzz_bench_baseline.log \
    fuzz "$TARGET" -d "$BASELINE_DIR" -c -n "$ITERS"
echo ""

# Run enhanced (all features) with coverage-attachment check
echo "[*] Running enhanced (elo + meta-elo + bandit + mopt${EXTRA_FLAGS:+$EXTRA_FLAGS})..."
run_with_retry /tmp/fuzz_bench_enhanced.log \
    fuzz "$TARGET" -d "$ENHANCED_DIR" -c -n "$ITERS" --elo --meta-elo --mc-bandit --mopt $EXTRA_FLAGS
echo ""

# ── Extract metrics ───────────────────────────────────────────────────
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
printf "%-25s %12s %12s %10s\n" "Collision risk" "${b_collision:-0}%" "${e_collision:-0}%" ""

echo ""
echo "Full logs: /tmp/fuzz_bench_baseline.log, /tmp/fuzz_bench_enhanced.log"
