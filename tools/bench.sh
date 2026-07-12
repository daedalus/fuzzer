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
#   enhanced:  elo + meta-elo + bandit + mopt
#   enhanced+: elo + meta-elo + bandit + mopt + markov + replicator + shapley
#              + renyi + transfer-entropy + grammar
#   optimal:   elo + mopt + replicator + markov (ensemble 0,1,2,3) + markov-gen
#              Best edges at -n 1k (74 vs 61 baseline) and -n 10k (184 vs 167 baseline)

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

# ── SHM cleanup ───────────────────────────────────────────────────────
# Remove all orphaned SHM segments owned by the current user.
# Previous fuzzer runs (especially those killed by signals) leave
# segments behind. Accumulation can cause shmget to fail or the
# target to attach to stale segments.
cleanup_shm() {
    local before
    before=$(ipcs -m 2>/dev/null | grep -c "$(whoami)" || true)
    # Remove all segments owned by current user
    ipcs -m 2>/dev/null | grep "$(whoami)" | awk '{print $2}' | while read -r shmid; do
        ipcrm -m "$shmid" 2>/dev/null || true
    done
    local after
    after=$(ipcs -m 2>/dev/null | grep -c "$(whoami)" || true)
    if [[ "$before" -gt 0 ]]; then
        echo "[*] Cleaned $((before - after)) orphaned SHM segments ($before -> $after)"
    fi
}

# ── SHM verification ──────────────────────────────────────────────────
# After a fuzzer run, verify that the SHM bitmap actually received data.
# This is more reliable than checking log messages — it checks the
# actual SHM segment that was created during the run.
verify_shm() {
    local log="$1"
    local label="$2"

    # Extract the SHM ID from the log
    local shm_id
    shm_id=$(grep -oP "SHM bitmap, id=\K[0-9]+" "$log" | tail -1)

    if [[ -z "$shm_id" ]]; then
        echo "FAIL: $label — no SHM ID found in log (coverage not enabled?)"
        return 1
    fi

    # Try to attach and check if bitmap has any non-zero bytes
    local has_data
    has_data=$(python3 -c "
import ctypes, ctypes.util
libc = ctypes.CDLL(ctypes.util.find_library('c') or 'libc.so.6', use_errno=True)
libc.shmat.restype = ctypes.c_void_p
ptr = libc.shmat($shm_id, None, 0)
if ptr is None or ptr == -1:
    print('FAIL')
else:
    size = 4096  # default map size
    bitmap = (ctypes.c_uint8 * size).from_address(ptr)
    non_zero = sum(1 for i in range(size) if bitmap[i] != 0)
    libc.shmdt(ptr)
    if non_zero > 0:
        print(f'OK:{non_zero}')
    else:
        print('EMPTY')
" 2>/dev/null)

    if [[ "$has_data" == FAIL ]]; then
        echo "FAIL: $label — SHM segment $shm_id could not be attached"
        return 1
    elif [[ "$has_data" == EMPTY ]]; then
        echo "FAIL: $label — SHM segment $shm_id has 0 non-zero bytes (coverage-blind)"
        return 1
    else
        local nedges="${has_data#OK:}"
        echo "[+] $label — SHM verified: $nedges non-zero bytes in bitmap"
        return 0
    fi
}

# ── Coverage-attachment sanity check ──────────────────────────────────
# Combine log-based and SHM-based checks for maximum reliability.
check_coverage() {
    local log="$1"
    local label="$2"

    # Check for explicit SHM failure messages in the log
    if grep -qi "SHM not attached\|AFL shim area is NULL\|shmat.*failed\|Coverage data will be empty" "$log"; then
        echo "FAIL: $label — SHM coverage did not attach (coverage-blind run)"
        return 1
    fi

    # Verify actual SHM bitmap has data
    if ! verify_shm "$log" "$label"; then
        return 1
    fi

    return 0
}

# ── Run with retry ────────────────────────────────────────────────────
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

        echo "[*] Coverage did not attach. Cleaning SHM and retrying..."
        cleanup_shm
        sleep 2
        attempt=$((attempt + 1))
    done

    echo "FAIL: Coverage failed to attach after $MAX_RETRIES attempts."
    echo "  Last log: $log"
    return 1
}

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
echo "[*] Running enhanced (elo + meta-elo + bandit + mopt${EXTRA_FLAGS:+$EXTRA_FLAGS})..."
run_with_retry /tmp/fuzz_bench_enhanced.log \
    fuzz "$TARGET" -d "$ENHANCED_DIR" -c -n "$ITERS" --elo --meta-elo --mc-bandit --mopt $EXTRA_FLAGS $REPORT_FLAG
echo ""

# Clean SHM between runs
cleanup_shm
sleep 1

# Run enhanced+ (markov + replicator + shapley + renyi + transfer-entropy + grammar)
echo "[*] Running enhanced+ (all enhanced + markov + replicator + shapley + renyi + transfer-entropy + grammar)..."
run_with_retry /tmp/fuzz_bench_enhanced+.log \
    fuzz "$TARGET" -d "$ENHANCEDP_DIR" -c -n "$ITERS" \
    --elo --meta-elo --mc-bandit --mopt \
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

extract() {
    grep -oP "$1" "$2" | tail -1
}

# Extract CI values from crash/timeout rate lines (format: "rate%  ±1σ: lo% ±2σ: lo% ±3σ: lo%")
extract_ci() {
    local log="$1"
    local pattern="$2"
    local line
    line=$(grep -P "$pattern" "$log" 2>/dev/null | tail -1)
    if [[ -z "$line" ]]; then
        echo "  -  -  -"
        return
    fi
    # Extract the three CI values: ±1σ, ±2σ, ±3σ
    local ci1 ci2 ci3
    ci1=$(echo "$line" | grep -oP '±1σ:\s+\K[0-9.]+')
    ci2=$(echo "$line" | grep -oP '±2σ:\s+\K[0-9.]+')
    ci3=$(echo "$line" | grep -oP '±3σ:\s+\K[0-9.]+')
    echo "${ci1:--} ${ci2:--} ${ci3:--}"
}

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

# Extract CI for crash rates
b_crash_ci=$(extract_ci /tmp/fuzz_bench_baseline.log "Crash rate:")
e_crash_ci=$(extract_ci /tmp/fuzz_bench_enhanced.log "Crash rate:")
p_crash_ci=$(extract_ci /tmp/fuzz_bench_enhanced+.log "Crash rate:")
o_crash_ci=$(extract_ci /tmp/fuzz_bench_optimal.log "Crash rate:")

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
