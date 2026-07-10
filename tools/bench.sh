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
rm -rf "$BASELINE_DIR" "$ENHANCED_DIR"
mkdir -p "$BASELINE_DIR" "$ENHANCED_DIR"
cleanup_shm

echo "============================================================"
echo " Benchmark: baseline vs enhanced"
echo " Target:    $TARGET"
echo " Iterations: $ITERS"
echo " Extra flags: ${EXTRA_FLAGS:-none}"
echo "============================================================"
echo ""

# Run baseline
echo "[*] Running baseline (no features)..."
run_with_retry /tmp/fuzz_bench_baseline.log \
    fuzz "$TARGET" -d "$BASELINE_DIR" -c -n "$ITERS"
echo ""

# Clean SHM between runs to prevent stale segment interference
cleanup_shm
sleep 1

# Run enhanced
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
