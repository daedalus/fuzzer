#!/usr/bin/env bash
# Shared helper functions for tools/bench.sh and tools/bench_sweep.sh.
# Source this file; do not execute it directly.

# ── SHM cleanup ───────────────────────────────────────────────────────
# Remove all orphaned SHM segments owned by the current user.
# Previous fuzzer runs (especially those killed by signals) leave
# segments behind. Accumulation can cause shmget to fail or the
# target to attach to stale segments.
cleanup_shm() {
    local before shmids
    before=$(ipcs -m 2>/dev/null | grep -c "$(whoami)" || true)
    # Capture matching SHM IDs into a variable first: under `set -o pipefail`,
    # piping straight into `while read` would abort the script (via `set -e`)
    # whenever grep finds no matches (the common case with no stale segments).
    shmids=$(ipcs -m 2>/dev/null | grep "$(whoami)" | awk '{print $2}' || true)
    if [[ -n "$shmids" ]]; then
        while read -r shmid; do
            ipcrm -m "$shmid" 2>/dev/null || true
        done <<< "$shmids"
    fi
    local after
    after=$(ipcs -m 2>/dev/null | grep -c "$(whoami)" || true)
    if [[ "$before" -gt 0 ]]; then
        echo "[*] Cleaned $((before - after)) orphaned SHM segments ($before -> $after)"
    fi
}

# ── Metric extraction ──────────────────────────────────────────────────
# Extract the last match of a PCRE pattern from a log file.
extract() {
    grep -oP "$1" "$2" 2>/dev/null | tail -1
}

# Extract CI values from a "Crash rate:" line (format: "rate% ±1σ: lo% ±2σ: lo% ±3σ: lo%").
# Usage: extract_ci <log> [pattern] [delimiter]
#   pattern   defaults to "Crash rate:"
#   delimiter defaults to "|" (used to join the three CI values, e.g. for CSV rows).
#             Pass " " for space-separated output suitable for direct display.
extract_ci() {
    local log="$1"
    local pattern="${2:-Crash rate:}"
    local delim="${3:-|}"
    local line
    line=$(grep -P "$pattern" "$log" 2>/dev/null | tail -1)
    if [[ -z "$line" ]]; then
        printf -- "-%s-%s-\n" "$delim" "$delim"
        return
    fi
    local ci1 ci2 ci3
    ci1=$(echo "$line" | grep -oP '±1σ:\s+\K[0-9.]+')
    ci2=$(echo "$line" | grep -oP '±2σ:\s+\K[0-9.]+')
    ci3=$(echo "$line" | grep -oP '±3σ:\s+\K[0-9.]+')
    printf -- "%s%s%s%s%s\n" "${ci1:--}" "$delim" "${ci2:--}" "$delim" "${ci3:--}"
}

# ── SHM verification ──────────────────────────────────────────────────
# After a fuzzer run, verify that the SHM bitmap actually received data.
# This is more reliable than checking log messages — it checks the
# actual SHM segment that was created during the run.
verify_shm() {
    local log="$1"
    local label="$2"

    local shm_id
    shm_id=$(grep -oP "SHM bitmap, id=\K[0-9]+" "$log" | tail -1)

    if [[ -z "$shm_id" ]]; then
        echo "FAIL: $label — no SHM ID found in log (coverage not enabled?)"
        return 1
    fi

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

    if grep -qi "SHM not attached\|AFL shim area is NULL\|shmat.*failed\|Coverage data will be empty" "$log"; then
        echo "FAIL: $label — SHM coverage did not attach (coverage-blind run)"
        return 1
    fi

    if ! verify_shm "$log" "$label"; then
        return 1
    fi

    return 0
}

# ── Run with retry ────────────────────────────────────────────────────
# Runs `python -m fuzzer_tool "$@"`, verifying coverage attached; retries
# on coverage-blind runs up to MAX_RETRIES (default 3) with SHM cleanup
# between attempts.
BENCH_MAX_RETRIES="${BENCH_MAX_RETRIES:-3}"

run_with_retry() {
    local log="$1"
    shift
    local attempt=1

    while [[ $attempt -le $BENCH_MAX_RETRIES ]]; do
        echo "[*] Attempt $attempt/$BENCH_MAX_RETRIES..."
        python -m fuzzer_tool "$@" 2>&1 | tee "$log"

        if [[ ! -s "$log" ]]; then
            echo "[*] Run produced no log output (crashed before startup, or the log could not be written). Retrying..."
        elif check_coverage "$log" "attempt $attempt"; then
            return 0
        fi

        echo "[*] Coverage did not attach. Cleaning SHM and retrying..."
        cleanup_shm
        sleep 2
        attempt=$((attempt + 1))
    done

    echo "FAIL: Coverage failed to attach after $BENCH_MAX_RETRIES attempts."
    echo "  Last log: $log"
    return 1
}

# ── Sweep combo runner ─────────────────────────────────────────────────
# Runs a single named feature combination for the sweep scripts, appending
# a CSV row to $RESULTS_DIR/sweep.csv. Requires $TARGET, $ITERS, $DICT,
# $GRAMMAR, $REPORT_FLAG, and $RESULTS_DIR to be set by the caller.
run_combo() {
    local name="$1"
    shift
    local flags=("$@")
    local dir="/tmp/fuzz_sweep_${name}"
    local log="$RESULTS_DIR/${name}.log"

    rm -rf "$dir"
    mkdir -p "$dir"
    cleanup_shm

    echo "[*] Running: $name"
    python -m fuzzer_tool fuzz "$TARGET" -d "$dir" -c -n "$ITERS" \
        $DICT $GRAMMAR "${flags[@]}" $REPORT_FLAG 2>&1 | tee "$log" || true

    local edges corpus eps dur p50 coll crash_ci
    edges=$(extract "Edges discovered:\s+\K[0-9]+" "$log")
    corpus=$(extract "Corpus:\s+\K[0-9]+" "$log")
    eps=$(extract "Avg eps:\s+\K[0-9.]+" "$log")
    dur=$(extract "Duration:\s+\K[0-9s]+" "$log")
    p50=$(extract "Exec time p50:\s+\K[0-9.]+ms" "$log")
    coll=$(extract "Collision risk:\s+\K[0-9.]+" "$log")
    crash_ci=$(extract_ci "$log")

    local ci1 ci2 ci3
    ci1=$(echo "$crash_ci" | cut -d'|' -f1)
    ci2=$(echo "$crash_ci" | cut -d'|' -f2)
    ci3=$(echo "$crash_ci" | cut -d'|' -f3)

    echo "${name},${edges:-0},${corpus:-0},${eps:-0},${dur:-0},${p50:-0},${coll:-0},${ci1},${ci2},${ci3}" >> "$RESULTS_DIR/sweep.csv"
    printf "  -> edges=%-5s corpus=%-5s eps=%-8s dur=%-8s crash_ci=[%s,%s,%s]\n" "${edges:-?}" "${corpus:-?}" "${eps:-?}" "${dur:-?}" "${ci1:--}" "${ci2:--}" "${ci3:--}"

    cleanup_shm
    sleep 1
}
