#!/usr/bin/env bash
# Phase 2 of the sweep — remaining fast combinations.
set -euo pipefail

TARGET="targets/png_read"
ITERS=1000
BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="/tmp/fuzz_sweep_results"
DICT="-D dictionaries/png.dict"
GRAMMAR="-g dictionaries/png.gram"
REPORT_FLAG="${BENCH_REPORT:-}"

cd "$BASE_DIR"

cleanup_shm() {
    ipcs -m 2>/dev/null | grep "$(whoami)" | awk '{print $2}' | while read -r shmid; do
        ipcrm -m "$shmid" 2>/dev/null || true
    done
}

extract() {
    grep -oP "$1" "$2" 2>/dev/null | tail -1
}

extract_ci() {
    local log="$1"
    local line
    line=$(grep -P "Crash rate:" "$log" 2>/dev/null | tail -1)
    if [[ -z "$line" ]]; then
        echo "-|-|-"
        return
    fi
    local ci1 ci2 ci3
    ci1=$(echo "$line" | grep -oP '±1σ:\s+\K[0-9.]+')
    ci2=$(echo "$line" | grep -oP '±2σ:\s+\K[0-9.]+')
    ci3=$(echo "$line" | grep -oP '±3σ:\s+\K[0-9.]+')
    echo "${ci1:--}|${ci2:--}|${ci3:--}"
}

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

echo "=== Phase 5: Game theory additions ==="
run_combo "gt1_sched_shapley" --elo --mc-bandit --shapley
run_combo "gt2_sched_rep_shapley" --elo --mc-bandit --replicator --shapley
run_combo "gt3_sched_all_game" --elo --meta-elo --mc-bandit --replicator --shapley

echo ""
echo "=== Phase 6: Full combinations (top candidates) ==="
run_combo "f1_enhanced" --elo --meta-elo --mc-bandit --mopt
run_combo "f2_enhanced_plus" --elo --meta-elo --mc-bandit --mopt \
    --markov --markov-gen --markov-order 0,1,2,3 \
    --replicator --shapley --renyi-weight
run_combo "f3_lean_best" --elo --mc-bandit --markov --markov-gen --markov-order 0,1,2,3 --renyi-weight
run_combo "f5_elo_markov_renyi" --elo --markov --markov-gen --markov-order 0,1,2,3 --renyi-weight
run_combo "f6_bandit_markov_renyi" --mc-bandit --markov --markov-gen --markov-order 0,1,2,3 --renyi-weight
run_combo "f7_elo_bandit_markov_renyi" --elo --mc-bandit --markov --markov-gen --markov-order 0,1,2,3 --renyi-weight
run_combo "f8_elo_meta_bandit_markov_rep_shapley" --elo --meta-elo --mc-bandit --markov --markov-gen --markov-order 0,1,2,3 --replicator --shapley
run_combo "f9_elo_mopt_markov_renyi" --elo --mopt --markov --markov-gen --markov-order 0,1,2,3 --renyi-weight
run_combo "f10_elo_bandit_markov_shapley" --elo --mc-bandit --markov --markov-gen --markov-order 0,1,2,3 --shapley
run_combo "f11_elo_bandit_rep_markov_renyi" --elo --mc-bandit --replicator --markov --markov-gen --markov-order 0,1,2,3 --renyi-weight
run_combo "f12_elo_bandit_markov_renyi_shapley" --elo --mc-bandit --markov --markov-gen --markov-order 0,1,2,3 --renyi-weight --shapley
run_combo "f14_elo_mopt_rep_markov" --elo --mopt --replicator --markov --markov-gen --markov-order 0,1,2,3
run_combo "f15_elo_bandit_markov_mi_renyi" --elo --mc-bandit --markov --markov-gen --markov-order 0,1,2,3 --mi-guided --renyi-weight
run_combo "f16_mopt_markov_renyi_shapley" --mopt --markov --markov-gen --markov-order 0,1,2,3 --renyi-weight --shapley
run_combo "f17_elo_mopt_rep_markov_renyi" --elo --mopt --replicator --markov --markov-gen --markov-order 0,1,2,3 --renyi-weight
run_combo "f18_elo_meta_mopt_markov_renyi" --elo --meta-elo --mopt --markov --markov-gen --markov-order 0,1,2,3 --renyi-weight

echo ""
echo "=== Phase 7: Advanced tuning ==="
run_combo "t1_pairwise0" --elo --mc-bandit --pairwise-blend 0.0
run_combo "t2_pairwise25" --elo --mc-bandit --pairwise-blend 0.25
run_combo "t3_pairwise50" --elo --mc-bandit --pairwise-blend 0.5
run_combo "t4_pairwise75" --elo --mc-bandit --pairwise-blend 0.75
run_combo "t5_anneal250" --elo --mc-bandit --anneal-budget 250
run_combo "t6_anneal1000" --elo --mc-bandit --anneal-budget 1000
run_combo "t7_best_blend" --elo --mc-bandit --pairwise-blend 0.25 --markov --markov-gen --markov-order 0,1,2,3 --renyi-weight
run_combo "t8_best_blend_rep" --elo --mc-bandit --pairwise-blend 0.25 --markov --markov-gen --markov-order 0,1,2,3 --replicator --renyi-weight

# Final: the very best candidates, run twice for variance
echo ""
echo "=== Final: Top 3 candidates x2 for variance ==="
run_combo "z1_best_a" --elo --mopt --replicator --markov --markov-gen --markov-order 0,1,2,3
run_combo "z1_best_b" --elo --mopt --replicator --markov --markov-gen --markov-order 0,1,2,3
run_combo "z2_second_a" --elo --mopt --markov --markov-gen --markov-order 0,1,2,3
run_combo "z2_second_b" --elo --mopt --markov --markov-gen --markov-order 0,1,2,3
run_combo "z3_elo_meta_bandit_a" --elo --meta-elo --mc-bandit
run_combo "z3_elo_meta_bandit_b" --elo --meta-elo --mc-bandit
run_combo "z4_elo_mopt_renyi_a" --elo --mopt --markov --markov-gen --markov-order 0,1,2,3 --renyi-weight
run_combo "z4_elo_mopt_renyi_b" --elo --mopt --markov --markov-gen --markov-order 0,1,2,3 --renyi-weight

echo ""
echo "============================================================"
echo " FINAL RESULTS SORTED BY EDGES (descending)"
echo "============================================================"
echo ""
sort -t, -k2 -rn "$RESULTS_DIR/sweep.csv" | head -40
echo ""
echo "Full results: $RESULTS_DIR/sweep.csv"
