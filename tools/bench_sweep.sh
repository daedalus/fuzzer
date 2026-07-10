#!/usr/bin/env bash
# Systematic feature combination sweep for -n 1k benchmarks.
# Tests individual features and combinations to find the best configuration.
#
# Usage: tools/bench_sweep.sh

set -euo pipefail

TARGET="targets/png_read"
ITERS=1000
BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="/tmp/fuzz_sweep_results"
DICT="-D dictionaries/png.dict"
GRAMMAR="-g dictionaries/png.gram"

cd "$BASE_DIR"
mkdir -p "$RESULTS_DIR"

# Clean SHM
cleanup_shm() {
    ipcs -m 2>/dev/null | grep "$(whoami)" | awk '{print $2}' | while read -r shmid; do
        ipcrm -m "$shmid" 2>/dev/null || true
    done
}

# Extract metrics from log
extract() {
    grep -oP "$1" "$2" 2>/dev/null | tail -1
}

# Results CSV
echo "combo,edges,corpus,eps,duration,exec_p50,collision" > "$RESULTS_DIR/sweep.csv"

# Run a single combination
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
        $DICT $GRAMMAR "${flags[@]}" 2>&1 | tee "$log" || true

    local edges corpus eps dur p50 coll
    edges=$(extract "Edges discovered:\s+\K[0-9]+" "$log")
    corpus=$(extract "Corpus:\s+\K[0-9]+" "$log")
    eps=$(extract "Avg eps:\s+\K[0-9.]+" "$log")
    dur=$(extract "Duration:\s+\K[0-9s]+" "$log")
    p50=$(extract "Exec time p50:\s+\K[0-9.]+ms" "$log")
    coll=$(extract "Collision risk:\s+\K[0-9.]+" "$log")

    echo "${name},${edges:-0},${corpus:-0},${eps:-0},${dur:-0},${p50:-0},${coll:-0}" >> "$RESULTS_DIR/sweep.csv"
    printf "  -> edges=%-5s corpus=%-5s eps=%-8s dur=%-8s\n" "${edges:-?}" "${corpus:-?}" "${eps:-?}" "${dur:-?}"

    cleanup_shm
    sleep 1
}

echo "============================================================"
echo " Feature Combination Sweep: -n $ITERS on $TARGET"
echo "============================================================"
echo ""

# ── Phase 1: Individual features ──────────────────────────────────
echo "=== Phase 1: Individual features ==="

run_combo "baseline"
run_combo "elo" --elo
run_combo "meta_elo" --elo --meta-elo
run_combo "bandit" --mc-bandit
run_combo "mopt" --mopt
run_combo "markov" --markov --markov-gen --markov-order 0,1,2,3
run_combo "replicator" --replicator
run_combo "shapley" --shapley
run_combo "renyi" --renyi-weight
run_combo "transfer_entropy" --transfer-entropy
run_combo "grammar" # grammar added via $GRAMMAR
run_combo "sensitivity" --sensitivity
run_combo "secretary" --secretary
run_combo "mi_guided" --mi-guided
run_combo "mc_cem" --mc-cem
run_combo "anneal" --anneal-budget 500
run_combo "markov_gen_only" --markov-gen

# ── Phase 2: Scheduling combinations ──────────────────────────────
echo ""
echo "=== Phase 2: Scheduling combinations ==="

run_combo "s1_elo_bandit" --elo --mc-bandit
run_combo "s2_elo_mopt" --elo --mopt
run_combo "s3_elo_meta_bandit" --elo --meta-elo --mc-bandit
run_combo "s4_elo_meta_mopt" --elo --meta-elo --mopt
run_combo "s5_bandit_mopt" --mc-bandit --mopt
run_combo "s6_elo_bandit_replicator" --elo --mc-bandit --replicator
run_combo "s7_elo_mopt_replicator" --elo --mopt --replicator

# ── Phase 3: Generation + scheduling ──────────────────────────────
echo ""
echo "=== Phase 3: Generation + scheduling ==="

run_combo "g1_elo_bandit_markov" --elo --mc-bandit --markov --markov-gen --markov-order 0,1,2,3
run_combo "g2_elo_mopt_markov" --elo --mopt --markov --markov-gen --markov-order 0,1,2,3
run_combo "g3_elo_bandit_rep_markov" --elo --mc-bandit --replicator --markov --markov-gen --markov-order 0,1,2,3
run_combo "g4_elo_mopt_rep_markov" --elo --mopt --replicator --markov --markov-gen --markov-order 0,1,2,3
run_combo "g5_markov_only" --markov --markov-gen --markov-order 0,1,2,3
run_combo "g6_markov_order012" --markov --markov-gen --markov-order 0,1,2
run_combo "g7_markov_order01" --markov --markov-gen --markov-order 0,1
run_combo "g8_markov_order3" --markov --markov-gen --markov-order 3

# ── Phase 4: Information theory + scheduling ──────────────────────
echo ""
echo "=== Phase 4: Information theory additions ==="

run_combo "i1_sched_renyi" --elo --mc-bandit --renyi-weight
run_combo "i2_sched_transfer" --elo --mc-bandit --transfer-entropy
run_combo "i3_sched_renyi_transfer" --elo --mc-bandit --renyi-weight --transfer-entropy
run_combo "i4_sched_mi" --elo --mc-bandit --mi-guided
run_combo "i5_sched_renyi_mi" --elo --mc-bandit --renyi-weight --mi-guided

# ── Phase 5: Game theory additions ────────────────────────────────
echo ""
echo "=== Phase 5: Game theory additions ==="

run_combo "gt1_sched_shapley" --elo --mc-bandit --shapley
run_combo "gt2_sched_rep_shapley" --elo --mc-bandit --replicator --shapley
run_combo "gt3_sched_all_game" --elo --meta-elo --mc-bandit --replicator --shapley

# ── Phase 6: Full combos (best of each) ───────────────────────────
echo ""
echo "=== Phase 6: Full combinations ==="

run_combo "f1_enhanced" --elo --meta-elo --mc-bandit --mopt
run_combo "f2_enhanced_plus" --elo --meta-elo --mc-bandit --mopt \
    --markov --markov-gen --markov-order 0,1,2,3 \
    --replicator --shapley --renyi-weight --transfer-entropy
run_combo "f3_lean_best" --elo --mc-bandit --markov --markov-gen --markov-order 0,1,2,3 --renyi-weight
run_combo "f4_full_kitchen" --elo --meta-elo --mc-bandit --mopt \
    --markov --markov-gen --markov-order 0,1,2,3 \
    --replicator --shapley --renyi-weight --transfer-entropy \
    --mi-guided --sensitivity --secretary --mc-cem --anneal-budget 500
run_combo "f5_elo_markov_renyi" --elo --markov --markov-gen --markov-order 0,1,2,3 --renyi-weight
run_combo "f6_bandit_markov_renyi_transfer" --mc-bandit --markov --markov-gen --markov-order 0,1,2,3 --renyi-weight --transfer-entropy
run_combo "f7_elo_bandit_markov_renyi_transfer" --elo --mc-bandit --markov --markov-gen --markov-order 0,1,2,3 --renyi-weight --transfer-entropy
run_combo "f8_elo_meta_bandit_markov_rep_shapley" --elo --meta-elo --mc-bandit --markov --markov-gen --markov-order 0,1,2,3 --replicator --shapley
run_combo "f9_elo_mopt_markov_renyi_transfer" --elo --mopt --markov --markov-gen --markov-order 0,1,2,3 --renyi-weight --transfer-entropy
run_combo "f10_elo_bandit_markov_shapley_transfer" --elo --mc-bandit --markov --markov-gen --markov-order 0,1,2,3 --shapley --transfer-entropy
run_combo "f11_elo_bandit_rep_markov_renyi" --elo --mc-bandit --replicator --markov --markov-gen --markov-order 0,1,2,3 --renyi-weight
run_combo "f12_elo_bandit_markov_renyi_shapley" --elo --mc-bandit --markov --markov-gen --markov-order 0,1,2,3 --renyi-weight --shapley
run_combo "f13_elo_bandit_markov_transfer_shapley_renyi" --elo --mc-bandit --markov --markov-gen --markov-order 0,1,2,3 --transfer-entropy --shapley --renyi-weight
run_combo "f14_elo_mopt_rep_markov_transfer" --elo --mopt --replicator --markov --markov-gen --markov-order 0,1,2,3 --transfer-entropy
run_combo "f15_elo_bandit_markov_mi_renyi" --elo --mc-bandit --markov --markov-gen --markov-order 0,1,2,3 --mi-guided --renyi-weight

# ── Phase 7: Pairwise blend + annealing variations ────────────────
echo ""
echo "=== Phase 7: Advanced tuning ==="

run_combo "t1_pairwise0" --elo --mc-bandit --pairwise-blend 0.0
run_combo "t2_pairwise25" --elo --mc-bandit --pairwise-blend 0.25
run_combo "t3_pairwise50" --elo --mc-bandit --pairwise-blend 0.5
run_combo "t4_pairwise75" --elo --mc-bandit --pairwise-blend 0.75
run_combo "t5_anneal250" --elo --mc-bandit --anneal-budget 250
run_combo "t6_anneal1000" --elo --mc-bandit --anneal-budget 1000
run_combo "t7_best_blend" --elo --mc-bandit --pairwise-blend 0.25 --markov --markov-gen --markov-order 0,1,2,3 --renyi-weight --transfer-entropy

# ── Summary ───────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " RESULTS SORTED BY EDGES (descending)"
echo "============================================================"
echo ""
sort -t, -k2 -rn "$RESULTS_DIR/sweep.csv" | head -20
echo ""
echo "Full results: $RESULTS_DIR/sweep.csv"
echo "Logs: $RESULTS_DIR/*.log"
