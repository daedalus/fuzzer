#!/usr/bin/env python3
"""Stress test RandPool for starvation and endurance limits.

Continuously draws from RandPool while monitoring entropy and refill
frequency to find when (and if) the pool starves under sustained load.

Two modes:

1. **exhaust** (default): Draw repeatedly with the entropy gate disabled
   (default), tracking when refills happen and what entropy looks like.
   Establishes the baseline: how many refills per N draws, entropy range,
   and throughput.

2. **block**: Draw with an impossible min_entropy threshold to confirm the
   blocking gate raises RuntimeError at the predicted point, and measure
   how long it survives before starvation is detected.

Usage:
    python3 tools/stress_randpool_exhaustion.py [--mode {exhaust,block}]
    [--min-entropy M] [--target-draws N] [--seed N] [--verbose]

Tab-separated summary:
    mode\tseed\tdraws\trefills\tentropy\tendurance\tduration\tstatus
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fuzzer_tool.core.rand_pool import RandPool

# Simulated fuzzer hotpath workload per iteration (matches bench_randpool.py):
# 1 randrange + 1 choice + 150 randint(0,255) + 50 randint(1,64) +
# 2 sample draws. Note: shuffle(20 els) delegates to numpy C-level, no pool draws.
# Total: 204 pool draws per iteration.
WORKLOAD_PER_ITER = 204


def hotpath_draw(pool: RandPool) -> None:
    """Execute one fuzzer-hotpath workload iteration."""
    buf = 512
    choices = list(range(50))
    pool.randrange(buf)  # 1 draw for select_position (buf=512 > 256 → pool_l)
    pool.choice(choices)  # 1 draw for select_op (50 elts ≤ 256 → pool_l)
    for _ in range(150):
        pool.randint(0, 255)  # 150 draws (width=256 → _m256_l fast path)
    for _ in range(50):
        pool.randint(1, 64)  # 50 draws (width=64 → _pool_l % width)
    pool.sample(range(buf), 2)  # 2 draws via _draw (k==2 fast path)
    buf2 = list(range(20))
    pool.shuffle(buf2)  # n>=8 → numpy C-level shuffle, no pool draws


def run_exhaustion(seed: int, target_draws: int, verbose: bool):
    """Run exhaustion test with entropy gate disabled (default)."""
    pool = RandPool(seed=seed, min_entropy=None)
    start = time.perf_counter()
    remaining = target_draws

    if verbose:
        print(f"  Exhaustion mode: seed={seed}, target={target_draws:,} draws")

    while remaining > 0:
        batch = min(remaining, WORKLOAD_PER_ITER * 100)
        iters = batch // WORKLOAD_PER_ITER
        for _ in range(iters):
            hotpath_draw(pool)
        remaining -= batch

    elapsed = time.perf_counter() - start
    entropy = pool.measure_entropy() if pool.pool_entropy() is not None else 0.0

    # Count refills: every _POOL_ENTRIES draws triggers one
    total_draws = target_draws
    refills = total_draws // 4096

    endurance = total_draws / elapsed if elapsed > 0 else 0.0
    status = "ok"

    if entropy < 0.5:
        status = "low_entropy"

    if verbose:
        print(f"  Refills: {refills:,}")
        print(f"  Final entropy: {entropy:.4f}")
        print(f"  Duration: {elapsed:.2f}s")
        print(f"  Endurance: {endurance:,.0f} draws/s")
        print(f"  Status: {status}")

    return {
        "mode": "exhaust",
        "seed": seed,
        "draws": total_draws,
        "refills": refills,
        "entropy": entropy,
        "endurance": endurance,
        "duration": elapsed,
        "status": status,
    }


def run_block(seed: int, min_entropy: float, target_draws: int, verbose: bool):
    """Run blocking gate test with impossible threshold."""
    threshold = min_entropy if min_entropy > 0 else 99.9
    pool = RandPool(seed=seed, min_entropy=threshold)
    start = time.perf_counter()
    error = None
    draws_done = 0

    if verbose:
        print(f"  Block mode: seed={seed}, min_entropy={threshold}, target={target_draws:,}")

    try:
        iters = target_draws // WORKLOAD_PER_ITER
        for _ in range(iters):
            hotpath_draw(pool)
            draws_done += WORKLOAD_PER_ITER
    except RuntimeError as e:
        error = str(e)

    elapsed = time.perf_counter() - start
    entropy = pool.pool_entropy() or 0.0
    endurance = draws_done / elapsed if elapsed > 0 else 0.0

    status = "block_gate_raises" if error else "ok"

    if verbose:
        print(f"  Draws before failure: {draws_done:,}")
        print(f"  Elapsed: {elapsed:.2f}s")
        print(f"  Error: {error or 'none'}")

    return {
        "mode": "block",
        "seed": seed,
        "draws": draws_done,
        "refills": 0,
        "entropy": entropy,
        "endurance": endurance,
        "duration": elapsed,
        "status": status,
        "error": error,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["exhaust", "block"], default="exhaust", help="Test mode")
    parser.add_argument("--seed", type=int, default=42, help="RandPool seed")
    parser.add_argument("--target-draws", type=int, default=1_000_000, help="Target draw count")
    parser.add_argument(
        "--min-entropy", type=float, default=0.0, help="Min entropy threshold (block mode)"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    print("=" * 80)
    print(" RandPool Exhaustion/Starvation Stress Test")
    print("=" * 80)
    print()

    if args.mode == "exhaust":
        result = run_exhaustion(args.seed, args.target_draws, args.verbose)
    else:
        result = run_block(args.seed, args.min_entropy, args.target_draws, args.verbose)

    # Tab-separated summary for script processing
    error_field = f"\terror={result.get('error', '')}" if result.get("error") else ""
    print(
        f"{result['mode']}\t"
        f"seed={result['seed']}\t"
        f"draws={result['draws']:,}\t"
        f"refills={result['refills']:,}\t"
        f"entropy={result['entropy']:.4f}\t"
        f"endurance={result['endurance']:,.0f}\t"
        f"duration={result['duration']:.2f}s\t"
        f"status={result['status']}"
        f"{error_field}"
    )


if __name__ == "__main__":
    main()
