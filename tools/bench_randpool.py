#!/usr/bin/env python3
"""Microbenchmark: RandPool vs standard random for the fuzzer hotpath.

Simulates the exact random-call pattern seen in the profiler over 2000
fuzz iterations:
  - 20  randrange(n) per iter  (select_position fallback)
  - 20  choice(list)  per iter  (operator selection)
  - 150 randint(0, 255) per iter (mutation operators)
  - 50  randint(1, n)  per iter  (block sizes, etc.)
  - 2   sample(n, 2)  per iter   (swap two distinct positions)
  - 2   shuffle(list) per iter   (region shuffle in havoc)
"""
import time
import random as std_random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.fuzzer_tool.core.rand_pool import RandPool

N_ITERS = 2000
# Average buffer length for position calculations
BUF_LEN = 512
# Number of mutations per input
N_MUTATIONS = 20

# Pre-build a list for choice() tests
CHOICE_POOL = list(range(50))


def bench_standard(iters):
    """Standard random.randint/randrange/choice calls."""
    std_random.seed(42)
    total = 0
    for _ in range(iters):
        # select_position fallback: 1 randrange per op
        for _ in range(N_MUTATIONS):
            total += std_random.randrange(BUF_LEN)
        # select_op choice
        for _ in range(N_MUTATIONS):
            total += std_random.choice(CHOICE_POOL)
        # mutation operators: byte values, positions, sizes
        for _ in range(150):
            total += std_random.randint(0, 255)
        for _ in range(50):
            total += std_random.randint(1, 64)
        # sample & shuffle
        for _ in range(2):
            i, j = std_random.sample(range(BUF_LEN), 2)
            total += i + j
        for _ in range(2):
            buf = list(range(20))
            std_random.shuffle(buf)
            total += buf[0]
    return total


def bench_pool(iters):
    """RandPool batched calls."""
    pool = RandPool()
    total = 0
    for _ in range(iters):
        for _ in range(N_MUTATIONS):
            total += pool.randrange(BUF_LEN)
        for _ in range(N_MUTATIONS):
            total += pool.choice(CHOICE_POOL)
        for _ in range(150):
            total += pool.randint(0, 255)
        for _ in range(50):
            total += pool.randint(1, 64)
        for _ in range(2):
            i, j = pool.sample(BUF_LEN, 2)
            total += i + j
        for _ in range(2):
            buf = list(range(20))
            pool.shuffle(buf)
            total += buf[0]
    return total


def bench_standard_compact(iters):
    """Standard random, but using randrange(n) directly instead of randint(0, n-1)."""
    std_random.seed(42)
    total = 0
    for _ in range(iters):
        for _ in range(N_MUTATIONS):
            total += std_random.randrange(BUF_LEN)
        for _ in range(N_MUTATIONS):
            total += std_random.choice(CHOICE_POOL)
        for _ in range(150):
            total += std_random.randrange(256)
        for _ in range(50):
            total += std_random.randrange(1, 65)
        for _ in range(2):
            i, j = std_random.sample(range(BUF_LEN), 2)
            total += i + j
        for _ in range(2):
            buf = list(range(20))
            std_random.shuffle(buf)
            total += buf[0]
    return total


# ── Raw throughput (single call type) ──────────────────────────────────
print("=" * 72)
print(" RANDPOOL MICROBENCHMARK")
print("=" * 72)
print()
print(f"  Iterations: {N_ITERS} (simulating 2000 fuzz_one cycles)")
print(f"  Total randrange calls: {N_ITERS * N_MUTATIONS:,}")
print(f"  Total choice calls:    {N_ITERS * N_MUTATIONS:,}")
print(f"  Total randint calls:   {N_ITERS * 200:,}")
print(f"  Total sample/shuffle:  {N_ITERS * 4:,}")

# Warmup caches / JIT
bench_standard(100)
bench_pool(100)
bench_standard_compact(100)

# Full benchmark
trials = 5
results_std = []
results_pool = []
results_stdc = []

for t in range(trials):
    t0 = time.perf_counter()
    r1 = bench_standard(N_ITERS)
    t1 = time.perf_counter()
    results_std.append(t1 - t0)

    t0 = time.perf_counter()
    r2 = bench_pool(N_ITERS)
    t1 = time.perf_counter()
    results_pool.append(t1 - t0)

    t0 = time.perf_counter()
    r3 = bench_standard_compact(N_ITERS)
    t1 = time.perf_counter()
    results_stdc.append(t1 - t0)

    # Verify correctness (deterministic seed)
    assert r1 == r3, f"standard and compact produced different results: {r1} vs {r3}"

def mean(vals):
    return sum(vals) / len(vals)

m_std = mean(results_std)
m_pool = mean(results_pool)
m_stdc = mean(results_stdc)

print(f"\n  {'Method':<30s} {'Mean':>10s}  {'Speedup':>10s}")
print(f"  {'-'*30} {'-'*10}  {'-'*10}")
print(f"  {'Standard randint/randrange/choice':<30s} {m_std:>8.3f}s  {'1.00×':>10s}")
print(f"  {'Standard (randrange only)':<30s} {m_stdc:>8.3f}s  {m_std/m_stdc:>8.2f}×")
print(f"  {'RandPool (batched)':<30s} {m_pool:>8.3f}s  {m_std/m_pool:>8.2f}×")

# ── Per-operation throughput ──────────────────────────────────────────
print(f"\n{'─' * 72}")
print(" PER-OPERATION THROUGHPUT")
print(f"{'─' * 72}")
print()

N = 200_000  # 200K calls per benchmark

def bench_op(label, std_fn, pool_fn):
    # warmup
    for _ in range(10_000):
        std_fn()
        pool_fn()

    t0 = time.perf_counter()
    for _ in range(N):
        std_fn()
    t1 = time.perf_counter()
    std_time = t1 - t0

    t0 = time.perf_counter()
    for _ in range(N):
        pool_fn()
    t1 = time.perf_counter()
    pool_time = t1 - t0

    ns_std = (std_time / N) * 1e9
    ns_pool = (pool_time / N) * 1e9
    speedup = std_time / pool_time if pool_time > 0 else float('inf')

    print(f"  {label:<30s}  std={ns_std:>6.0f}ns  pool={ns_pool:>6.0f}ns  {speedup:>5.1f}×")

buf_512 = list(range(512))
_pool_inst = RandPool()
_rand_inst = type('_', (), {'randrange': lambda _, n: std_random.randrange(n)})()
_choice_seq = list(range(50))

def _std_choice():
    return std_random.choice(_choice_seq)

def _pool_choice():
    return _pool_inst.choice(_choice_seq)

bench_op("randrange(512)",       lambda: std_random.randrange(512), lambda: _pool_inst.randrange(512))
bench_op("randint(0, 255)",      lambda: std_random.randint(0, 255), lambda: _pool_inst.randint(0, 255))
bench_op("randint(1, 64)",       lambda: std_random.randint(1, 64), lambda: _pool_inst.randint(1, 64))
bench_op("choice(50 els)",       _std_choice, _pool_choice)
bench_op("sample(512, 2)",       lambda: std_random.sample(range(512), 2), lambda: _pool_inst.sample(512, 2))
bench_op("shuffle(20 els)",      lambda: std_random.shuffle(list(range(20))), lambda: _pool_inst.shuffle(list(range(20))))

print()