#!/usr/bin/env python3
"""Benchmark profile cache: compare cold-start vs cached startup times.

Usage:
    python3 tools/bench_cache.py [target] [corpus] [iterations]

Measures wall-clock time from process start until the fuzzing loop begins.
"""
import subprocess
import sys
import time
import os

TARGET = sys.argv[1] if len(sys.argv) > 1 else 'targets/png_read_nosan.so'
CORPUS = sys.argv[2] if len(sys.argv) > 2 else 'corpus/png_read_smt_3'
ITERS = sys.argv[3] if len(sys.argv) > 3 else '200'
BASE = '/home/dclavijo/my_code/fuzzer'

os.chdir(BASE)

# Derive cache path (same logic as TargetProfiler._cache_path)
cache_path = os.path.join(
    os.path.dirname(TARGET),
    os.path.basename(TARGET) + '.profile_cache'
)

def clean():
    """Remove cache and stale SHM segments."""
    if os.path.exists(cache_path):
        os.unlink(cache_path)
        print(f"  Removed: {cache_path}")
    subprocess.run(
        ['sh', '-c', 'for i in /dev/shm/*; do [ -f "$i" ] && echo "$i" | grep -q fuzzer && rm -f "$i"; done 2>/dev/null'],
        capture_output=True,
    )


def run(label, extra_args=None):
    """Time a fuzzer run and return (wall_time, exit_code)."""
    cmd = [
        sys.executable, '-m', 'fuzzer_tool', 'fuzz', TARGET,
        '-c', '-d', CORPUS,
        '-n', ITERS,
        '--stats-interval', '99999',
    ]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n  Command: {' '.join(cmd)}")
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0
    return wall, r.stdout, r.stderr, r.returncode


def extract_times(output):
    """Extract timing info from fuzzer output."""
    lines = output.split('\n')
    results = {}
    for line in lines:
        if 'Profile:' in line and 'functions' in line:
            results['profile_line'] = line.strip()
        if 'Raw target speed:' in line:
            results['raw_speed'] = line.strip()
        if 'execs:' in line and 'eps:' in line:
            results['progress'] = line.strip()
        if 'RUN SUMMARY' in line:
            results['summary'] = True
        if 'Duration:' in line and 's' in line:
            results['duration'] = line.strip()
        if 'Avg eps:' in line:
            results['avg_eps'] = line.strip()
    return results


# ── Benchmark ────────────────────────────────────────────────────────────
print("=" * 72)
print(" PROFILE CACHE BENCHMARK")
print(f" Target: {TARGET}")
print(f" Corpus: {CORPUS} ({len(os.listdir(os.path.join(CORPUS, 'seeds')))} seeds)"
      if os.path.isdir(os.path.join(CORPUS, 'seeds')) else "")
print(f" Iterations: {ITERS}")
print("=" * 72)

# Run 1: fresh (no cache), 3 trials
print("\n─── RUN 1: COLD START (no cache) ──────────────────────────────────")
times_cold = []
for i in range(3):
    clean()
    wall, out, err, rc = run(f"cold-{i}")
    times_cold.append(wall)
    meta = extract_times(out)
    print(f"  Trial {i+1}: {wall:.2f}s  |  {meta.get('profile_line', '?')}")
    print(f"             {meta.get('raw_speed', '?')}")
    print(f"             {'rc=' + str(rc) if rc else ''}")

# Run 2: cached (second run creates cache, third run uses it)
print("\n─── RUN 2: WARM CACHE ──────────────────────────────────────────────")
times_warm = []
for i in range(3):
    clean_shm_only = subprocess.run(
        ['sh', '-c', 'for i in /dev/shm/*; do [ -f "$i" ] && echo "$i" | grep -q fuzzer && rm -f "$i"; done 2>/dev/null'],
        capture_output=True,
    )
    # First invocation creates cache; 2nd and 3rd use it
    wall, out, err, rc = run(f"warm-{i}")
    times_warm.append(wall)
    meta = extract_times(out)
    print(f"  Trial {i+1}: {wall:.2f}s  |  {meta.get('profile_line', '?')}")
    print(f"             {meta.get('raw_speed', '?')}")
    print(f"             {'rc=' + str(rc) if rc else ''}")

# Run 3: cached (force cache hit, using --resume on existing state)
print("\n─── RUN 3: CACHED + RESUME ─────────────────────────────────────────")
times_resume = []
for i in range(3):
    clean_shm_only2 = subprocess.run(
        ['sh', '-c', 'for i in /dev/shm/*; do [ -f "$i" ] && echo "$i" | grep -q fuzzer && rm -f "$i"; done 2>/dev/null'],
        capture_output=True,
    )
    wall, out, err, rc = run(f"resume-{i}", extra_args=['--resume'])
    times_resume.append(wall)
    meta = extract_times(out)
    print(f"  Trial {i+1}: {wall:.2f}s  |  {meta.get('profile_line', '?')}")
    print(f"             {meta.get('raw_speed', '?')}")
    print(f"             {'rc=' + str(rc) if rc else ''}")


# ── Results ──────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print(" RESULTS")
print("=" * 72)
print()
print(f"  {'Mode':<20s} {'Mean':>8s}  {'Min':>8s}  {'Max':>8s}  {'Trials':>6s}")
print(f"  {'-'*20} {'-'*8}  {'-'*8}  {'-'*8}  {'-'*6}")

def stats(vals):
    return sum(vals)/len(vals), min(vals), max(vals)

m_cold, mn_cold, mx_cold = stats(times_cold)
m_warm, mn_warm, mx_warm = stats(times_warm)
m_res,  mn_res,  mx_res  = stats(times_resume)

print(f"  {'Cold (no cache)':<20s} {m_cold:>8.2f}s {mn_cold:>8.2f}s {mx_cold:>8.2f}s {len(times_cold):>6d}")
print(f"  {'Warm (cache hit)':<20s} {m_warm:>8.2f}s {mn_warm:>8.2f}s {mx_warm:>8.2f}s {len(times_warm):>6d}")
print(f"  {'Cached + resume':<20s} {m_res:>8.2f}s {mn_res:>8.2f}s {mx_res:>8.2f}s {len(times_resume):>6d}")

if m_cold > 0:
    speedup_cache = (m_cold / m_warm - 1) * 100
    speedup_resume = (m_cold / m_res - 1) * 100
    print(f"\n  Cold → Cache hit: {speedup_cache:.0f}% faster")
    print(f"  Cold → Resume:    {speedup_resume:.0f}% faster")
    print(f"  Cache saves ~{m_cold - m_warm:.1f}s per run ({(m_cold - m_warm):.1f}s)")

# Cache file size
if os.path.exists(cache_path):
    size_kb = os.path.getsize(cache_path) / 1024
    print(f"\n  Cache file: {cache_path} ({size_kb:.1f} KB)")

print()
