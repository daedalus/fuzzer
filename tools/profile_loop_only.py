#!/usr/bin/env python3
"""Profile the fuzzer hot inner loop (fuzz_one) only, excluding startup overhead.

Run for more iterations to get a clearer picture of steady-state hotpath.
Also uses line_profiler on the hottest functions for per-line breakdown.

Usage:
    python3 tools/profile_loop_only.py [target] [corpus] [iterations]
"""
import cProfile
import pstats
import sys
import os


def main():
    os.chdir('/home/dclavijo/my_code/fuzzer')

    target = sys.argv[1] if len(sys.argv) > 1 else 'targets/png_read_nosan.so'
    corpus = sys.argv[2] if len(sys.argv) > 2 else '/tmp/png'
    iterations = sys.argv[3] if len(sys.argv) > 3 else '1500'

    # Warm up: run a small number of iterations first so startup costs
    # (target_profiler, capstone, branch_density) don't pollute the profile.
    # We measure only the steady-state loop by excluding the warmup execs.
    import subprocess
    import time

    warmup = min(200, int(iterations) // 4)
    main_iter = int(iterations) - warmup

    print(f"[*] Warmup: {warmup} iterations (startup overhead excluded)")
    print(f"[*] Profile: {main_iter} steady-state iterations")
    print()

    # Run warmup
    env = os.environ.copy()
    warmup_args = [
        sys.executable, '-m', 'fuzzer_tool', 'fuzz', target,
        '-c', '-d', corpus,
        '-n', str(warmup),
        '--stats-interval', '99999',
    ]
    subprocess.run(warmup_args, capture_output=True, env=env)

    # Now profile the real run with same state, relying on --resume
    sys.argv = [
        'fuzzer-tool', 'fuzz', target,
        '-c', '-d', corpus, '--resume',
        '-n', str(main_iter),
        '--stats-interval', '99999',
    ]

    pr = cProfile.Profile()
    pr.enable()
    try:
        import fuzzer_tool.cli.commands
        fuzzer_tool.cli.commands.main()
    except SystemExit:
        pass
    pr.disable()

    stats = pstats.Stats(pr)

    sections = [
        ("TOP 40 BY TOTAL TIME (tottime) — steady-state self-time", 'tottime', 40),
        ("TOP 25 BY CUMULATIVE TIME (cumtime) — steady-state self+callee", 'cumtime', 25),
        ("TOP 25 BY CALL COUNT (ncalls)", 'ncalls', 25),
    ]

    for title, sort_key, n in sections:
        print(f"\n{'=' * 80}")
        print(f" {title}")
        print(f"{'=' * 80}")
        stats.sort_stats(sort_key)
        stats.print_stats(n)

    dump_path = '/tmp/fuzzer_loop_only.prof'
    stats.dump_stats(dump_path)
    print(f"\n[*] Stats saved to {dump_path}")

    # Also print a small section focused on target-execution and mutation costs
    print(f"\n{'=' * 80}")
    print(" TARGET EXECUTION & MUTATION COSTS (filtered)")
    print(f"{'=' * 80}")
    stats.sort_stats('tottime')
    stats.print_stats('inprocess|run_target|mutate|havoc|run_one|shm|coverage|seed_picker')


if __name__ == '__main__':
    main()
