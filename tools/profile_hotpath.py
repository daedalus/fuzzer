#!/usr/bin/env python3
"""Profile fuzzer execution hotpath with cProfile — tottime + cumtime + call count.

Usage:
    python3 tools/profile_hotpath.py [target] [corpus] [iterations]

Examples:
    python3 tools/profile_hotpath.py targets/png_read_nosan.so /tmp/png 500
"""
import cProfile
import pstats
import sys
import os


def main():
    os.chdir('/home/dclavijo/my_code/fuzzer')

    target = sys.argv[1] if len(sys.argv) > 1 else 'targets/png_read_nosan.so'
    corpus = sys.argv[2] if len(sys.argv) > 2 else '/tmp/png'
    iterations = sys.argv[3] if len(sys.argv) > 3 else '500'

    sys.argv = [
        'fuzzer-tool', 'fuzz', target,
        '-c', '-d', corpus,
        '-n', iterations,
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
    
    # Sort and print by total time (tottime) — pure self-time: the hot hotpath
    print("\n" + "=" * 80)
    print(" TOP 60 BY TOTAL TIME (tottime) — self-time, no children")
    print("=" * 80)
    stats.sort_stats('tottime')
    stats.print_stats(60)
    
    # Sort and print by cumulative time — functions that spend time in callees
    print("\n" + "=" * 80)
    print(" TOP 40 BY CUMULATIVE TIME (cumtime) — self + callee time")
    print("=" * 80)
    stats.sort_stats('cumtime')
    stats.print_stats(40)
    
    # Sort by call count — most frequently called functions
    print("\n" + "=" * 80)
    print(" TOP 30 BY CALL COUNT (ncalls)")
    print("=" * 80)
    stats.sort_stats('ncalls')
    stats.print_stats(30)

    # Save stats dump for later interactive analysis
    dump_path = '/tmp/fuzzer_hotpath.prof'
    stats.dump_stats(dump_path)
    print(f"\n[*] Stats saved to {dump_path}")
    print("    Load interactively:  python3 -c \"import pstats; p=pstats.Stats('{}'); p.sort_stats('tottime').print_stats(40)\"".format(dump_path))


if __name__ == '__main__':
    main()
