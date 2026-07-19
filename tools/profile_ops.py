#!/usr/bin/env python3
"""Profile fuzzer execution with cUsage: python3 tools/profile_ops.py [target] [corpus] [iterations]

Examples:
    python3 tools/profile_ops.py targets/png_read_nosan.so /tmp/png 200
    python3 tools/profile_ops.py targets/fuzz_shm_nosan.so /tmp/fgrep_cli3 500
"""
import cProfile
import pstats
import sys
import os

def main():
    os.chdir('/home/dclavijo/my_code/fuzzer')

    target = sys.argv[1] if len(sys.argv) > 1 else 'targets/png_read_nosan.so'
    corpus = sys.argv[2] if len(sys.argv) > 2 else '/tmp/png'
    iterations = sys.argv[3] if len(sys.argv) > 3 else '200'

    sys.argv = [
        'fuzzer-tool', 'fuzz', target,
        '-c', '-d', corpus,
        '-n', iterations,
        '--stats-interval', '99999'
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
    stats.sort_stats('tottime')
    stats.print_stats(20)

if __name__ == '__main__':
    main()
