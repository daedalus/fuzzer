"""CLI commands for fuzzer-tool."""

import argparse
import os
import sys
from pathlib import Path

from fuzzer_tool.core.mutations import load_dictionary
from fuzzer_tool.services.fuzzer import Fuzzer


def main() -> int:
    parser = argparse.ArgumentParser(description="Coverage-guided binary fuzzer")
    parser.add_argument("target", help="Path to target binary")
    parser.add_argument(
        "-d",
        "--corpus",
        default=None,
        help="Corpus directory (default: ~/fuzzing/<target>/corpus)",
    )
    parser.add_argument(
        "-o",
        "--crashes",
        default=None,
        help="Crashes directory (default: ~/fuzzing/<target>/crashes)",
    )
    parser.add_argument("-m", "--max-len", type=int, default=4096, help="Max input length")
    parser.add_argument("-t", "--timeout", type=float, default=5, help="Timeout in seconds")
    parser.add_argument(
        "-n",
        "--iterations",
        type=int,
        default=0,
        help="Number of iterations (0=infinite)",
    )
    parser.add_argument(
        "-M",
        "--mutations",
        type=int,
        default=8,
        help="Mutations per input",
    )
    parser.add_argument(
        "-c",
        "--coverage",
        action="store_true",
        help="Enable coverage-guided mode",
    )
    parser.add_argument(
        "--deep-coverage",
        action="store_true",
        help="Enable capstone-based basic block discovery (requires -c)",
    )
    parser.add_argument(
        "--max-bps",
        type=int,
        default=50000,
        help="Max breakpoints for deep coverage (default: 50000)",
    )
    parser.add_argument(
        "-D",
        "--dict",
        help="Dictionary file (one token per line, NAME=value or raw bytes)",
    )
    parser.add_argument(
        "-F",
        "--file-mode",
        action="store_true",
        help="Write input to temp file instead of stdin",
    )
    parser.add_argument(
        "-A",
        "--target-args",
        nargs=argparse.REMAINDER,
        help="Target arguments (use {file} as placeholder for temp file)",
    )
    parser.add_argument(
        "--markov",
        action="store_true",
        help="Enable Markov chain mutation (trained on corpus)",
    )
    parser.add_argument(
        "--markov-gen",
        action="store_true",
        help="Enable Markov chain seed generation (15%% of seeds)",
    )
    parser.add_argument(
        "--markov-order",
        type=int,
        default=1,
        help="Markov chain order (default: 1)",
    )
    parser.add_argument(
        "--mc-bandit",
        action="store_true",
        help="Enable Thompson sampling for mutation operator selection",
    )
    parser.add_argument(
        "--mc-cem",
        action="store_true",
        help="Enable cross-entropy method for byte distribution learning",
    )
    parser.add_argument(
        "--mc-elite-frac",
        type=float,
        default=0.1,
        help="Fraction of elite set to fit CEM (default: 0.1)",
    )
    parser.add_argument(
        "--mc-refit-int",
        type=int,
        default=1000,
        help="Refit CEM every N executions (default: 1000)",
    )
    parser.add_argument(
        "--stats-file",
        default=None,
        help="Save stats to JSON file periodically",
    )
    parser.add_argument(
        "--stats-interval",
        type=int,
        default=1000,
        help="Stats dump interval in iterations (default: 1000)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.target):
        print(f"[-] Target not found: {args.target}")
        sys.exit(1)

    if not os.access(args.target, os.X_OK):
        print(f"[-] Target not executable: {args.target}")
        sys.exit(1)

    target_name = os.path.basename(os.path.abspath(args.target))
    fuzz_dir = Path.home() / "fuzzing" / target_name
    corpus_dir = args.corpus or str(fuzz_dir / "corpus")
    crashes_dir = args.crashes or str(fuzz_dir / "crashes")

    dictionary = []
    if args.dict:
        if not os.path.isfile(args.dict):
            print(f"[-] Dictionary not found: {args.dict}")
            sys.exit(1)
        dictionary = load_dictionary(args.dict)
        print(f"[*] Loaded {len(dictionary)} tokens from {args.dict}")

    use_markov = args.markov or args.markov_gen

    fuzzer = Fuzzer(
        target=args.target,
        corpus_dir=corpus_dir,
        crashes_dir=crashes_dir,
        max_len=args.max_len,
        timeout=args.timeout,
        mutations_per_input=args.mutations,
        use_coverage=args.coverage,
        deep_coverage=args.deep_coverage,
        max_bps=args.max_bps,
        dictionary=dictionary,
        file_mode=args.file_mode,
        target_args=args.target_args,
        markov_order=args.markov_order if use_markov else 0,
        markov_generate=args.markov_gen,
        mc_bandit=args.mc_bandit,
        mc_cem=args.mc_cem,
        mc_elite_frac=args.mc_elite_frac,
        mc_refit_interval=args.mc_refit_int,
        stats_file=args.stats_file,
        stats_interval=args.stats_interval,
    )
    fuzzer.run(iterations=args.iterations)
    return 0
