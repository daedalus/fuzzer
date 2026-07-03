"""CLI commands for fuzzer-tool."""

import argparse
import os
import shutil
import sys
from pathlib import Path

from fuzzer_tool.core.mutations import load_dictionary
from fuzzer_tool.services.fuzzer import Fuzzer


def _add_common_args(parser):
    """Add arguments shared by fuzz and subcommands."""
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
    parser.add_argument("-t", "--timeout", type=float, default=5, help="Timeout in seconds")
    parser.add_argument(
        "-F", "--file-mode", action="store_true", help="Write input to temp file instead of stdin"
    )
    parser.add_argument(
        "-A",
        "--target-args",
        nargs="*",
        help="Target arguments ({file} placeholder)",
    )
    parser.add_argument("-c", "--coverage", action="store_true", help="Enable coverage-guided mode")


def _get_dirs(args, target):
    """Resolve corpus/crashes directories."""
    target_name = os.path.basename(os.path.abspath(target))
    fuzz_dir = Path.home() / "fuzzing" / target_name
    corpus_dir = args.corpus or str(fuzz_dir / "corpus")
    crashes_dir = args.crashes or str(fuzz_dir / "crashes")
    return corpus_dir, crashes_dir


def _validate_target(target):
    """Check target binary exists and is executable."""
    if not os.path.isfile(target):
        print(f"[-] Target not found: {target}")
        sys.exit(1)
    if not os.access(target, os.X_OK):
        print(f"[-] Target not executable: {target}")
        sys.exit(1)


def cmd_fuzz(args):
    """Main fuzzing command."""
    if not args.inprocess and not args.inprocess_direct:
        _validate_target(args.target)
    corpus_dir, crashes_dir = _get_dirs(args, args.target)

    dictionary = []
    if args.dict:
        if not os.path.isfile(args.dict):
            print(f"[-] Dictionary not found: {args.dict}")
            sys.exit(1)
        dictionary = load_dictionary(args.dict)
        print(f"[*] Loaded {len(dictionary)} tokens from {args.dict}")

    use_markov = args.markov or args.markov_gen

    # Auto-tune timeout if requested
    timeout = args.timeout
    if args.auto_timeout:
        timeout = _auto_tune_timeout(args.target, args.file_mode, args.target_args)
        print(f"[*] Auto-tuned timeout: {timeout:.2f}s")

    # Load grammar if specified
    grammar = None
    if args.grammar:
        from fuzzer_tool.core.grammar import load_grammar

        grammar = load_grammar(args.grammar)
        print(f"[*] Grammar loaded: {len(grammar.rules)} rules")

    # Parallel mode
    if args.jobs and args.jobs > 1:
        from fuzzer_tool.services.parallel import run_parallel

        run_parallel(
            target=args.target,
            jobs=args.jobs,
            corpus_dir=corpus_dir,
            crashes_dir=crashes_dir,
            max_len=args.max_len,
            timeout=timeout,
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
            coverage_report=args.coverage_report,
            iterations=args.iterations,
            sync_interval=args.sync_interval,
            seed=args.seed,
        )
        return 0

    fuzzer = Fuzzer(
        target=args.target,
        corpus_dir=corpus_dir,
        crashes_dir=crashes_dir,
        max_len=args.max_len,
        timeout=timeout,
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
        coverage_report=args.coverage_report,
        coverage_log=args.coverage_log,
        grammar=grammar,
        persistent=args.persistent,
        cmplog=args.cmplog,
        max_corpus=args.max_corpus,
        no_shm=args.no_shm,
        resume=args.resume,
        trace_crashes=args.trace,
        inprocess=args.inprocess,
        inprocess_direct=args.inprocess_direct,
        inprocess_func=args.inprocess_func,
        seed=args.seed,
        extra_crash_codes=args.crash_codes,
        replay_n=args.replay_n,
        schedule_ablation=getattr(args, 'schedule_ablation', None),
    )
    fuzzer.run(iterations=args.iterations)

    if args.report is not None:
        from fuzzer_tool.services.report import generate_report

        report = generate_report(fuzzer, corpus_dir, crashes_dir)
        if args.report == "-":
            print(report)
        else:
            Path(args.report).parent.mkdir(parents=True, exist_ok=True)
            Path(args.report).write_text(report)
            print(f"[*] Report saved to {args.report}")

    return 0


def _auto_tune_timeout(target, file_mode=False, target_args=None, runs=10):
    """Run the target N times on empty input and set timeout to 5x median."""
    import time as _time

    from fuzzer_tool.adapters.process import run_target_file, run_target_stdin

    tmp_dir = Path("/tmp") / f"tune_{os.getpid()}"
    if file_mode:
        tmp_dir.mkdir(parents=True, exist_ok=True)

    times = []
    for _ in range(runs):
        start = _time.monotonic()
        if file_mode:
            run_target_file(target, b"\n", 30, str(tmp_dir), target_args or [])
        else:
            run_target_stdin(target, b"\n", 30)
        elapsed = _time.monotonic() - start
        times.append(elapsed)

    if tmp_dir.exists():
        import shutil

        shutil.rmtree(tmp_dir, ignore_errors=True)

    times.sort()
    median = times[len(times) // 2]
    return max(5 * median, 0.5)


def cmd_import(args):
    """Import corpus from AFL/libFuzzer/honggfuzz."""
    from fuzzer_tool.services.import_corpus import (
        import_from_afl,
        import_from_honggfuzz,
        import_from_libfuzzer,
    )

    if args.format == "afl":
        seeds, crashes = import_from_afl(args.source_dir, args.corpus, args.crashes)
        print(f"[+] Imported {seeds} seeds, {crashes} crashes from AFL output")
    elif args.format == "libfuzzer":
        imported = import_from_libfuzzer(args.source_dir, args.corpus)
        print(f"[+] Imported {imported} seeds from libFuzzer corpus")
    elif args.format == "honggfuzz":
        imported, _ = import_from_honggfuzz(args.source_dir, args.corpus, args.crashes)
        print(f"[+] Imported {imported} seeds from honggfuzz")
    return 0


def cmd_tmin(args):
    """Crash minimizer subcommand."""
    _validate_target(args.target)
    from fuzzer_tool.services.tmin import tmin

    grammar = None
    if args.grammar:
        from fuzzer_tool.core.grammar import load_grammar
        grammar = load_grammar(args.grammar)
        print(f"[*] Grammar loaded: {len(grammar.rules)} rules (tree-level shrinking enabled)")

    minimized = tmin(
        target=args.target,
        crash_file=args.crash_file,
        timeout=args.timeout,
        file_mode=args.file_mode,
        target_args=args.target_args,
        use_coverage=args.coverage,
        max_stages=args.max_stages,
        grammar=grammar,
    )

    if minimized is None:
        return 1

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_bytes(minimized)
        print(f"[+] Saved to {args.output}")
    else:
        sys.stdout.buffer.write(minimized)
    return 0


def cmd_minimize(args):
    """Corpus minimization subcommand."""
    _validate_target(args.target)
    from fuzzer_tool.services.minimize import minimize_corpus

    kept, removed = minimize_corpus(
        target=args.target,
        corpus_dir=args.corpus,
        timeout=args.timeout,
        file_mode=args.file_mode,
        target_args=args.target_args,
        use_coverage=args.coverage,
        output_dir=args.output,
    )

    if removed == 0:
        print("[*] Corpus already minimal")
    return 0


def cmd_replay(args):
    """Replay a crash input against the target."""
    _validate_target(args.target)

    crash_path = Path(args.crash_file)
    if not crash_path.is_file():
        print(f"[-] Crash file not found: {args.crash_file}", file=sys.stderr)
        return 1

    data = crash_path.read_bytes()
    print(f"[*] Replaying {len(data)} bytes from {args.crash_file}")

    from fuzzer_tool.adapters.process import SIGNAL_CRASH_CODES, run_target_file, run_target_stdin
    from fuzzer_tool.core.sanitizer import SanitizerReport

    env = os.environ.copy()
    tmp_dir = None
    try:
        if args.file_mode:
            tmp_dir = Path("/tmp") / f"replay_{os.getpid()}"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            returncode, stderr = run_target_file(
                target=args.target,
                data=data,
                timeout=args.timeout,
                tmp_dir=str(tmp_dir),
                target_args=args.target_args or [],
                env=env,
            )
        else:
            returncode, stderr = run_target_stdin(
                target=args.target, data=data, timeout=args.timeout, env=env
            )
    finally:
        if tmp_dir and tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)

    if returncode == -1 and stderr == "timeout":
        print(f"[*] Target timed out after {args.timeout}s")
        return 1

    report = SanitizerReport.parse(stderr)
    if report and report.is_valid():
        print(f"[+] Crash reproduced: {report.sanitizer}:{report.error_type}")
        print(f"    Fault address: {report.fault_addr}")
        if report.frames:
            print("    Stack trace:")
            for i, frame in enumerate(report.frames[:8]):
                print(f"      #{i} {frame}")
        return 0

    if abs(returncode) in SIGNAL_CRASH_CODES:
        print(f"[+] Crash reproduced: signal {abs(returncode)}")
        return 0

    print(f"[*] No crash detected (returncode={returncode})")
    if stderr.strip():
        print(f"    stderr: {stderr[:200]}")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="fuzzer-tool",
        description="Coverage-guided binary fuzzer with crash analysis tools",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- fuzz (default) ---
    fuzz_parser = subparsers.add_parser("fuzz", help="Run coverage-guided fuzzing")
    fuzz_parser.add_argument("target", help="Path to target binary")
    fuzz_parser.add_argument(
        "-d", "--corpus", default=None, help="Corpus directory (default: ~/fuzzing/<target>/corpus)"
    )
    fuzz_parser.add_argument(
        "-o",
        "--crashes",
        default=None,
        help="Crashes directory (default: ~/fuzzing/<target>/crashes)",
    )
    fuzz_parser.add_argument("-m", "--max-len", type=int, default=4096, help="Max input length")
    fuzz_parser.add_argument("-t", "--timeout", type=float, default=5, help="Timeout in seconds")
    fuzz_parser.add_argument(
        "-n", "--iterations", type=int, default=0, help="Number of iterations (0=infinite)"
    )
    fuzz_parser.add_argument("-M", "--mutations", type=int, default=8, help="Mutations per input")
    fuzz_parser.add_argument(
        "-c", "--coverage", action="store_true", help="Enable coverage-guided mode"
    )
    fuzz_parser.add_argument(
        "--deep-coverage", action="store_true", help="Enable capstone-based basic block discovery"
    )
    fuzz_parser.add_argument(
        "--max-bps", type=int, default=50000, help="Max breakpoints for deep coverage"
    )
    fuzz_parser.add_argument(
        "--no-shm",
        action="store_true",
        help="Skip AFL SHM coverage, use ptrace instead (for uninstrumented binaries)",
    )
    fuzz_parser.add_argument("-D", "--dict", help="Dictionary file")
    fuzz_parser.add_argument(
        "-F", "--file-mode", action="store_true", help="Write input to temp file instead of stdin"
    )
    fuzz_parser.add_argument(
        "-A",
        "--target-args",
        nargs="*",
        help="Target arguments ({file} placeholder)",
    )
    fuzz_parser.add_argument("--markov", action="store_true", help="Enable Markov chain mutation")
    fuzz_parser.add_argument(
        "--markov-gen", action="store_true", help="Enable Markov chain seed generation"
    )
    fuzz_parser.add_argument(
        "--markov-order", type=int, default=1, help="Markov chain order (default: 1)"
    )
    fuzz_parser.add_argument(
        "--mc-bandit", action="store_true", help="Enable Thompson sampling bandit"
    )
    fuzz_parser.add_argument("--mc-cem", action="store_true", help="Enable cross-entropy method")
    fuzz_parser.add_argument(
        "--mc-elite-frac", type=float, default=0.1, help="CEM elite fraction (default: 0.1)"
    )
    fuzz_parser.add_argument(
        "--mc-refit-int", type=int, default=1000, help="CEM refit interval (default: 1000)"
    )
    fuzz_parser.add_argument(
        "--stats-file", default=None, help="Save stats to JSON file periodically"
    )
    fuzz_parser.add_argument(
        "--stats-interval", type=int, default=1000, help="Stats dump interval (default: 1000)"
    )
    fuzz_parser.add_argument(
        "--coverage-report",
        default=None,
        metavar="FILE",
        help="Dump edge coverage map to JSON file on exit",
    )
    fuzz_parser.add_argument(
        "--auto-timeout", action="store_true", help="Auto-tune timeout by probing target at startup"
    )
    fuzz_parser.add_argument(
        "--cmplog",
        action="store_true",
        help="Enable comparison tracing via LD_PRELOAD (memcmp/strcmp/strncmp/memchr interception)",
    )
    fuzz_parser.add_argument(
        "--max-corpus",
        type=int,
        default=0,
        help="Auto-minimize corpus when it exceeds N entries (0=unlimited)",
    )
    fuzz_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from saved fuzzer state (corpus, stats, edge tracker)",
    )
    fuzz_parser.add_argument(
        "--trace",
        action="store_true",
        help="Generate GDB backtrace + strace reports for crash inputs",
    )
    fuzz_parser.add_argument(
        "--crash-codes",
        nargs="+",
        type=int,
        default=None,
        help="Additional exit codes to treat as crashes (e.g. --crash-codes 1 126)",
    )
    fuzz_parser.add_argument(
        "--coverage-log",
        default=None,
        metavar="FILE",
        help="Append (timestamp, edge_count) lines to file for coverage-over-time plots",
    )
    fuzz_parser.add_argument(
        "--report",
        default=None,
        nargs="?",
        const="-",
        metavar="FILE",
        help="Generate explainability report after run (default: stdout, or specify output file)",
    )
    fuzz_parser.add_argument(
        "--replay-n",
        type=int,
        default=0,
        metavar="N",
        help="Replay each crash N times for reproducibility scoring (default: 0 = off)",
    )
    fuzz_parser.add_argument(
        "--schedule-ablation",
        default=None,
        metavar="FILE",
        help="Log per-iteration scheduling signal contributions to CSV for backtesting",
    )
    fuzz_parser.add_argument(
        "-g",
        "--grammar",
        default=None,
        help="Grammar spec (built-in: json, http_request, elf) or path to .gram file",
    )
    fuzz_parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel fuzzing workers (default: 1)",
    )
    fuzz_parser.add_argument(
        "--sync-interval",
        type=int,
        default=30,
        help="Seconds between corpus sync in parallel mode (default: 30)",
    )
    fuzz_parser.add_argument(
        "--persistent",
        action="store_true",
        help="Use persistent mode for AFL-loop targets (no fork per iteration)",
    )
    fuzz_parser.add_argument(
        "--inprocess",
        action="store_true",
        help="Call target function in-process (C .so or Python module:function)",
    )
    fuzz_parser.add_argument(
        "--inprocess-direct",
        action="store_true",
        help="Direct ctypes.CDLL call — zero overhead, target must not SIGSEGV",
    )
    fuzz_parser.add_argument(
        "--inprocess-func",
        default="LLVMFuzzerTestOneInput",
        help="Function name for in-process mode (default: LLVMFuzzerTestOneInput)",
    )
    fuzz_parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducibility (default: 42)",
    )
    fuzz_parser.set_defaults(func=cmd_fuzz)

    # --- tmin ---
    tmin_parser = subparsers.add_parser("tmin", help="Minimize a crash to smallest reproducer")
    tmin_parser.add_argument("target", help="Path to target binary")
    tmin_parser.add_argument("crash_file", help="Path to crashing input file")
    tmin_parser.add_argument("-t", "--timeout", type=float, default=5, help="Timeout in seconds")
    tmin_parser.add_argument(
        "-F", "--file-mode", action="store_true", help="Write input to temp file instead of stdin"
    )
    tmin_parser.add_argument(
        "-A",
        "--target-args",
        nargs=argparse.REMAINDER,
        help="Target arguments ({file} placeholder)",
    )
    tmin_parser.add_argument("-c", "--coverage", action="store_true", help="Enable SHM coverage")
    tmin_parser.add_argument(
        "--max-stages", type=int, default=128, help="Max reduction stages (default: 128)"
    )
    tmin_parser.add_argument(
        "-g", "--grammar",
        default=None,
        help="Grammar for tree-level shrinking (built-in: json, http_request, elf or .gram file)",
    )
    tmin_parser.add_argument(
        "-O", "--output", default=None, help="Output file for minimized input (default: stdout)"
    )
    tmin_parser.set_defaults(func=cmd_tmin)

    # --- minimize ---
    min_parser = subparsers.add_parser(
        "minimize", help="Minimize corpus by removing redundant inputs"
    )
    min_parser.add_argument("target", help="Path to target binary")
    min_parser.add_argument("-d", "--corpus", required=True, help="Corpus directory")
    min_parser.add_argument("-t", "--timeout", type=float, default=5, help="Timeout in seconds")
    min_parser.add_argument(
        "-F", "--file-mode", action="store_true", help="Write input to temp file instead of stdin"
    )
    min_parser.add_argument(
        "-A",
        "--target-args",
        nargs=argparse.REMAINDER,
        help="Target arguments ({file} placeholder)",
    )
    min_parser.add_argument("-c", "--coverage", action="store_true", help="Enable SHM coverage")
    min_parser.add_argument(
        "-o", "--output", default=None, help="Output directory (default: overwrite in-place)"
    )
    min_parser.set_defaults(func=cmd_minimize)

    # --- replay ---
    replay_parser = subparsers.add_parser("replay", help="Replay a crash input against the target")
    replay_parser.add_argument("target", help="Path to target binary")
    replay_parser.add_argument("crash_file", help="Path to crash input file")
    replay_parser.add_argument("-t", "--timeout", type=float, default=5, help="Timeout in seconds")
    replay_parser.add_argument(
        "-F", "--file-mode", action="store_true", help="Write input to temp file instead of stdin"
    )
    replay_parser.add_argument(
        "-A",
        "--target-args",
        nargs=argparse.REMAINDER,
        help="Target arguments ({file} placeholder)",
    )
    replay_parser.set_defaults(func=cmd_replay)

    # --- import ---
    import_parser = subparsers.add_parser(
        "import", help="Import corpus from AFL/libFuzzer/honggfuzz"
    )
    import_parser.add_argument("source_dir", help="Source directory")
    import_parser.add_argument("-d", "--corpus", required=True, help="Destination corpus directory")
    import_parser.add_argument(
        "-o", "--crashes", default=None, help="Destination crashes directory"
    )
    import_parser.add_argument(
        "--format",
        choices=["afl", "libfuzzer", "honggfuzz"],
        default="afl",
        help="Source format (default: afl)",
    )
    import_parser.set_defaults(func=cmd_import)

    args = parser.parse_args()

    # Default to fuzz if no subcommand given
    if args.command is None:
        # Re-parse with fuzz defaults for backwards compatibility
        sys.argv.insert(1, "fuzz")
        args = parser.parse_args()

    return args.func(args)
