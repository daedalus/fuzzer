"""CLI commands for fuzzer-tool."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from fuzzer_tool.core.mutations import load_dictionary
from fuzzer_tool.services.fuzzer import Fuzzer


def _detect_asan(target: str) -> bool:
    """Detect if a binary is ASAN-instrumented by checking for __asan_init symbol."""
    for flags in [[], ["-D"]]:
        try:
            r = subprocess.run(["nm"] + flags + [target], capture_output=True, timeout=10)
            if r.returncode == 0:
                if b"__asan_init" in r.stdout or b"__asan_register_globals" in r.stdout:
                    return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    return False


def _detect_asan_static(target: str) -> bool:
    """Check if ASAN is statically linked (defined T/t, not U).

    Targets built with --whole-archive libasan.a have __asan_init
    and related symbols defined in the .so itself, not as unresolved
    references that need LD_PRELOAD.
    """
    for flags in [[], ["-D"]]:
        try:
            r = subprocess.run(["nm"] + flags + [target], capture_output=True, timeout=10)
            if r.returncode == 0:
                out = r.stdout.decode(errors="replace")
                for line in out.splitlines():
                    parts = line.split()
                    if len(parts) >= 3 and parts[-1] == "__asan_init":
                        # parts[0] = address, parts[1] = symbol type (T/t = defined, U = undefined)
                        sym_type = parts[1]
                        if sym_type in ("T", "t", "D", "d", "B", "b"):
                            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    return False


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
    parser.add_argument("-t", "--timeout", type=float, default=1, help="Timeout in seconds")
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
    if hasattr(args, "targets") and len(args.targets) > 1:
        # Multi-target: require explicit --corpus, derive from first target name
        target_name = "multi_" + os.path.basename(os.path.abspath(args.targets[0]))
    else:
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
    # Normalize targets: support both old single-target and new multi-target
    import glob as _glob

    _NON_BINARY_EXT = {
        ".c",
        ".h",
        ".py",
        ".sh",
        ".md",
        ".txt",
        ".json",
        ".dict",
        ".gram",
        ".bak",
        ".bak2",
        ".log",
        ".csv",
        ".html",
        ".xml",
        ".yaml",
        ".yml",
        ".toml",
        ".cfg",
        ".ini",
        ".conf",
        ".so",
        ".o",
        ".a",
        ".dylib",
        ".dll",
        ".class",
        ".jar",
    }

    if not hasattr(args, "targets") or args.targets is None:
        args.targets = [args.target]
    # Expand glob patterns (e.g. targets/fuzz_*)
    _GLOB_CHARS = set("*?[")
    expanded = []
    for t in args.targets:
        if any(c in t for c in _GLOB_CHARS):
            matches = _glob.glob(t)
            if matches:
                for m in sorted(matches):
                    ext = os.path.splitext(m)[1].lower()
                    if ext in _NON_BINARY_EXT:
                        continue
                    if not os.path.isfile(m):
                        continue
                    expanded.append(m)
        else:
            expanded.append(t)
    if not expanded:
        print("[-] No executable targets found from glob pattern")
        sys.exit(1)
    # Filter out non-executable files (source files, scripts, etc.)
    if not args.inprocess and not args.inprocess_direct:
        executable = [t for t in expanded if os.access(t, os.X_OK)]
        skipped = [t for t in expanded if not os.access(t, os.X_OK)]
        for s in skipped:
            print(f"[*] Skipping non-executable: {s}")
        if not executable:
            print("[-] No executable targets found")
            sys.exit(1)
        expanded = executable
    args.targets = expanded
    args.target = args.targets[0]
    corpus_dir, crashes_dir = _get_dirs(args, args.target)

    # Auto-detect ASAN instrumentation
    target_is_asan = _detect_asan(args.target)
    if target_is_asan:
        print(f"[*] ASAN detected in {args.target}")

        # For .so targets loaded via ctypes, ASAN must be first in library list.
        # Set LD_PRELOAD to ensure ASAN loads before Python's libraries.
        # Skip this if ASAN is statically linked (defined T/t symbol, not U).
        is_so = args.target.lower().endswith((".so", ".dylib", ".dll"))
        asan_static = _detect_asan_static(args.target) if is_so else False
        if is_so and not asan_static:
            # Find full path to libasan (ctypes.util.find_library may return relative name)
            libasan = "/usr/lib/x86_64-linux-gnu/libasan.so.8"
            if not os.path.exists(libasan):
                import ctypes.util

                libasan = ctypes.util.find_library("asan") or libasan
            existing = os.environ.get("LD_PRELOAD", "")
            if libasan not in existing:
                if existing:
                    os.environ["LD_PRELOAD"] = f"{libasan}:{existing}"
                else:
                    os.environ["LD_PRELOAD"] = libasan
                print(f"[*] LD_PRELOAD={libasan} (ASAN must load first for .so targets)")
        elif asan_static:
            print("[*] ASAN statically linked — no LD_PRELOAD needed")

    # ASAN calls _exit() which kills inprocess-direct mode.
    # Fall back to subprocess mode so stderr is captured.
    if target_is_asan and getattr(args, "inprocess_direct", False):
        print("[*] ASAN + --inprocess-direct: falling back to subprocess mode")
        args.inprocess_direct = False
        # Enable inprocess mode so InProcessRunner is created with direct=False
        args.inprocess = True

    dictionary = []
    if args.dict:
        if not os.path.isfile(args.dict):
            print(f"[-] Dictionary not found: {args.dict}")
            sys.exit(1)
        dictionary = load_dictionary(args.dict)
        print(f"[*] Loaded {len(dictionary)} tokens from {args.dict}")

    # QEA and GA are mutually exclusive — QEA takes precedence
    if getattr(args, "qea", False) and getattr(args, "ga", False):
        print("[*] --qea and --ga both set: --ga disabled (QEA takes precedence)")
        args.ga = False

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
            markov_order=args.markov_order if use_markov else "0",
            markov_generate=args.markov_gen,
            markov_blend=getattr(args, "markov_blend", False),
            mc_bandit=args.mc_bandit,
            mc_cem=args.mc_cem,
            mc_elite_frac=args.mc_elite_frac,
            mc_refit_interval=args.mc_refit_int,
            mc_decay_interval=getattr(args, "mc_decay_interval", 100),
            pairwise_blend=getattr(args, "pairwise_blend", 0.0),
            stats_file=args.stats_file,
            stats_interval=args.stats_interval,
            coverage_report=args.coverage_report,
            iterations=args.iterations,
            sync_interval=args.sync_interval,
            seed=args.seed,
            secretary=getattr(args, "secretary", False),
            secretary_window=getattr(args, "secretary_window", 500),
            secretary_exploration=getattr(args, "secretary_exploration", 0.368),
            resize_map_on_stall=getattr(args, "resize_map_on_stall", False),
        )
        return 0

    plot_graph_path = None
    coverage_log_arg = args.coverage_log
    if getattr(args, "plot_graph", None) is not None:
        plot_graph_path = (
            str(Path(corpus_dir) / "report.html") if args.plot_graph == "-" else args.plot_graph
        )
        if not coverage_log_arg:
            coverage_log_arg = str(Path(corpus_dir) / ".plot_graph_coverage_log.csv")

    fuzzer = Fuzzer(
        target=args.target,
        multi_targets=args.targets if len(args.targets) > 1 else None,
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
        mopt=getattr(args, "mopt", False),
        targets=getattr(args, "targets", None),
        anneal_budget=getattr(args, "anneal_budget", 0),
        mc_elite_frac=args.mc_elite_frac,
        mc_refit_interval=args.mc_refit_int,
        mc_decay_interval=getattr(args, "mc_decay_interval", 100),
        pairwise_blend=getattr(args, "pairwise_blend", 0.0),
        stats_file=args.stats_file,
        stats_interval=args.stats_interval,
        coverage_report=args.coverage_report,
        coverage_log=coverage_log_arg,
        grammar=grammar,
        persistent=args.persistent,
        cmplog=args.cmplog,
        cmplog_max_tokens=getattr(args, "cmplog_max_tokens", 0),
        cmplog_max_pairs=getattr(args, "cmplog_max_pairs", 0),
        max_corpus=args.max_corpus,
        max_corpus_bytes=getattr(args, "max_corpus_bytes", 0),
        minimize_every_execs=getattr(args, "minimize_every_execs", 0),
        prune_corpus_max_memory=getattr(args, "prune_corpus_max_memory", 80),
        no_shm=args.no_shm,
        resume=args.resume,
        trace_crashes=args.trace,
        learn_format=getattr(args, "learn_format", False),
        corpus_ppmd=getattr(args, "corpus_ppmd", False),
        inprocess=args.inprocess,
        inprocess_direct=args.inprocess_direct,
        inprocess_func=args.inprocess_func,
        seed=args.seed,
        extra_crash_codes=args.crash_codes,
        replay_n=args.replay_n,
        schedule_ablation=getattr(args, "schedule_ablation", None),
        replicator=getattr(args, "replicator", False),
        shapley=getattr(args, "shapley", False),
        bayesian=getattr(args, "bayesian", False),
        mi_guided=getattr(args, "mi_guided", False),
        renyi_weight=getattr(args, "renyi_weight", False),
        transfer_entropy=getattr(args, "transfer_entropy", False),
        elo=getattr(args, "elo", False),
        secretary=getattr(args, "secretary", False),
        secretary_window=getattr(args, "secretary_window", 500),
        secretary_exploration=getattr(args, "secretary_exploration", 0.368),
        sensitivity=getattr(args, "sensitivity", False),
        ga=getattr(args, "ga", False),
        qea=getattr(args, "qea", False),
        wfc=getattr(args, "wfc", False),
        ga_pop_size=getattr(args, "ga_pop_size", 200),
        ga_gen_size=getattr(args, "ga_gen_size", 500),
        ga_elite_frac=getattr(args, "ga_elite_frac", 0.1),
        ga_crossover_rate=getattr(args, "ga_crossover_rate", 0.7),
        ga_mutation_rate=getattr(args, "ga_mutation_rate", 0.3),
        ga_tournament_size=getattr(args, "ga_tournament_size", 3),
        ga_speciation_threshold=getattr(args, "ga_speciation_threshold", 0.3),
        continue_until_crash=getattr(args, "continue_until_crash", False),
        calibrate=getattr(args, "calibrate", 0),
        stall_threshold=getattr(args, "stall", 1000),
        map_size=getattr(args, "map_size", 0),
        max_collision_risk=getattr(args, "max_collision_risk", 30),
        debug=getattr(args, "debug", False),
        enable_regex_bomb=getattr(args, "enable_regex_bomb_mutations", False),
        refresh_profile=getattr(args, "refresh_profile", False),
        resize_map_on_stall=getattr(args, "resize_map_on_stall", False),
        enable_smt_z3=getattr(args, "enable_smt_z3", False),
        mod_solving=getattr(args, "mod_solving", "heuristic"),
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

    if plot_graph_path is not None:
        from fuzzer_tool.core.plotting import generate_html_report

        written = generate_html_report(fuzzer, coverage_log_arg, plot_graph_path)
        print(f"[*] Plot graph saved to {written}")

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
    return max(5 * median, 0.05)


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
        rate_distortion=getattr(args, "rate_distortion", False),
        target_frac=getattr(args, "target_frac", 0.95),
    )

    if removed == 0:
        print("[*] Corpus already minimal")
    return 0


def cmd_verify(args):
    """Re-run crashes with ASAN target to confirm memory bugs.

    Takes a crashes directory (from fast no-ASAN fuzzing) and an
    ASAN-instrumented target, re-runs each crash to confirm it's
    a real memory bug detected by the sanitizer.
    """
    _validate_target(args.asan_target)

    crashes_dir = Path(args.crashes_dir)
    if not crashes_dir.is_dir():
        print(f"[-] Crashes directory not found: {args.crashes_dir}", file=sys.stderr)
        return 1

    crash_files = sorted(crashes_dir.glob("*.bin")) + sorted(crashes_dir.glob("*crash*"))
    # Deduplicate and filter to actual files
    crash_files = [f for f in set(crash_files) if f.is_file()]
    if not crash_files:
        print(f"[-] No crash files found in {args.crashes_dir}")
        return 1

    print(f"[*] Verifying {len(crash_files)} crashes against {args.asan_target}")

    from fuzzer_tool.adapters.process import SIGNAL_CRASH_CODES, run_target_file, run_target_stdin
    from fuzzer_tool.core.sanitizer import SanitizerReport

    confirmed = 0
    failed = 0
    errors = 0

    for crash_file in crash_files:
        data = crash_file.read_bytes()
        if not data:
            continue

        try:
            if args.file_mode:
                tmp_dir = Path("/tmp") / f"verify_{os.getpid()}"
                tmp_dir.mkdir(parents=True, exist_ok=True)
                try:
                    returncode, stderr = run_target_file(
                        target=args.asan_target,
                        data=data,
                        timeout=args.timeout,
                        tmp_dir=str(tmp_dir),
                        env=os.environ.copy(),
                    )
                finally:
                    import shutil

                    if tmp_dir.exists():
                        shutil.rmtree(tmp_dir, ignore_errors=True)
            else:
                returncode, stderr, _ = run_target_stdin(
                    target=args.asan_target,
                    data=data,
                    timeout=args.timeout,
                    env=os.environ.copy(),
                )
        except Exception as e:
            print(f"  [!] {crash_file.name}: execution error: {e}")
            errors += 1
            continue

        report = SanitizerReport.parse(stderr)
        if report and report.is_valid():
            print(f"  [+] {crash_file.name}: {report.sanitizer}:{report.error_type}")
            confirmed += 1
        elif abs(returncode) in SIGNAL_CRASH_CODES:
            print(f"  [+] {crash_file.name}: signal {abs(returncode)}")
            confirmed += 1
        elif returncode == -1 and stderr == "timeout":
            print(f"  [-] {crash_file.name}: timeout (not a crash)")
            failed += 1
        else:
            print(f"  [-] {crash_file.name}: no crash (rc={returncode})")
            failed += 1

    print(f"\n[*] Results: {confirmed} confirmed, {failed} not reproduced, {errors} errors")
    return 0 if confirmed > 0 else 1


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


def cmd_rank(args):
    """Rank corpus seeds by interestingness."""
    _validate_target(args.target)
    import hashlib
    import json
    import math
    import time

    from fuzzer_tool.adapters.filesystem import load_corpus
    from fuzzer_tool.core.bloom import BloomFilter
    from fuzzer_tool.core.edge_tracker import EdgeTracker
    from fuzzer_tool.core.elf import estimate_map_size

    corpus_dir = Path(args.corpus)
    if not corpus_dir.is_dir():
        print(f"[-] Corpus not found: {corpus_dir}", file=sys.stderr)
        return 1

    state_path = corpus_dir / "state.json"
    edge_path = corpus_dir / "edge_tracker.json"

    bloom = BloomFilter(capacity=100_000)
    corpus, seen_hashes = load_corpus(corpus_dir, bloom)
    if not corpus:
        print("[-] Empty corpus", file=sys.stderr)
        return 1

    map_size = estimate_map_size(args.target)
    et = EdgeTracker(map_size=map_size)
    if edge_path.exists():
        et.load(str(edge_path))

    # Load seed metadata from state.json
    seed_meta = {}
    now = time.time()
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
            saved = state.get("seed_meta", {})
            for seed in corpus:
                key = seed.hex()
                if key in saved:
                    sm = saved[key]
                    seed_meta[seed] = {
                        "fuzz_count": sm.get("fuzz_count", 0),
                        "coverage_edges": sm.get("coverage_edges", 0),
                        "added_at": sm.get("added_at", now),
                    }
                else:
                    seed_meta[seed] = {"fuzz_count": 0, "coverage_edges": 0, "added_at": now}
        except (OSError, json.JSONDecodeError):
            pass

    if not seed_meta:
        for seed in corpus:
            seed_meta[seed] = {"fuzz_count": 0, "coverage_edges": 0, "added_at": now}

    def score(seed):
        meta = seed_meta.get(seed, {})
        fuzz_count = max(meta.get("fuzz_count", 0), 1)
        coverage = meta.get("coverage_edges", 0)
        age = now - meta.get("added_at", now)
        key = hashlib.sha256(seed).hexdigest()[:16]

        # Edge tracker signals (only if this seed is tracked)
        seed_edges = et.seed_edges.get(key, set())
        edge_count = len(seed_edges)
        rare = sum(1 for e in seed_edges if et._global_edge_hits.get(e, 0) <= 2)
        sub = et.compute_subsumption_weight(key) if seed_edges else 1.0
        prox = et.compute_coverage_proximity(key) if seed_edges else 0.0

        # Composite score:
        #   - coverage_edges (from state.json) is the primary signal:
        #     seeds that discovered more edges are more interesting
        #   - edge tracker signals (rarity, subsumption, proximity) add
        #     granularity when available
        #   - fuzz_count penalizes over-explored seeds
        #   - seed length penalizes very large inputs (harder to mutate)
        w = 1.0 + coverage * 2.0  # primary: edge discovery
        if edge_count > 0:
            w *= (1.0 + rare * 0.5) * sub * (0.5 + prox)
        w /= math.sqrt(fuzz_count)
        # Slight penalty for very large seeds (diminishing returns)
        w *= 1.0 / (1.0 + len(seed) / 4096.0)

        return {
            "score": w,
            "edges": edge_count or coverage,
            "rare": rare,
            "fuzz_count": fuzz_count,
            "coverage": coverage,
            "subsumption": sub,
            "proximity": prox,
        }

    scored = [(score(s), s) for s in corpus]
    scored.sort(key=lambda x: x[0]["score"], reverse=True)

    n = min(args.top, len(scored))
    n_edges = (
        len(et.cumulative_edges) if hasattr(et.cumulative_edges, "__len__") else et.cumulative_edges
    )
    print(f"[*] Corpus: {len(corpus)} seeds, {len(et.seed_edges)} tracked, {n_edges} edges\n")
    print(
        f"{'#':>4}  {'Score':>7}  {'Edges':>5}  {'Rare':>4}  {'Fuzz':>5}  "
        f"{'Sub':>5}  {'Prox':>5}  {'Hash':>16}  Preview"
    )
    print("-" * 95)

    for i, (s, seed) in enumerate(scored[:n]):
        h = hashlib.sha256(seed).hexdigest()[:16]
        # Show hex preview for binary, text preview for printable
        raw = seed[:32]
        printable = sum(1 for b in raw if 32 <= b < 127)
        if printable > len(raw) * 0.7:
            pstr = raw.decode("ascii", errors="replace")
            if len(seed) > 32:
                pstr += "..."
        else:
            pstr = raw.hex()
            if len(seed) > 32:
                pstr += "..."
        print(
            f"{i + 1:>4}  {s['score']:>7.2f}  {s['edges']:>5}  {s['rare']:>4}  "
            f"{s['fuzz_count']:>5}  {s['subsumption']:>5.2f}  "
            f"{s['proximity']:>5.2f}  {h}  {pstr}"
        )

    if args.dump:
        out = Path(args.dump)
        with open(out, "w") as f:
            for i, (s, seed) in enumerate(scored[:n]):
                h = hashlib.sha256(seed).hexdigest()[:16]
                f.write(seed)
                print(f"  wrote seed #{i + 1} ({len(seed)} bytes) -> {out}.{i}")
        # Also write each seed to a separate file
        for i, (s, seed) in enumerate(scored[:n]):
            seed_path = out.parent / f"{out.name}.{i}"
            seed_path.write_bytes(seed)
        print(f"[*] Dumped top {n} seeds to {out}.{0}..{n - 1}")

    return 0


def cmd_ppmd(args):
    """Analyze corpus compressibility with PPMD and generate distribution graph."""
    import math
    from pathlib import Path

    from fuzzer_tool.adapters.filesystem import load_corpus
    from fuzzer_tool.core.bloom import BloomFilter
    from fuzzer_tool.core.corpus_compression import CorpusCompressor

    corpus_dir = Path(args.corpus)
    if not corpus_dir.exists():
        print(f"Error: corpus directory {corpus_dir} not found")
        return 1

    bloom = BloomFilter(capacity=100_000)
    bloom.init_fuzzy(max_recent=200)
    corpus, _ = load_corpus(corpus_dir, bloom)

    if not corpus:
        print(f"No seeds found in {corpus_dir}")
        return 1

    print(f"Corpus: {corpus_dir} ({len(corpus)} seeds)")
    cc = CorpusCompressor()

    # Compute ratios
    ratios = []
    sizes = []
    for seed in corpus:
        ratio = cc.compute_seed_ratio(seed)
        ratios.append(ratio)
        sizes.append(len(seed))

    ratios.sort()
    n = len(ratios)
    mean_r = sum(ratios) / n
    var_r = sum((r - mean_r) ** 2 for r in ratios) / n
    std_r = math.sqrt(var_r)

    print("\n--- PPMD Compression Statistics ---")
    print(f"  Seeds:           {n}")
    print(f"  Mean ratio:      {mean_r:.4f}")
    print(f"  Std deviation:   {std_r:.4f}")
    print(f"  Median ratio:    {ratios[n // 2]:.4f}")
    print(f"  Min ratio:       {ratios[0]:.4f} (most compressible)")
    print(f"  Max ratio:       {ratios[-1]:.4f} (most novel)")
    print(f"  Total raw:       {sum(sizes):,} bytes")
    print(f"  Total compressed:{sum(int(s * r) for s, r in zip(sizes, ratios)):,} bytes")
    print(f"  Corpus ratio:    {sum(int(s * r) for s, r in zip(sizes, ratios)) / sum(sizes):.4f}")

    # Top N most/least novel
    scored = list(enumerate(ratios))
    scored.sort(key=lambda x: x[1])
    top = min(args.top, n)
    print(f"\n  Top {top} most novel (highest ratio):")
    for i in range(max(0, n - top), n):
        idx, r = scored[i]
        print(f"    [{idx:4d}] ratio={r:.4f}  size={sizes[idx]}B")

    print(f"\n  Top {top} most redundant (lowest ratio):")
    for i in range(top):
        idx, r = scored[i]
        print(f"    [{idx:4d}] ratio={r:.4f}  size={sizes[idx]}B")

    # Generate graph
    if args.graph:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np

            fig, ax = plt.subplots(figsize=(10, 6))

            # Histogram of PPMD ratios
            ax.hist(
                ratios,
                bins=min(50, max(10, n // 5)),
                alpha=0.7,
                color="#2196F3",
                edgecolor="white",
                label="PPMD ratios",
            )

            # Normal distribution curve
            x = np.linspace(max(0, mean_r - 3 * std_r), min(1, mean_r + 3 * std_r), 200)
            if std_r > 0:
                y = np.exp(-0.5 * ((x - mean_r) / std_r) ** 2) / (std_r * math.sqrt(2 * math.pi))
                y_scaled = y * n * (ratios[-1] - ratios[0]) / min(50, max(10, n // 5))
                ax.plot(
                    x, y_scaled, "r-", linewidth=2, label=f"Normal (μ={mean_r:.3f}, σ={std_r:.3f})"
                )

            ax.set_xlabel("PPMD Compression Ratio", fontsize=12)
            ax.set_ylabel("Count", fontsize=12)
            ax.set_title(f"Corpus PPMD Compression Distribution ({n} seeds)", fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add stats annotation
            stats_text = (
                f"Mean: {mean_r:.4f}\n"
                f"Std:  {std_r:.4f}\n"
                f"Min:  {ratios[0]:.4f}\n"
                f"Max:  {ratios[-1]:.4f}\n"
                f"Median: {ratios[n // 2]:.4f}"
            )
            ax.text(
                0.98,
                0.95,
                stats_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

            plt.tight_layout()
            plt.savefig(args.graph, dpi=150)
            print(f"\n  Graph saved to: {args.graph}")
        except ImportError:
            print("\n  Warning: matplotlib not installed, skipping graph generation")
        except Exception as e:
            print(f"\n  Error generating graph: {e}")

    return 0


def cmd_estimate(args):
    """Estimate executions to first crash."""
    from fuzzer_tool.core.crash_eta import (
        estimate_execs_to_first_crash,
        estimate_risky_density,
    )
    from fuzzer_tool.core.target_profiler import TargetProfiler
    from fuzzer_tool.services.fuzzer import Fuzzer

    print(f"Target: {args.target}")
    print(f"Corpus: {args.corpus}")
    print(f"Calibration: {args.calibrate} execs\n")

    # Static analysis
    print("Running static analysis...")
    profiler = TargetProfiler(args.target)
    profile = profiler.profile()
    rho = estimate_risky_density(profile)
    print(f"  Risky density (ρ): {rho:.4f}")
    print(f"  Functions analyzed: {len(profile.functions)}")
    print(f"  Error-related strings: {len(profile.rodata_strings)}\n")

    # Calibration pass
    print(f"Running calibration ({args.calibrate} execs)...")
    fuzzer = Fuzzer(
        target=args.target,
        corpus_dir=args.corpus,
        crashes_dir=args.corpus + "/crashes",
        timeout=5,
        calibrate=args.calibrate,
    )
    fuzzer._run_calibration(args.calibrate)

    # Get stats
    gt = fuzzer._edge_tracker.good_turing_estimate()
    dr = fuzzer.discovery_rate()
    print(f"  Edges discovered: {gt['n']}")
    print(f"  Estimated total: {gt['n'] + gt['estimated_undiscovered']}")
    print(f"  Discovery rate: {dr:.1f} edges/1k execs")
    print(f"  GT confidence: {gt['confidence']}\n")

    # Estimate
    eta = estimate_execs_to_first_crash(profile, gt, dr, args.calibrate)
    print("=== Crash ETA Estimate ===")
    print(f"  Point estimate: {eta.point_est:,} execs")
    print(f"  Range: {eta.low:,} - {eta.high:,} execs")
    print(f"  Confidence: {eta.confidence}")
    print(f"  Reasoning: {eta.reasoning}")


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(
        prog="fuzzer-tool",
        description="Coverage-guided binary fuzzer with crash analysis tools",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- fuzz (default) ---
    fuzz_parser = subparsers.add_parser("fuzz", help="Run coverage-guided fuzzing")
    fuzz_parser.add_argument("targets", nargs="+", help="Path(s) to target binary(ies)")
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
    fuzz_parser.add_argument("-t", "--timeout", type=float, default=1, help="Timeout in seconds")
    fuzz_parser.add_argument(
        "-n", "--iterations", type=int, default=0, help="Number of iterations (0=infinite)"
    )
    fuzz_parser.add_argument(
        "--continue-until-crash",
        action="store_true",
        help="Ignore -n, fuzz until the first crash is found",
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
        "--markov-order",
        type=str,
        default="1",
        help="Markov chain order(s), comma-separated (e.g. '0,1,2' for ensemble)",
    )
    fuzz_parser.add_argument(
        "--markov-blend",
        action="store_true",
        help="Blend probability distributions across orders (slower but smoother)",
    )
    fuzz_parser.add_argument(
        "--mc-bandit", action="store_true", help="Enable Thompson sampling bandit"
    )
    fuzz_parser.add_argument(
        "--pairwise-blend",
        type=float,
        default=0.0,
        help="Blend factor for pairwise operator transitions (0.0=pure Thompson, 1.0=pure pairwise)",
    )
    fuzz_parser.add_argument("--mc-cem", action="store_true", help="Enable cross-entropy method")
    fuzz_parser.add_argument(
        "--mopt",
        action="store_true",
        help="Enable MOpt PSO operator scheduling (alternative to bandit)",
    )
    fuzz_parser.add_argument(
        "--replicator",
        action="store_true",
        help="Enable replicator dynamics operator scheduling (evolutionary game theory)",
    )
    fuzz_parser.add_argument(
        "--shapley",
        action="store_true",
        help="Enable Shapley value operator attribution (fair credit distribution)",
    )
    fuzz_parser.add_argument(
        "--bayesian",
        action="store_true",
        help="Enable Bayesian methods: Thompson-sampled seed selection, hierarchical operator priors, Bayesian coverage growth model",
    )
    fuzz_parser.add_argument(
        "--mi-guided",
        action="store_true",
        help="Enable mutual information guided mutation (target high-MI byte positions)",
    )
    fuzz_parser.add_argument(
        "--renyi-weight",
        action="store_true",
        help="Enable Rényi entropy weighting in seed selection (boost cold-edge seeds)",
    )
    fuzz_parser.add_argument(
        "--transfer-entropy",
        action="store_true",
        help="Enable transfer entropy causal tracking (byte→edge influence detection)",
    )
    fuzz_parser.add_argument(
        "--elo",
        action="store_true",
        help="Enable Elo scheduling: arbitrates between operator strategies (bandit/MOpt/replicator) AND seed strategies (ga/weighted/pareto/format)",
    )
    fuzz_parser.add_argument(
        "--secretary",
        action="store_true",
        help="Enable secretary-problem optimal stopping for seed/operator/corpus scheduling",
    )
    fuzz_parser.add_argument(
        "--secretary-window",
        type=int,
        default=500,
        help="Sliding window size for secretary quality observations (default: 500)",
    )
    fuzz_parser.add_argument(
        "--secretary-exploration",
        type=float,
        default=0.368,
        help="Exploration fraction threshold for secretary stopping (default: 0.368 = 1/e)",
    )
    fuzz_parser.add_argument(
        "--sensitivity",
        action="store_true",
        help="Enable per-byte sensitivity analysis (Lyapunov exponent) for mutation targeting",
    )
    fuzz_parser.add_argument(
        "--ga",
        action="store_true",
        help="Enable genetic algorithm lifecycle mode",
    )
    fuzz_parser.add_argument(
        "--qea",
        action="store_true",
        help="Enable quantum-inspired evolutionary algorithm (QEA) encoding mode",
    )
    fuzz_parser.add_argument(
        "--wfc",
        action="store_true",
        help="Enable Wave Function Collapse structural generation (chunk reordering, pixel generation)",
    )
    fuzz_parser.add_argument(
        "--enable-smt-z3",
        action="store_true",
        help="Enable z3-based SMT solving: arithmetic constraint solving on cmplog pairs "
        "and computed-field repair for WFC output",
    )
    fuzz_parser.add_argument(
        "--mod-solving",
        choices=["heuristic", "trace", "concolic"],
        default="concolic",
        help="Modulo constraint solving mode (requires --enable-smt-z3). "
        "concolic: full constraint model with z3 solver (default); "
        "heuristic: try common divisors on (remainder, 0) pairs; "
        "trace: use PC-correlated DIV/IDIV from static analysis",
    )
    fuzz_parser.add_argument(
        "--ga-pop-size",
        type=int,
        default=200,
        help="GA population size (default: 200)",
    )
    fuzz_parser.add_argument(
        "--ga-gen-size",
        type=int,
        default=500,
        help="Fuzz iterations per GA generation (default: 500)",
    )
    fuzz_parser.add_argument(
        "--ga-elite-frac",
        type=float,
        default=0.1,
        help="GA elite fraction (default: 0.1)",
    )
    fuzz_parser.add_argument(
        "--ga-crossover-rate",
        type=float,
        default=0.7,
        help="GA crossover probability (default: 0.7)",
    )
    fuzz_parser.add_argument(
        "--ga-mutation-rate",
        type=float,
        default=0.3,
        help="GA mutation probability (default: 0.3)",
    )
    fuzz_parser.add_argument(
        "--ga-tournament-size",
        type=int,
        default=3,
        help="GA tournament selection size (default: 3)",
    )
    fuzz_parser.add_argument(
        "--ga-speciation-threshold",
        type=float,
        default=0.3,
        help="MinHash Jaccard threshold for species grouping (default: 0.3)",
    )
    fuzz_parser.add_argument(
        "--targets",
        nargs="+",
        default=None,
        metavar="FUNC",
        help="Target functions for directed fuzzing (names or hex addresses)",
    )
    fuzz_parser.add_argument(
        "--anneal-budget",
        type=int,
        default=0,
        metavar="N",
        help="Annealing budget in iterations (0=no annealing, default). "
        "Temperature decays linearly from 1.0 to 0.1 over N iterations.",
    )
    fuzz_parser.add_argument(
        "--mc-elite-frac", type=float, default=0.1, help="CEM elite fraction (default: 0.1)"
    )
    fuzz_parser.add_argument(
        "--mc-refit-int", type=int, default=1000, help="CEM refit interval (default: 1000)"
    )
    fuzz_parser.add_argument(
        "--mc-decay-interval",
        type=int,
        default=100,
        help="Bandit decay interval: apply arm_decay every N calls (default: 100)",
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
        "--cmplog-max-tokens",
        type=int,
        default=0,
        help="Max unique cmplog operand tokens (default 10000, 0=default)",
    )
    fuzz_parser.add_argument(
        "--cmplog-max-pairs",
        type=int,
        default=0,
        help="Max unique cmplog operand pairs (default 5000, 0=default)",
    )
    fuzz_parser.add_argument(
        "--max-corpus",
        type=int,
        default=0,
        help="Auto-minimize corpus when it exceeds N entries (0=unlimited)",
    )
    fuzz_parser.add_argument(
        "--max-corpus-bytes",
        type=int,
        default=0,
        help="Auto-minimize corpus when total seed bytes exceeds N (0=unlimited)",
    )
    fuzz_parser.add_argument(
        "--minimize-every-execs",
        type=int,
        default=0,
        help="Fire corpus minimization every N executions (0=disabled)",
    )
    fuzz_parser.add_argument(
        "--prune-corpus-on-max-memory",
        type=int,
        default=80,
        help="Auto-prune corpus when RSS exceeds N%% of total RAM (0=disabled, default=80)",
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
        "--learn-format",
        action="store_true",
        help="Enable format structure learner (schema-harness methodology)",
    )
    fuzz_parser.add_argument(
        "--corpus-ppmd",
        action="store_true",
        help="Enable PPMD-based corpus compression for seed novelty scoring",
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
    fuzz_parser.add_argument(
        "--plot-graph",
        nargs="?",
        const="-",
        default=None,
        metavar="FILE",
        help=(
            "Write a self-contained HTML report with SVG charts of edges, "
            "corpus size, exec rate, crashes, and operator success rates "
            "over the run (default: <corpus_dir>/report.html). Works "
            "standalone -- does not require --coverage-log to be set "
            "separately, an internal log is used automatically if needed."
        ),
    )
    fuzz_parser.add_argument(
        "--calibrate",
        type=int,
        default=0,
        metavar="N",
        help="Run N calibration execs (seed replay + cheap mutations) to bootstrap "
        "coverage stats before the main fuzz loop (default: 0 = off)",
    )
    fuzz_parser.add_argument(
        "--stall",
        type=int,
        default=1000,
        metavar="N",
        help="Detect stall after N execs without new edges and activate "
        "recovery mode with more aggressive mutations (default: 1000)",
    )
    fuzz_parser.add_argument(
        "--resize-map-on-stall",
        action="store_true",
        default=False,
        help="Resize the SHM coverage bitmap when stall recovery triggers, "
        "reducing hash collision risk and potentially exposing new edges. "
        "Uses birthday-bound (n^2/0.02) to compute the new size.",
    )
    fuzz_parser.add_argument(
        "--map-size",
        type=int,
        default=0,
        metavar="N",
        help="Initial edge bitmap size in bytes (default: 0 = auto-size from branch density)",
    )
    fuzz_parser.add_argument(
        "--max-collision-risk",
        type=int,
        default=30,
        metavar="N",
        help="Resize bitmap when collision risk exceeds N%% (default: 30)",
    )
    fuzz_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output (SHM attach, coverage tracing, etc.)",
    )
    fuzz_parser.add_argument(
        "--refresh-profile",
        action="store_true",
        help="Force re-analysis of target binary (skip cached profile)",
    )
    fuzz_parser.add_argument(
        "--enable-regex-bomb-mutations",
        action="store_true",
        help="Enable regex backtracking bomb mutations (ReDoS patterns that cause explosive memory usage)",
    )
    fuzz_parser.set_defaults(func=cmd_fuzz)

    # --- tmin ---
    tmin_parser = subparsers.add_parser("tmin", help="Minimize a crash to smallest reproducer")
    tmin_parser.add_argument("target", help="Path to target binary")
    tmin_parser.add_argument("crash_file", help="Path to crashing input file")
    tmin_parser.add_argument("-t", "--timeout", type=float, default=1, help="Timeout in seconds")
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
        "-g",
        "--grammar",
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
    min_parser.add_argument("-t", "--timeout", type=float, default=1, help="Timeout in seconds")
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
    min_parser.add_argument(
        "--rate-distortion",
        action="store_true",
        help="Use rate-distortion optimal pruning (preserves coverage diversity)",
    )
    min_parser.add_argument(
        "--target-frac",
        type=float,
        default=0.95,
        help="Target coverage fraction for rate-distortion (default: 0.95)",
    )
    min_parser.set_defaults(func=cmd_minimize)

    # --- replay ---
    replay_parser = subparsers.add_parser("replay", help="Replay a crash input against the target")
    replay_parser.add_argument("target", help="Path to target binary")
    replay_parser.add_argument("crash_file", help="Path to crash input file")
    replay_parser.add_argument("-t", "--timeout", type=float, default=1, help="Timeout in seconds")
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

    # --- verify ---
    verify_parser = subparsers.add_parser(
        "verify", help="Re-run crashes with ASAN target to confirm memory bugs"
    )
    verify_parser.add_argument("asan_target", help="Path to ASAN-instrumented target binary")
    verify_parser.add_argument("crashes_dir", help="Directory containing crash input files")
    verify_parser.add_argument("-t", "--timeout", type=float, default=1, help="Timeout in seconds")
    verify_parser.add_argument(
        "-F", "--file-mode", action="store_true", help="Write input to temp file instead of stdin"
    )
    verify_parser.set_defaults(func=cmd_verify)

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

    # --- rank ---
    rank_parser = subparsers.add_parser("rank", help="Rank corpus seeds by interestingness")
    rank_parser.add_argument("target", help="Path to target binary")
    rank_parser.add_argument("-d", "--corpus", required=True, help="Corpus directory")
    rank_parser.add_argument(
        "-n", "--top", type=int, default=10, help="Number of top seeds to show"
    )
    rank_parser.add_argument(
        "--dump",
        default=None,
        metavar="PREFIX",
        help="Dump top seeds to files named PREFIX.0, PREFIX.1, ...",
    )
    rank_parser.set_defaults(func=cmd_rank)

    # --- estimate ---
    est_parser = subparsers.add_parser(
        "estimate",
        help="Estimate executions to first crash",
    )
    est_parser.add_argument("target", help="Path to target binary")
    est_parser.add_argument("--corpus", required=True, help="Corpus directory")
    est_parser.add_argument(
        "--calibrate",
        type=int,
        default=1000,
        help="Number of calibration executions (default: 1000)",
    )
    est_parser.set_defaults(func=cmd_estimate)

    # --- ppmd ---
    ppmd_parser = subparsers.add_parser(
        "ppmd",
        help="Analyze corpus compressibility with PPMD",
    )
    ppmd_parser.add_argument(
        "-d",
        "--corpus",
        required=True,
        help="Corpus directory",
    )
    ppmd_parser.add_argument(
        "-g",
        "--graph",
        default=None,
        help="Output graph PNG file (e.g. graph.png)",
    )
    ppmd_parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Show top N most/least novel seeds (default: 10)",
    )
    ppmd_parser.set_defaults(func=cmd_ppmd)

    args = parser.parse_args()

    # Default to fuzz if no subcommand given
    if args.command is None:
        # Re-parse with fuzz defaults for backwards compatibility
        sys.argv.insert(1, "fuzz")
        args = parser.parse_args()

    return args.func(args)
