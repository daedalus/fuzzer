"""CLI commands for fuzzer-tool."""

import argparse
import builtins
import datetime
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from fuzzer_tool.core.mutations import load_dictionary
from fuzzer_tool.services.fuzzer import Fuzzer

_original_print = builtins.print
_patched_print = builtins.print


def _enable_timestamp_print() -> None:
    """Monkey-patch builtins.print to prefix every message with a timestamp."""
    global _patched_print
    import logging as _logging

    def _timestamped_print(*args, **kwargs):
        ts = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        if args:
            first = args[0]
            if first.startswith("\n"):
                args = (ts + " " + first.lstrip("\n"),) + args[1:]
            else:
                args = (f"{ts} {first}",) + args[1:]
        else:
            args = (ts,)
        _original_print(*args, **kwargs)

    _patched_print = _timestamped_print
    builtins.print = _patched_print

    root = _logging.getLogger()
    if not any(isinstance(h, _logging.StreamHandler) for h in root.handlers):
        handler = _logging.StreamHandler()
        handler.setFormatter(
            _logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        root.addHandler(handler)


def _mail_config_from_args(args):
    """Build MailConfig from CLI, or None when --send-email-on-crash is unset."""
    to = getattr(args, "send_email_on_crash", None)
    if not to:
        return None
    from fuzzer_tool.services.sendmail import MailConfig

    return MailConfig.from_cli(
        to=to,
        smtp_server=getattr(args, "send_mail_smtp_server", None),
        auth=getattr(args, "send_mail_auth", None),
        require_tls=getattr(args, "send_mail_require_tls", False),
        from_addr=getattr(args, "send_email_from", None),
        subject=getattr(args, "send_email_subject", None),
    )


def _load_hash_list(path: str | None) -> set[str] | None:
    """Load a file of hex hashes (one per line) into a set."""
    if not path:
        return None
    try:
        lines = Path(path).read_text().splitlines()
        return {line.strip() for line in lines if line.strip() and not line.startswith("#")}
    except FileNotFoundError:
        print(f"[!] Warning: hash list file not found: {path}")
        return None


def _detect_asan(target: str) -> bool:
    """Detect if a binary is ASAN-instrumented by checking for __asan_init symbol."""
    for flags in [[], ["-D"]]:
        try:
            r = subprocess.run(["nm"] + flags + [target], capture_output=True, timeout=10)
            if r.returncode == 0 and (
                b"__asan_init" in r.stdout or b"__asan_register_globals" in r.stdout
            ):
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    return False


def _detect_ubsan(target: str) -> bool:
    """Detect if a binary is UBSAN-instrumented by checking for __ubsan_handle_* symbols."""
    for flags in [[], ["-D"]]:
        try:
            r = subprocess.run(["nm"] + flags + [target], capture_output=True, timeout=10)
            if r.returncode == 0 and b"__ubsan_handle" in r.stdout:
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


def _run_fuzzer(fuzzer, args):
    """Run the campaign, optionally under cProfile.

    Extracted from cmd_fuzz so the --log-json handle can be closed in a
    single finally, rather than duplicating cleanup down both the profiled
    and unprofiled arms.
    """
    if not getattr(args, "profile_hotpath", False):
        fuzzer.run(iterations=args.iterations)
        return

    import cProfile
    import pstats

    pr = cProfile.Profile()
    pr.enable()
    try:
        fuzzer.run(iterations=args.iterations)
    finally:
        pr.disable()
        stats = pstats.Stats(pr)
        builtins.print = _original_print
        print("\n" + "=" * 80)
        print(" TOP 60 BY TOTAL TIME (tottime) — self-time, no children")
        print("=" * 80)
        stats.sort_stats("tottime")
        stats.print_stats(60)
        print("\n" + "=" * 80)
        print(" TOP 40 BY CUMULATIVE TIME (cumtime) — self + callee time")
        print("=" * 80)
        stats.sort_stats("cumtime")
        stats.print_stats(40)
        print("\n" + "=" * 80)
        print(" TOP 30 BY CALL COUNT (ncalls)")
        print("=" * 80)
        stats.sort_stats("ncalls")
        stats.print_stats(30)
        profile_out = getattr(args, "profile_out", "/tmp/fuzzer_hotpath.prof")
        stats.dump_stats(profile_out)
        builtins.print = _patched_print
        print(f"[*] cProfile stats saved to {profile_out}")


def cmd_fuzz(args):
    """Main fuzzing command."""
    # Applied before anything can spawn a child, so the setting is in force
    # for the whole run including early teardown.
    if getattr(args, "no_kill_children", False):
        from fuzzer_tool.services.fuzzer import set_kill_children_enabled

        set_kill_children_enabled(False)
        print("[*] Child process groups will NOT be killed on exit")

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
    # The verify_asan_link_order=0 shim and libasan are preloaded
    # via ctypes in fuzzer.py to make ASAN .so targets loadable
    # in direct mode.  ASAN-detected bugs may still abort the
    # process — the user accepts this tradeoff with --inprocess-direct.
    if target_is_asan and getattr(args, "inprocess_direct", False):
        print("[*] ASAN detected — will preload ASAN runtime for in-process mode")

    dictionary = []
    if args.dict:
        if not os.path.isfile(args.dict):
            print(f"[-] Dictionary not found: {args.dict}")
            sys.exit(1)
        dictionary = load_dictionary(args.dict)
        print(f"[*] Loaded {len(dictionary)} tokens from {args.dict}")

    # QEA and GA now run simultaneously when both are set — they share the
    # corpus and edge tracker but maintain independent populations, competing
    # for coverage discoveries and corpus slots.
    if getattr(args, "qea", False) and getattr(args, "ga", False):
        print("[*] --qea and --ga both set: both enabled (competing mode)")

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
        if getattr(args, "profile_hotpath", False):
            print(
                "[*] --profile-hotpath ignored in parallel mode (--jobs > 1); "
                "profiling applies to single-process fuzz runs"
            )
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
            sharpe_kelly_blend=getattr(args, "sharpe_kelly_blend", 0.0),
            stats_file=args.stats_file,
            stats_interval=args.stats_interval,
            coverage_report=args.coverage_report,
            iterations=args.iterations,
            sync_interval=args.sync_interval,
            seed=args.seed,
            secretary=getattr(args, "secretary", False),
            secretary_window=getattr(args, "secretary_window", 500),
            secretary_exploration=getattr(args, "secretary_exploration", 0.368),
            overlap_density=getattr(args, "overlap_density", False),
            overlap_density_mode=getattr(args, "overlap_mode", "modifier"),
            overlap_min_jaccard=getattr(args, "overlap_min_jaccard", 0.25),
            overlap_density_blend=getattr(args, "overlap_blend", 0.5),
            resize_map_on_stall=getattr(args, "resize_map_on_stall", True),
            exp3=getattr(args, "exp3", False),
            invasion=getattr(args, "invasion", False),
            exp3_gamma=getattr(args, "exp3_gamma", 0.1),
            eps_greedy=getattr(args, "eps_greedy", False),
            eps_greedy_epsilon0=getattr(args, "eps_greedy_epsilon0", 1.0),
            eps_greedy_decay=getattr(args, "eps_greedy_decay", 0.9995),
            hierarchical_bandit=getattr(args, "hierarchical_bandit", False),
            gp_ucb=getattr(args, "gp_ucb", False),
            ducb=getattr(args, "ducb", False),
            ducb_gamma=getattr(args, "ducb_gamma", 0.9999),
            swucb=getattr(args, "swucb", False),
            swucb_window=getattr(args, "swucb_window", 4000),
            cucb=getattr(args, "cucb", False),
            cucb_gamma=getattr(args, "cucb_gamma", 0.9995),
            gp_length_scale=getattr(args, "gp_length_scale", 1.0),
            gp_beta=getattr(args, "gp_beta", 2.0),
            contextual=getattr(args, "contextual", False),
            contextual_alpha=getattr(args, "contextual_alpha", 1.0),
            contextual_lambda=getattr(args, "contextual_lambda", 1.0),
            asan_target=getattr(args, "asan_target", None),
            ubsan_target=getattr(args, "ubsan_target", None),
            chi2_operator_interval=getattr(args, "chi2_operator_interval", 0),
            lineage=getattr(args, "lineage", False),
            lineage_backtrack=getattr(args, "lineage_backtrack", False),
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

    # --elo all: enable every available meta-scheduler and seed scheduler so Elo
    # arbitrates them (previously only the operator meta-schedulers were enabled,
    # leaving the seed strategies listed-but-disabled with phantom matches)
    if getattr(args, "elo", None) == "all":
        args.mc_bandit = True
        args.mc_cem = True
        args.mopt = True
        args.replicator = True
        args.exp3 = True
        args.eps_greedy = True
        args.hierarchical_bandit = True
        args.gp_ucb = True
        args.ducb = True
        args.swucb = True
        args.cucb = True
        args.contextual = True
        args.invasion = True
        args.ga = True
        args.qea = True
        args.bayesian = True
        args.boltzmann = True
        args.markov_gen = True
        # Mutation-side schedulers/features that are not Elo-arbitrated but are
        # part of the scheduling stack; flip them on so --elo all is the
        # everything-on switch (power schedule fast = classic AFL default)
        args.metropolis = True
        args.shapley = True
        args.mi_guided = True
        args.secretary = True
        args.wfc = True
        args.lineage = True
        args.mcts = True
        args.schedule = "fast"

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
        cmaes=getattr(args, "cma_es", False),
        cmaes_pop_size=getattr(args, "cmaes_pop_size", 8),
        cmaes_generation_size=getattr(args, "cmaes_generation_size", 200),
        cmaes_step_size=getattr(args, "cmaes_step_size", 0.3),
        cmaes_elite_frac=getattr(args, "cmaes_elite_frac", 0.5),
        targets=getattr(args, "target_functions", None),
        use_cfg_cache=not getattr(args, "no_cfg_cache", False),
        anneal_budget=getattr(args, "anneal_budget", 0),
        boltzmann=getattr(args, "boltzmann", False),
        metropolis=getattr(args, "metropolis", False),
        mc_elite_frac=args.mc_elite_frac,
        mc_refit_interval=args.mc_refit_int,
        mc_decay_interval=getattr(args, "mc_decay_interval", 100),
        pairwise_blend=getattr(args, "pairwise_blend", 0.0),
        sharpe_kelly_blend=getattr(args, "sharpe_kelly_blend", 0.0),
        stats_file=args.stats_file,
        stats_interval=args.stats_interval,
        coverage_report=args.coverage_report,
        coverage_log=coverage_log_arg,
        stack_heartbeat=(
            str(Path(corpus_dir) / ".fuzz_stack.txt")
            if getattr(args, "stack_heartbeat", None) == "__auto__"
            else getattr(args, "stack_heartbeat", None)
        ),
        grammar=grammar,
        persistent=args.persistent,
        net_host=getattr(args, "net_host", None),
        net_port=getattr(args, "net_port", None),
        net_proto=getattr(args, "net_proto", "tcp"),
        net_keepalive=getattr(args, "net_keepalive", False),
        net_settle_ms=getattr(args, "net_settle_ms", 10),
        calibrate_stability=getattr(args, "calibrate_stability", 0),
        cmplog=args.cmplog,
        cmplog_max_tokens=getattr(args, "cmplog_max_tokens", 0),
        cmplog_max_pairs=getattr(args, "cmplog_max_pairs", 0),
        cmplog_workdir=getattr(args, "cmplog_workdir", None)
        or None,  # Will fall back to cachedir in CmplogCollector
        max_corpus=args.max_corpus,
        max_corpus_bytes=getattr(args, "max_corpus_bytes", 0),
        minimize_every_execs=getattr(args, "minimize_every_execs", 0),
        prune_corpus_max_memory=getattr(args, "prune_corpus_on_max_memory", 80),
        bootstrap=getattr(args, "bootstrap", False),
        bootstrap_k=getattr(args, "bootstrap_k", 1),
        no_shm=args.no_shm,
        use_ptrace=args.ptrace,
        adaptive_havoc=not getattr(args, "no_adaptive_havoc", False),
        adaptive_timeout=getattr(args, "adaptive_timeout", False),
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
        asan_target=getattr(args, "asan_target", None),
        ubsan_target=getattr(args, "ubsan_target", None),
        crash_blocklist=_load_hash_list(getattr(args, "crash_blocklist", None)),
        crash_allowlist=_load_hash_list(getattr(args, "crash_allowlist", None)),
        save_smaller=getattr(args, "save_smaller", False),
        honggfuzz=getattr(args, "honggfuzz", False),
        hw_perf=getattr(args, "hw_perf", False),
        schedule_ablation=getattr(args, "schedule_ablation", None),
        schedule=getattr(args, "schedule", "base"),
        aflgo_cooling=getattr(args, "aflgo_cooling", "exp"),
        t_x_minutes=getattr(args, "t_x", 60.0),
        replicator=getattr(args, "replicator", False),
        exp3=getattr(args, "exp3", False),
        invasion=getattr(args, "invasion", False),
        exp3_gamma=getattr(args, "exp3_gamma", 0.1),
        eps_greedy=getattr(args, "eps_greedy", False),
        eps_greedy_epsilon0=getattr(args, "eps_greedy_epsilon0", 1.0),
        eps_greedy_decay=getattr(args, "eps_greedy_decay", 0.9995),
        hierarchical_bandit=getattr(args, "hierarchical_bandit", False),
        gp_ucb=getattr(args, "gp_ucb", False),
        ducb=getattr(args, "ducb", False),
        ducb_gamma=getattr(args, "ducb_gamma", 0.9999),
        swucb=getattr(args, "swucb", False),
        swucb_window=getattr(args, "swucb_window", 4000),
        cucb=getattr(args, "cucb", False),
        cucb_gamma=getattr(args, "cucb_gamma", 0.9995),
        gp_length_scale=getattr(args, "gp_length_scale", 1.0),
        gp_beta=getattr(args, "gp_beta", 2.0),
        contextual=getattr(args, "contextual", False),
        contextual_alpha=getattr(args, "contextual_alpha", 1.0),
        contextual_lambda=getattr(args, "contextual_lambda", 1.0),
        shapley=getattr(args, "shapley", False),
        bayesian=getattr(args, "bayesian", False),
        mi_guided=getattr(args, "mi_guided", False),
        renyi_weight=getattr(args, "renyi_weight", False),
        transfer_entropy=getattr(args, "transfer_entropy", False),
        elo=getattr(args, "elo", False),
        lineage=getattr(args, "lineage", False),
        lineage_backtrack=getattr(args, "lineage_backtrack", False),
        secretary=getattr(args, "secretary", False),
        secretary_window=getattr(args, "secretary_window", 500),
        secretary_exploration=getattr(args, "secretary_exploration", 0.368),
        overlap_density=getattr(args, "overlap_density", False),
        overlap_density_mode=getattr(args, "overlap_mode", "modifier"),
        overlap_min_jaccard=getattr(args, "overlap_min_jaccard", 0.25),
        overlap_density_blend=getattr(args, "overlap_blend", 0.5),
        sensitivity=getattr(args, "sensitivity", False),
        region_profile=getattr(args, "region_profile", False),
        fluctuation=getattr(args, "fluctuation_theorems", False),
        deterministic=getattr(args, "deterministic", True),
        forkserver=getattr(args, "forkserver", True),
        ga=getattr(args, "ga", False),
        qea=getattr(args, "qea", False),
        wfc=getattr(args, "wfc", False),
        path_negation=getattr(args, "path_negation", False),
        mcts=getattr(args, "mcts", False),
        ga_pop_size=getattr(args, "ga_pop_size", 200),
        ga_gen_size=getattr(args, "ga_gen_size", 500),
        ga_elite_frac=getattr(args, "ga_elite_frac", 0.1),
        ga_crossover_rate=getattr(args, "ga_crossover_rate", 0.7),
        ga_mutation_rate=getattr(args, "ga_mutation_rate", 0.3),
        ga_tournament_size=getattr(args, "ga_tournament_size", 3),
        ga_speciation_threshold=getattr(args, "ga_speciation_threshold", 0.3),
        qea_rotation_angle=getattr(args, "qea_rotation_angle", 0.05),
        qea_strong_bias=getattr(args, "qea_strong_bias", None),
        qea_elite_reset=getattr(args, "qea_elite_reset", 0),
        qea_correlation=getattr(args, "qea_correlation", False),
        qea_correlation_delta=getattr(args, "qea_correlation_delta", 0.02),
        qea_correlation_max=getattr(args, "qea_correlation_max", 2.0),
        qea_correlation_sweeps=getattr(args, "qea_correlation_sweeps", 3),
        qea_cooling=getattr(args, "qea_cooling", False),
        qea_cooling_decay=getattr(args, "qea_cooling_decay", 0.98),
        qea_cooling_min_angle=getattr(args, "qea_cooling_min_angle", 0.005),
        continue_until_crash=getattr(args, "continue_until_crash", False),
        calibrate=getattr(args, "calibrate", 0),
        stall_threshold=getattr(args, "stall", 1000),
        map_size=getattr(args, "map_size", 0),
        max_collision_risk=getattr(args, "max_collision_risk", 30),
        debug=getattr(args, "debug", False),
        enable_regex_bomb=getattr(args, "enable_regex_bomb_mutations", False),
        colorize=getattr(args, "colorize", False),
        colorize_max_execs=getattr(args, "colorize_max_execs", 512),
        weizz_tags=getattr(args, "weizz_tags", False),
        weizz_tags_max_len=getattr(args, "weizz_tags_max_len", 8192),
        email_on_crash=_mail_config_from_args(args),
        enable_x86_mutator=getattr(args, "x86_mutate", False),
        enable_arm_mutator=getattr(args, "arm_mutate", False),
        refresh_profile=getattr(args, "refresh_profile", False),
        corpus_boost=getattr(args, "corpus_boost", 0),
        boost_mean=getattr(args, "boost_mean", None),
        boost_std=getattr(args, "boost_std", None),
        boost_pad=getattr(args, "boost_pad", "repeat"),
        seed_skip_size=getattr(args, "seed_skip_size", 0),
        seed_truncate_size=getattr(args, "seed_truncate_size", 0),
        seed_slide_size=getattr(args, "seed_slide_size", 0),
        seed_slide_max_seeds=getattr(args, "seed_slide_max_seeds", 0),
        resize_map_on_stall=getattr(args, "resize_map_on_stall", False),
        reseed_on_stall=getattr(args, "reseed_on_stall", False),
        enable_smt_z3=getattr(args, "enable_smt_z3", False),
        mod_solving=getattr(args, "mod_solving", "heuristic"),
        chi2_operator_interval=getattr(args, "chi2_operator_interval", 0),
        quiet_stats=False,
        no_save_state=getattr(args, "no_save_state", False),
        dedup_execs=not getattr(args, "no_dedup_execs", False),
        perf_novelty=not getattr(args, "no_perf_novelty", False),
        reject_code=getattr(args, "reject_code", None),
    )
    # shlex.join, not " ".join: this string is now persisted into state.json
    # and printed as the command that reproduces the run, so an argument
    # containing a space or a quote (a --target-args value, a corpus path)
    # has to survive being copy-pasted back into a shell.
    fuzzer.invocation = shlex.join(sys.argv)

    # --log-json: opened here rather than inside the Fuzzer so the handle's
    # lifetime is bounded by the try/finally below. "-" means stderr, which
    # keeps the stream separate from the human stats line on stdout so a
    # consumer can read either one cleanly.
    log_json_path = getattr(args, "log_json", None)
    if log_json_path == "-":
        fuzzer._log_json_fh = sys.stderr
    elif log_json_path:
        Path(log_json_path).parent.mkdir(parents=True, exist_ok=True)
        # Line-buffered append: a campaign is usually killed rather than
        # exited, so records must already be on disk when that happens, and
        # a --resume run should extend the series rather than truncate it.
        # noqa SIM115: the handle deliberately outlives this statement; it
        # is owned by the try/finally below, which is the context manager.
        fuzzer._log_json_fh = open(log_json_path, "a", buffering=1)  # noqa: SIM115

    try:
        _run_fuzzer(fuzzer, args)
    finally:
        fh = getattr(fuzzer, "_log_json_fh", None)
        if fh is not None and fh is not sys.stderr:
            fh.close()
        fuzzer._log_json_fh = None

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
        build_autotoken_dictionary,
        import_from_afl,
        import_from_honggfuzz,
        import_from_libfuzzer,
        write_dictionary,
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

    if getattr(args, "autotokens", None):
        tokens = build_autotoken_dictionary(args.corpus)
        write_dictionary(tokens, args.autotokens)
        print(f"[+] Wrote {len(tokens)} autotokens to {args.autotokens}")
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
        lineage=getattr(args, "lineage", False),
        corpus_dir=getattr(args, "corpus_dir", None),
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


def cmd_root_cause(args):
    """Root-cause byte diff subcommand: isolate the minimal edit from a
    non-crashing baseline that is responsible for the crash."""
    _validate_target(args.target)
    from fuzzer_tool.services.root_cause import root_cause

    result = root_cause(
        target=args.target,
        crash_file=args.crash_file,
        corpus_dir=getattr(args, "corpus_dir", None),
        baseline_file=getattr(args, "baseline", None),
        timeout=args.timeout,
        file_mode=args.file_mode,
        target_args=args.target_args,
        use_coverage=args.coverage,
        max_stages=args.max_stages,
    )

    if result is None:
        return 1

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(result["report"])
        print(f"[+] Report saved to {args.output}")
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
    import math
    import time

    from fuzzer_tool.adapters.filesystem import load_corpus
    from fuzzer_tool.core.bloom import BloomFilter
    from fuzzer_tool.core.edge_tracker import EdgeTracker
    from fuzzer_tool.core.elf import estimate_map_size
    from fuzzer_tool.core.state_store import StateStore

    corpus_dir = Path(args.corpus)
    if not corpus_dir.is_dir():
        print(f"[-] Corpus not found: {corpus_dir}", file=sys.stderr)
        return 1

    store = StateStore(corpus_dir)
    store.load()

    bloom = BloomFilter(capacity=100_000)
    corpus, seen_hashes, irreplaceable_hashes = load_corpus(corpus_dir, bloom)
    if not corpus:
        print("[-] Empty corpus", file=sys.stderr)
        return 1

    map_size = estimate_map_size(args.target)
    et = EdgeTracker(map_size=map_size)
    et_data = store.get("edge_tracker")
    if et_data is not None:
        et.from_dict(et_data)

    # Load seed metadata from state store
    from fuzzer_tool.services.corpus_manager import seed_key as seed_key_for

    seed_meta = {}
    now = time.time()
    state = store.get("corpus")
    if state:
        saved = state.get("seed_meta", {})
        for seed in corpus:
            # Same key scheme as CorpusManager.save_state: content hash,
            # falling back to the legacy seed.hex() key for older states.
            key = seed_key_for(seed)
            if key not in saved:
                legacy = seed.hex()
                if legacy in saved:
                    key = legacy
            if key in saved:
                sm = saved[key]
                seed_meta[seed] = {
                    "fuzz_count": sm.get("fuzz_count", 0),
                    "coverage_edges": sm.get("coverage_edges", 0),
                    "added_at": sm.get("added_at", now),
                }
            else:
                seed_meta[seed] = {"fuzz_count": 0, "coverage_edges": 0, "added_at": now}

    if not seed_meta:
        for seed in corpus:
            seed_meta[seed] = {"fuzz_count": 0, "coverage_edges": 0, "added_at": now}

    def score(seed):
        meta = seed_meta.get(seed, {})
        fuzz_count = max(meta.get("fuzz_count", 0), 1)
        coverage = meta.get("coverage_edges", 0)
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
            for i, (_score, seed) in enumerate(scored[:n]):
                h = hashlib.sha256(seed).hexdigest()[:16]
                f.write(seed)
                print(f"  wrote seed #{i + 1} ({len(seed)} bytes) -> {out}.{i}")
        # Also write each seed to a separate file
        for i, (_score, seed) in enumerate(scored[:n]):
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
    corpus, _, _ = load_corpus(corpus_dir, bloom)

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
    print(
        f"  Total compressed:{sum(int(s * r) for s, r in zip(sizes, ratios, strict=False)):,} bytes"
    )
    print(
        f"  Corpus ratio:    {sum(int(s * r) for s, r in zip(sizes, ratios, strict=False)) / sum(sizes):.4f}"
    )

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


def _probe_so_function_sw(target: str) -> str:
    """Probe a shared object for the best fuzz entry point (sweep)."""
    import subprocess as _sp

    try:
        r = _sp.run(["nm", "-D", target], capture_output=True, text=True, timeout=5)
        syms = r.stdout
    except OSError:
        syms = ""
    if "LLVMFuzzerTestOneInput" in syms:
        return "LLVMFuzzerTestOneInput"
    for line in syms.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[-1].startswith("fuzz_"):
            return parts[-1]
    return "LLVMFuzzerTestOneInput"


def _run_so_target(so_path: str, data: bytes, timeout: float) -> tuple[int, str, int]:
    """Execute a .so fuzz target in a subprocess via ctypes.

    Spawns a Python subprocess that loads the .so, calls the fuzz function,
    and exits with the return code. The subprocess provides crash isolation
    — target SIGSEGV/SIGABRT kills only the child, not the fuzzer.

    Returns (returncode, stderr, child_pid).
    """
    import subprocess as _sp
    import sys as _sys

    func_name = _probe_so_function_sw(so_path)

    _script = (
        "import ctypes,sys;"
        "d=sys.stdin.buffer.read();"
        "lib=ctypes.CDLL(sys.argv[1]);"
        "fn=getattr(lib,sys.argv[2]);"
        "fn.restype=ctypes.c_int;"
        "fn.argtypes=[ctypes.POINTER(ctypes.c_uint8),ctypes.c_size_t];"
        "buf=(ctypes.c_uint8*len(d))(*d);"
        "rc=fn(buf,len(buf));"
        "sys.exit(max(0,min(rc,125)))"
    )
    try:
        proc = _sp.Popen(
            [_sys.executable, "-c", _script, so_path, func_name],
            stdin=_sp.PIPE,
            stdout=_sp.DEVNULL,
            stderr=_sp.PIPE,
        )
        _stdout, stderr_b = proc.communicate(input=data, timeout=timeout)
        rc = proc.returncode or 0
        return rc, stderr_b.decode(errors="replace"), proc.pid
    except _sp.TimeoutExpired:
        proc.kill()
        proc.wait()
        return -1, "timeout", proc.pid
    except Exception as e:
        return -2, str(e), 0


def cmd_sweep(args):
    """Linearly scan corpus seeds for missed crashes.

    Loads every seed from the corpus, runs it against the target without
    mutations, scheduler, coverage tracking, or minification. Discovers
    seeds that happen to crash the target — inputs added to the corpus
    during fuzzing that triggered no coverage event but still crash.
    """
    _validate_target(args.target)

    # Detect .so target and switch to in-process ctypes execution
    target_is_so = args.target.lower().endswith((".so", ".dylib", ".dll"))

    corpus_dir = Path(args.corpus)
    if not corpus_dir.is_dir():
        print(f"[-] Corpus dir not found: {args.corpus}", file=sys.stderr)
        return 1

    from fuzzer_tool.adapters.filesystem import hash_data, load_corpus

    seeds, _, _ = load_corpus(corpus_dir, bloom=None, add_default=False)
    if not seeds:
        print("[-] No seeds found in corpus")
        return 0

    # Optionally limit the number of seeds to sweep
    max_seeds = getattr(args, "max_seeds", 0)
    if max_seeds > 0 and len(seeds) > max_seeds:
        seeds = seeds[:max_seeds]

    crashes_dir = Path(args.crashes) if args.crashes else corpus_dir / "crashes"
    crashes_dir.mkdir(parents=True, exist_ok=True)

    import time as _time

    from fuzzer_tool.adapters.process import (
        SIGNAL_CRASH_CODES,
        run_target_file,
        run_target_stdin,
    )
    from fuzzer_tool.core.sanitizer import SanitizerReport

    found = 0
    total = len(seeds)
    seeds.sort(key=lambda s: hash_data(s))
    _sweep_start = _time.monotonic()

    for i, seed in enumerate(seeds):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"\r[*] Sweeping seed {i + 1}/{total}...", end="", file=sys.stderr)
            sys.stderr.flush()

        try:
            if target_is_so:
                returncode, stderr, _ = _run_so_target(args.target, seed, timeout=args.timeout)
            elif args.file_mode:
                tmp_dir = Path("/tmp") / f"sweep_{os.getpid()}"
                tmp_dir.mkdir(parents=True, exist_ok=True)
                try:
                    returncode, stderr, _ = run_target_file(
                        target=args.target,
                        data=seed,
                        timeout=args.timeout,
                        tmp_dir=str(tmp_dir),
                        target_args=args.target_args or [],
                        env=os.environ.copy(),
                    )
                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
            else:
                returncode, stderr, _ = run_target_stdin(
                    target=args.target,
                    data=seed,
                    timeout=args.timeout,
                    env=os.environ.copy(),
                )
        except Exception as e:
            print(f"\n  [!] Error on seed {i + 1}/{total}: {e}", file=sys.stderr)
            continue

        # Check for crash
        report = SanitizerReport.parse(stderr)
        is_crash = bool(report and report.is_valid())
        if not is_crash:
            is_crash = abs(returncode) in SIGNAL_CRASH_CODES
        if not is_crash:
            is_crash = returncode < 0
        if not is_crash:
            is_crash = any(
                sig in stderr
                for sig in [
                    "SIGSEGV",
                    "SIGABRT",
                    "Segmentation fault",
                    "Aborted",
                ]
            )

        if is_crash:
            found += 1
            h = hash_data(seed)
            sig = report.signature if report and report.is_valid() else f"signal{abs(returncode)}"
            crash_name = f"crash_{h[:12]}_{sig}"
            crash_path = crashes_dir / crash_name
            if not crash_path.exists():
                crash_path.write_bytes(seed)
            print(f"\n  [+] Crash: rc={returncode}, hash={h[:12]} -> {crash_name}")

    _elapsed = _time.monotonic() - _sweep_start
    _eps = total / _elapsed if _elapsed > 0 else 0
    print(
        f"\n[*] Sweep complete: {total} seeds processed, {found} crashes found ({_eps:.0f} seeds/s)"
    )
    return 0


# store_true / BooleanOptionalAction dests that represent an opt-in feature,
# strategy, or diagnostic (i.e. "off by default, flipping it on only adds
# behavior"). --hail-mary force-enables every one of these that the user
# didn't already touch on the command line. Deliberately excluded: the
# "--no-*" opt-outs (forkserver, deterministic, shm, cfg-cache, adaptive-havoc,
# kill-children, dedup-execs, save-state, perf-novelty) since those are
# already on by default and this flag is additive, not destructive;
# --resume, since forcing it on would fail outright with no prior state;
# --refresh-profile and --profile-hotpath, since hail-mary is already slow
# and adding full profiling/cProfile overhead per iteration makes it
# untrackable rather than exploratory.
_HAIL_MARY_FLAGS = (
    "continue_until_crash",
    "deep_coverage",
    "ptrace",
    "adaptive_timeout",
    "file_mode",
    "markov",
    "markov_gen",
    "markov_blend",
    "mc_bandit",
    "mc_cem",
    "mopt",
    "cma_es",
    "replicator",
    "shapley",
    "bayesian",
    "mi_guided",
    "renyi_weight",
    "transfer_entropy",
    "lineage",
    "lineage_backtrack",
    "exp3",
    "eps_greedy",
    "hierarchical_bandit",
    "gp_ucb",
    "ducb",
    "swucb",
    "cucb",
    "contextual",
    "invasion",
    "overlap_density",
    "secretary",
    "sensitivity",
    "region_profile",
    "ga",
    "qea",
    "mcts",
    "fluctuation_theorems",
    "wfc",
    "path_negation",
    "enable_smt_z3",
    "qea_correlation",
    "qea_cooling",
    "boltzmann",
    "metropolis",
    "auto_timeout",
    "bootstrap",
    "trace",
    "learn_format",
    "corpus_ppmd",
    "persistent",
    "net_keepalive",
    "inprocess",
    "inprocess_direct",
    "save_smaller",
    "honggfuzz",
    "hw_perf",
    "debug",
    "colorize",
    "weizz_tags",
    "enable_regex_bomb_mutations",
    "x86_mutate",
    "arm_mutate",
    "reseed_on_stall",
)


def _apply_hail_mary(args: argparse.Namespace, fuzz_parser: argparse.ArgumentParser) -> None:
    """Force-enable every opt-in fuzzing flag left at its default value."""
    for dest in _HAIL_MARY_FLAGS:
        if getattr(args, dest, None) == fuzz_parser.get_default(dest):
            setattr(args, dest, True)

    # --elo takes a value ('all' turns on every meta/seed/mutation scheduler
    # it arbitrates between), not a plain bool -- special-case it.
    if args.elo == fuzz_parser.get_default("elo"):
        args.elo = "all"

    # --cmplog is a tri-state (None=auto, True=on, False=off); force it on.
    if args.cmplog == fuzz_parser.get_default("cmplog"):
        args.cmplog = True

    # Boltzmann/Metropolis annealing is inert without a nonzero budget --
    # give it one so enabling the flags actually does something.
    if args.anneal_budget == fuzz_parser.get_default("anneal_budget"):
        args.anneal_budget = 10000

    print(
        "[hail-mary] every left-at-default fuzzing option has been force-enabled; "
        "expect a slow, noisy, exploratory run."
    )


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(
        prog="fuzzer-tool",
        description="Coverage-guided binary fuzzer with crash analysis tools",
    )
    parser.add_argument(
        "--print-timestamp",
        action="store_true",
        help="Prefix every stdout/stderr message with a wall-clock timestamp",
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
    fuzz_parser.add_argument(
        "--hail-mary",
        action="store_true",
        help="Kitchen-sink mode: flip on every optional scheduler, mutation "
        "strategy, and diagnostic flag at once (all bandits, GA/QEA/CMA-ES, "
        "Markov, lineage/MCTS, SMT/path-negation, honggfuzz/hw-perf, tracing, "
        "etc.). Explicit flags you also pass are left as you set them; only "
        "options still at their default get force-enabled. Very slow, very "
        "noisy, exploratory last resort -- not a normal run mode.",
    )
    fuzz_parser.add_argument("-M", "--mutations", type=int, default=8, help="Mutations per input")
    fuzz_parser.add_argument(
        "-c",
        "--coverage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Coverage-guided mode: AFL SHM edge bitmap drives corpus admission "
            "and scheduling (default: on). --no-coverage runs blind — crash "
            "detection still works, but the corpus will not grow. -c is "
            "accepted and is now a no-op."
        ),
    )
    fuzz_parser.add_argument(
        "--deep-coverage",
        action="store_true",
        help="Enable basic block discovery via x86-64 decoder",
    )
    fuzz_parser.add_argument(
        "--max-bps", type=int, default=50000, help="Max breakpoints for deep coverage"
    )
    fuzz_parser.add_argument(
        "--no-shm",
        action="store_true",
        help="Skip AFL SHM coverage, use ptrace instead (for uninstrumented binaries)",
    )
    fuzz_parser.add_argument(
        "--no-cfg-cache",
        action="store_true",
        help="Disable the on-disk CFG decode cache (~/.cache/fuzzer_cfgcache); "
        "directed-mode distance setup always decodes fresh",
    )
    fuzz_parser.add_argument(
        "--ptrace",
        action="store_true",
        help="Enable ptrace self-trace in the persistent loader for per-crash "
        "fault-address/register capture on .so targets (adds per-exec overhead; "
        "disabled by default)",
    )
    fuzz_parser.add_argument(
        "--no-adaptive-havoc",
        action="store_true",
        help="Draw havoc's 11 inline sub-mutations uniformly instead of weighting "
        "them by measured new-coverage rate (adaptive weighting is on by default)",
    )
    fuzz_parser.add_argument(
        "--adaptive-timeout",
        action="store_true",
        help="Retune --timeout during the run from the measured execution-time "
        "distribution (p99 + 1 sd) instead of holding the value fixed. Off by "
        "default: the suggestion is derived from one target's observed timings "
        "and is not a safe global",
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
    fuzz_parser.add_argument(
        "--sharpe-kelly-blend",
        type=float,
        default=0.0,
        metavar="W",
        help="Blend weight for Sharpe/Kelly risk-adjusted operator selection "
        "(0.0=off, 1.0=pure Sharpe/Kelly). When > 0, per-operator reward "
        "moments are tracked and Thompson draws are blended with "
        "risk-normalized scores to down-weight high-variance lottery tickets.",
    )
    fuzz_parser.add_argument("--mc-cem", action="store_true", help="Enable cross-entropy method")
    fuzz_parser.add_argument(
        "--mopt",
        action="store_true",
        help="Enable MOpt PSO operator scheduling (alternative to bandit)",
    )
    fuzz_parser.add_argument(
        "--cma-es",
        action="store_true",
        help="Enable CMA-ES operator scheduling (covariance-adapted continuous optimization)",
    )
    fuzz_parser.add_argument(
        "--cmaes-pop-size",
        type=int,
        default=8,
        metavar="N",
        help="CMA-ES population size (default: 8)",
    )
    fuzz_parser.add_argument(
        "--cmaes-generation-size",
        type=int,
        default=200,
        metavar="N",
        help="CMA-ES evaluations per generation (default: 200)",
    )
    fuzz_parser.add_argument(
        "--cmaes-step-size",
        type=float,
        default=0.3,
        metavar="SIGMA",
        help="CMA-ES initial step size sigma (default: 0.3)",
    )
    fuzz_parser.add_argument(
        "--cmaes-elite-frac",
        type=float,
        default=0.5,
        metavar="FRAC",
        help="CMA-ES elite fraction mu/lambda (default: 0.5)",
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
        nargs="?",
        const=True,
        default=False,
        metavar="all",
        help="Enable Elo scheduling: arbitrates between operator and seed strategies. "
        "Pass 'all' to also enable all available meta-schedulers (bandit, MOpt, "
        "replicator, EXP3, eps-greedy, hierarchical, GP-UCB), seed schedulers "
        "(GA, QEA, Bayesian, Markov, Boltzmann) and the mutation scheduling stack "
        "(Metropolis, Shapley, MI-guided, secretary, WFC, lineage, fast power schedule).",
    )
    fuzz_parser.add_argument(
        "--lineage",
        action="store_true",
        help="Track a weighted mutation lineage tree (parent/ops/sites/new-edge weight "
        "per seed): branch-level pruning, causal crash-path replay in tmin, "
        "LCA-based diversity scoring",
    )
    fuzz_parser.add_argument(
        "--lineage-backtrack",
        action="store_true",
        help="Widen exploration by backing off exhausted lineage branches: "
        "seeds whose subtree has gained no edges are penalised by depth, "
        "shifting selection back toward shallow seeds with unexplored "
        "siblings (implies --lineage)",
    )
    fuzz_parser.add_argument(
        "--exp3", action="store_true", help="Enable EXP3 adversarial bandit operator scheduling"
    )
    fuzz_parser.add_argument(
        "--invasion",
        action="store_true",
        help="Enable invasion percolation operator selection as an additional Elo-arbitrated "
        "strategy: always picks the operator with the lowest resistance (inverse observed "
        "success rate) from the MC bandit's stats. Requires --mc-bandit; has no effect "
        "without it.",
    )
    fuzz_parser.add_argument(
        "--exp3-gamma",
        type=float,
        default=0.1,
        help="EXP3 exploration rate in [0,1] (default: 0.1)",
    )
    fuzz_parser.add_argument(
        "--eps-greedy",
        action="store_true",
        help="Enable epsilon-greedy operator scheduling with annealing",
    )
    fuzz_parser.add_argument(
        "--eps-greedy-epsilon0",
        type=float,
        default=1.0,
        help="Initial epsilon for epsilon-greedy (default: 1.0)",
    )
    fuzz_parser.add_argument(
        "--eps-greedy-decay",
        type=float,
        default=0.9995,
        help="Epsilon decay rate per pull (default: 0.9995)",
    )
    fuzz_parser.add_argument(
        "--hierarchical-bandit",
        action="store_true",
        help="Enable hierarchical bandit operator scheduling (category -> operator)",
    )
    fuzz_parser.add_argument(
        "--gp-ucb",
        action="store_true",
        help="Enable GP-UCB operator scheduling with kernel covariance",
    )
    fuzz_parser.add_argument(
        "--gp-length-scale",
        type=float,
        default=1.0,
        help="GP kernel RBF length scale (default: 1.0)",
    )
    fuzz_parser.add_argument(
        "--gp-beta",
        type=float,
        default=2.0,
        help="GP-UCB exploration parameter (default: 2.0)",
    )
    fuzz_parser.add_argument(
        "--ducb",
        action="store_true",
        help=(
            "Enable discounted-UCB operator scheduling (Garivier & Moulines): "
            "exponentially discounted counts and rewards, so an operator whose "
            "yield collapses is re-tried instead of exploited forever"
        ),
    )
    fuzz_parser.add_argument(
        "--ducb-gamma",
        type=float,
        default=0.9999,
        help=(
            "D-UCB discount per pull; effective memory is 1/(1-gamma) pulls "
            "and must stay well above the operator count (default: 0.9999)"
        ),
    )
    fuzz_parser.add_argument(
        "--swucb",
        action="store_true",
        help=(
            "Enable sliding-window UCB operator scheduling (Garivier & "
            "Moulines): only the last --swucb-window pulls count"
        ),
    )
    fuzz_parser.add_argument(
        "--swucb-window",
        type=int,
        default=4000,
        help="SW-UCB window length in pulls (default: 4000)",
    )
    fuzz_parser.add_argument(
        "--cucb",
        action="store_true",
        help=(
            "Enable combinatorial UCB operator scheduling (Chen et al.): "
            "scores the round's whole operator stack as one superarm and "
            "recovers per-operator rates by inclusion contrast, instead of "
            "handing every operator in the stack the same shared outcome"
        ),
    )
    fuzz_parser.add_argument(
        "--cucb-gamma",
        type=float,
        default=0.9995,
        help="CUCB discount per mutation round (default: 0.9995)",
    )
    fuzz_parser.add_argument(
        "--contextual",
        action="store_true",
        help=(
            "Enable LinUCB contextual bandit operator scheduling "
            "(per-arm ridge regression over seed features: size, entropy, "
            "format, edge coverage, lineage depth, cmplog availability, "
            "corpus-size percentile, and per-op cost)"
        ),
    )
    fuzz_parser.add_argument(
        "--contextual-alpha",
        type=float,
        default=1.0,
        help="LinUCB exploration weight on the confidence bound (default: 1.0)",
    )
    fuzz_parser.add_argument(
        "--contextual-lambda",
        type=float,
        default=1.0,
        help="LinUCB ridge regularization strength (default: 1.0)",
    )
    fuzz_parser.add_argument(
        "--overlap-density",
        action="store_true",
        default=False,
        help="Enable FMM-clustered pairwise overlap density in seed selection (boosts novel seeds, penalises redundant ones)",
    )
    fuzz_parser.add_argument(
        "--overlap-mode",
        choices=["modifier", "pareto4d"],
        default="modifier",
        help="How to integrate overlap density: 'modifier' (weight penalty) or 'pareto4d' (4th Pareto dimension) (default: modifier)",
    )
    fuzz_parser.add_argument(
        "--overlap-min-jaccard",
        type=float,
        default=0.25,
        help="Minimum Jaccard similarity for LSH clustering in overlap density (default: 0.25)",
    )
    fuzz_parser.add_argument(
        "--overlap-blend",
        type=float,
        default=0.5,
        help="Blend factor for overlap density weight modifier, 0-1 (default: 0.5)",
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
        "--region-profile",
        action="store_true",
        help=(
            "Enable statistical region profiling for mutation targeting "
            "(labels seed windows incompressible/tabular/textual/repetitive "
            "and weights byte selection accordingly)"
        ),
    )
    fuzz_parser.add_argument(
        "--chi2-operator-interval",
        type=int,
        default=0,
        metavar="N",
        help="Run chi-squared operator heterogeneity test every N iterations (0=disabled, default: 0)",
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
        "--mcts",
        action="store_true",
        help="Enable MCTS/UCT seed scheduling over the mutation lineage tree "
        "(implies --lineage; competes as an Elo-arbitrated seed strategy)",
    )
    fuzz_parser.add_argument(
        "--fluctuation-theorems",
        action="store_true",
        help="Enable Jarzynski/Crooks fluctuation-theorem diagnostics over mutation "
        "trajectories (speculative; opt-in diagnostics only by default)",
    )
    fuzz_parser.add_argument(
        "--wfc",
        action="store_true",
        help="Enable Wave Function Collapse structural generation (chunk reordering, pixel generation)",
    )
    fuzz_parser.add_argument(
        "--no-kill-children",
        action="store_true",
        help="Do not SIGKILL child process groups on exit. Use when the fuzzer "
        "is embedded, supervised, or run under a debugger; target processes may "
        "then outlive the fuzzer",
    )
    fuzz_parser.add_argument(
        "--path-negation",
        action="store_true",
        help="Concolic path-condition negation: solve for inputs that take the "
        "opposite side of a branch the run actually took (requires --cmplog and z3)",
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
        "--qea-rotation-angle",
        type=float,
        default=0.05,
        help="QEA rotation gate magnitude; 0.0 disables amplitude feedback "
        "entirely (the zero-coupling endpoint) (default: 0.05)",
    )
    fuzz_parser.add_argument(
        "--qea-strong-bias",
        type=float,
        default=None,
        help="QEA amplitude bias toward a parent's committed bytes; 0.5 is "
        "no bias, i.e. uniform amplitudes (default: ALPHA_STRONG)",
    )
    fuzz_parser.add_argument(
        "--qea-elite-reset",
        type=int,
        default=0,
        metavar="N",
        help="Breed the full QEA population every N generations instead of "
        "carrying elites forward; 0 disables (default: 0)",
    )
    fuzz_parser.add_argument(
        "--qea-correlation",
        action="store_true",
        help="Enable intra-byte coupling: each QEA individual learns an 8x8 "
        "pairwise correlation matrix per byte (Hebbian-updated alongside the "
        "rotation gate) so collapse can bias toward bit combinations that "
        "worked together, not just individually-likely bits. Off by default "
        "-- existing amplitude-only behavior is unchanged either way.",
    )
    fuzz_parser.add_argument(
        "--qea-correlation-delta",
        type=float,
        default=0.02,
        help="Per-pair coupling learning rate when --qea-correlation is set (default: 0.02)",
    )
    fuzz_parser.add_argument(
        "--qea-correlation-max",
        type=float,
        default=2.0,
        help="Clip magnitude for a single coupling entry when "
        "--qea-correlation is set (default: 2.0)",
    )
    fuzz_parser.add_argument(
        "--qea-correlation-sweeps",
        type=int,
        default=3,
        help="Gibbs sweeps per correlated collapse when --qea-correlation "
        "is set; more sweeps mix closer to the joint distribution's "
        "stationary point at proportionally higher cost (default: 3)",
    )
    fuzz_parser.add_argument(
        "--qea-cooling",
        action="store_true",
        help="Enable algorithmic cooling: decay the rotation gate's step "
        "angle across generations instead of holding it constant, so "
        "search takes larger steps early and smaller ones as the "
        "population settles. Anchored to --qea-elite-reset's cycle -- Δθ "
        "resets to full strength at every elite reset -- so cooling and "
        "resets cooperate instead of one flattening the other; with "
        "--qea-elite-reset=0 the decay instead runs against the raw "
        "generation count for the rest of the session. Off by default -- "
        "existing constant-angle behavior is unchanged either way.",
    )
    fuzz_parser.add_argument(
        "--qea-cooling-decay",
        type=float,
        default=0.98,
        help="Per-generation multiplicative decay applied to the rotation "
        "angle when --qea-cooling is set (default: 0.98)",
    )
    fuzz_parser.add_argument(
        "--qea-cooling-min-angle",
        type=float,
        default=0.005,
        help="Floor (radians) the rotation angle never decays past when "
        "--qea-cooling is set (default: 0.005)",
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
        "--target-functions",
        dest="target_functions",
        nargs="+",
        default=None,
        metavar="FUNC",
        help="Target functions for directed fuzzing — names, hex addresses, "
        "or file.c:line (via DWARF). Note: use --target-functions (not the "
        "positional 'targets' binary list).",
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
        "--boltzmann",
        action="store_true",
        default=False,
        help="Boltzmann seed selection: P(seed) ∝ exp(-E/T) with E=log(fuzz_count+1). "
        "Requires --anneal-budget > 0.",
    )
    fuzz_parser.add_argument(
        "--metropolis",
        action="store_true",
        default=False,
        help="Metropolis corpus admission: accept non-improving inputs with P=exp(-ΔE/T). "
        "Requires --anneal-budget > 0.",
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
        "--stack-heartbeat",
        nargs="?",
        const="__auto__",
        metavar="FILE",
        default=None,
        help=(
            "Write the main-thread Python stack to FILE every few seconds "
            "(default: <corpus>/.fuzz_stack.txt). Since SIGKILL (kill -9) is "
            "uncatchable, the last stack file shows where the fuzzer was "
            "executing. A live trace is also available any time via "
            "`kill -USR1 <pid>`, and SIGTERM/SIGINT dump the stack by default."
        ),
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
    # Tri-state: None = auto (enable when the target is detected as
    # cmplog/tracecmp-instrumented), True = force on, False = force off.
    # Magic-value and checksum branches are where edge discovery plateaus on
    # real formats, and _detect_cmplog() identifies instrumented targets
    # reliably, so making the user remember the flag cost coverage for no
    # reason. --no-cmplog is the opt-out, matching --no-forkserver.
    fuzz_parser.add_argument(
        "--calibrate-stability",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Re-run each newly accepted seed N times and mask edges that do "
            "not reproduce (nondeterministic: ASLR, time, uninitialized "
            "memory). 0 disables. Use 8: measured against a target with "
            "known-unstable edges, N=3 recovers the full unstable set only "
            "35%% of the time and misses it entirely 6%% of the time, while "
            "N=8 recovers it 87%% of the time. See "
            "docs/sweeps/synthetic_target_ground_truth_2026-08-19.md. "
            "Costs N extra executions per accepted seed."
        ),
    )
    fuzz_parser.add_argument(
        "--cmplog",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Comparison tracing via LD_PRELOAD "
            "(memcmp/strcmp/strncmp/memchr interception). "
            "Default: on when the target is detected as instrumented; "
            "--no-cmplog forces it off."
        ),
    )
    fuzz_parser.add_argument(
        "--no-forkserver",
        dest="forkserver",
        action="store_false",
        default=True,
        help=(
            "Disable the C forkserver on the default execution path, falling back to "
            "posix_spawn per execution. On by default; the forkserver replaces a full "
            "ELF load + dynamic linker + libc init per exec with a fork from an "
            "already-loaded process."
        ),
    )
    fuzz_parser.add_argument(
        "--no-deterministic",
        dest="deterministic",
        action="store_false",
        default=True,
        help=(
            "Disable the AFL-style deterministic stage (bitflip/byte-flip/arithmetic/"
            "interesting-value sweep) for favored seeds before havoc. On by default; "
            "a full pass costs 8*len(seed) execs per favored seed."
        ),
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
        "--cmplog-workdir",
        type=str,
        default=None,
        help="Directory for cmplog runtime log files (default ~/.cache/fuzzer_cmplog)",
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
        "--bootstrap",
        action="store_true",
        help="Enable bootstrap percolation post-pass in corpus minimization "
        "(removes transitively-redundant seeds that single-pass set-cover leaves behind)",
    )
    fuzz_parser.add_argument(
        "--bootstrap-k",
        type=int,
        default=1,
        help="Minimum unique edges required to keep a seed during bootstrap percolation (default: 1)",
    )
    fuzz_parser.add_argument(
        "--corpus-boost",
        type=int,
        default=0,
        metavar="MAX_LEN",
        help="Resize corpus seed lengths to follow a normal distribution capped at MAX_LEN",
    )
    fuzz_parser.add_argument(
        "--boost-mean",
        type=float,
        default=None,
        help="Target mean for normal size distribution (default: corpus_boost/2)",
    )
    fuzz_parser.add_argument(
        "--boost-std",
        type=float,
        default=None,
        help="Target std for normal size distribution (default: corpus_boost/6)",
    )
    fuzz_parser.add_argument(
        "--boost-pad",
        choices=["repeat", "zero", "random"],
        default="repeat",
        help="Padding mode for short seeds: repeat (cycle self), zero (\\x00), random",
    )
    fuzz_parser.add_argument(
        "--seed-skip-size",
        type=int,
        default=0,
        help="Skip seeds larger than this size (0=disabled)",
    )
    fuzz_parser.add_argument(
        "--seed-truncate-size",
        type=int,
        default=0,
        help="Truncate seeds to this size (0=disabled)",
    )
    fuzz_parser.add_argument(
        "--seed-slide-size",
        type=int,
        default=0,
        help="Slide a window of this size over each seed, replacing the corpus with windows (0=disabled)",
    )
    fuzz_parser.add_argument(
        "--seed-slide-max-seeds",
        type=int,
        default=0,
        help="Cap the total number of seeds after sliding (0=unlimited)",
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
        "--log-json",
        metavar="FILE",
        default=None,
        help="Write one JSON object per stats interval to FILE ('-' for stderr) "
        "for machine-parseable output. Appends, so --resume extends the series.",
    )
    fuzz_parser.add_argument(
        "--no-save-state",
        action="store_true",
        help="Do not persist fuzzer state at exit (no state.pkl.gz written)",
    )
    fuzz_parser.add_argument(
        "--no-dedup-execs",
        action="store_true",
        help="Do not filter already-executed mutants through the exec bloom filter",
    )
    fuzz_parser.add_argument(
        "--reject-code",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Exit code the harness uses to say it rejected the input "
            "(Zest validity channel: coverage reached on accepted inputs is "
            "tracked separately, and a valid input covering new valid ground "
            "is saved even when its total coverage is old)"
        ),
    )
    fuzz_parser.add_argument(
        "--no-perf-novelty",
        action="store_true",
        help=(
            "Do not treat a substantially increased per-edge hit count as novelty "
            "(disables PerfFuzz-style algorithmic-complexity discovery)"
        ),
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
        "--asan-target",
        default=None,
        metavar="PATH",
        help="ASAN-instrumented target variant for sanitizer crash replay",
    )
    fuzz_parser.add_argument(
        "--ubsan-target",
        default=None,
        metavar="PATH",
        help="UBSAN-instrumented target variant for sanitizer crash replay",
    )
    fuzz_parser.add_argument(
        "--crash-blocklist",
        default=None,
        metavar="FILE",
        help="File with stack hashes (one per line) to skip when saving crashes",
    )
    fuzz_parser.add_argument(
        "--crash-allowlist",
        default=None,
        metavar="FILE",
        help="File with stack hashes that override the blocklist (always save these)",
    )
    fuzz_parser.add_argument(
        "--send-email-on-crash",
        default=None,
        metavar="EMAIL",
        help=(
            "Send a triage report email (with .bin/.txt/.sh/.hex attachments) "
            "to EMAIL whenever a *novel* crash is saved. Uses the system MTA "
            "(sendmail -t) unless --send-mail-smtp-server is set."
        ),
    )
    fuzz_parser.add_argument(
        "--send-email-from",
        default=None,
        metavar="EMAIL",
        help=(
            "From: address for --send-email-on-crash (default: auth user if it "
            "looks like an email, otherwise fuzzer-tool@<hostname>)"
        ),
    )
    fuzz_parser.add_argument(
        "--send-email-subject",
        default=None,
        metavar="TEXT",
        help=(
            "Subject line for --send-email-on-crash. Optional placeholders: "
            "{target}, {target_base}, {base_name}, {returncode}, {exec_count}. "
            "Default: [fuzzer-tool] crash {base_name} ({target_base})"
        ),
    )
    fuzz_parser.add_argument(
        "--send-mail-smtp-server",
        default=None,
        metavar="HOST[:PORT]",
        help=(
            "SMTP server for --send-email-on-crash (default port 25; use "
            "HOST:587 with --send-mail-require-tls for submission). When "
            "omitted, the system sendmail/MTA is used instead."
        ),
    )
    fuzz_parser.add_argument(
        "--send-mail-auth",
        default=None,
        metavar="USER:PASSWORD",
        help="SMTP AUTH credentials for --send-mail-smtp-server (USER:PASSWORD)",
    )
    fuzz_parser.add_argument(
        "--send-mail-require-tls",
        action="store_true",
        help=(
            "Require STARTTLS (or use implicit SSL on port 465) when talking "
            "to --send-mail-smtp-server"
        ),
    )
    fuzz_parser.add_argument(
        "--save-smaller",
        action="store_true",
        help="Replace crash triggers with smaller inputs for the same stack hash",
    )
    fuzz_parser.add_argument(
        "--honggfuzz",
        action="store_true",
        help="Enable honggfuzz power factors (novelty decay, freshness, fertility, density, entropy penalty, timeout penalty)",
    )
    fuzz_parser.add_argument(
        "--hw-perf",
        action="store_true",
        help="Enable hardware performance counters (instructions, branches, branch_misses) via perf_event_open. Requires CAP_PERFMON or root.",
    )
    fuzz_parser.add_argument(
        "--schedule-ablation",
        default=None,
        metavar="FILE",
        help="Log per-iteration scheduling signal contributions to CSV for backtesting",
    )
    fuzz_parser.add_argument(
        "--schedule",
        default="base",
        choices=("base", "fast", "coe", "rare", "mopt", "lin", "quad", "go", "aflgo", "entropic"),
        help="Power schedule: base|fast|coe|rare|mopt|lin|quad|go|aflgo|entropic "
        "(aflgo = exact AFLGo distance annealing, see --t-x; "
        "entropic = libFuzzer -entropic, log-scaled rare-feature energy)",
    )
    fuzz_parser.add_argument(
        "--aflgo-cooling",
        default="exp",
        choices=("exp", "log", "lin", "quad"),
        help="Cooling schedule for the aflgo power factor (default: exp)",
    )
    fuzz_parser.add_argument(
        "--t-x",
        type=float,
        default=60.0,
        metavar="MINUTES",
        help="AFLGo time-to-exploitation in minutes; temperature cools to "
        "1/20 of its start over this window (default: 60)",
    )
    fuzz_parser.add_argument(
        "--differential",
        default=None,
        metavar="TARGET_B",
        help="Differential fuzzing: run each input through a second target and flag divergence",
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
        "--net-host",
        default=None,
        metavar="IP",
        help="Network mode: destination IPv4 address of a persistent TCP/UDP target",
    )
    fuzz_parser.add_argument(
        "--net-port",
        type=int,
        default=None,
        metavar="PORT",
        help="Network mode: destination port (requires --net-host)",
    )
    fuzz_parser.add_argument(
        "--net-proto",
        default="tcp",
        choices=("tcp", "udp"),
        help="Network mode: transport protocol (default: tcp)",
    )
    fuzz_parser.add_argument(
        "--net-keepalive",
        action="store_true",
        help="Network mode: reuse one connection across iterations instead of reconnecting each time",
    )
    fuzz_parser.add_argument(
        "--net-settle-ms",
        type=int,
        default=10,
        metavar="MS",
        help="Network mode: grace period after send() before reading coverage (default: 10; "
        "no reply is ever read, so this is a fixed guess, not a real sync signal)",
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
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resize the SHM coverage bitmap when stall recovery triggers, "
        "reducing hash collision risk and potentially exposing new edges. "
        "Uses birthday-bound (n^2/0.02) to compute the new size. "
        "(default: enabled; use --no-resize-map-on-stall to disable)",
    )
    fuzz_parser.add_argument(
        "--reseed-on-stall",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reseed the RNGs when stall recovery triggers, forcing the "
        "mutation stream to explore a different region of the input space "
        "(default: disabled)",
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
        "--profile-hotpath",
        action="store_true",
        help="Profile the fuzz run with cProfile (tottime/cumtime/ncalls tables + .prof dump)",
    )
    fuzz_parser.add_argument(
        "--profile-out",
        default="/tmp/fuzzer_hotpath.prof",
        metavar="PATH",
        help="cProfile dump path for --profile-hotpath (default: /tmp/fuzzer_hotpath.prof)",
    )
    fuzz_parser.add_argument(
        "--colorize",
        action="store_true",
        help=(
            "Colorize seeds before redqueen matching (AFL++/Redqueen): replace every "
            "byte that does not change the execution path, so coincidental occurrences "
            "of a comparison operand stop counting as input-to-state matches. Costs "
            "executions; off by default"
        ),
    )
    fuzz_parser.add_argument(
        "--colorize-max-execs",
        type=int,
        default=512,
        metavar="N",
        help="Per-seed execution budget for --colorize (default: 512)",
    )
    fuzz_parser.add_argument(
        "--weizz-tags",
        action="store_true",
        help=(
            "Build Weizz-style structure tags from cmplog (+ optional colorize) "
            "and enable field/chunk operators that consume them. Off by default; "
            "see docs/handover/handover_weizz_structure_aware_port_2026-08-31.md"
        ),
    )
    fuzz_parser.add_argument(
        "--weizz-tags-max-len",
        type=int,
        default=8192,
        metavar="N",
        help=(
            "Skip Weizz tag collection for seeds longer than N bytes "
            "(Weizz -L analogue; default: 8192)"
        ),
    )
    fuzz_parser.add_argument(
        "--enable-regex-bomb-mutations",
        action="store_true",
        help="Enable regex backtracking bomb mutations (ReDoS patterns that cause explosive memory usage)",
    )
    fuzz_parser.add_argument(
        "--x86-mutate",
        action="store_true",
        help="Enable x86 instruction-stream mutations (decode-and-mutate binary code)",
    )
    fuzz_parser.add_argument(
        "--arm-mutate",
        action="store_true",
        help="Enable ARM instruction-stream mutations (decode-and-mutate binary code)",
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
    tmin_parser.add_argument(
        "--lineage",
        action="store_true",
        help="Causal crash-path replay: walk the mutation lineage chain from the "
        "parent seed and rehydrate pruned intermediates by hash",
    )
    tmin_parser.add_argument(
        "--corpus-dir",
        default=None,
        help="Corpus directory for lineage rehydration of pruned intermediates",
    )
    tmin_parser.set_defaults(func=cmd_tmin)

    # --- root-cause ---
    rc_parser = subparsers.add_parser(
        "root-cause",
        help="Isolate the minimal byte diff from a non-crashing input that causes a crash",
    )
    rc_parser.add_argument("target", help="Path to target binary")
    rc_parser.add_argument("crash_file", help="Path to crashing input file")
    rc_parser.add_argument("-t", "--timeout", type=float, default=1, help="Timeout in seconds")
    rc_parser.add_argument(
        "-F", "--file-mode", action="store_true", help="Write input to temp file instead of stdin"
    )
    rc_parser.add_argument(
        "-A",
        "--target-args",
        nargs=argparse.REMAINDER,
        help="Target arguments ({file} placeholder)",
    )
    rc_parser.add_argument("-c", "--coverage", action="store_true", help="Enable SHM coverage")
    rc_parser.add_argument(
        "-d",
        "--corpus-dir",
        default=None,
        help="Corpus directory to search for the nearest non-crashing seed",
    )
    rc_parser.add_argument(
        "-b",
        "--baseline",
        default=None,
        help="Explicit non-crashing input to diff against (skips corpus search)",
    )
    rc_parser.add_argument(
        "--max-stages", type=int, default=200, help="Max ddmin stages (default: 200)"
    )
    rc_parser.add_argument(
        "-O", "--output", default=None, help="Save the root-cause report to a file"
    )
    rc_parser.set_defaults(func=cmd_root_cause)

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
    import_parser.add_argument(
        "--autotokens",
        default=None,
        metavar="FILE",
        help="Also tokenize the destination corpus into a whole-token "
        "AFL-format dictionary written to FILE",
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

    # --- sweep ---
    sweep_parser = subparsers.add_parser("sweep", help="Linearly scan corpus for missed crashes")
    sweep_parser.add_argument("target", help="Path to target binary")
    sweep_parser.add_argument("-d", "--corpus", required=True, help="Corpus directory")
    sweep_parser.add_argument("-o", "--crashes", default=None, help="Crashes output directory")
    sweep_parser.add_argument(
        "-t", "--timeout", type=float, default=1, help="Timeout per seed in seconds"
    )
    sweep_parser.add_argument(
        "-F",
        "--file-mode",
        action="store_true",
        help="Write input to temp file instead of stdin",
    )
    sweep_parser.add_argument(
        "-A",
        "--target-args",
        nargs=argparse.REMAINDER,
        help="Target arguments ({file} placeholder)",
    )
    sweep_parser.add_argument(
        "-n", "--max-seeds", type=int, default=0, help="Max seeds to sweep (0=all)"
    )
    sweep_parser.set_defaults(func=cmd_sweep)

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

    if getattr(args, "print_timestamp", False):
        _enable_timestamp_print()

    if args.command == "fuzz" and getattr(args, "hail_mary", False):
        _apply_hail_mary(args, fuzz_parser)

    return args.func(args)
