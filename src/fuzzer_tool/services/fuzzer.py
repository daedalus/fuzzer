"""Fuzzer orchestration: coordinates mutations, execution, and coverage."""

import atexit
import contextlib
import logging
import math
import os
import random
import resource
import shutil
import signal
import sys
import tempfile
import threading
import time
from array import array
from pathlib import Path

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

from fuzzer_tool.adapters.process import (
    _child_pids,
    disable_aslr,
)
from fuzzer_tool.adapters.shm import ShmCoverage
from fuzzer_tool.core.bloom import BloomFilter
from fuzzer_tool.core.markov import MarkovChain, MarkovEnsemble
from fuzzer_tool.core.mi import MI_MAX_POSITIONS, MutualInformationTracker
from fuzzer_tool.core.operator_registry import REGISTRY
from fuzzer_tool.core.running_stats import RunningMoments
from fuzzer_tool.core.sanitizer import SanitizerReport
from fuzzer_tool.core.schedulers import (
    ContextualLinUCBScheduler,
    EpsilonGreedyScheduler,
    Exp3Scheduler,
    GPUCBScheduler,
    HierarchicalBanditScheduler,
    MonteCarloScheduler,
    MOptScheduler,
    ReplicatorScheduler,
)
from fuzzer_tool.core.schedules import SeedScorer, compute_mean_log_n_fuzz
from fuzzer_tool.core.secretary import DEFAULT_EXPLORATION_FRAC, SecretaryStopping
from fuzzer_tool.core.seed_quality import BayesianSeedQuality
from fuzzer_tool.core.shapley import ShapleyAttribution
from fuzzer_tool.services.corpus_manager import CorpusManager
from fuzzer_tool.services.operators import OperatorEngine
from fuzzer_tool.services.ptrace_coverage import (
    PtraceCoverage,
)
from fuzzer_tool.services.runner import TargetRunner, ptrace_available
from fuzzer_tool.services.seed_picker import SeedPicker
from fuzzer_tool.services.stats import StatsReporter

log = logging.getLogger(__name__)

_shutdown = False

# Strategy names pre-registered with the Elo tracker (single source of truth
# for the pre-registration loop and the meta-scheduler log line).
_OPERATOR_STRATEGY_NAMES = (
    "replicator",
    "bandit",
    "mopt",
    "cem",
    "exp3",
    "eps_greedy",
    "hierarchical",
    "gp_ucb",
    "contextual",
)
_SEED_STRATEGY_NAMES = (
    "ga",
    "qea",
    "weighted",
    "pareto",
    "format",
    "bayesian",
    "markov",
    "boltzmann",
)


_kill_children_enabled = os.environ.get("FUZZER_DISABLE_KILL_CHILDREN", "") not in (
    "1",
    "true",
    "yes",
)
"""Whether teardown SIGKILLs child process groups.

On by default: a fuzzer that exits leaving target processes behind will
exhaust the machine over a long campaign. It is switchable because the
teardown is destructive and not always wanted — when the fuzzer is embedded
in a larger process, driven by a supervisor that manages its own children,
or run under a debugger where killing the group would take the debugger with
it. The environment variable is read at import because the handlers install
at import; ``set_kill_children_enabled`` changes it afterwards, and the CLI
``--no-kill-children`` flag routes through that.
"""


def set_kill_children_enabled(enabled: bool) -> None:
    """Enable or disable the destructive part of teardown.

    Shutdown signalling still happens when disabled — only the SIGKILL of
    child process groups is suppressed, so the fuzzing loop still stops
    cleanly.
    """
    global _kill_children_enabled
    _kill_children_enabled = bool(enabled)


def _kill_children(sig=None, frame=None):
    global _shutdown
    _shutdown = True
    # SIGTERM/SIGINT are catchable: show where the fuzzer was executing
    # before tearing down children — a live answer to "what is it doing?".
    if sig is not None:
        try:
            import faulthandler

            faulthandler.dump_traceback()
        except Exception:
            pass
    if not _kill_children_enabled:
        return
    try:
        own_pgid = os.getpgrp()
    except OSError:
        own_pgid = None
    for pid in _child_pids():
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            pgid = os.getpgid(pid)
            # Children call os.setsid(), so a child's pgid is its own. If it
            # matches ours the pid was recorded before setsid ran, or the pid
            # has been reused — killing that group would SIGKILL the fuzzer
            # and everything sharing its group.
            if own_pgid is not None and pgid == own_pgid:
                continue
            os.killpg(pgid, signal.SIGKILL)


def install_cleanup_handlers() -> bool:
    """Register teardown on atexit, SIGTERM and SIGINT.

    Returns False if the handlers could not be installed. ``signal.signal``
    only works on the main thread, so importing this module from a worker
    thread previously raised at import time; that is now reported rather
    than fatal.
    """
    atexit.register(_kill_children)
    try:
        signal.signal(signal.SIGTERM, _kill_children)
        signal.signal(signal.SIGINT, _kill_children)
    except (ValueError, OSError):
        return False
    return True


install_cleanup_handlers()

# On-demand live trace: `kill -USR1 <fuzzer-pid>` dumps every thread's
# Python stack to stderr (faulthandler).  SIGKILL (kill -9) is uncatchable
# in-process — for that, run with --stack-heartbeat, whose periodic
# main-thread stack file survives the kill.
try:
    import faulthandler as _faulthandler

    _faulthandler.register(signal.SIGUSR1)
except (AttributeError, OSError):
    pass


def _handle_sigsegv(signum, frame):
    """Handle SIGSEGV in the fuzzer process itself."""
    import traceback

    print("\n[FATAL] Segmentation fault in fuzzer process!")
    print(f"Signal: {signum}")
    if frame:
        print(f"Frame: {frame}")
    traceback.print_stack(frame)
    sys.exit(1)


signal.signal(signal.SIGSEGV, _handle_sigsegv)


def _cleanup_tmp_dir(path: Path) -> None:
    """Remove temp directory on exit."""

    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        log.debug("Failed to clean up %s", path, exc_info=True)


# ── Entropy rate tracking constants ─────────────────────────────────
ENTROPY_HISTORY_MAX = 200  # max samples before trimming
ENTROPY_HISTORY_TRIM = 100  # keep this many after trim
ENTROPY_WINDOW = 4  # samples for rate-of-change computation
ENTROPY_FLAT_THRESHOLD = 0.001  # rate below which entropy is "flat"

# ── Memory bounds ────────────────────────────────────────────────────
CRASH_RATE_HISTORY_MAX = 500  # max entries in _crash_rate_execs/_crash_rate_counts
MAX_CRASH_SIGS = 10_000  # max unique crash signatures before pruning old entries
KERNEL_CRASHES_MAX = 500  # max kernel-verified crashes retained
SEED_SECRETARY_MAX = 500  # max per-seed SecretaryStopping entries
SEEN_HASHES_MAX = 200_000  # max unique seed hashes retained
EXEC_BLOOM_CAPACITY = 500_000  # executed-input filter capacity before generational wipe
EXEC_DEDUP_RETRIES = 3  # re-rolls of the mutation before executing a repeat anyway
ELO_MATCH_WINDOW_MAX = 1_000  # max Elo match history entries
META_STRATEGY_CHOICES_MAX = 1_000  # max meta-strategy choice history entries
# ── Allan variance detector ───────────────────────────────────────────
ALLAN_BUFFER_POW = 8  # 2^8 = 256 samples
ALLAN_MIN_SAMPLES = 8  # minimum before noise_type() returns a result
# ── Stall reseeding (--reseed-on-stall) ───────────────────────────────
# splitmix64 constants: the derived seed must decorrelate from `self.seed`
# even though it is a small additive offset away from it.  A bare
# `seed + count` would hand adjacent stalls near-identical Mersenne
# Twister states.
SEED_MIX_GAMMA = 0x9E3779B97F4A7C15  # odd, golden-ratio derived
SEED_MIX_A = 0xBF58476D1CE4E5B9
SEED_MIX_B = 0x94D049BB133111EB
SEED_MASK_64 = 0xFFFFFFFFFFFFFFFF
SEED_MASK_32 = 0xFFFFFFFF  # np.random.seed accepts [0, 2**32)


def _detect_afl(target_path: str) -> bool:
    """Check if a binary has AFL edge coverage instrumentation."""
    import subprocess

    try:
        result = subprocess.run(
            ["nm", target_path],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return "__afl_area" in result.stdout or "__afl_map_shm" in result.stdout
    except (OSError, subprocess.TimeoutExpired):
        return False


def _detect_distance(target_path: str) -> bool:
    """Check if a binary has the AFLGo distance channel compiled in.

    Distance builds define __afl_dist_flush (the shim under
    __AFL_DISTANCE_MODE) and define __sanitizer_cov_trace_pc (trace-pc
    instrumentation); the bare trace_pc symbol distinguishes them from
    plain trace-pc-guard builds, which only carry the *_guard callbacks.
    """
    import re
    import subprocess

    try:
        result = subprocess.run(
            ["nm", target_path],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if "__afl_dist_flush" in result.stdout:
            return True
        return bool(re.search(r"^[tT] __sanitizer_cov_trace_pc$", result.stdout, re.MULTILINE))
    except (OSError, subprocess.TimeoutExpired):
        return False


def _detect_cmplog(target_path: str) -> bool:
    """Check if a binary has cmplog or tracecmp built in.

    Recognizes either the symbol-based shim (__cmplog_reset) or the
    compiler-IR shim (__tracecmp_reset).
    """
    import subprocess

    cmplog_symbols = ("__cmplog_reset", "__tracecmp_reset")
    try:
        for flags in [[], ["-D"]]:
            result = subprocess.run(
                ["nm"] + flags + [target_path], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                for sym in cmplog_symbols:
                    if sym in result.stdout:
                        return True
    except (OSError, subprocess.TimeoutExpired):
        pass
    return False


def _detect_tracecmp_target(target_path: str) -> bool:
    """Check if a binary was compiled with -fsanitize-coverage=trace-cmp.

    Targets compiled with trace-cmp have undefined (U) references to
    __sanitizer_cov_trace_cmp{1,2,4,8} that must be resolved at runtime
    by tracecmp_shim.so or LD_PRELOAD.
    """
    import subprocess

    target_syms = ("__sanitizer_cov_trace_cmp1",)
    try:
        for flags in [[], ["-D"]]:
            result = subprocess.run(
                ["nm"] + flags + [target_path], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                for sym in target_syms:
                    if sym in result.stdout:
                        return True
    except (OSError, subprocess.TimeoutExpired):
        pass
    return False


def _detect_asan(target_path: str) -> bool:
    """Detect if a binary is ASAN-instrumented by checking for __asan_init symbol."""
    import subprocess

    for flags in [[], ["-D"]]:
        try:
            r = subprocess.run(["nm"] + flags + [target_path], capture_output=True, timeout=10)
            if r.returncode == 0 and (
                b"__asan_init" in r.stdout or b"__asan_register_globals" in r.stdout
            ):
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    return False


def _detect_ubsan(target_path: str) -> bool:
    """Detect if a binary is UBSAN-instrumented by checking for __ubsan_handle_* symbols."""
    import subprocess

    for flags in [[], ["-D"]]:
        try:
            r = subprocess.run(["nm"] + flags + [target_path], capture_output=True, timeout=10)
            if r.returncode == 0 and b"__ubsan_handle" in r.stdout:
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    return False


class Fuzzer:
    def _warn_no_coverage(self) -> None:
        """Warn that an in-process target is running without coverage.

        Emitted once per run. Without an SHM segment nothing populates the
        edge bitmap, so every coverage-guided subsystem downstream — seed
        scheduling, MI/TE/sensitivity position weighting, Elo/bandit operator
        scheduling, stall detection, corpus admission — runs on a
        constant-zero signal. The run looks fast and healthy while
        discovering nothing, so the failure is otherwise entirely silent.
        """
        if getattr(self, "_no_cov_warned", False):
            return
        self._no_cov_warned = True
        msg = (
            "No coverage enabled: running blind. Edge discovery, "
            "coverage-guided scheduling and corpus growth are all inactive. "
            "Pass -c/--coverage to enable the AFL SHM bitmap."
        )
        log.warning(msg)
        print(f"[!] WARNING: {msg}")

    @staticmethod
    def _probe_so_function(target):
        """Probe a shared object for the best fuzz entry point.

        Uses nm -D to scan symbols without loading the library (avoids
        ASAN issues with ctypes.CDLL loading order).
        Falls back to fuzz_shm_run if nothing is found.
        """
        import subprocess

        try:
            result = subprocess.run(
                ["nm", "-D", target],
                capture_output=True,
                text=True,
                timeout=5,
            )
            symbols = result.stdout
        except (OSError, subprocess.TimeoutExpired):
            return "fuzz_shm_run"

        # Prefer the standard wrapper
        if "fuzz_shm_run" in symbols:
            return "fuzz_shm_run"

        # Scan for any fuzz_* symbol
        for line in symbols.splitlines():
            parts = line.split()
            if len(parts) >= 3:
                name = parts[-1]
                if name.startswith("fuzz_"):
                    return name

        return "fuzz_shm_run"

    def __init__(
        self,
        target,
        corpus_dir,
        crashes_dir,
        max_len=4096,
        timeout=1,
        mutations_per_input=8,
        use_coverage=False,
        deep_coverage=False,
        max_bps=50000,
        dictionary=None,
        file_mode=False,
        target_args=None,
        markov_order=1,
        markov_generate=False,
        markov_blend=False,
        mc_bandit=False,
        mc_cem=False,
        mopt=False,
        targets=None,
        anneal_budget=0,
        boltzmann=False,
        metropolis=False,
        mc_elite_frac=0.1,
        mc_refit_interval=1000,
        mc_decay_interval=100,
        pairwise_blend=0.0,
        stats_file=None,
        stats_interval=1000,
        coverage_report=None,
        coverage_log=None,
        stack_heartbeat=None,
        grammar=None,
        persistent=False,
        net_host=None,
        net_port=None,
        net_proto="tcp",
        net_keepalive=False,
        net_settle_ms=10,
        inprocess=False,
        inprocess_direct=False,
        inprocess_func="LLVMFuzzerTestOneInput",
        cmplog=False,
        cmplog_max_tokens=0,
        cmplog_max_pairs=0,
        cmplog_workdir=None,
        asan_target=None,
        ubsan_target=None,
        max_corpus=0,
        max_corpus_bytes=0,
        minimize_every_execs=0,
        prune_corpus_max_memory=80,
        no_shm=False,
        use_ptrace=False,
        adaptive_havoc=True,
        resume=False,
        trace_crashes=False,
        learn_format=False,
        corpus_ppmd=False,
        seed=42,
        extra_crash_codes=None,
        replay_n=0,
        crash_blocklist=None,
        crash_allowlist=None,
        save_smaller=False,
        honggfuzz=False,
        hw_perf=False,
        schedule_ablation=None,
        schedule="base",
        aflgo_cooling="exp",
        t_x_minutes=60.0,
        differential_target=None,
        replicator=False,
        shapley=False,
        bayesian=False,
        mi_guided=False,
        renyi_weight=False,
        transfer_entropy=False,
        secretary=False,
        secretary_window=500,
        secretary_exploration=None,
        elo=False,
        exp3=False,
        exp3_gamma=0.1,
        eps_greedy=False,
        eps_greedy_epsilon0=1.0,
        eps_greedy_decay=0.9995,
        hierarchical_bandit=False,
        gp_ucb=False,
        gp_length_scale=1.0,
        gp_beta=2.0,
        contextual=False,
        contextual_alpha=1.0,
        contextual_lambda=1.0,
        overlap_density=False,
        overlap_density_mode="modifier",
        overlap_min_jaccard=0.25,
        overlap_density_blend=0.5,
        lineage=False,
        lineage_backtrack=False,
        mcts=False,
        sensitivity=False,
        ga=False,
        qea=False,
        wfc=False,
        ga_pop_size=200,
        ga_gen_size=500,
        ga_elite_frac=0.1,
        ga_crossover_rate=0.7,
        ga_mutation_rate=0.3,
        ga_tournament_size=3,
        ga_speciation_threshold=0.3,
        qea_rotation_angle=0.05,
        qea_strong_bias=None,
        qea_elite_reset=0,
        calibrate=0,
        stall_threshold=1000,
        resize_map_on_stall=True,
        reseed_on_stall=False,
        map_size=0,
        max_collision_risk=30,
        continue_until_crash=False,
        multi_targets=None,
        debug=False,
        enable_regex_bomb=False,
        enable_x86_mutator=False,
        enable_arm_mutator=False,
        enable_smt_z3=False,
        path_negation=False,
        mod_solving="concolic",
        corpus_boost=0,
        boost_mean=None,
        boost_std=None,
        boost_pad="repeat",
        refresh_profile=False,
        chi2_operator_interval=0,
        quiet_stats=False,
        no_save_state=False,
        dedup_execs=True,
        # Appended rather than grouped with the other mutation-targeting
        # flags: this signature is positional, so inserting a parameter
        # mid-list silently shifts every caller argument after it.
        region_profile=False,
    ):
        self.target = target
        self.debug = debug
        # Persistent-loader ptrace self-trace for fault-address/register
        # capture (PTRACE_TRACEME on every forked call). Off by default —
        # it adds per-exec overhead and can be blocked by yama ptrace_scope.
        # Crash triage (_run_triage_ptrace) still fires a one-off re-run on
        # each crash regardless of this flag; this only controls the
        # always-on per-iteration trace in the persistent loader.
        self.use_ptrace = use_ptrace
        # Weight havoc's 11 inline sub-mutations by their measured hit rate
        # instead of drawing them uniformly. On by default: the priors start
        # uniform, the per-draw cost is a bisect over 11 floats, and the
        # branch mix is otherwise the one part of the mutation stack that
        # ignores the feedback every layer above it collects. --no-adaptive-
        # havoc restores the flat `r[0] % 11` split for A/B runs.
        self._adaptive_havoc = adaptive_havoc
        self.refresh_profile = refresh_profile
        self.quiet_stats = quiet_stats
        # Multi-target support: list of target binaries to fuzz with shared corpus
        self.multi_targets = multi_targets  # None for single-target
        self._active_target_idx = 0  # round-robin index
        self._target_shm_covs = {}  # target_path -> ShmCoverage (per-target)
        self._target_profiles = {}  # target_path -> TargetProfile
        # Pin the address-space layout BEFORE anything spawns, dlopens, or
        # profiles a target. personality(ADDR_NO_RANDOMIZE) is inherited by
        # every child through fork and survives execve, so this one call
        # covers posix_spawn (which has no preexec_fn hook), Popen, the
        # in-process subprocess loader, and the forkserver.
        #
        # Required for correctness, not just reproducibility: afl_shim.c's
        # caller-context edge hashing derives edge_id from a runtime return
        # address, and _seen_edge_ids is compared across target processes.
        # With ASLR on, every exec of a CTX build reports a fresh edge set.
        # See adapters/process.disable_aslr.
        self._aslr_disabled = disable_aslr()

        # Record boot time at init — before any child processes are spawned.
        # Use -2s tolerance so crashes logged just before this read are included.
        try:
            with open("/proc/uptime") as f:
                self._run_boot_start = float(f.read().split()[0]) - 2.0
        except OSError:
            self._run_boot_start = 0.0
        self.corpus_dir = Path(corpus_dir)
        self.crashes_dir = Path(crashes_dir)
        self.resume = resume
        self.continue_until_crash = continue_until_crash
        self._calibrate = calibrate
        self._stall_threshold = stall_threshold
        self._resize_map_on_stall = resize_map_on_stall
        self._reseed_on_stall = reseed_on_stall
        self._max_collision_risk = max_collision_risk
        self._last_new_edge_exec = 0
        self._stall_recovery_active = False
        self._stall_recovery_count = 0  # times recovery was activated
        self._stall_recovery_execs = 0  # execs spent in recovery mode
        self._stall_reseed_count = 0  # times the RNG was reseeded on stall
        self._last_stall_seed = None  # seed applied by the most recent reseed
        self.extra_crash_codes = set(extra_crash_codes) if extra_crash_codes else set()
        self.max_len = max_len
        # Floor for the adaptive max_len in corpus_manager: that value
        # tracks the corpus size distribution and must be allowed to
        # fall again when the corpus shrinks, but never below what the
        # caller asked for.
        self._max_len_floor = max_len
        self.timeout = timeout
        self.mutations_per_input = mutations_per_input
        self.use_coverage = use_coverage
        self.dictionary = dictionary or []
        self.file_mode = file_mode
        self.target_args = target_args or []
        self.max_corpus = max_corpus
        self.max_corpus_bytes = max_corpus_bytes
        self.minimize_every_execs = minimize_every_execs
        self.prune_corpus_max_memory = prune_corpus_max_memory
        self._last_memory_prune_exec = 0
        self._last_bloat_warn_exec = 0
        self._minimize_pending = False
        self.coverage_report = Path(coverage_report) if coverage_report else None
        self.coverage_log = Path(coverage_log) if coverage_log else None
        if self.coverage_log:
            self.coverage_log.parent.mkdir(parents=True, exist_ok=True)
        self.grammar = grammar
        self.persistent = persistent
        self.net_host = net_host
        self.net_port = net_port
        self.net_proto = net_proto
        self.net_keepalive = net_keepalive
        self.net_settle_ms = net_settle_ms
        self.enable_regex_bomb = enable_regex_bomb
        self.enable_x86_mutator = enable_x86_mutator
        self.enable_arm_mutator = enable_arm_mutator
        self.seed = seed
        random.seed(seed)
        if _HAS_NUMPY:
            np.random.seed(seed)  # seed numpy for deterministic RandPool
        # GA lifecycle parameters
        self._ga_enabled = ga
        self._ga_pop_size = ga_pop_size
        self._ga_gen_size = ga_gen_size
        self._ga_elite_frac = ga_elite_frac
        self._ga_crossover_rate = ga_crossover_rate
        self._ga_mutation_rate = ga_mutation_rate
        self._ga_tournament_size = ga_tournament_size
        self._ga_speciation_threshold = ga_speciation_threshold
        # QEA coupling magnitudes, exposed so the zero endpoint is an arm.
        self._qea_rotation_angle = qea_rotation_angle
        self._qea_strong_bias = qea_strong_bias
        self._qea_elite_reset = qea_elite_reset
        self.ga = None  # Initialized in run() when --ga is set

        # QEA lifecycle
        self._qea_enabled = qea
        self.qea = None  # Initialized in run() when --qea is set

        # Differential fuzzing
        self._diff_target = differential_target
        self._diff_tracker = None

        # WFC structural generation mode
        self._wfc_enabled = wfc

        # Corpus size boost: normal-distribution seed resizing
        self._corpus_boost = corpus_boost
        self._boost_mean = boost_mean
        self._boost_std = boost_std
        self._boost_pad = boost_pad

        # Static analysis: profile target for string extraction, function
        # boundaries, input format hints, and call graph structure.
        # Run this BEFORE estimate_map_size so it can reuse the decoded data.
        from fuzzer_tool.core.target_profiler import TargetProfiler

        self._profile = TargetProfiler(target).profile_cached(refresh=self.refresh_profile)

        # Edge bitmap size: use provided value or auto-size from branch density.
        # Pass the profile to skip the redundant full-text disassembly when
        # the profile already has text_size and total_branches.
        if map_size > 0:
            self.map_size = map_size
        else:
            from fuzzer_tool.core.elf import estimate_map_size

            self.map_size = estimate_map_size(target, profile=self._profile)

        # Auto-populate dictionary from extracted strings and magic bytes
        if self._profile.interesting_strings:
            for s in self._profile.interesting_strings[:200]:
                token = s.encode("utf-8", errors="replace")
                if token not in self.dictionary:
                    self.dictionary.append(token)
        if self._profile.magic_bytes:
            for mb in self._profile.magic_bytes:
                if mb not in self.dictionary:
                    self.dictionary.append(mb)

        # Auto-populate dictionary from disassembly-extracted constants
        if self._profile.extracted_constants:
            for c in self._profile.extracted_constants:
                if c not in self.dictionary and len(c) >= 2:
                    self.dictionary.append(c)

        # Auto-populate dictionary from parser token tables (Bison/Yacc)
        if self._profile.parser_tokens:
            for t in self._profile.parser_tokens:
                if t not in self.dictionary:
                    self.dictionary.append(t)

        # Cmplog: comparison tracing via LD_PRELOAD
        self._cmplog = None
        self._redqueen_index = 0
        self._cmplog_skip_counter = 0  # adaptive cmplog collection skip
        if cmplog:
            from fuzzer_tool.core.cmplog import CmplogCollector

            self._cmplog = CmplogCollector(
                max_tokens=cmplog_max_tokens, max_pairs=cmplog_max_pairs, workdir=cmplog_workdir
            )
            if self._cmplog.start():
                print("[*] Cmplog: comparison tracing enabled (memcmp/strcmp/strncmp/memchr)")
                from fuzzer_tool.core.rq_encodings import encoders_summary

                encoders = encoders_summary()
                print(
                    f"[*]   Redqueen encoders: {len(encoders)} ({', '.join(e['name'] for e in encoders)})"
                )
            else:
                print("[!] Cmplog: failed to compile shim, disabling")
                self._cmplog = None

        # Checksum learner: recovers unknown linear checksum polynomials
        # from observed (data, checksum) pairs via Berlekamp-Massey / GCD.
        self.checksum_learner = None
        try:
            from fuzzer_tool.core.checksum_learner import ChecksumLearner

            self.checksum_learner = ChecksumLearner(self)
        except Exception as exc:
            log.debug("ChecksumLearner init failed: %s", exc)

        # SMT solver: arithmetic constraint solving on cmplog pairs
        self._smt_solver = None
        self._enable_smt_z3 = enable_smt_z3

        # Path-condition negation. Independent of --enable-smt-z3: that flag
        # selects the modulo-solving strategy for cmplog pairs, whereas this
        # solves for inputs that flip a branch outright.
        self._path_solver = None
        if path_negation:
            from fuzzer_tool.core.path_constraints import PathConstraintSolver, _z3

            if _z3() is None:
                print("[!] Path negation requested but z3 is unavailable — disabled")
            else:
                self._path_solver = PathConstraintSolver()
                print("[*] Path negation: solving for branch-flipping inputs")

        self._mod_solving = mod_solving if enable_smt_z3 else "heuristic"
        if enable_smt_z3:
            from fuzzer_tool.core.smt_solver import Z3Solver

            self._smt_solver = Z3Solver(mod_solving_mode=mod_solving)
            if self._smt_solver._available:
                print(f"[*] SMT solver: modulo solving mode '{mod_solving}'")

                # Trace mode: pre-compute PC→divisor map from static analysis
                if mod_solving == "trace":
                    from fuzzer_tool.core.elf import extract_div_constants

                    targets = multi_targets or [target]
                    div_map: dict[int, int] = {}
                    weak_set: set[int] = set()
                    for t in targets:
                        try:
                            d, w = extract_div_constants(t)
                            div_map.update(d)
                            weak_set.update(w)
                        except Exception:
                            log.debug("Failed to extract DIV constants from %s", t)
                    if div_map or weak_set:
                        from fuzzer_tool.core.smt_solver import (
                            set_pc_divisor_map,
                            set_weak_mod_set,
                        )

                        set_pc_divisor_map(div_map)
                        set_weak_mod_set(weak_set)
                        print(
                            f"[*] SMT solver: loaded {len(div_map)} PC→divisor mappings"
                            f" + {len(weak_set)} weak modulus PCs"
                        )
                    else:
                        print("[!] SMT solver: no DIV/IDIV with known divisor in target(s)")
            else:
                print(
                    "[!] SMT solver: z3-solver not installed — install with: pip install z3-solver"
                )
                self._smt_solver = None

        if self._smt_solver is not None and self._cmplog is None:
            print("[!] SMT solver: --enable-smt-z3 requires --cmplog; disabling SMT path")
            self._smt_solver = None
            self._enable_smt_z3 = False

        if self.file_mode:
            self._tmp_dir = Path(tempfile.mkdtemp(prefix="fuzzer_"))
            atexit.register(_cleanup_tmp_dir, self._tmp_dir)
        else:
            self._tmp_dir = Path("/tmp") / f"fuzzer_{os.getpid()}"

        self.ptrace_cov: PtraceCoverage | None = None
        self.shm_cov: ShmCoverage | None = None
        self._forkserver = None
        if self.use_coverage:
            if no_shm:
                self._setup_ptrace(target, deep_coverage, max_bps)
            else:
                try:
                    self.shm_cov = ShmCoverage(size=self.map_size)
                    print(f"[*] Coverage: AFL SHM bitmap, id={self.shm_cov.env_id}")
                except OSError:
                    self._setup_ptrace(target, deep_coverage, max_bps, fallback_hint=True)

        # Per-target SHM for multi-target mode
        if self.multi_targets and self.use_coverage and not no_shm:
            for t in self.multi_targets:
                try:
                    self._target_shm_covs[t] = ShmCoverage(size=self.map_size)
                except OSError:
                    log.warning("Failed to create SHM for %s, using shared SHM", t)
            if self._target_shm_covs:
                print(f"[*] Multi-target: {len(self._target_shm_covs)} per-target SHM regions")

        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        self.crashes_dir.mkdir(parents=True, exist_ok=True)

        # Single-file state store (replaces per-component JSON files).
        # Loaded eagerly so all components can fetch their section via get().
        from fuzzer_tool.core.state_store import StateStore

        self._state_store = StateStore(self.corpus_dir, enabled=not no_save_state)
        if self.resume:
            self._state_store.load()

        self.corpus: list[bytes] = []
        self.seen_hashes: set[str] = set()
        self.irreplaceable_hashes: set[str] = set()
        self.bloom = BloomFilter(capacity=100_000)
        self.bloom.init_fuzzy(max_recent=200)
        # Executed-input filter.  The mutation space is not uniform: stall
        # recovery, dictionary ops and the deterministic stages collapse onto
        # very short buffers, so the same mutant is handed to the target
        # hundreds of times.  A membership test costs ~1us against a ~1.5ms
        # exec, so re-rolling the mutation on a hit is close to free.
        # Generational: wiped once `capacity` inputs are absorbed, which keeps
        # the realised FP rate at 1e-3 over an unbounded exec stream.
        self._exec_bloom = BloomFilter(capacity=EXEC_BLOOM_CAPACITY, error_rate=1e-3)
        self._dedup_execs = dedup_execs
        self._dedup_hits = 0
        self._dedup_gaveup = 0
        self.crash_hashes: set[str] = set()
        self.crash_sigs: dict[str, int] = {}
        self.crash_frames: dict[str, list[str]] = {}  # sig -> frames for clustering
        # Crash-stop state captured by the ptrace runner (fault address from
        # PTRACE_GETSIGINFO, registers from PTRACE_GETREGS); consumed by
        # corpus_manager.save_crash for the sidecar + signature.
        self._last_fault_addr: int | None = None
        self._last_regs: dict[str, int] = {}
        # Lazy probe: whether ptrace crash triage (re-running direct_lite
        # crashes through the ptrace-attached loader) is usable here.
        self._triage_ok: bool | None = None
        self.exec_count = 0
        self.crash_count = 0
        self.timeout_count = 0
        self.start_time = time.time()
        self.last_report: SanitizerReport | None = None
        self.op_counts: dict[str, int] = {}
        self.op_success: dict[str, int] = {}
        self.op_edges: dict[str, int] = {}
        self._peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self._discovery_execs: array = array("Q")  # exec_count per discovery snapshot
        self._discovery_edges: array = array("Q")  # cumulative edges per snapshot
        self._crash_rate_execs: array = array("Q")  # exec_count per crash-rate sample
        self._crash_rate_counts: array = array("Q")  # crash_count per sample
        self._duplicate_reject_count = 0
        self._total_corpus_attempts = 0
        self._pruned_count = 0
        self._exec_baseline = 0
        self._peak_eps = 0.0
        self._total_exec_time = 0.0
        self._replay_budget_ms: float = 0.2  # max 200ms per batch for crash replay
        self._crash_replays: dict[str, list[int]] = {}  # sig -> list of replay return codes
        self.replay_n: int = replay_n  # --replay-N: replay each crash N times
        self.asan_target: str | None = asan_target  # --asan-target: ASAN-instrumented variant
        self.ubsan_target: str | None = ubsan_target  # --ubsan-target: UBSAN-instrumented variant
        self._crash_sanitizer_replays: dict[str, dict] = {}  # sig -> {data, asan, ubsan}
        self.crash_blocklist: set[str] = crash_blocklist or set()
        self.crash_allowlist: set[str] = crash_allowlist or set()
        self.save_smaller: bool = save_smaller
        self.honggfuzz: bool = honggfuzz
        self.hw_perf: bool = hw_perf
        self.crash_min_sizes: dict[str, int] = {}  # stack_hash -> min trigger size
        # Honggfuzz power factor stats (for display)
        self._hf_novelty_boosts: int = 0
        self._hf_freshness_boosts: int = 0
        self._hf_fertility_boosts: int = 0
        self._hf_density_boosts: int = 0
        self._hf_entropy_penalties: int = 0
        self._hf_timeout_penalties: int = 0

        # Execution time tracking for adaptive timeout calibration
        from fuzzer_tool.core.execution_time import ExecutionTimeTracker

        self._exec_time_tracker = ExecutionTimeTracker()

        self._last_child_pid: int | None = None

        self.stats_file = Path(stats_file) if stats_file else None
        self.stats_interval = stats_interval
        # Optional stack heartbeat: a daemon thread writes the main-thread
        # stack to this file every few seconds, so a SIGKILL (uncatchable)
        # still leaves the last executing location on disk.
        self._stack_heartbeat_path = Path(stack_heartbeat) if stack_heartbeat else None
        self._last_stats_exec = 0
        self._eps = 0.0
        # Kalman filter for denoised EPS tracking.
        # Uses 2D RobustKF to handle scheduling jitter / GC pauses / bursty
        # throughput.  The adaptive-R variant learns the actual measurement
        # noise online.  Filtered rate replaces the raw sliding-window in
        # dict pruning, stats-interval calc, etc.
        self._eps_kf = None  # lazy-initialized after first stats tick
        self._last_eps_count = 0  # exec_count at last EPS KF update
        self._last_eps_time = 0.0  # monotonic time at last EPS KF update
        # exec_count at the start of this process (0 for fresh runs, loaded
        # value for --resume).  EPS display and interval math subtract it so
        # resumed runs don't divide the cumulative count by fresh wall time.
        self._resume_baseline_exec = 0

        # Rolling avg-eps samples (one per stats tick). The first ticks often
        # show inflated EPS (bursty warm-up, startup time in the denominator),
        # so the stats line is held until the window fills and the effective
        # stats interval is driven by the window mean, not a single reading.
        self._eps_history: array = array("d")
        self._eps_history_max = 10  # stabilization window (in stats ticks)

        # Bayesian seed quality estimation
        self._seed_quality = BayesianSeedQuality()

        # Schedule ablation: per-iteration CSV log of signal contributions
        self._ablation_path = Path(schedule_ablation) if schedule_ablation else None
        self._ablation_file = None
        if self._ablation_path:
            self._ablation_path.parent.mkdir(parents=True, exist_ok=True)
            self._ablation_file = open(self._ablation_path, "w")  # noqa: SIM115
            self._ablation_file.write(
                "iter,seed_idx,seed_hash,fuzz_count,coverage_edges,age_s,"
                "base_w,burst,penalty,subsumption,diversity,spatial,mdl,"
                "final_w,new_coverage,new_crash\n"
            )
            self._ablation_file.flush()

        # Support multiple markov orders via comma-separated list or single int
        if isinstance(markov_order, str):
            orders = [int(o.strip()) for o in markov_order.split(",")]
        elif isinstance(markov_order, list):
            orders = markov_order
        else:
            orders = [markov_order]
        if len(orders) > 1:
            self.markov = MarkovEnsemble(orders=orders, blend=markov_blend)
        else:
            self.markov = MarkovChain(order=orders[0])
        self.markov_generate = markov_generate
        self.markov_trained = False

        # ── Extracted modules ──────────────────────────────────────────
        self._operators = OperatorEngine(self)
        self._seed_picker = SeedPicker(self)
        self._runner = TargetRunner(self)
        self._stats = StatsReporter(self)
        self._corpus_manager = CorpusManager(self)

        # Hardware performance counters (optional, requires CAP_PERFMON)
        self._perf_counters = None
        self._last_perf_deltas: dict[str, int] = {}
        if self.hw_perf:
            from fuzzer_tool.adapters.perf_event import PerfCounters

            self._perf_counters = PerfCounters()
            if not self._perf_counters.available:
                log.warning("Hardware perf counters not available (needs CAP_PERFMON or root)")
                self._perf_counters = None
                self.hw_perf = False

        # Seed-level energy multiplier: scales mutations_per_input per seed
        self._seed_scorer = SeedScorer(
            schedule=schedule or "base",
            aflgo_cooling=aflgo_cooling,
            t_x_minutes=t_x_minutes,
        )
        self._power_schedule = schedule
        self._last_perf_score = 100.0  # default multiplier (1x)

        # Seed key cache: maps seed bytes -> hex digest.  Initialised early
        # because _boost_corpus_sizes() invalidates it on corpus resizing.
        self._seed_key_cache: dict[bytes, str] = {}

        # Per-byte sensitivity tracker (Lyapunov exponent). Constructed
        # before _init_seed_metadata so load_state (resume) can restore
        # sensitivity.json — it crashed with AttributeError otherwise.
        self._use_sensitivity = sensitivity
        from fuzzer_tool.core.sensitivity import ByteSensitivityTracker

        self._sensitivity = ByteSensitivityTracker(
            max_seeds=50, max_bytes=max_len, sample_rate=0.02
        )

        # Statistical region profiling (randomness.profile_buffer): labels
        # each window of a seed incompressible / tabular / textual /
        # repetitive and weights byte selection accordingly, so mutation
        # effort moves off compressed payloads and onto offset and length
        # tables. Off by default -- the profile costs ~1 ms per 4 KiB, which
        # is only worth paying on structured targets, and it is cached per
        # seed in OperatorEngine rather than recomputed per mutation.
        self._use_region_profile = region_profile

        # Weighted mutation lineage tree (parent/ops/sites/new-edge weight
        # per seed). Initialised early so the post-metadata rebuild can
        # consume it; rebuilt from persisted seed_meta after metadata init.
        self._use_lineage = lineage
        # Backtracking widens exploration: when a lineage branch stops
        # producing edges, its seeds are penalised geometrically by depth so
        # weight shifts back toward shallow seeds with unexplored siblings.
        # Implies --lineage (needs the tree to know what a branch is).
        self._use_lineage_backtrack = bool(lineage_backtrack and lineage)
        self._lineage_backtrack_decay = 0.7
        self._lineage_backtrack_min_fuzz = 8
        # MCTS seed scheduling walks the lineage genealogy, so it is
        # meaningless without the tree; --mcts implies --lineage.
        self._use_mcts = bool(mcts)
        self._mcts = None
        if self._use_mcts and not lineage:
            lineage = True
            log.info("--mcts implies --lineage (MCTS schedules over the lineage tree)")

        self._lineage = None
        if lineage:
            from fuzzer_tool.core.lineage import LineageTree

            self._lineage = LineageTree()
            log.info("Mutation lineage tree enabled")

        if self._use_mcts and self._lineage is not None:
            from fuzzer_tool.core.schedulers.mcts import MCTSSeedScheduler

            self._mcts = MCTSSeedScheduler()
            log.info("MCTS seed scheduling enabled")

        self._load_corpus()
        if self._corpus_boost > 0 and self.corpus:
            self._boost_corpus_sizes()
        self._init_seed_metadata()
        # Rebuild the lineage tree from persisted seed_meta (single source
        # of truth; never re-derived from runs to avoid double-counting).
        if self._use_lineage and self._lineage is not None:
            self._lineage.rebuild_from_meta(self.seed_meta, self._seed_key)
        # Load persisted Markov state from state store; skip retrain if loaded
        markov_data = self._state_store.get("markov")
        if markov_data is not None:
            loaded = True
            self.markov.from_dict(markov_data)
        else:
            loaded = False
        if self.corpus and not loaded:
            self.markov.train_corpus(self.corpus)
        self.markov_trained = self.markov.is_trained()

        # Seed aggregate cache: compute from initial seed_meta
        self._refresh_agg_cache()

        self.mc_bandit = mc_bandit
        self.mc_cem = mc_cem
        self._use_mopt = mopt
        self.mc = (
            MonteCarloScheduler(
                elite_frac=mc_elite_frac,
                refit_interval=mc_refit_interval,
                pairwise_blend=pairwise_blend,
                decay_interval=mc_decay_interval,
            )
            if (mc_bandit or mc_cem or mopt)
            else None
        )
        self._mopt = None
        if mopt:
            self._mopt = MOptScheduler(n_particles=5, window_size=200)
            log.info("MOpt PSO scheduling enabled (5 particles, window=200)")
        self._use_replicator = replicator
        self._seed_strategy = None
        self._seed_strategy_pool: list[str] = []
        self._seed_strategies_used: set[str] = set()
        self._use_boltzmann = boltzmann
        self._metropolis = metropolis
        self._op_dispatch = self._build_dispatch()
        self._replicator = None
        if replicator:
            self._replicator = ReplicatorScheduler(window_size=200, learning_rate=0.1)
            log.info("Replicator dynamics scheduling enabled (window=200, eta=0.1)")
        # EXP3 adversarial bandit
        self._use_exp3 = exp3
        self._exp3 = None
        if exp3:
            self._exp3 = Exp3Scheduler(gamma=exp3_gamma)
            log.info("EXP3 adversarial bandit enabled (gamma=%.2f)", exp3_gamma)
        # Epsilon-greedy with annealing
        self._use_eps_greedy = eps_greedy
        self._eps_greedy = None
        if eps_greedy:
            self._eps_greedy = EpsilonGreedyScheduler(
                epsilon_0=eps_greedy_epsilon0, decay=eps_greedy_decay
            )
            log.info(
                "Epsilon-greedy enabled (epsilon0=%.2f, decay=%.4f)",
                eps_greedy_epsilon0,
                eps_greedy_decay,
            )
        # Hierarchical bandit
        self._use_hierarchical = hierarchical_bandit
        self._hierarchical = None
        if hierarchical_bandit:
            self._hierarchical = HierarchicalBanditScheduler()
            log.info(
                "Hierarchical bandit enabled (%d categories)",
                len(HierarchicalBanditScheduler.CATEGORIES),
            )
        # GP-UCB bandit
        self._use_gp_ucb = gp_ucb
        self._gp_ucb = None
        if gp_ucb:
            self._gp_ucb = GPUCBScheduler(length_scale=gp_length_scale, beta=gp_beta)
            log.info("GP-UCB enabled (l=%.2f, beta=%.2f)", gp_length_scale, gp_beta)

        self._use_contextual = contextual
        self._contextual = None
        if contextual:
            from fuzzer_tool.services.operators import CONTEXT_DIM

            self._contextual = ContextualLinUCBScheduler(
                dim=CONTEXT_DIM, alpha=contextual_alpha, lambda_reg=contextual_lambda
            )
            log.info(
                "Contextual LinUCB enabled (dim=%d, alpha=%.2f, lambda=%.2f)",
                CONTEXT_DIM,
                contextual_alpha,
                contextual_lambda,
            )
        # Running mean/stddev of corpus seed sizes, updated in
        # corpus_manager.save_to_corpus(). Feeds the contextual scheduler's
        # "position in corpus size distribution" feature via a cheap
        # logistic approximation of the CDF, instead of sorting the whole
        # corpus on every mutation.
        self._corpus_size_stats = RunningMoments()

        self._use_shapley = shapley
        self._shapley = ShapleyAttribution(n_samples=100, window_size=500) if shapley else None
        self._use_bayesian = bayesian
        self._use_mi = mi_guided
        self._mi = (
            # Cap tracked positions: max_len auto-grows to 65536 and the MI
            # joint is positions x 256 byte values x MAX_EDGES_PER_CELL cells —
            # unbounded positions is a multi-GB memory blowup.
            MutualInformationTracker(
                max_positions=min(max_len, MI_MAX_POSITIONS), min_observations=50
            )
            if mi_guided
            else None
        )
        # Load persisted MI state from state store (resume-gated — an oversized
        # mi.json otherwise becomes a multi-GB object tree at every startup)
        if self._use_mi and self._mi and self.resume:
            mi_data = self._state_store.get("mi")
            if mi_data is not None:
                self._mi.from_dict(mi_data)
                log.info(
                    "MI tracker loaded from state store (%d positions)", self._mi.max_positions
                )

        # Crash MI tracker: I(byte_position; crash_outcome)
        from fuzzer_tool.core.crash_eta import CrashMITracker

        self._crash_mi = CrashMITracker(max_positions=max_len, min_observations=20)
        crash_mi_data = self._state_store.get("crash_mi")
        if crash_mi_data is not None:
            self._crash_mi.load(crash_mi_data)
            log.info(
                "Crash MI tracker loaded: %d execs, %d crashes",
                self._crash_mi.total_execs,
                self._crash_mi.total_crashes,
            )

        # Length-edge tracker: input_length → coverage edges
        from fuzzer_tool.core.length_mi import LengthEdgeTracker

        self._length_tracker = LengthEdgeTracker()
        lt_data = self._state_store.get("length_tracker")
        if lt_data is not None:
            self._length_tracker.load(lt_data)
            log.info("Length-edge tracker loaded: %d execs", self._length_tracker.total_execs)

        self._use_renyi_weight = renyi_weight
        self._use_transfer_entropy = transfer_entropy
        self._te = None
        self._te_byte_edges: dict[int, dict[int, int]] = {}  # pos → {edge: count}
        if transfer_entropy:
            from fuzzer_tool.core.transfer_entropy import TransferEntropy

            self._te = TransferEntropy(history_length=1)
            self._te_input_history: list[bytes] = []
            self._te_edge_history: list[bytes] = []
            self._te_history_max = 500
            log.info("Transfer entropy tracking enabled")

        # FrameShift: universal length-field auto-adjustment
        from fuzzer_tool.core.frameshift import FrameShift

        self._frameshift = FrameShift(max_relations=64)
        self._last_ops_used: list[str] = []
        # Subset of _last_ops_used that actually changed the buffer. Set by
        # OperatorEngine.mutate() when _track_op_effect is on; consumed by
        # _record_outcome() to keep no-op operators out of the winner set.
        self._last_ops_effective: set[str] = set()
        # Bitmask of havoc sub-mutations applied this round, set by
        # OperatorEngine._apply_single_mutation and consumed by
        # _record_outcome(). Havoc's inner loop has no visibility into the
        # coverage verdict at mutation time, so credit is deferred here the
        # same way it is for top-level operators.
        self._last_havoc_subops: int = 0
        self._last_ops_with_sites: list[tuple[str, int]] = []
        self._last_op_costs: dict[str, float] = {}
        # EMA of wall-clock seconds per call, per operator. Populated in
        # OperatorMixin.mutate(). Used to convert bandit rewards from
        # edges-per-selection to edges-per-unit-time so expensive operators
        # (gradient_descent, condstmt_solve, path_negate, crc_learn) don't
        # get rated on the same scale as bit_flip.
        self._op_time_ema: dict[str, float] = {}
        self._last_new_edge_count = 0
        self._last_hamming_distance: int = -1
        self._last_mutation_offset: int = 0

        # Critical slowing down detector
        from fuzzer_tool.core.critical_slowing import CriticalSlowingDown

        self._csd = CriticalSlowingDown(window_size=50, rise_threshold=1.5, min_observations=20)

        # Allan variance detector for stall detection (edge discovery rate)
        from fuzzer_tool.core.allan_variance import AllanVarianceDetector

        self._allan = AllanVarianceDetector(
            max_buffer_pow=ALLAN_BUFFER_POW, min_samples=ALLAN_MIN_SAMPLES
        )
        self._last_allan_edge_count = 0

        # ── Running aggregate cache for seed metadata ──────────────────
        # Avoids O(n·m) recomputation of corpus-wide sums every iteration.
        # Updated by delta in fuzz_one() and invalidated when the corpus
        # structure changes (add/remove/replace seeds).
        self._cached_total_time: float = 0.0
        self._cached_total_fuzz: int = 0
        self._cached_total_edges: int = 0
        self._cached_mean_log_n_fuzz: float = 0.0
        self._agg_cache_valid: bool = False

        # ── Vectorized random number pool for mutation hotpath ────────
        # Generates random values in batches (one numpy C-level call per
        # batch) instead of per-call Python-level random() invocations.
        from fuzzer_tool.core.rand_pool import RandPool

        self._rand_pool = RandPool(seed=seed)

        # ── Dictionary scratch buffer (vectorized choice) ─────────────
        # Refilled in mutate() via one randint_list call; consumed by
        # dict-aware operators instead of calling random.choice(f.dictionary).
        self._dict_scratch: list[int] = []
        self._dict_scratch_idx = 0

        # Format structure learner (schema-harness methodology)
        self._format_learner = None
        if learn_format:
            from fuzzer_tool.core.format_learner import FormatLearner

            self._format_learner = FormatLearner(max_timeline=10000)

        # Corpus PPMD compression for seed novelty scoring
        self._ppmd = None
        if corpus_ppmd:
            from fuzzer_tool.core.corpus_compression import CorpusCompressor

            self._ppmd = CorpusCompressor()

        # Per-operator buffer-change tracking costs one xxh3 digest per
        # mutation (~3.4us at 64KiB, no copy). Only pay it when something
        # actually consumes the credit assignment.
        self._track_op_effect = bool(
            elo
            or (self.mc and self.mc_bandit)
            or self._mopt
            or self._replicator
            or self._exp3
            or self._eps_greedy
            or self._hierarchical
            or self._gp_ucb
            or self._contextual
            or self._use_shapley
        )

        # Elo rating system for operator scheduling
        self._use_elo = elo
        self._elo = None
        if elo:
            from fuzzer_tool.core.elo import BayesianEloTracker

            self._elo = BayesianEloTracker(
                initial_mu=1500,
                initial_sigma=350,
                beta=200,
                tau=5.0,
                min_matches=10,
            )

            log.info("Elo rating system enabled (k=16, decay=0.99)")
            self._elo_decay_interval = 100  # apply decay every N iterations
            self._elo_decay_counter = 0
            self._elo_match_window: list[tuple[str, str, float, bool]] = []
            elo_data = self._state_store.get("elo")
            if elo_data is not None:
                self._elo.from_dict(elo_data)
                log.info("Elo tracker loaded from state store (%d operators)", len(self._elo.mu))

            # Pre-register all strategy names so Elo can arbitrate immediately
            # (without this, select_strategy requires min_matches before considering a strategy)
            for s in _OPERATOR_STRATEGY_NAMES:
                self._elo._strategy_mu.setdefault(s, self._elo.initial_mu)
                self._elo._strategy_sigma_sq.setdefault(s, self._elo.initial_sigma**2)
                self._elo._strategy_match_count.setdefault(s, 0)
            for s in _SEED_STRATEGY_NAMES:
                key = f"seed_{s}"
                self._elo._strategy_mu.setdefault(key, self._elo.initial_mu)
                self._elo._strategy_sigma_sq.setdefault(key, self._elo.initial_sigma**2)
                self._elo._strategy_match_count.setdefault(key, 0)

        # Chi-squared operator heterogeneity test interval
        self._chi2_operator_interval = chi2_operator_interval
        if chi2_operator_interval > 0:
            print(f"[*] Chi-squared operator test: every {chi2_operator_interval} execs")

        # Elo arbitrates between all available strategies when enabled
        self._meta_strategy: str | None = None
        # Per-exec cache: resolved once in mutate(), reused for all mutations
        self._meta_strategy_cached: str | None = None
        # Operator schedulers actually selected this run (for the convergence
        # report, which must show only used schedulers)
        self._meta_strategy_used: set[str] = set()
        if self._use_elo:
            log.info(
                "Meta-scheduler enabled: Elo arbitrating across %d operator and %d seed strategies",
                len(_OPERATOR_STRATEGY_NAMES),
                len(_SEED_STRATEGY_NAMES),
            )

        # Secretary-problem optimal stopping
        self._secretary = secretary
        self._secretary_window = secretary_window
        self._secretary_exploration = (
            secretary_exploration if secretary_exploration is not None else DEFAULT_EXPLORATION_FRAC
        )
        self._seed_secretary: dict[str, SecretaryStopping] = {}
        self._op_secretary: dict[str, SecretaryStopping] = {}
        self._corpus_secretary = (
            SecretaryStopping(
                window_size=secretary_window,
                exploration_frac=self._secretary_exploration,
                min_observations=30,
            )
            if secretary
            else None
        )

        # FMM-clustered pairwise overlap density
        self._use_overlap_density = overlap_density
        self._overlap_mode = overlap_density_mode
        self._overlap_min_jaccard = overlap_min_jaccard
        self._overlap_density_blend = overlap_density_blend
        self._overlap_density_cache: dict[str, float] = {}

        # Entropy rate tracking: (exec_count, shannon_entropy) samples
        self._entropy_execs: array = array("Q")  # exec_count per entropy sample
        self._entropy_vals: array = array("d")  # shannon entropy per sample

        # Directed distance for targeted fuzzing
        self._distance = None
        self._distance_targets = targets
        self._anneal_progress = 0.0  # 0.0 = pure coverage, 1.0 = pure distance
        # Running min/max of observed per-seed distances (AFLGo queue
        # normalization); the no-data sentinel (20.0) is excluded.
        self._dist_min_observed: float | None = None
        self._dist_max_observed: float | None = None
        if targets:
            from fuzzer_tool.core.distance import TargetDistance

            self._distance = TargetDistance(target, targets)
            self._dist_table_shm = None
            if self._distance.load():
                print(
                    f"[*] Directed mode: {len(self._distance.target_addrs)} target(s), "
                    f"{len(self._distance.functions)} functions mapped"
                )
                if self._distance._bb_value:
                    try:
                        from fuzzer_tool.adapters.shm import DistanceTableShm

                        # Keys are trace-pc call-site addresses relative
                        # to the object base (__sancov_pcs); the shim
                        # looks up pc - dladdr_base.
                        table = self._distance.pc_distance_table()
                        if not table:
                            base = self._distance._base_addr or 0
                            table = {
                                bb_start - base: dist
                                for bb_start, dist in self._distance._bb_value.items()
                            }
                        self._dist_table_shm = DistanceTableShm(table)
                        if self._dist_table_shm.shm_id >= 0:
                            os.environ["__AFL_DIST_SHM_ID"] = self._dist_table_shm.env_id
                            print(
                                f"[*] AFLGo distance table: {len(table)} sites "
                                "uploaded (SHM-tail channel active)"
                            )
                    except OSError as e:
                        log.warning("Distance table upload failed: %s", e)
            else:
                print(
                    "[!] Directed mode: failed to load target distances, falling back to coverage"
                )
                self._distance = None

        # Simulated annealing temperature schedule
        self._anneal_budget = anneal_budget  # 0 = no annealing (temperature always 1.0)
        self._temperature = 1.0

        # Crash tracing: GDB backtrace + strace on crash inputs
        self._tracer = None
        if trace_crashes:
            from fuzzer_tool.core.trace import CrashTracer

            self._tracer = CrashTracer(target)

        def _register_arms(scheduler, priors=None):
            """Register all mutation arms on a scheduler (mc, mopt, replicator, elo).

            Args:
                scheduler: Scheduler exposing init_arm(name).
                priors: Optional dict of operator name -> (prior_alpha,
                    prior_beta) overrides. Only meaningful for the
                    Beta-Bernoulli Thompson-sampling scheduler; ignored for
                    schedulers whose init_arm() doesn't accept a prior.
            """
            priors = priors or {}

            def _init(op):
                prior = priors.get(op) if getattr(scheduler, "supports_priors", False) else None
                if prior is not None and len(prior) == 2:
                    scheduler.init_arm(op, *prior)
                else:
                    scheduler.init_arm(op)

            for op in REGISTRY.names():
                _init(op)

        from fuzzer_tool.core.target_profiler import format_operator_priors

        _format_priors = format_operator_priors(self._profile)

        if self.mc and self.mc_bandit:
            _register_arms(self.mc, _format_priors)
        if self._mopt:
            _register_arms(self._mopt)
        if self._replicator:
            _register_arms(self._replicator)
        if self._exp3:
            _register_arms(self._exp3)
        if self._eps_greedy:
            _register_arms(self._eps_greedy)
        if self._hierarchical:
            _register_arms(self._hierarchical)
        if self._gp_ucb:
            _register_arms(self._gp_ucb)
        if self._contextual:
            _register_arms(self._contextual)
        if self._elo:
            _register_arms(self._elo)
        del _format_priors  # free priors dict after arm registration

        self._persistent_runner = None
        if self.persistent:
            from fuzzer_tool.adapters.persistent import PersistentRunner

            self._persistent_runner = PersistentRunner(target=self.target, timeout=self.timeout)
            if self._persistent_runner.start():
                print("[*] Persistent mode: target started")
            else:
                print("[!] Persistent mode: failed to start target, falling back to fork")
                self._persistent_runner = None

        self._network_runner = None
        if getattr(self, "net_host", None) and getattr(self, "net_port", None):
            from fuzzer_tool.adapters.network import NetworkRunner
            from fuzzer_tool.core.kalman import RobustKF

            # Create a settle KF to adapt the per-iteration settle time.
            # After each run_one() the runner records wall-clock duration
            # as an observation; the KF's estimate smooths jitter and
            # provides a filtered settle time via _settle().
            initial_settle = getattr(self, "net_settle_ms", 10) / 1000
            settle_kf = RobustKF(
                dim=1,
                process_noise=initial_settle * 0.05,
                measurement_noise=initial_settle * 0.5,
                huber_threshold=3.0,
                adaptive_r_gain=0.02,
            )
            settle_kf.update(initial_settle)

            self._network_runner = NetworkRunner(
                host=self.net_host,
                port=self.net_port,
                proto=getattr(self, "net_proto", "tcp"),
                keepalive=getattr(self, "net_keepalive", False),
                settle=initial_settle,
                settle_kf=settle_kf,
            )
            print(
                f"[*] Network mode: fuzzing {self._network_runner.proto}://"
                f"{self.net_host}:{self.net_port} "
                f"(keepalive={self._network_runner.keepalive}, no reply read)"
            )

        self._inprocess_runner = None
        # Detect ASAN and set LD_PRELOAD BEFORE probing/loading (ctypes.CDLL) for any
        # .so/.dylib/.dll target — needed by both the auto-detect path below and the
        # explicit --inprocess/--inprocess-direct path, or ASAN aborts on first call
        # with "ASan runtime does not come first" instead of running the target.
        target_is_asan = False
        if self.target.lower().endswith((".so", ".dylib", ".dll")):
            target_is_asan = _detect_asan(self.target)
            if target_is_asan:
                libasan = "/usr/lib/x86_64-linux-gnu/libasan.so.8"
                if not os.path.exists(libasan):
                    import ctypes.util

                    libasan = ctypes.util.find_library("asan") or libasan
                # Read original LD_PRELOAD from process-start environment
                # (/proc/self/environ), not os.environ which may have been
                # modified by commands.py's ASAN detection before we get here.
                _original_ld_preload = ""
                try:
                    with open("/proc/self/environ", "rb") as _f:
                        for _entry in _f.read().split(b"\0"):
                            if _entry.startswith(b"LD_PRELOAD="):
                                _original_ld_preload = _entry[len(b"LD_PRELOAD=") :].decode()
                                break
                except OSError:
                    _original_ld_preload = os.environ.get("LD_PRELOAD", "")
                _asan_was_preloaded = libasan in _original_ld_preload
                if not _asan_was_preloaded:
                    existing = os.environ.get("LD_PRELOAD", "")
                    os.environ["LD_PRELOAD"] = f"{libasan}:{existing}" if existing else libasan
            # Preload ASAN runtime via ctypes for in-process loading (both
            # auto-detect and --inprocess-direct paths). This loads the
            # verify_asan_link_order=0 shim so ASAN's "does not come first"
            # check is suppressed, then loads libasan via RTLD_GLOBAL so the
            # target .so's DT_NEEDED libasan.so.8 is satisfied at dlopen time.
            _asan_ctypes_loaded = False
            if target_is_asan and not _asan_was_preloaded:
                import ctypes as _ctypes
                import subprocess as _subprocess
                import tempfile as _tempfile

                from fuzzer_tool.adapters.shim_factory import _find_compiler

                _asan_opts_shim_src = (
                    b"const char *__asan_default_options() {  "
                    b'return "halt_on_error=0:abort_on_error=0:verify_asan_link_order=0";}'
                )
                _fd, _shim_path = _tempfile.mkstemp(suffix=".so", prefix="asan_opts_")
                os.close(_fd)
                try:
                    _compiler = _find_compiler()
                    # Strip ASAN from compiler subprocess (clang/gcc aren't
                    # built with ASAN; libasan's LeakSanitizer causes false
                    # leak reports that make the compiler exit non-zero).
                    _env = os.environ.copy()
                    _env.pop("ASAN_OPTIONS", None)
                    _env.pop("LSAN_OPTIONS", None)
                    _ld_preload = _env.get("LD_PRELOAD", "")
                    if _ld_preload:
                        _parts = [p for p in _ld_preload.split(":") if "libasan" not in p]
                        _env["LD_PRELOAD"] = ":".join(_parts) if _parts else ""
                    _r = _subprocess.run(
                        [_compiler, "-shared", "-fPIC", "-O2", "-o", _shim_path, "-xc", "-"],
                        input=_asan_opts_shim_src,
                        capture_output=True,
                        timeout=30,
                        env=_env,
                    )
                    if _r.returncode != 0:
                        raise OSError(f"compiler failed: {_r.stderr.decode(errors='replace')}")
                    _ctypes.CDLL(_shim_path, mode=_ctypes.RTLD_GLOBAL)
                    _ctypes.CDLL(libasan, mode=_ctypes.RTLD_GLOBAL)
                    _asan_ctypes_loaded = True
                    print(f"[*] ASAN preloaded for in-process: {libasan}")

                    # With halt_on_error=0 (set via __asan_default_options shim
                    # above), ASAN reports bugs to stderr but does not abort().
                    # The target function returns normally (rc=0) and stderr
                    # contains the full ASAN report. The existing crash detection
                    # pipeline (SanitizerReport.parse() in runner.py) detects
                    # the crash from captured stderr. No death callback is
                    # needed — ASAN only fires death callbacks in the fatal
                    # path (halt_on_error=1).
                except OSError as e:
                    print(f"[!] ASAN ctypes preload failed: {e}")
                finally:
                    with contextlib.suppress(OSError):
                        os.unlink(_shim_path)
            # UBSAN detection — set UBSAN_OPTIONS so errors abort the
            # target for crash detection.  The UBSAN runtime was already
            # preloaded via LD_PRELOAD by ldpreload_wrapper.py.
            target_is_ubsan = _detect_ubsan(self.target)
            if target_is_ubsan:
                ubsan_opts = os.environ.get("UBSAN_OPTIONS", "")
                opt_parts = [p for p in ubsan_opts.split(":") if p] if ubsan_opts else []
                seen = {p.split("=")[0] for p in opt_parts}
                for opt in ("halt_on_error=1", "abort_on_error=1", "print_stacktrace=1"):
                    key = opt.split("=")[0]
                    if key not in seen:
                        opt_parts.append(opt)
                        seen.add(key)
                os.environ["UBSAN_OPTIONS"] = ":".join(opt_parts)
        # Auto-detect .so targets and use in-process mode
        if not inprocess and self.target.lower().endswith((".so", ".dylib", ".dll")):
            from fuzzer_tool.adapters.inprocess import InProcessRunner

            cov_env_id = self.shm_cov.env_id if self.shm_cov else None
            if not cov_env_id:
                # Without an SHM segment nothing populates the edge bitmap, so
                # every coverage-guided subsystem downstream (seed scheduling,
                # MI/TE/sensitivity position weighting, Elo/bandit operator
                # scheduling, stall detection, corpus admission) runs on a
                # constant-zero signal. That degrades silently — the run looks
                # healthy and fast while discovering nothing — so say so.
                self._warn_no_coverage()
            # Probe the shared object for a fuzz function name
            auto_func = self._probe_so_function(self.target)
            # Decide whether to use direct_lite (in-process ctypes) mode.
            # ASAN-instrumented .so targets need the ASAN runtime loaded
            # before the target. If LD_PRELOAD already contained libasan
            # at process start (external wrapper), it's already available.
            # Otherwise, load a tiny shim that exports __asan_default_options
            # (returning "verify_asan_link_order=0") before libasan.so, so
            # ASAN skips the post-startup first-load check. Safe for fuzzing:
            # ASAN only needs target-side bug detection, not Python-side.
            use_direct_lite = True  # NEVER EVER CHANGE THIS!!!
            # ASAN ctypes preloading was done above (before the branch). If it
            # failed, fall back to persistent mode where LD_PRELOAD handles it.
            # Even if ctypes preloading succeeds, ASAN detection does NOT work
            # in direct_lite mode when loaded mid-process: the compiled-in
            # shadow offset (0x7fff8000) doesn't match the runtime ASAN shadow
            # mapping for mid-process-loaded libasan (the ASAN heap is placed
            # at addresses whose shadow lands outside the mapped shadow region
            # on 48-bit systems). See docs/ASAN-LIMITATION.md §Layer 1.
            # Subprocess mode with LD_PRELOAD (set above at line 1027) is the
            # reliable path: ASAN initializes at process start in the child,
            # the shadow mapping is correct, and halt_on_error=0 (from the
            # ctypes-loaded shim) prevents ASAN from aborting the child.
            if target_is_asan and not _asan_was_preloaded:
                use_direct_lite = False
            # Cmplog: if the .so has cmplog compiled in, direct_lite works
            # because the shim is part of the .so itself. If the shim is
            # externally LD_PRELOAD'd, that also works. Otherwise we need
            # a process boundary (or preload the shim via ctypes).
            if self._cmplog is not None:
                has_cmplog = _detect_cmplog(self.target)
                has_tracecmp = _detect_tracecmp_target(self.target)
                if has_cmplog or has_tracecmp:
                    if has_cmplog:
                        print("[*] Cmplog: compiled into target .so (direct_lite compatible)")
                    else:
                        print(
                            "[*] Trace-cmp: compiled into target .so (direct_lite compatible, preloading shim)"
                        )
                else:
                    ld_preload = os.environ.get("LD_PRELOAD", "")
                    shim_in_preload = "cmplog_shim" in ld_preload or "tracecmp_shim" in ld_preload
                    if not shim_in_preload:
                        use_direct_lite = False
                    else:
                        print("[*] Cmplog: externally LD_PRELOAD'd (direct_lite compatible)")
            # Set _CMPLOG_OUT in os.environ so the cmplog constructor can
            # open the log file. Must happen regardless of execution mode:
            # direct_lite loads the .so in-process, persistent mode loads
            # it in a subprocess that inherits os.environ.
            if self._cmplog is not None:
                self._cmplog.setup_env_for_run()
                if use_direct_lite:
                    self._cmplog.preload_shims()
            self._inprocess_runner = InProcessRunner(
                target=self.target,
                function_name=auto_func,
                timeout=self.timeout,
                shm_size=self.map_size,
                direct_lite=use_direct_lite,
                coverage_env_id=cov_env_id,
                cov=bool(cov_env_id),
                debug=self.debug,
                capture_stderr=target_is_asan or target_is_ubsan,
                use_ptrace=self.use_ptrace,
            )
            if use_direct_lite:
                mode = "direct_lite"
            elif self._inprocess_runner._persistent:
                mode = "persistent"
            else:
                mode = "subprocess loader"
            print(f"[*] Auto-detected .so target: in-process mode ({mode}) with {auto_func}")
        elif inprocess:
            from fuzzer_tool.adapters.inprocess import InProcessRunner

            cov_env_id = self.shm_cov.env_id if self.shm_cov else None
            if not cov_env_id:
                # Without an SHM segment nothing populates the edge bitmap, so
                # every coverage-guided subsystem downstream (seed scheduling,
                # MI/TE/sensitivity position weighting, Elo/bandit operator
                # scheduling, stall detection, corpus admission) runs on a
                # constant-zero signal. That degrades silently — the run looks
                # healthy and fast while discovering nothing — so say so.
                self._warn_no_coverage()
            # For .so targets, probe for the correct fuzz function name
            # when the user didn't explicitly specify one.
            func = inprocess_func
            if (
                self.target.lower().endswith((".so", ".dylib", ".dll"))
                and func == "LLVMFuzzerTestOneInput"
            ):
                func = self._probe_so_function(self.target)
            # ASAN ctypes preloading was done above (before the branch). If the
            # verify_asan_link_order=0 shim was loaded successfully, direct mode
            # works even for ASAN .so targets. The user explicitly requested
            # --inprocess-direct, so try direct regardless — ASAN-detected bugs
            # may abort the process, but that's the user's accepted tradeoff.
            direct_ok = inprocess_direct
            self._inprocess_runner = InProcessRunner(
                target=self.target,
                function_name=func,
                timeout=self.timeout,
                shm_size=self.map_size,
                direct=direct_ok,
                coverage_env_id=cov_env_id,
                cov=bool(cov_env_id),
                debug=self.debug,
                use_ptrace=self.use_ptrace,
            )
            mode = "direct ctypes" if direct_ok else "subprocess loader"
            cov_note = f", SHM cov id={cov_env_id}" if cov_env_id else ""
            print(f"[*] In-process mode ({mode}{cov_note}): {self.target}::{func}")
            if self._inprocess_runner._persistent:
                print("[*] Persistent loader: enabled (1 process, many calls)")

        # Forkserver: use C fuzz_loader for default execution path when available.
        # Currently disabled: fuzz_loader reads bitmap from file while target
        # writes to SHM — these are disconnected. Enable when fuzz_loader.c
        # Forkserver disabled for multi-target mode (requires single binary)
        # if not self._inprocess_runner and not self._persistent_runner and not self.ptrace_cov:
        #     from fuzzer_tool.adapters.forkserver import ForkserverRunner
        #     self._forkserver = ForkserverRunner(target, timeout=self.timeout)
        #     if self._forkserver.start():
        #         log.info("Forkserver started for default execution path")

    def _setup_ptrace(self, target, deep_coverage, max_bps, fallback_hint=False):
        cov = PtraceCoverage(target, deep_coverage=deep_coverage, max_bps=max_bps)
        if cov.bb_addrs:
            self.ptrace_cov = cov
            mode = "deep (pure decoder)" if cov.deep_coverage else "function-entry"
            print(f"[*] Coverage: {len(cov.bb_addrs)} breakpoints ({mode}), map={cov.map_size}")
        else:
            print(
                "[!] Coverage: no symbols found in ELF, "
                "coverage disabled (use -g to compile with symbols)"
            )
            if fallback_hint:
                print(
                    "[!] For closed-source binaries, use AFL++ QEMU mode: afl-qemu-trace ./target"
                )

    def _load_corpus(self):
        return self._corpus_manager.load_corpus()

    def _init_seed_metadata(self):
        return self._corpus_manager.init_seed_metadata()

    def _seed_key(self, data: bytes) -> str:
        """Return cached content hash for *data*."""
        cached = self._seed_key_cache.get(data)
        if cached is not None:
            return cached
        key = self._corpus_manager.seed_key(data)
        self._seed_key_cache[data] = key
        return key

    def _invalidate_seed_key_cache(self) -> None:
        """Clear the seed key cache — call when corpus structure changes."""
        self._seed_key_cache.clear()

    def _boost_corpus_sizes(self) -> None:
        """Resize each corpus seed to a target size drawn from N(boost_mean, boost_std),
        clamped to [1, corpus_boost]. Target sizes are shuffled to avoid ordering bias
        (e.g. all small seeds paired with small targets)."""
        if not self._corpus_boost:
            return
        n = len(self.corpus)
        if n == 0:
            return
        mean = self._boost_mean if self._boost_mean is not None else self._corpus_boost / 2.0
        std = self._boost_std if self._boost_std is not None else self._corpus_boost / 6.0
        std = max(std, 1.0)
        target_sizes = [
            max(1, min(int(random.gauss(mean, std)), self._corpus_boost)) for _ in range(n)
        ]
        random.shuffle(target_sizes)
        self.corpus = [
            self._resize_seed(s, t) for s, t in zip(self.corpus, target_sizes, strict=False)
        ]
        self._invalidate_seed_key_cache()

    def _resize_seed(self, seed: bytes, target_size: int) -> bytes:
        """Truncate or pad *seed* to *target_size* bytes.

        Padding modes (controlled by ``self._boost_pad``):
          * repeat — cycle the existing bytes (AFL-style, default)
          * zero   — zero-pad
          * random — fill with random bytes
        """
        if len(seed) == target_size:
            return seed
        if len(seed) > target_size:
            return seed[:target_size]
        need = target_size - len(seed)
        if self._boost_pad == "zero":
            return seed + b"\x00" * need
        if self._boost_pad == "random":
            return seed + bytes(random.randrange(256) for _ in range(need))
        # "repeat" (default): AFL-style cyclic padding
        if len(seed) == 0:
            return b"\x00" * target_size
        repeats = (need // len(seed)) + 1
        return seed + (seed * repeats)[:need]

    def _save_state(self):
        return self._corpus_manager.save_state()

    def _load_state(self):
        return self._corpus_manager.load_state()

    def _run_target(self, data: bytes):
        return self._runner.run_target(data)

    def _check_differential(self, data: bytes):
        """Run data on differential target and track divergence."""
        if not self._diff_tracker or not self._diff_target:
            return
        from fuzzer_tool.services.differential import diff_run

        diverged, desc = diff_run(self.target, self._diff_target, data)
        rc_b, stderr_b = 0, ""
        if diverged:
            pass  # diff_run already logs; stats tracked in _diff_tracker
        # track drift
        self._diff_tracker.record(0, "", rc_b, stderr_b)

    def _ptrace_handle_breakpoint(self, pid: int, libc, cov: PtraceCoverage, regs_buf):
        return self._runner._ptrace_handle_breakpoint(pid, libc, cov, regs_buf)

    def _run_target_ptrace(self, data: bytes):
        return self._runner._run_target_ptrace(data)

    def _is_interesting(self, returncode: int, stderr: str):
        return self._runner.is_interesting(returncode, stderr)

    def _is_crash(self, returncode: int, stderr: str):
        return self._runner.is_crash(returncode, stderr)

    def mutate(self, data: bytes):
        return self._operators.mutate(data)

    def _dedup_mutate(self, data: bytes) -> bytes:
        """Mutate *data*, re-rolling mutants the exec bloom has already seen.

        A hit means the mutant was almost certainly executed before, so the
        exec would buy no coverage and no bandit signal.  Re-rolling is close
        to free: ``mutate()`` costs ~2 orders of magnitude less than a target
        run.  After ``EXEC_DEDUP_RETRIES`` consecutive hits the last mutant is
        executed anyway, so a saturated filter degrades to the old behaviour
        rather than spinning.

        A bloom false positive (rate 1e-3) discards a genuinely novel mutant.
        That is harmless — it stays reachable on later iterations — and the
        filter never yields a false negative, so nothing already executed
        slips through as new.
        """
        mutated = self.mutate(data)
        if not self._dedup_execs:
            return mutated
        for _ in range(EXEC_DEDUP_RETRIES):
            if not self._exec_bloom.update_bytes(bytes(mutated), reset_on_full=True):
                return mutated
            self._dedup_hits += 1
            mutated = self.mutate(data)
        self._dedup_gaveup += 1
        return mutated

    def _cost_adjusted_weight(self, op: str, base_weight: float) -> float:
        """Scale a bandit reward by inverse operator cost.

        Converts "edges per selection" into an "edges per unit time"
        proxy: an operator's raw reward is multiplied by
        median_cost / this_op's_cost, so an operator costing the median
        amount is unaffected (ratio ~= 1.0), a cheap operator (bit_flip,
        ~us) gets a bonus, and an expensive operator (gradient_descent,
        condstmt_solve, path_negate, crc_learn, ~ms-s) gets penalized in
        proportion to how many cheap mutations could have run in the same
        wall-clock budget. This is fed to every per-op reward call
        (mc/mopt/replicator/exp3/eps_greedy/hierarchical/gp_ucb/elo) so
        the whole tournament sees cost, not just Elo.

        Falls back to the unscaled weight until at least a few operators
        have timing data, and clamps the ratio so a single outlier can't
        zero out or blow up the reward scale.
        """
        if base_weight <= 0.0 or len(self._op_time_ema) < 2:
            return base_weight
        cost = self._op_time_ema.get(op)
        if cost is None or cost <= 0.0:
            return base_weight
        costs = sorted(c for c in self._op_time_ema.values() if c > 0.0)
        if not costs:
            return base_weight
        median_cost = costs[len(costs) // 2]
        ratio = median_cost / cost
        ratio = max(0.05, min(ratio, 20.0))
        return base_weight * ratio

    def _build_ops(self, data: bytes):
        return self._operators.build_ops(data)

    def _select_op(self, ops: list[str]):
        return self._operators.select_op(ops)

    def _select_position(self, buf: bytearray, data: bytes):
        return self._operators.select_position(buf, data)

    # ── Operator handlers ──────────────────────────────────────────────
    # Each handler: (buf, byte_idx, data) -> None (in-place) or bytes (replace buf)

    def _op_bit_flip(self, buf, byte_idx, _data):
        return self._operators._op_bit_flip(buf, byte_idx, _data)

    def _op_bit_offset_flip(self, buf, _byte_idx, _data):
        return self._operators._op_bit_offset_flip(buf, _byte_idx, _data)

    def _op_bit_offset_span(self, buf, _byte_idx, _data):
        return self._operators._op_bit_offset_span(buf, _byte_idx, _data)

    def _op_byte_flip(self, buf, byte_idx, _data):
        return self._operators._op_byte_flip(buf, byte_idx, _data)

    def _op_interesting_8(self, buf, byte_idx, _data):
        return self._operators._op_interesting_8(buf, byte_idx, _data)

    def _op_interesting_16(self, buf, _byte_idx, _data):
        return self._operators._op_interesting_16(buf, _byte_idx, _data)

    def _op_interesting_32(self, buf, _byte_idx, _data):
        return self._operators._op_interesting_32(buf, _byte_idx, _data)

    def _op_arithmetic(self, buf, _byte_idx, _data):
        return self._operators._op_arithmetic(buf, _byte_idx, _data)

    def _op_random_bytes(self, buf, _byte_idx, _data):
        return self._operators._op_random_bytes(buf, _byte_idx, _data)

    def _op_block_insert(self, buf, _byte_idx, _data):
        return self._operators._op_block_insert(buf, _byte_idx, _data)

    def _op_block_delete(self, buf, _byte_idx, _data):
        return self._operators._op_block_delete(buf, _byte_idx, _data)

    def _op_block_duplicate(self, buf, _byte_idx, _data):
        return self._operators._op_block_duplicate(buf, _byte_idx, _data)

    def _op_dict_insert(self, buf, _byte_idx, _data):
        return self._operators._op_dict_insert(buf, _byte_idx, _data)

    def _op_dict_replace(self, buf, _byte_idx, _data):
        return self._operators._op_dict_replace(buf, _byte_idx, _data)

    def _op_dict_overwrite(self, buf, _byte_idx, _data):
        return self._operators._op_dict_overwrite(buf, _byte_idx, _data)

    def _op_dict_prepend(self, buf, _byte_idx, _data):
        return self._operators._op_dict_prepend(buf, _byte_idx, _data)

    def _op_dict_append(self, buf, _byte_idx, _data):
        return self._operators._op_dict_append(buf, _byte_idx, _data)

    def _op_checksum_repair(self, buf, _byte_idx, _data):
        return self._operators._op_checksum_repair(buf, _byte_idx, _data)

    def _op_token_dup(self, buf, _byte_idx, _data):
        return self._operators._op_token_dup(buf, _byte_idx, _data)

    def _op_markov_bytes(self, buf, _byte_idx, _data):
        return self._operators._op_markov_bytes(buf, _byte_idx, _data)

    def _op_cem_bytes(self, buf, _byte_idx, _data):
        return self._operators._op_cem_bytes(buf, _byte_idx, _data)

    def _op_splice(self, buf, _byte_idx, data):
        return self._operators._op_splice(buf, _byte_idx, data)

    def _op_crossover(self, buf, _byte_idx, data):
        return self._operators._op_crossover(buf, _byte_idx, data)

    def _op_type_replace(self, buf, _byte_idx, _data):
        return self._operators._op_type_replace(buf, _byte_idx, _data)

    def _op_ascii_num(self, buf, _byte_idx, _data):
        return self._operators._op_ascii_num(buf, _byte_idx, _data)

    def _op_byte_shuffle(self, buf, _byte_idx, _data):
        return self._operators._op_byte_shuffle(buf, _byte_idx, _data)

    def _op_byte_delete(self, buf, _byte_idx, _data):
        return self._operators._op_byte_delete(buf, _byte_idx, _data)

    def _op_byte_insert(self, buf, _byte_idx, _data):
        return self._operators._op_byte_insert(buf, _byte_idx, _data)

    def _op_insert_ascii_num(self, buf, _byte_idx, _data):
        return self._operators._op_insert_ascii_num(buf, _byte_idx, _data)

    def _op_transpose_16(self, buf, _byte_idx, _data):
        return self._operators._op_transpose_16(buf, _byte_idx, _data)

    def _op_transpose_32(self, buf, _byte_idx, _data):
        return self._operators._op_transpose_32(buf, _byte_idx, _data)

    def _op_transpose_64(self, buf, _byte_idx, _data):
        return self._operators._op_transpose_64(buf, _byte_idx, _data)

    def _op_bit_transpose_8(self, buf, _byte_idx, _data):
        return self._operators._op_bit_transpose_8(buf, _byte_idx, _data)

    def _op_bit_transpose_16(self, buf, _byte_idx, _data):
        return self._operators._op_bit_transpose_16(buf, _byte_idx, _data)

    def _op_bit_transpose_32(self, buf, _byte_idx, _data):
        return self._operators._op_bit_transpose_32(buf, _byte_idx, _data)

    def _op_bit_transpose_64(self, buf, _byte_idx, _data):
        return self._operators._op_bit_transpose_64(buf, _byte_idx, _data)

    def _op_length_grow(self, buf, _byte_idx, _data):
        return self._operators._op_length_grow(buf, _byte_idx, _data)

    def _op_length_shrink(self, buf, _byte_idx, _data):
        return self._operators._op_length_shrink(buf, _byte_idx, _data)

    def _op_repeat_clone(self, buf, _byte_idx, _data):
        return self._operators._op_repeat_clone(buf, _byte_idx, _data)

    def _op_truncate(self, buf, _byte_idx, _data):
        return self._operators._op_truncate(buf, _byte_idx, _data)

    def _op_length_boundary(self, buf, _byte_idx, _data):
        return self._operators._op_length_boundary(buf, _byte_idx, _data)

    def _op_swap_regions(self, buf, _byte_idx, _data):
        return self._operators._op_swap_regions(buf, _byte_idx, _data)

    def _op_swap_bytes(self, buf, _byte_idx, _data):
        return self._operators._op_swap_bytes(buf, _byte_idx, _data)

    def _op_endianness_swap(self, buf, _byte_idx, _data):
        return self._operators._op_endianness_swap(buf, _byte_idx, _data)

    def _op_grammar_mutate(self, buf, _byte_idx, _data):
        return self._operators._op_grammar_mutate(buf, _byte_idx, _data)

    def _op_grammar_tree_mutate(self, buf, _byte_idx, _data):
        return self._operators._op_grammar_tree_mutate(buf, _byte_idx, _data)

    def _op_png_chunk_mutate(self, buf, _byte_idx, _data):
        return self._operators._op_png_chunk_mutate(buf, _byte_idx, _data)

    def _op_jpeg_chunk_mutate(self, buf, _byte_idx, _data):
        return self._operators._op_jpeg_chunk_mutate(buf, _byte_idx, _data)

    def _op_jpeg_crc_fix(self, buf, _byte_idx, _data):
        return self._operators._op_jpeg_crc_fix(buf, _byte_idx, _data)

    def _op_gzip_chunk_mutate(self, buf, _byte_idx, _data):
        return self._operators._op_gzip_chunk_mutate(buf, _byte_idx, _data)

    def _op_bmp_chunk_mutate(self, buf, _byte_idx, _data):
        return self._operators._op_bmp_chunk_mutate(buf, _byte_idx, _data)

    def _op_zlib_chunk_mutate(self, buf, _byte_idx, _data):
        return self._operators._op_zlib_chunk_mutate(buf, _byte_idx, _data)

    def _op_png_crc_fix(self, buf, _byte_idx, _data):
        return self._operators._op_png_crc_fix(buf, _byte_idx, _data)

    def _op_redqueen(self, buf, _byte_idx, data):
        return self._operators._op_redqueen(buf, _byte_idx, data)

    def _op_havoc(self, buf, _byte_idx, data):
        return self._operators._op_havoc(buf, _byte_idx, data)

    # ── Dispatch table: op name → handler method ───────────────────────
    def _build_dispatch(self):
        return self._operators.build_dispatch()

    def _havoc_mutate(self, buf: bytearray):
        return self._operators.havoc_mutate(buf)

    def _apply_single_mutation(self, buf: bytearray):
        return self._operators._apply_single_mutation(buf)

    def save_crash(self, data: bytes, returncode: int, stderr: str):
        return self._corpus_manager.save_crash(data, returncode, stderr)

    def _prune_crash_data(self) -> None:
        """Trim crash structures when they exceed MAX_CRASH_SIGS.

        Keeps the most frequent crash signatures and evicts all data for
        the least frequent ones. Leaves crash_hashes intact (small memory
        footprint, prevents duplicate disk writes).
        """
        if len(self.crash_sigs) <= MAX_CRASH_SIGS:
            return
        # Keep top 75% of signatures sorted by frequency descending
        keep_count = max(MAX_CRASH_SIGS * 3 // 4, 1)
        sorted_sigs = sorted(self.crash_sigs.items(), key=lambda x: -x[1])
        kept = {sig for sig, _ in sorted_sigs[:keep_count]}
        evicted = set(self.crash_sigs) - kept
        self.crash_sigs = dict(sorted_sigs[:keep_count])
        # Evict associated data for dropped signatures
        for sig in evicted:
            self.crash_frames.pop(sig, None)
            self.crash_min_sizes.pop(sig, None)
            self._crash_replays.pop(sig, None)

    def save_to_corpus(self, data: bytes, parent: bytes | None = None):
        return self._corpus_manager.save_to_corpus(data, parent)

    def _trim_new_coverage(self, data: bytes, parent: bytes):
        return self._corpus_manager.trim_new_coverage(data, parent)

    def _auto_minimize_corpus(self):
        return self._corpus_manager.auto_minimize_corpus()

    def _defer_minimize(self):
        """Schedule auto_minimize_corpus for the next main-loop iteration.
        This avoids pruning seeds that were just added but not yet fuzzed."""
        self._minimize_pending = True

    def _record_lineage_insert(self, child: bytes, parent: bytes | None, corpus_len_before: int):
        """Insert *child* into the lineage tree when it joined the corpus.

        Gated on the flag; no-op when the seed was rejected as a duplicate
        (corpus length unchanged) or under qea where seeds bypass f.corpus.
        Node weight = new coverage edges contributed by this iteration.
        """
        if not self._use_lineage or self._lineage is None:
            return
        if len(self.corpus) <= corpus_len_before:
            return
        ops = [op for op, _ in self._last_ops_with_sites]
        sites = [s for _, s in self._last_ops_with_sites]
        parent_key = self._seed_key(parent) if parent is not None else None
        self._lineage.insert(
            parent_key, self._seed_key(child), ops, sites, self._last_new_edge_count
        )

    def _flush_pending_minimize(self):
        """Run deferred minimize if one is pending."""
        if self._minimize_pending:
            self._minimize_pending = False
            self._auto_minimize_corpus()

    def _deprioritize_near_duplicates(self):
        return self._corpus_manager.deprioritize_near_duplicates()

    def _check_memory_and_prune(self):
        """Check RSS against total RAM and prune corpus if threshold exceeded.

        Uses /proc/meminfo for total RAM and getrusage for current RSS.
        Only triggers once per 1000 execs to avoid constant polling overhead.
        """
        if self.prune_corpus_max_memory <= 0:
            return
        if self.exec_count - self._last_memory_prune_exec < 1000:
            return
        self._last_memory_prune_exec = self.exec_count

        try:
            rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        total_kb = int(line.split()[1])
                        break
                else:
                    return
        except (OSError, ValueError):
            return

        used_pct = (rss_kb / total_kb) * 100
        if used_pct >= self.prune_corpus_max_memory:
            before = len(self.corpus)
            log.warning(
                "Memory usage %.1f%% exceeds %d%% threshold — pruning corpus",
                used_pct,
                self.prune_corpus_max_memory,
            )
            self._auto_minimize_corpus()
            after = len(self.corpus)
            if after < before:
                print(
                    f"\n[*] MEMORY PRUNE: {before} → {after} seeds "
                    f"(RSS {rss_kb // 1024}MB / {total_kb // 1024}MB, {used_pct:.1f}%)"
                )

    def _select_next_target(self):
        """Select the next target for multi-target round-robin fuzzing."""
        if not self.multi_targets:
            return
        # Weighted round-robin: prefer targets with fewer total edges discovered
        if len(self.multi_targets) > 1 and self.exec_count > 100:
            # Weight by inverse of cumulative edges (less-covered targets get more execs)
            weights = []
            for t in self.multi_targets:
                shm = self._target_shm_covs.get(t)
                edges = shm.cumulative_edges if shm else 0
                weights.append(1.0 / max(edges, 1))
            total = sum(weights)
            r = random.random() * total
            cumulative = 0.0
            for idx, w in enumerate(weights):
                cumulative += w
                if r <= cumulative:
                    self._active_target_idx = idx
                    break
        else:
            self._active_target_idx = (self._active_target_idx + 1) % len(self.multi_targets)
        self.target = self.multi_targets[self._active_target_idx]

    def _pick_seed(self):
        return self._seed_picker.pick_seed()

    def _pick_markov_seed(self):
        return self._seed_picker._pick_markov_seed()

    def _pick_pareto_only(self):
        return self._seed_picker._pick_pareto_only()

    def _format_aware_seed(self):
        return self._seed_picker._format_aware_seed()

    def _compute_weights(self, now: float):
        return self._seed_picker._compute_weights(now)

    @staticmethod
    def _pareto_front(scores: list[tuple[float, float, float]], window: int = 100):
        return SeedPicker._pareto_front(scores, window)

    def _pick_from_pareto_front(self, weights: list[float], now: float):
        return self._seed_picker._pick_from_pareto_front(weights, now)

    def _weighted_pick_seed(self):
        return self._seed_picker.weighted_pick_seed()

    def _refresh_agg_cache(self) -> None:
        """Recompute running aggregates from seed_meta."""
        self._cached_total_time = sum(m.get("total_time", 0.0) for m in self.seed_meta.values())
        self._cached_total_fuzz = sum(m.get("fuzz_count", 1) for m in self.seed_meta.values())
        self._cached_total_edges = sum(m.get("coverage_edges", 0) for m in self.seed_meta.values())
        n_fuzz_vals = [m.get("fuzz_count", 0) for m in self.seed_meta.values()]
        self._cached_mean_log_n_fuzz = compute_mean_log_n_fuzz(n_fuzz_vals)
        self._agg_cache_valid = True

    def _reset_cmplog(self):
        """Flush cmplog buffer to disk before collecting tokens.

        In direct_lite mode with cmplog compiled into the target .so,
        the shim buffers CMP lines in a 256KB internal buffer. This flushes
        that buffer to disk so collect_tokens() can read the data.

        Does NOT truncate the file — collect_tokens() handles truncation
        after reading.
        """
        if self._cmplog is None:
            return

        # Flush tracecmp shim's internal buffer to disk
        self._cmplog.flush_shims()

        # Flush the target .so's compiled-in shim buffer
        runner = self._inprocess_runner
        if runner and runner.direct_lite and runner._lib:
            try:
                if hasattr(runner._lib, "__tracecmp_flush"):
                    runner._lib.__tracecmp_flush()
            except (AttributeError, OSError):
                pass

    def fuzz_one(self, data: bytes) -> bool:
        # Invalidate Elo K-factor cache at the start of each iteration
        # so record_strategy_match calls recompute K from the current
        # prediction errors if record_match hasn't been called yet.
        if self._use_elo and self._elo:
            self._elo._eff_k_cache = None
        self._last_parent_seed = data
        self._last_new_edge_count = 0  # reset; set when record_edges finds new edges
        meta = self.seed_meta.get(data)
        if meta is not None:
            meta["fuzz_count"] += 1
            self._cached_total_fuzz += 1

        t_start = time.monotonic()
        self._cov_before_fuzz = (
            len(self._edge_tracker._global_edge_hits)
            if hasattr(self._edge_tracker, "_global_edge_hits")
            else 0
        )
        mutated = self._dedup_mutate(data)
        returncode, stderr = self._run_target(mutated)
        t_elapsed = time.monotonic() - t_start
        self.exec_count += 1
        if self._stall_recovery_active:
            self._stall_recovery_execs += 1

        # Per-seed wall-clock cost
        if meta is not None:
            meta["total_time"] = meta.get("total_time", 0.0) + t_elapsed
            self._cached_total_time += t_elapsed

        # Record execution time for adaptive timeout calibration
        self._exec_time_tracker.record(t_elapsed)

        if self.mc:
            self.mc.execs_since_refit += 1

        # Flush tracecmp buffer before collecting tokens (direct_lite mode)
        self._reset_cmplog()

        # Collect cmplog tokens — periodic sampling once pool is saturated.
        # collect_tokens() reads + parses the whole cmplog file (~14-23ms
        # with 5000 pairs); running it every iteration when the pool is
        # already saturated destroys throughput.  Adaptive: eager while
        # building the pool, then sample every N iterations.
        cmplog_found = False
        smt_found = False
        if self._cmplog:
            pr_count = len(self._cmplog.pairs)
            _interval = 1 if pr_count < 500 else (5 if pr_count < 2000 else 20)
            self._cmplog_skip_counter += 1
            _collect_now = self._cmplog_skip_counter >= _interval
            if _collect_now:
                self._cmplog_skip_counter = 0
                new_tokens = self._cmplog.collect_tokens()
            else:
                new_tokens = []
            cmplog_found = bool(new_tokens)
            if not hasattr(self, "_dict_set"):
                self._dict_set = set(self.dictionary)
                self._dict_eps_window: list[float] = []
                self._dict_last_prune = 0
            for token in new_tokens:
                if token and token not in self._dict_set:
                    self.dictionary.append(token)
                    self._dict_set.add(token)

            # Feed checksum learner: format-aware pairs from current input
            # + cmplog heuristic pairs (gated on _collect_now to avoid
            # repeated work when the pool is saturated).
            if self.checksum_learner and _collect_now:
                fmt_pairs = self.checksum_learner.extract_format_pairs(data)
                if fmt_pairs:
                    self.checksum_learner.add_pairs(fmt_pairs)
                cmp_pairs = self.checksum_learner.extract_cmplog_pairs(data)
                if cmp_pairs:
                    self.checksum_learner.add_pairs(cmp_pairs)

            # Dynamic cap: scale with recent throughput.
            # High EPS → larger dictionary (more mutations explore more).
            # Low EPS → smaller dictionary (reduce overhead).
            # Window: last 500 iterations. Range: [64, 1024].
            window = 500
            if self.exec_count > 0 and self.exec_count % 100 == 0:
                elapsed = time.time() - self.start_time
                eps = (self.exec_count - self._resume_baseline_exec) / elapsed if elapsed > 0 else 0
                self._dict_eps_window.append(eps)
                if len(self._dict_eps_window) > 10:
                    self._dict_eps_window.pop(0)

            if self._dict_eps_window and self.exec_count - self._dict_last_prune >= window:
                # Use Kalman-filtered EPS if available, fall back to window avg.
                if (
                    hasattr(self, "_eps_filtered")
                    and self._eps_filtered is not None
                    and self._eps_filtered > 0
                ):
                    avg_eps = self._eps_filtered
                else:
                    avg_eps = sum(self._dict_eps_window) / len(self._dict_eps_window)
                # Map EPS to cap: 10 eps → 128, 30 eps → 256, 100+ eps → 1024
                dyn_cap = max(64, min(1024, int(avg_eps * 8)))
                if len(self.dictionary) > dyn_cap:
                    keep = max(dyn_cap // 2, 32)
                    self.dictionary = self.dictionary[-keep:]
                    self._dict_set = set(self.dictionary)
                    self._dict_last_prune = self.exec_count
            # Record redqueen matches: (offset, operand_a, operand_b)
            # for input-to-state matching during mutation.
            # Only scan new pairs (not yet seen) to avoid O(5000) per iteration.
            matches = list(meta.get("redqueen_matches", [])) if meta is not None else []
            seen = {(m[1], m[2]) for m in matches}  # dedup by (A, B)
            if (
                self._cmplog.pairs
                and meta is not None
                and self._redqueen_index < len(self._cmplog.pairs)
            ):
                for op_a, op_b in self._cmplog.pairs[self._redqueen_index :]:
                    if len(op_a) < 2 or (op_a, op_b) in seen:
                        continue

                    # Pass 1: find op_a literally in mutated input (original redqueen)
                    pos = 0
                    matched = False
                    while pos <= len(mutated) - len(op_a):
                        idx = mutated.find(op_a, pos)
                        if idx == -1:
                            break
                        matches.append((idx, op_a, op_b))
                        seen.add((op_a, op_b))
                        matched = True
                        pos = idx + 1
                        if len(matches) >= 50:
                            break
                    if matched or len(matches) >= 50:
                        if len(matches) >= 50:
                            break
                        continue

                    # Pass 2: try finding op_b instead (reverse direction)
                    if len(op_b) >= 2:
                        pos = 0
                        while pos <= len(mutated) - len(op_b):
                            idx = mutated.find(op_b, pos)
                            if idx == -1:
                                break
                            matches.append((idx, op_b, op_a))  # swap: replace op_b with op_a
                            seen.add((op_b, op_a))
                            matched = True
                            pos = idx + 1
                            if len(matches) >= 50:
                                break
                    if matched or len(matches) >= 50:
                        if len(matches) >= 50:
                            break
                        continue

                self._redqueen_index = len(self._cmplog.pairs)

            # SMT sampling pass: runs every iteration regardless of redqueen gate.
            # Adaptive sample size based on historical solve rate.
            if self._smt_solver is not None and self._cmplog.pairs:
                self._smt_solver.reset_batch()
                # Tune sample budget: if solve rate is sustained >50% we can
                # invest more; if <10% we're mostly wasting time on that input.
                _smt_q = self._smt_solver.queries_attempted
                if _smt_q > 50:
                    _rate = self._smt_solver.queries_solved / _smt_q
                    if _rate < 0.1:
                        _budget = 2
                    elif _rate > 0.5:
                        _budget = 10
                    else:
                        _budget = 5
                else:
                    _budget = 5
                sample = self._cmplog.pairs[:]
                import random as _rand

                _rand.shuffle(sample)
                smt_counter = 0
                for op_a, op_b in sample:
                    if smt_counter >= _budget:
                        break
                    if (op_a, op_b) in seen:
                        continue
                    smt_counter += 1
                    pc = self._cmplog.pair_pc(op_a, op_b)
                    result = self._smt_solver.solve_cmplog_pair(op_a, op_b, pc=pc)
                    smt_found = False
                    if result is not None:
                        solved = result["solved_bytes"]
                        for candidate, target in [(op_a, solved), (op_b, solved)]:
                            if len(candidate) < 2:
                                continue
                            pos = 0
                            while pos <= len(mutated) - len(candidate):
                                idx = mutated.find(candidate, pos)
                                if idx == -1:
                                    break
                                matches.append((idx, candidate, target))
                                seen.add((candidate, target))
                                smt_found = True
                                pos = idx + 1
                                if len(matches) >= 50:
                                    break
                            if smt_found or len(matches) >= 50:
                                break
                        if len(matches) >= 50:
                            break
            # Concolic mode: after accumulating trace entries, solve and inject
            if (
                self._smt_solver is not None
                and self._smt_solver.mod_solving_mode == "concolic"
                and self._smt_solver.concolic_trace is not None
                and self._smt_solver.concolic_trace.has_entries()
            ):
                concolic_result = self._smt_solver.solve_concolic(mutated)
                if concolic_result is not None and concolic_result != mutated:
                    self._smt_solver.queries_solved += 1
                    self._smt_solver.batch_solved += 1
                    # Inject the concolic solution as a replacement mutation
                    matches.append((0, mutated, concolic_result))
                    smt_found = True
                else:
                    self._smt_solver.queries_failed += 1

            # Path negation: solve for an input that takes the opposite side
            # of a branch this run actually took. Unlike the concolic block
            # above — which pins every byte to a literal, giving a fully
            # determined system that reproduces the observed operands — this
            # leaves the operand window symbolic and asserts the *negated*
            # predicate, so z3 searches for a value reaching the sibling
            # branch rather than replaying one already seen.
            if self._path_solver is not None and self._cmplog is not None:
                from fuzzer_tool.core.path_constraints import records_from_collector

                records = records_from_collector(self._cmplog)
                if records:
                    negated = self._path_solver.solve_first(records, mutated)
                    if negated is not None and negated != mutated:
                        matches.append((0, mutated, negated))
                        smt_found = True
            if meta is not None:
                meta["redqueen_matches"] = matches[:50]
                # Keep legacy field for state compat
                meta["redqueen_offsets"] = [m[0] for m in meta["redqueen_matches"]]

        if self.exec_count % 100 == 0:
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if rss > self._peak_rss:
                self._peak_rss = rss
            elapsed = time.time() - self.start_time
            eps = (self.exec_count - self._resume_baseline_exec) / elapsed if elapsed > 0 else 0
            if eps > self._peak_eps:
                self._peak_eps = eps
            self._crash_rate_execs.append(self.exec_count)
            self._crash_rate_counts.append(self.crash_count)
            if len(self._crash_rate_execs) > CRASH_RATE_HISTORY_MAX:
                del self._crash_rate_execs[:250]
                del self._crash_rate_counts[:250]

        for op in set(self._last_ops_used):
            self.op_counts[op] = self.op_counts.get(op, 0) + 1

        # Track cmplog as its own operator
        if cmplog_found:
            self.op_counts["cmplog"] = self.op_counts.get("cmplog", 0) + 1

        # Track SMT solver as its own operator
        if smt_found:
            self.op_counts["smt_solver"] = self.op_counts.get("smt_solver", 0) + 1

        is_timeout = returncode == -1 and stderr == "timeout"
        if is_timeout:
            self.timeout_count += 1
            # Mark the parent seed as timeout-causing for power schedule
            parent_meta = self.seed_meta.get(self._last_parent_seed)
            if parent_meta is not None:
                parent_meta["timed_out"] = True

        is_crash = self._is_crash(returncode, stderr)
        is_interesting = self._is_interesting(returncode, stderr)
        # Check new coverage (per-target SHM in multi-target mode).
        # Use is_new_coverage_with_edges() on SHM to get both the boolean
        # and the edge set in one buffer scan, avoiding redundant scans.
        self._current_edges_cache = None  # will be set below if SHM scanned
        if self.multi_targets:
            active_shm = self._target_shm_covs.get(self.target)
            if active_shm:
                has_new, edge_ids = active_shm.is_new_coverage_with_edges()
                self._current_edges_cache = edge_ids
                has_new_coverage = has_new
            else:
                # bool(): `x and x.f()` yields None (not False) when x is
                # None, and that None propagates into `success`, which
                # MonteCarloScheduler.record() feeds to float().
                has_new_coverage = bool(
                    (self.ptrace_cov and self.ptrace_cov.is_new_coverage())
                    or (self.shm_cov and self.shm_cov.is_new_coverage())
                )
        elif self.shm_cov:
            has_new, edge_ids = self.shm_cov.is_new_coverage_with_edges()
            self._current_edges_cache = edge_ids
            has_new_coverage = has_new
        else:
            has_new_coverage = bool(self.ptrace_cov and self.ptrace_cov.is_new_coverage())

        # Bayesian seed quality feedback: record whether this parent seed
        # produced new coverage (Thompson sampling posterior update).
        if self._seed_quality:
            parent_key = self._seed_key(data)
            self._seed_quality.init_seed(parent_key)
            self._seed_quality.record_outcome(parent_key, discovered=bool(has_new_coverage))

        # Mark cmplog tokens/pairs present during a coverage gain as more
        # valuable — they survive eviction longer.
        if has_new_coverage and self._cmplog:
            self._cmplog.mark_coverage_gain()

        # Record crash MI: I(byte_position; crash_outcome)
        if self._crash_mi:
            self._crash_mi.record(mutated, is_crash)

        # Format learner: only record when coverage actually changes
        if self._format_learner and self._last_ops_used and has_new_coverage:
            current_edges = (
                self._current_edges_cache
                if self._current_edges_cache is not None
                else self._get_current_edge_set()
            )
            new_edges = set()
            lost_edges = set()
            if hasattr(self, "_prev_edge_set"):
                new_edges = current_edges - self._prev_edge_set
                lost_edges = self._prev_edge_set - current_edges
            self._prev_edge_set = current_edges

            cov_after = (
                len(self._edge_tracker._global_edge_hits)
                if hasattr(self._edge_tracker, "_global_edge_hits")
                else 0
            )
            parent_meta = self.seed_meta.get(self._last_parent_seed)
            stride = parent_meta.get("record_stride") if parent_meta else None
            if stride != self._format_learner.record_stride:
                self._format_learner.set_record_stride(stride)
            self._format_learner.record_transition(
                input_bytes=mutated,
                mutation_op=self._last_ops_used[0] if self._last_ops_used else "unknown",
                mutation_offset=self._last_mutation_offset,
                mutation_width=len(mutated),
                coverage_before=self._cov_before_fuzz,
                coverage_after=cov_after,
                new_edges=new_edges,
                lost_edges=lost_edges,
            )
        elif self._format_learner:
            self._prev_edge_set = (
                self._current_edges_cache
                if self._current_edges_cache is not None
                else self._get_current_edge_set()
            )

        # Write ablation log row: signal data + outcome
        if self._ablation_file and hasattr(self, "_last_pick_signals"):
            ps = self._last_pick_signals
            self._ablation_file.write(
                f"{self.exec_count},{ps['seed_idx']},{ps['seed_hash']},"
                f"{ps['fuzz_count']},{ps['coverage_edges']},{ps['age_s']},"
                f"{ps['base_w']},{ps['burst']},{ps['penalty']},"
                f"{ps['subsumption']},{ps['diversity']},{ps['spatial']},"
                f"{ps['mdl']},{ps['final_w']},"
                f"{1 if has_new_coverage else 0},{1 if is_crash else 0}\n"
            )
            if self.exec_count % 100 == 0:
                self._ablation_file.flush()

        # Record edges for per-seed tracking
        if has_new_coverage:
            seed_key = self._seed_key(data)
            # Prefer sparse edge set with counts (SHM), fall back to byte bitmap (ptrace)
            if self.shm_cov and not self.ptrace_cov:
                hit_counts = self.shm_cov.get_edge_counts()
                hit_edges = set(hit_counts.keys())
            else:
                hit_edges = (
                    self._current_edges_cache
                    if self._current_edges_cache is not None
                    else self._get_current_edge_set()
                )
                hit_counts = None
            if hit_edges:
                # Read stack depth and path hash from SHM metadata (if available)
                stack_depth = 0
                path_hash = 0
                if self.shm_cov:
                    stack_depth = self.shm_cov.read_stack_depth()
                    path_hash = self.shm_cov.read_path_hash()
                # Fallback: compute path hash from edge IDs in Python
                if path_hash == 0 and hit_edges and not isinstance(hit_edges, bytes):
                    path_hash = (
                        self.shm_cov.compute_path_hash_from_edges(hit_edges) if self.shm_cov else 0
                    )
                new = self._edge_tracker.record_edges(
                    seed_key,
                    hit_edges,
                    target_name=os.path.basename(self.target) if self.multi_targets else "",
                    hit_counts=hit_counts,
                    stack_depth=stack_depth,
                    path_hash=path_hash,
                    hw_instructions=self._last_perf_deltas.get("instructions", 0),
                    hw_branches=self._last_perf_deltas.get("branches", 0),
                    hw_branch_misses=self._last_perf_deltas.get("branch_misses", 0),
                )
                if new:
                    self._last_new_edge_exec = self.exec_count
                    self._last_new_edge_count = len(new)
                    # Attribute new edges to the operators that ran this iteration.
                    # Proportional split: edges ÷ unique ops in _last_ops_used.
                    unique_ops = list(dict.fromkeys(self._last_ops_used))
                    if unique_ops:
                        share = len(new) / len(unique_ops)
                        for op in unique_ops:
                            self.op_edges[op] = self.op_edges.get(op, 0.0) + share
                    # Separate counter for cmplog-involved edge discoveries
                    # (cumulative with the op attribution above — cmplog is a
                    #  signal source, not a mutation op, so it can overlap).
                    if cmplog_found:
                        self.op_edges["cmplog"] = self.op_edges.get("cmplog", 0.0) + len(new)
                    if smt_found:
                        self.op_edges["smt_solver"] = self.op_edges.get("smt_solver", 0.0) + len(
                            new
                        )
                    if self._stall_recovery_active:
                        print(
                            f"\n[*] RECOVERED: found {len(new)} new edges at exec "
                            f"{self.exec_count}, resuming normal mode"
                        )
                        self._stall_recovery_active = False
                if meta is not None and new:
                    meta["coverage_edges"] += len(new)
                    self._cached_total_edges += len(new)
                    meta["momentum"] = 0.8 * meta["momentum"] + 0.2 * 1.0
                elif meta is not None:
                    meta["momentum"] = 0.8 * meta["momentum"]
                # Secretary-problem: track seed discovery rate for optimal stopping
                if self._secretary and seed_key:
                    if seed_key not in self._seed_secretary:
                        self._seed_secretary[seed_key] = SecretaryStopping(
                            window_size=self._secretary_window,
                            exploration_frac=self._secretary_exploration,
                        )
                        if len(self._seed_secretary) > SEED_SECRETARY_MAX:
                            # Evict oldest 100 entries (dict preserves insertion order)
                            for k in list(self._seed_secretary)[:100]:
                                del self._seed_secretary[k]
                    fuzz_count = max(meta["fuzz_count"], 1) if meta else 1
                    discovery_rate = len(new) / fuzz_count
                    self._seed_secretary[seed_key].observe(discovery_rate)

        # Update edge lifetime tracking for every execution
        if self._inprocess_runner or self.ptrace_cov or self.shm_cov:
            current_edges = (
                self._current_edges_cache
                if self._current_edges_cache is not None
                else self._get_current_edge_set()
            )
            if current_edges:
                self._edge_tracker.record_edge_lifetimes(current_edges, self.exec_count)

        # Track input-length → edge discovery correlation
        if has_new_coverage and self._length_tracker:
            new_edges = (
                self._current_edges_cache
                if self._current_edges_cache is not None
                else self._get_current_edge_set()
            )
            if new_edges:
                self._length_tracker.record(len(mutated), new_edges)

        # Compute directed distance for targeted fuzzing.  Prefer the
        # runtime average from the SHM tail (AFLGo channel, exact per-BB
        # distances accumulated in the target) when the target carries
        # the distance table; otherwise derive it in Python from the
        # edge trace.
        if self._distance and meta is not None:
            runtime_avg = self._read_runtime_avg_distance()
            if runtime_avg is not None:
                meta["avg_distance"] = runtime_avg
                if self._dist_min_observed is None or runtime_avg < self._dist_min_observed:
                    self._dist_min_observed = runtime_avg
                if self._dist_max_observed is None or runtime_avg > self._dist_max_observed:
                    self._dist_max_observed = runtime_avg
            elif has_new_coverage:
                hit_bbs = (
                    self._current_edges_cache
                    if self._current_edges_cache is not None
                    else self._get_current_edge_set()
                )
                if hit_bbs:
                    # Record edge trace for distance computation
                    seed_key = self._seed_key(data)
                    edge_pairs = {(i, i) for i in hit_bbs}  # self-loops as BB proxies
                    self._edge_tracker.record_edge_trace(seed_key, edge_pairs)
                    # Compute average distance
                    avg_dist = self._distance.seed_distance({(i, i) for i in hit_bbs})
                    meta["avg_distance"] = avg_dist
                    if avg_dist < 20.0:  # exclude the no-valued-blocks sentinel
                        if self._dist_min_observed is None or avg_dist < self._dist_min_observed:
                            self._dist_min_observed = avg_dist
                        if self._dist_max_observed is None or avg_dist > self._dist_max_observed:
                            self._dist_max_observed = avg_dist

        # Update annealing progress for directed mode
        if self._distance and self.exec_count > 0:
            # Anneal over first 20% of max_len-scaled iterations
            anneal_target = max(5000, self.max_len * 10)
            self._anneal_progress = min(1.0, self.exec_count / anneal_target)

        success = bool(is_crash or is_interesting or has_new_coverage)

        # Per-operator credit. An operator that was selected but left the
        # buffer unchanged cannot have caused this round's outcome, so it
        # must not be recorded as a success -- but it must still be recorded
        # as a failure, or an operator that no-ops forever would never be
        # deprioritised and would keep consuming selection slots.
        effective = self._last_ops_effective if self._track_op_effect else None

        def _op_success(op: str) -> bool:
            return success and (effective is None or op in effective)

        # Surprisal-weighted reward: discoveries in sparse regions of the
        # coverage bitmap carry more information than discoveries near
        # already-saturated areas. Weight = 1 - density so rare discoveries
        # (low density) get higher credit; saturated regions (high density)
        # get lower credit.
        if success and self._edge_tracker and self._edge_tracker.map_size:
            density = self._edge_tracker.bitmap_density()
            surprisal_weight = max(0.05, 1.0 - density)
        else:
            surprisal_weight = 1.0 if success else 0.0

        if success and self._last_havoc_subops:
            # Havoc's inner branches, credited on the same signal the outer
            # bandits use. Trials are counted at application time, so a
            # branch whose guard fails accrues trials without hits and
            # decays -- the same treatment no-op operators get above.
            self._operators.credit_havoc_subops(self._last_havoc_subops)

        if success:
            # Same rule as the bandits: no-op operators didn't earn this.
            for op in effective if effective is not None else set(self._last_ops_used):
                self.op_success[op] = self.op_success.get(op, 0) + 1
            if cmplog_found:
                self.op_success["cmplog"] = self.op_success.get("cmplog", 0) + 1
            if smt_found:
                self.op_success["smt_solver"] = self.op_success.get("smt_solver", 0) + 1

        # Per-operator outcome and reward, computed once. Every scheduler
        # below scored the same operator identically, so this was seven
        # copies of the same dedup-and-weight loop.
        op_rewards = []
        for op in dict.fromkeys(self._last_ops_used):
            ok = _op_success(op)
            op_rewards.append(
                (op, ok, self._cost_adjusted_weight(op, surprisal_weight if ok else 0.0))
            )

        if self.mc and self.mc_bandit:
            for op, ok, w in op_rewards:
                self.mc.record(op, ok, weight=w)
                self.mc.record_brier(op, ok, weight=w)
                # Secretary-problem: track operator quality for optimal stopping
                if self._secretary:
                    if op not in self._op_secretary:
                        self._op_secretary[op] = SecretaryStopping(
                            window_size=self._secretary_window,
                            exploration_frac=self._secretary_exploration,
                            min_observations=50,
                        )
                    a = self.mc.arm_alpha.get(op, 1.0)
                    b = self.mc.arm_beta.get(op, 1.0)
                    self._op_secretary[op].observe(a / (a + b))

        if self._mopt and (not self._use_elo or self._meta_strategy == "mopt"):
            # MOpt is separate: it needs the particle each operator was drawn
            # from, so it pairs each op with its first particle rather than
            # iterating the deduped list.
            rewards_by_op = {op: (ok, w) for op, ok, w in op_rewards}
            seen = set()
            for op, pid in zip(self._last_ops_used, self._last_mopt_particles, strict=False):
                if op not in seen and op in rewards_by_op:
                    ok, w = rewards_by_op[op]
                    self._mopt.record(op, ok, particle_id=pid, weight=w)
                    seen.add(op)

        # Schedulers sharing the record(op, success, weight=...) signature.
        for scheduler in (
            self._replicator if self._use_replicator else None,
            self._exp3,
            self._eps_greedy,
            self._hierarchical,
            self._gp_ucb,
        ):
            if scheduler is None:
                continue
            for op, ok, w in op_rewards:
                scheduler.record(op, ok, weight=w)

        if self._contextual:
            # LinUCB takes a feature vector rather than a success flag.
            for op, ok, w in op_rewards:
                self._contextual.record(op, self._operators._context_vector(op), w if ok else 0.0)

        # Chi-squared operator heterogeneity test
        if (
            self._chi2_operator_interval > 0
            and self.exec_count > 0
            and self.exec_count % self._chi2_operator_interval == 0
        ):
            try:
                self._run_chi2_operator_test()
            except Exception as ex:
                log.debug("Chi-squared operator test failed: %s", ex)

        # Elo: record matches between operators that were used
        if self._use_elo and self._elo and len(self._last_ops_used) >= 2:
            unique_ops = list(dict.fromkeys(self._last_ops_used))  # preserve order, dedup
            # Winners are the operators that actually changed the buffer. This
            # used to be `set(self._last_ops_used)`, which is by construction
            # the same set as `unique_ops` -- so `losers` in record_round() was
            # always empty and the entire winners-beat-losers branch, including
            # the proportional edge_counts path, was unreachable from here.
            # Measured: 1000/1000 rounds fell through to the cross-iteration
            # fallback. Crediting no-op operators is also wrong on its own
            # terms: on a single-seed corpus, splice and crossover change
            # nothing 100% of the time yet were scored as full winners on every
            # successful round.
            #
            # Buffer-change is necessary, not sufficient -- an operator can
            # change a byte the target never reads. But it strictly dominates
            # "everyone wins", and it produces a usable split in 56.5% of
            # rounds (measured over 2500 execs); the rest fall through to the
            # cross-iteration path as before.
            winners = set(self._last_ops_effective) if success else set()
            # Record unconditionally, including rounds where nothing found
            # coverage. Guarding on `if winners:` meant Elo only ever saw
            # successful iterations -- a systematic positive bias, and no
            # learning at all during a stall (measured: 4000 execs sitting
            # in random_stall produced zero recorded matches across all 106
            # arms). record_round() handles the empty-winners case via the
            # cross-iteration comparison against the previous round.
            #
            # Failed rounds vastly outnumber successes and the
            # cross-iteration path is quadratic in operators per round, so
            # sampling them was tried. It is not worth it: direct
            # instrumentation puts record_round at 0.69s of a 48.7s,
            # 2500-exec run (1.4%), while sampling 1-in-4 narrowed the
            # rating spread from 49.4 to 38.1 points. Paying 1.4% for the
            # full signal is the better trade. Note wall-clock A/B is
            # useless for judging this -- run-to-run variance on the same
            # build spanned 23s-54s, which is why the figure above comes
            # from timing record_round itself rather than total runtime.
            # Cost-aware edge_counts: previously always None, so record_round
            # took the flat score_a=1.0 path for every winner regardless of
            # how expensive it was to run. Feeding a per-op edges/time proxy
            # activates record_round's existing proportional-scoring branch
            # (edges[op] / max_edges among winners) so an expensive winner
            # that merely tied a cheap winner's edge count scores lower.
            edge_counts: dict[str, float] | None = None
            if winners and self._last_new_edge_count:
                # Split across winners, not all selected operators: a no-op
                # operator contributed no edges, and including it in the
                # denominator diluted everyone else's share.
                raw_share = self._last_new_edge_count / len(winners)
                edge_counts = {op: self._cost_adjusted_weight(op, raw_share) for op in winners}
            self._elo.record_round(unique_ops, winners, edge_counts=edge_counts, crash=is_crash)
            # Apply periodic decay
            self._elo_decay_counter += 1
            if self._elo_decay_counter >= self._elo_decay_interval:
                self._elo_decay_counter = 0
                self._elo.apply_decay()

        # Meta-elo: record operator strategy-level match. Only when the current
        # strategy is a real selectable scheduler (random_stall is excluded, so
        # stall recovery never accrues phantom matches)
        if self._use_elo and self._elo and self._meta_strategy:
            self._record_operator_strategy_matches(surprisal_weight if success else 0.0)

        # Meta-elo: record seed strategy-level match
        if self._use_elo and self._elo and self._seed_strategy:
            score = surprisal_weight if success else 0.0
            self._record_seed_strategy_matches(score)

        if self._use_shapley and self._shapley:
            new_edges = self._get_current_edge_set()
            if new_edges:
                self._shapley.record(
                    set(effective) if effective is not None else set(self._last_ops_used),
                    len(new_edges),
                    new_edges,
                )
            elif self.exec_count > 0 and self._last_ops_used:
                # Even with no edges, record a zero to track operator impact
                self._shapley.record(set(self._last_ops_used), 0, set())

        if self._use_mi and self._mi:
            current_edges = self._get_current_edge_set()
            if current_edges:
                self._mi.record(data, current_edges, self.map_size)

        if self._use_transfer_entropy and self._te:
            current_edges = self._get_current_edge_set()
            if current_edges:
                self._te_input_history.append(data[:64] if len(data) > 64 else data)
                self._te_edge_history.append(current_edges)
                if len(self._te_input_history) > self._te_history_max:
                    self._te_input_history = self._te_input_history[-self._te_history_max :]
                    self._te_edge_history = self._te_edge_history[-self._te_history_max :]
                # Update byte→edge causal map periodically
                if len(self._te_input_history) % 100 == 0 and len(self._te_input_history) > 50:
                    self._update_te_causal_map()

        if is_crash:
            self.crash_count += 1
            # direct_lite crashes carry no fault address (no ptrace, and the
            # guarded call reports only the signal). Re-run the input once
            # through the ptrace-attached loader to capture si_addr + regs.
            if (
                self._last_fault_addr is None
                and self._triage_ok is not False
                and self._inprocess_runner is not None
                and self._inprocess_runner.direct_lite
                and str(self.target).lower().endswith((".so", ".dylib", ".dll"))
            ):
                if self._triage_ok is None:
                    self._triage_ok = ptrace_available()
                if self._triage_ok:
                    try:
                        TargetRunner(self)._run_triage_ptrace(mutated)
                    except Exception as e:
                        log.debug("crash triage failed: %s", e)
            crash_name = self.save_crash(mutated, returncode, stderr)
            self._prune_crash_data()
            # Generate GDB/strace trace report if enabled
            if self._tracer and crash_name:
                report = self._tracer.trace(mutated, returncode)
                self._tracer.save_report(report, str(self.crashes_dir), crash_name)
            if self.mc and self.mc_cem:
                self.mc.add_elite(mutated, 3, temperature=self._temperature)
                self.mc.maybe_refit()
            # Schedule crash replay for reproducibility check
            if self.replay_n > 0 and crash_name:
                sig = self.crash_sigs.get(crash_name, crash_name)
                if sig not in self._crash_replays:
                    self._crash_replays[sig] = []
            # Schedule sanitizer replay: re-run crash on ASAN/UBSAN targets
            if (self.asan_target or self.ubsan_target) and crash_name:
                sig = self.crash_sigs.get(crash_name, crash_name)
                if sig not in self._crash_sanitizer_replays:
                    self._crash_sanitizer_replays[sig] = {
                        "data": mutated,
                        "asan": None,
                        "ubsan": None,
                    }
            return True

        if is_interesting or has_new_coverage:
            _corpus_len_before = len(self.corpus)
            self.save_to_corpus(mutated, parent=data)
            self._record_lineage_insert(mutated, data, _corpus_len_before)
            # GA: add new-coverage individual to population
            if self.ga and has_new_coverage:
                edge_count = (
                    len(self._edge_tracker.seed_edges.get(self._edge_tracker._last_seed_key, set()))
                    if hasattr(self._edge_tracker, "_last_seed_key")
                    else 0
                )
                ind = self.ga.on_fuzz_result(mutated, True, edge_count, self._edge_tracker)
                if ind is not None:
                    self.ga.add_to_population(ind)
            # QEA: amplitude rotation feedback + new individual on coverage
            if self.qea:
                edge_count = (
                    len(self._edge_tracker.seed_edges.get(self._edge_tracker._last_seed_key, set()))
                    if hasattr(self._edge_tracker, "_last_seed_key")
                    else 0
                )
                qea_ind = self.qea.on_fuzz_result(
                    mutated, has_new_coverage, edge_count, self._edge_tracker
                )
                if qea_ind is not None:
                    self.qea.add_to_population(qea_ind)
            # Analyze byte sensitivity for seeds that found new coverage (optional)
            if has_new_coverage and self.shm_cov and self._use_sensitivity:
                try:
                    edges = self.shm_cov.get_edge_ids()
                    if edges:

                        def _exec_fn(data):
                            rc, _ = self._run_target(data)
                            if self.shm_cov:
                                return self.shm_cov.get_edge_ids()
                            return set()

                        self._sensitivity.analyze_seed(mutated, edges, _exec_fn)
                except Exception:
                    pass
            # Coverage-guided trimming: try to minimize inputs that hit new edges
            if has_new_coverage and len(mutated) > 10:
                self._trim_new_coverage(mutated, data)
            if self.mc and self.mc_cem:
                self.mc.add_elite(mutated, 2, temperature=self._temperature)
                self.mc.maybe_refit()
            # Periodic minimization based on edge stats
            if (
                self.minimize_every_execs > 0
                and (self.exec_count - self._exec_baseline) % self.minimize_every_execs == 0
                and len(self.corpus) > 1
            ):
                self._auto_minimize_corpus()
                self._deprioritize_near_duplicates()
            return True

        # ── Metropolis acceptance for non-improving / non-crashing inputs ──
        if self._metropolis and self._anneal_budget > 0 and not is_timeout:
            p_accept = math.exp(-1.0 / max(self._temperature, 0.01))
            if random.random() < p_accept:
                _corpus_len_before = len(self.corpus)
                self.save_to_corpus(mutated, parent=data)
                self._record_lineage_insert(mutated, data, _corpus_len_before)
                if self.mc and self.mc_cem:
                    self.mc.add_elite(mutated, 1, temperature=self._temperature)
                    self.mc.maybe_refit()
                return True

        # Periodic minimization (also for non-interesting iterations)
        if (
            self.minimize_every_execs > 0
            and (self.exec_count - self._exec_baseline) % self.minimize_every_execs == 0
            and len(self.corpus) > 1
        ):
            self._auto_minimize_corpus()

        # GA: trigger generation boundary for non-coverage iterations
        if self.ga:
            self.ga.on_fuzz_result(mutated, False, 0, self._edge_tracker)

        # QEA: trigger generation boundary and rotation for non-coverage
        if self.qea:
            self.qea.on_fuzz_result(mutated, False, 0, self._edge_tracker)

        return False

    def _record_discovery_snapshot(self):
        return self._stats.record_discovery_snapshot()

    def _run_calibration(self, max_execs: int = 1000):
        return self._stats.run_calibration(max_execs)

    def discovery_rate(self):
        return self._stats.discovery_rate()

    def _run_crash_replays(self, budget_ms: float = 200):
        return self._stats.run_crash_replays(budget_ms)

    def _run_sanitizer_replays(self, budget_ms: float = 200):
        """Replay crashes on ASAN/UBSAN targets for sanitizer reports.

        Runs in subprocess mode (fork+exec with LD_PRELOAD) because
        ASAN detection doesn't work in-process via ctypes/direct_lite.
        """
        from fuzzer_tool.adapters.process import run_target_stdin
        from fuzzer_tool.core.sanitizer import SanitizerReport

        t0 = time.monotonic()
        pending = [
            (sig, info)
            for sig, info in self._crash_sanitizer_replays.items()
            if info["asan"] is None or info["ubsan"] is None
        ]
        for sig, info in pending:
            if (time.monotonic() - t0) * 1000 > budget_ms:
                break

            data = info["data"]

            # Replay on ASAN target
            if self.asan_target and info["asan"] is None:
                env = os.environ.copy()
                self._setup_asan_env(env)
                try:
                    rc, stderr, _ = run_target_stdin(self.asan_target, data, self.timeout, env=env)
                    report = SanitizerReport.parse(stderr)
                    info["asan"] = {
                        "rc": rc,
                        "report": report.to_dict() if report and report.is_valid() else None,
                        "stderr": stderr[:4096],
                    }
                except Exception as e:
                    info["asan"] = {"rc": -2, "error": str(e)}

            # Replay on UBSAN target
            if self.ubsan_target and info["ubsan"] is None:
                env = os.environ.copy()
                self._setup_ubsan_env(env)
                try:
                    rc, stderr, _ = run_target_stdin(self.ubsan_target, data, self.timeout, env=env)
                    report = SanitizerReport.parse(stderr)
                    info["ubsan"] = {
                        "rc": rc,
                        "report": report.to_dict() if report and report.is_valid() else None,
                        "stderr": stderr[:4096],
                    }
                except Exception as e:
                    info["ubsan"] = {"rc": -2, "error": str(e)}

            # Save reports when both are done (or one is done and the other is absent)
            if (
                info["asan"] is not None
                and info["ubsan"] is not None
                or self.asan_target
                and info["asan"] is not None
                and not self.ubsan_target
                or self.ubsan_target
                and info["ubsan"] is not None
                and not self.asan_target
            ):
                self._save_sanitizer_reports(sig, info)

    @staticmethod
    def _setup_asan_env(env: dict) -> None:
        """Set LD_PRELOAD and ASAN_OPTIONS for an ASAN-instrumented target."""
        from fuzzer_tool.cli.ldpreload_wrapper import _resolve_asan

        libasan = _resolve_asan()
        if libasan:
            ld_preload = env.get("LD_PRELOAD", "")
            parts = [p for p in ld_preload.split(":") if p] if ld_preload else []
            parts.insert(0, libasan)
            env["LD_PRELOAD"] = ":".join(parts)
        asan_opts = env.get("ASAN_OPTIONS", "")
        opt_parts = [p for p in asan_opts.split(":") if p] if asan_opts else []
        seen = {p.split("=")[0] for p in opt_parts}
        for opt in ("halt_on_error=0", "abort_on_error=0", "detect_leaks=0"):
            key = opt.split("=")[0]
            if key not in seen:
                opt_parts.append(opt)
                seen.add(key)
        env["ASAN_OPTIONS"] = ":".join(opt_parts)

    @staticmethod
    def _setup_ubsan_env(env: dict) -> None:
        """Set UBSAN_OPTIONS for an UBSAN-instrumented target."""
        ubsan_opts = env.get("UBSAN_OPTIONS", "")
        opt_parts = [p for p in ubsan_opts.split(":") if p] if ubsan_opts else []
        seen = {p.split("=")[0] for p in opt_parts}
        for opt in ("halt_on_error=1", "abort_on_error=1", "print_stacktrace=1"):
            key = opt.split("=")[0]
            if key not in seen:
                opt_parts.append(opt)
                seen.add(key)
        env["UBSAN_OPTIONS"] = ":".join(opt_parts)

    def _save_sanitizer_reports(self, sig: str, info: dict) -> None:
        """Write ASAN/UBSAN reports as JSON alongside the crash file."""
        import json as _json

        report_path = self.crashes_dir / f"{sig[:16]}_sanitizer_report.json"
        try:
            existing = {}
            if report_path.exists():
                existing = _json.loads(report_path.read_text())
            existing.update(
                {
                    "asan_result": info.get("asan"),
                    "ubsan_result": info.get("ubsan"),
                }
            )
            report_path.write_text(_json.dumps(existing, indent=2))
        except Exception:
            log.warning("Failed to save sanitizer report for sig=%s", sig)

    def _print_run_summary(self):
        return self._stats.print_run_summary()

    def _dump_stats(self):
        return self._stats.dump_stats()

    def _dump_coverage_report(self):
        return self._stats.dump_coverage_report()

    def _append_coverage_log(self):
        return self._stats.append_coverage_log()

    def _update_te_causal_map(self):
        return self._stats.update_te_causal_map()

    def _get_te_weighted_position(self, input_length: int):
        return self._stats.get_te_weighted_position(input_length)

    def _get_current_edge_set(self) -> set[int]:
        """Return the set of currently-active edge IDs.

        Works for sparse-entry SHM (edge_id from struct entries) and
        byte-bitmap ptrace coverage (non-zero byte positions).
        """
        return self._stats.get_current_edge_set()

    def _read_runtime_avg_distance(self) -> float | None:
        """Read the per-execution average distance from the SHM tail.

        Returns the unscaled average when the target reported valued
        blocks (dist_count > 0), else None (fall back to Python-side
        distance computation).
        """
        shm = (
            self._target_shm_covs.get(self.target, self.shm_cov)
            if self.multi_targets
            else self.shm_cov
        )
        if shm is None:
            return None
        try:
            dist_sum, dist_count = shm.read_distance_tail()
        except (AttributeError, OSError):
            return None
        if dist_count <= 0:
            return None
        return dist_sum / dist_count / 100.0

    def _format_elapsed(self):
        return self._stats.format_elapsed()

    def print_stats(self):
        return self._stats.print_stats()

    def _last_avg_eps(self) -> float:
        """Mean of the last `_eps_history_max` avg-eps samples.

        Returns 0 until the window is full, so callers fall back to a fixed
        interval instead of chasing the unstable first EPS readings.
        """
        if len(self._eps_history) < self._eps_history_max:
            return 0.0
        return sum(self._eps_history) / len(self._eps_history)

    def _stats_effective_interval(self) -> int:
        """Stats-tick spacing in execs.

        The first tick uses ~1 second of work (1x EPS) so the first [*] execs
        line appears promptly; subsequent ticks space at ~10 seconds of work
        using 10x the mean of the last 10 avg-eps samples rather than a single
        raw reading.  Before the window fills, the fixed stats interval is
        used (the first samples are too unstable to trust).
        """
        if not self._eps_history:
            elapsed = max(time.time() - self.start_time, 1e-9)
            eps_now = (self.exec_count - self._resume_baseline_exec) / elapsed
            if eps_now <= 0:
                return self.stats_interval
            return max(1, int(eps_now))
        last_avg_eps = self._last_avg_eps()
        if last_avg_eps <= 0:
            return self.stats_interval
        return max(1, int(10 * last_avg_eps))

    def _record_entropy_sample(self, sh):
        """Append a Shannon-entropy sample and trim history to a bounded size."""
        self._entropy_execs.append(self.exec_count)
        self._entropy_vals.append(sh)
        if len(self._entropy_execs) > ENTROPY_HISTORY_MAX:
            self._entropy_execs = self._entropy_execs[-ENTROPY_HISTORY_TRIM:]
            self._entropy_vals = self._entropy_vals[-ENTROPY_HISTORY_TRIM:]

    def _compute_entropy_flat(self):
        """Return whether the recent Shannon-entropy rate of change is flat.

        Returns True if entropy has been flat over the last N samples,
        False if it is still changing (redistribution, not stagnation),
        or None if there aren't enough samples yet to measure the rate.
        """
        if len(self._entropy_execs) < ENTROPY_WINDOW:
            return None
        recent_execs = self._entropy_execs[-ENTROPY_WINDOW:]
        recent_vals = self._entropy_vals[-ENTROPY_WINDOW:]
        dt = recent_execs[-1] - recent_execs[0]
        if dt <= 0:
            return None
        dS = abs(recent_vals[-1] - recent_vals[0])
        entropy_rate = dS / dt
        return entropy_rate < ENTROPY_FLAT_THRESHOLD

    def _derive_stall_seed(self) -> int:
        """Return the seed to apply for the current stall reseed.

        When the run was given a seed, the new seed is a pure function of
        ``(self.seed, self._stall_reseed_count)`` — not of the draw history,
        which varies with target execution timing — so a seeded run stays
        reproducible across the reseed. Without a seed there is nothing to
        be reproducible against, so OS entropy is used.

        Returns:
            A seed in ``[0, 2**32)``, suitable for ``np.random.seed``.
        """
        if self.seed is None:
            return int.from_bytes(os.urandom(4), "little")
        x = (self.seed + self._stall_reseed_count * SEED_MIX_GAMMA) & SEED_MASK_64
        x = ((x ^ (x >> 30)) * SEED_MIX_A) & SEED_MASK_64
        x = ((x ^ (x >> 27)) * SEED_MIX_B) & SEED_MASK_64
        x ^= x >> 31
        return x & SEED_MASK_32

    def _reseed_after_stall(self) -> int:
        """Reseed every RNG the mutation hotpath draws from.

        A stall means the current mutation trajectory has stopped producing
        edges. Recovery already widens *what* is mutated; reseeding changes
        *which* stream those choices come from, so a resumed run does not
        replay the same exhausted sequence.

        Both generators are reseeded together, matching how ``__init__``
        seeds them: ``random`` drives the non-hotpath choices and
        ``np.random`` backs ``RandPool``. ``RandPool.reseed`` also drops the
        pre-fetched pool, which would otherwise keep dispensing old-stream
        values for another ``_POOL_ENTRIES`` draws.

        Returns:
            The seed that was applied.
        """
        self._stall_reseed_count += 1
        new_seed = self._derive_stall_seed()
        random.seed(new_seed)
        if _HAS_NUMPY:
            self._rand_pool.reseed(new_seed)
        self._last_stall_seed = new_seed
        print(f"[*] Reseeded RNG → {new_seed} (stall reseed #{self._stall_reseed_count})")
        return new_seed

    def _maybe_trigger_stall_recovery(self, execs_since_edge):
        """Activate stall recovery unless entropy shows active redistribution.

        Entropy rate confirmation: if entropy is still changing, this is
        redistribution among existing edges, not genuine stagnation, so
        recovery is skipped this round. If there aren't enough samples yet
        to measure the rate (`entropy_flat is None`), fall back to the
        no-new-edges signal alone.

        Allan variance confirmation: the noise type of the incremental edge
        discovery rate provides a leading indicator. White noise = normal
        exploration (skip stall). Flicker noise = correlated discoveries
        approaching saturation (reduce threshold). Random walk = integrated
        signal, likely genuine stall (bypass entropy gate).

        Dispersion index override: complements Allan variance by resolving
        a blind spot — a buffer dominated by zeros (genuine stall) and a
        buffer with rare bursts of discoveries (bursty exploration) both
        produce near-zero Allan deviation, but dispersion index D tells
        them apart: D › 1.5 = bursty (override stall), D « 0.3 = stall.
        """
        entropy_flat = self._compute_entropy_flat()
        if entropy_flat is False:
            return False

        # Consult Allan variance detector for noise-type signal
        noise = self._allan.noise_type()
        allan_slope = self._allan.noise_slope()
        dispersion = self._allan.dispersion()

        reason = "no new edges"
        if entropy_flat:
            reason += " + flat entropy"
        if noise != "unknown":
            reason += (
                f" + {noise} noise (slope={allan_slope:+.2f})"
                if allan_slope is not None
                else f" + {noise} noise"
            )
        if dispersion is not None:
            reason += f" + D={dispersion:.2f}"

        # Dispersion index override: a *significantly* overdispersed D
        # (chi-squared dispersion test, not a fixed cutoff — see
        # AllanVarianceDetector.is_overdispersed) means clusters of
        # discoveries with gaps — NOT a stall even if Allan says stalled.
        if self._allan.is_overdispersed():
            return False

        # Noise-type gating and threshold adjustment
        if noise == "active":
            # Random exploration is normal — don't stall even if entropy is flat
            return False

        effective_threshold = self._stall_threshold
        if noise == "fatiguing":
            # Pre-stall: discovery rate is trending down.
            # Halve the threshold to catch it earlier.
            effective_threshold = max(self._stall_threshold // 2, 100)
            if execs_since_edge < effective_threshold:
                return False

        if noise == "stalled":
            # Near-zero variance confirms genuine stall.
            # Bypass entropy gate and use minimal threshold.
            # Significantly underdispersed D (chi-squared test) further
            # confirms stall — use the most aggressive threshold.
            if self._allan.is_underdispersed():
                effective_threshold = max(self._stall_threshold // 8, 25)
            else:
                effective_threshold = max(self._stall_threshold // 4, 50)

        # Without any detector signal (Allan unknown + entropy unknown),
        # fall through to original behavior (trigger on no-new-edges alone).
        if noise not in ("fatiguing", "stalled", "unknown") and entropy_flat is None:
            return False

        # Check coverage growth model for saturation
        growth = self._edge_tracker.coverage_growth_model()
        if growth["confidence"] > 0.3 and growth["current_rate"] < 0.001:
            reason += " + near-saturation"

        self._stall_recovery_count += 1
        print(
            f"\n[*] STALL #{self._stall_recovery_count}: {reason} in "
            f"{execs_since_edge} execs, switching to random mode"
        )
        self._stall_recovery_active = True

        # Optionally reseed the RNGs so recovery explores a different
        # mutation stream rather than continuing the exhausted one.
        if self._reseed_on_stall:
            self._reseed_after_stall()

        # Optionally resize the coverage bitmap to reduce hash collision risk
        if self._resize_map_on_stall and self.shm_cov:
            # Ask the shim how many edges it had to throw away. Without this
            # the load factor is computed only from edges that survived, so
            # a saturated table looks under-full and never triggers a resize
            # — the exact case this recovery path exists for.
            dropped = self.shm_cov.read_dropped_edges()
            new_size = self._edge_tracker.recommended_map_size(dropped_edges=dropped)
            if dropped:
                pinned = " (counter pinned)" if self.shm_cov.drop_counter_saturated() else ""
                print(
                    f"[!] Coverage map saturated: {dropped:,} edges dropped{pinned} — "
                    f"coverage was lost, not merely delayed"
                )
            if new_size > self.shm_cov.size:
                current = self.shm_cov.size
                print(
                    f"[*] Resizing SHM {current:,} → {new_size:,} entries "
                    f"(stall-triggered, n={len(self._edge_tracker._global_edge_hits)}, "
                    f"dropped={dropped:,})"
                )
                self.shm_cov.resize(new_size)
                self.map_size = new_size
                # on_resize() sets map_size and clears only what a resize
                # actually invalidates — nothing, on the SHM path, since
                # edge IDs do not depend on map_size.
                self._edge_tracker.on_resize(new_size)
                # Drops recorded against the old table say nothing about the
                # new one; clearing keeps the next decision on fresh evidence.
                self.shm_cov.reset_diag()
                os.environ["__AFL_SHM_ID"] = self.shm_cov.env_id
                os.environ["AFL_MAP_SIZE"] = str(new_size)
                if self._inprocess_runner:
                    self._inprocess_runner.update_shm_after_resize(
                        self.shm_cov._ptr, new_size, self.shm_cov.env_id
                    )
        return True

    def _run_chi2_operator_test(self) -> None:
        """Chi-squared test: do operators have different success rates?

        Builds a 2×K contingency table (operators × success/failure) and
        tests the null hypothesis that all operators share the same success
        probability.  Results are logged at ``info`` when significant.
        """
        from fuzzer_tool.core.chi_squared import chi_squared_independence, cramers_v

        ops = sorted(set(self.op_counts.keys()) | set(self.op_success.keys()))
        if len(ops) < 2:
            return

        table: list[list[float]] = []
        for op in ops:
            total = self.op_counts.get(op, 0)
            success = self.op_success.get(op, 0)
            if total < 1:
                continue
            table.append([float(success), float(total - success)])

        if len(table) < 2:
            return
        if not any(row[1] > 0 for row in table):
            return

        try:
            chi2, p, dof = chi_squared_independence(table)
            n = sum(sum(r) for r in table)
            v = cramers_v(chi2, n, len(table), 2)

            if p < 0.05:
                log.info(
                    "χ² op heterogeneity: χ²=%.2f, p=%.4f, V=%.3f, "
                    "%d operators — significant (p<0.05)",
                    chi2,
                    p,
                    v,
                    len(table),
                )
            else:
                log.debug(
                    "χ² op heterogeneity: χ²=%.2f, p=%.4f, V=%.3f, %d operators — not significant",
                    chi2,
                    p,
                    v,
                    len(table),
                )
        except Exception as ex:
            log.debug("Chi-squared test failed: %s", ex)

    def _record_seed_strategy_matches(self, score: float) -> None:
        """Record the active seed strategy's Elo match against every OTHER
        eligible strategy in the current pool. Only the strategies that were
        actually selectable at pick time participate, so never-enabled
        strategies do not accrue phantom matches.
        """
        if not (self._use_elo and self._elo and self._seed_strategy):
            return
        seed_strategies = getattr(self, "_seed_strategy_pool", [])
        if self._seed_strategy not in seed_strategies:
            return
        for other in seed_strategies:
            if other != self._seed_strategy:
                self._elo.record_strategy_match(
                    f"seed_{self._seed_strategy}", f"seed_{other}", score
                )

    def _record_operator_strategy_matches(self, score: float) -> None:
        """Record the active operator scheduler's Elo match against every other
        enabled scheduler. Only schedulers actually selected this run
        participate (random_stall is never recorded), so enabled-but-unused
        schedulers do not accrue phantom matches.
        """
        if not (self._use_elo and self._elo and self._meta_strategy):
            return
        if self._meta_strategy not in self._meta_strategy_used:
            return
        all_strategies = []
        if self._use_replicator and self._replicator:
            all_strategies.append("replicator")
        if self.mc and self.mc_bandit:
            all_strategies.append("bandit")
        if self._use_mopt and self._mopt:
            all_strategies.append("mopt")
        if self.mc and self.mc_cem and self.mc.cem_fitted:
            all_strategies.append("cem")
        if self._exp3:
            all_strategies.append("exp3")
        if self._eps_greedy:
            all_strategies.append("eps_greedy")
        if self._hierarchical:
            all_strategies.append("hierarchical")
        if self._gp_ucb:
            all_strategies.append("gp_ucb")
        if self._contextual:
            all_strategies.append("contextual")
        for other in all_strategies:
            if other != self._meta_strategy:
                self._elo.record_strategy_match(self._meta_strategy, other, score)

    def _seed_convergence_rows(self) -> list[tuple[str, float, float, int]]:
        """(name, rating, delta, matches) for every seed strategy actually used
        this run. Strategies that were never selected (only ever recorded as
        phantom opponents) are excluded from the convergence report.
        """
        if not (self._use_elo and self._elo):
            return []
        rows = []
        for s in getattr(self, "_seed_strategies_used", set()):
            key = f"seed_{s}"
            count = self._elo._strategy_match_count.get(key, 0)
            if count > 0:
                rating = self._elo._strategy_mu.get(key, self._elo.initial_mu)
                rows.append((s, rating, rating - self._elo.initial_mu, count))
        return sorted(rows)

    def _operator_convergence_rows(self) -> list[tuple[str, float, float, int]]:
        """(name, rating, delta, matches) for every operator scheduler actually
        selected this run. Schedulers that were enabled but never selected are
        excluded from the convergence report.
        """
        if not (self._use_elo and self._elo):
            return []
        rows = []
        for s in getattr(self, "_meta_strategy_used", set()):
            count = self._elo._strategy_match_count.get(s, 0)
            if count > 0:
                rating = self._elo._strategy_mu.get(s, self._elo.initial_mu)
                rows.append((s, rating, rating - self._elo.initial_mu, count))
        return sorted(rows)

    def _selected_schedulers_str(self) -> str:
        """One-line summary of the active scheduling stack (startup banner)."""
        parts = []
        if getattr(self, "_power_schedule", "base") != "base":
            parts.append(f"power={self._power_schedule}")

        ops = []
        if self.mc_bandit:
            ops.append("bandit")
        if self.mc_cem:
            ops.append("cem")
        if getattr(self, "_use_mopt", False):
            ops.append("mopt")
        if getattr(self, "_use_replicator", False):
            ops.append("replicator")
        if getattr(self, "_use_exp3", False):
            ops.append("exp3")
        if getattr(self, "_eps_greedy", False):
            ops.append("eps_greedy")
        if getattr(self, "_hierarchical_bandit", False):
            ops.append("hierarchical")
        if getattr(self, "_gp_ucb", False):
            ops.append("gp_ucb")
        if getattr(self, "_contextual", False):
            ops.append("contextual")
        if getattr(self, "_use_shapley", False):
            ops.append("shapley")
        if ops:
            parts.append("ops=" + "+".join(ops))

        seeds = []
        if self.ga:
            seeds.append("ga")
        if self.qea:
            seeds.append("qea")
        if getattr(self, "_use_bayesian", False):
            seeds.append("bayesian")
        if getattr(self, "_use_boltzmann", False):
            seeds.append("boltzmann")
        if getattr(self, "_distance", None) is not None:
            seeds.append("aflgo")
        if seeds:
            parts.append("seeds=" + "+".join(seeds))

        if getattr(self, "_use_elo", False):
            parts.append("elo")
        if self.markov_generate:
            parts.append("markov-gen")

        return " | ".join(parts) if parts else "base"

    def _start_stack_heartbeat(self, interval: float = 3.0) -> None:
        """Daemon thread: periodically write the main-thread Python stack.

        SIGKILL cannot be handled in-process, so `kill -9` leaves nothing.
        This thread writes where the main thread is executing (only when the
        top frame moves) to a small file, giving the last known location
        after a hard kill.
        """
        if self._stack_heartbeat_path is None:
            return
        out = self._stack_heartbeat_path
        out.parent.mkdir(parents=True, exist_ok=True)

        def _beat() -> None:
            import traceback

            ident = threading.main_thread().ident
            last_key: tuple | None = None
            while True:
                time.sleep(interval)
                try:
                    frame = sys._current_frames().get(ident)
                    if frame is None:
                        continue
                    key = (frame.f_code.co_filename, frame.f_lineno)
                    if key == last_key:
                        continue
                    last_key = key
                    out.write_text(
                        "".join(traceback.format_stack(frame)[-8:])
                        + f"\n# heartbeat ts={time.time():.0f} execs={self.exec_count}\n"
                    )
                except Exception:
                    pass

        threading.Thread(target=_beat, daemon=True, name="stack-heartbeat").start()

    def run(self, iterations=0):
        self._start_stack_heartbeat()
        if self.multi_targets:
            print(f"[*] Multi-target: {len(self.multi_targets)} targets, shared corpus")
            for i, t in enumerate(self.multi_targets):
                afl = _detect_afl(t)
                tag = " [AFL]" if afl else " [no-AFL]"
                dist = _detect_distance(t)
                if dist:
                    tag += " [DIST]"
                print(f"  [{i}] {t}{tag}")
        else:
            print(f"[*] Target: {self.target}")
            if _detect_afl(self.target):
                print("[*] AFL instrumentation: detected")
            if _detect_distance(self.target):
                print("[*] Distance instrumentation: detected")
                if self._distance is None:
                    print(
                        "[*]   Directed mode idle: pass --target-functions "
                        "(function, address, or file.c:line) to engage the "
                        "distance channel (dist: stats + aflgo schedule/elo arm)"
                    )
        print(f"[*] Selected schedulers: {self._selected_schedulers_str()}")
        # Static branch density: conditional branches per KB of .text
        from fuzzer_tool.core.elf import branch_density

        if self.multi_targets:
            bd_total = 0
            bd_count = 0
            for t in self.multi_targets:
                bd = branch_density(t)
                if bd is not None:
                    name = os.path.basename(t)
                    print(f"[*] Branch density: {name} {bd:.1f} cond branches/KB")
                    bd_total += bd
                    bd_count += 1
            if bd_count > 1:
                print(f"[*] Branch density: avg {bd_total / bd_count:.1f} cond branches/KB")
        else:
            bd = branch_density(self.target)
            if bd is not None:
                print(f"[*] Branch density: {bd:.1f} cond branches/KB")
        print(f"[*] Edge bitmap: {self.map_size:,} bytes (auto-sized)")
        print(f"[*] Corpus: {self.corpus_dir} ({len(self.corpus)} seeds)")
        print(f"[*] Crashes: {self.crashes_dir}")
        print(f"[*] Max input length: {self.max_len}")
        print(f"[*] Timeout: {self.timeout}s")
        if self.honggfuzz:
            print(
                "[*] Honggfuzz power factors: enabled (novelty, freshness, fertility, density, entropy, timeout)"
            )
        if self.hw_perf:
            if self._perf_counters:
                print(f"[*] HW perf counters: {', '.join(self._perf_counters.counter_names)}")
            else:
                print(
                    "[*] HW perf counters: requested but not available (needs CAP_PERFMON or root)"
                )
        print(f"[*] Seed: {self.seed}")
        # Target profile summary
        if self._profile.functions:
            profile_cache = Path(self.target).with_suffix(
                Path(self.target).suffix + ".profile_cache"
            )
            tag = " [cached]" if profile_cache.exists() and not self.refresh_profile else ""
            print(
                f"[*] Profile: {len(self._profile.functions)} functions, "
                f"{len(self._profile.hot_functions)} hot, "
                f"format={self._profile.format_signature or 'unknown'}{tag}"
            )
        if self.grammar:
            print(f"[*] Grammar: {len(self.grammar.rules)} rules")
        if self._calibrate > 0:
            print(f"[*] Calibration: {self._calibrate} execs before main loop")
        if self.persistent:
            print("[*] Persistent mode: enabled")
        if self._inprocess_runner:
            print("[*] In-process mode: enabled")
        if self.dictionary:
            print(f"[*] Dictionary: {len(self.dictionary)} tokens")
        if self.markov_trained:
            if hasattr(self.markov, "chains"):
                orders_str = ",".join(str(o) for o in self.markov.orders)
                total_ctx = sum(c._contexts_seen for c in self.markov.chains.values())
                print(f"[*] Markov ensemble: orders=[{orders_str}], total_contexts={total_ctx}")
            else:
                print(
                    f"[*] Markov chain: order={self.markov.order}, "
                    f"transitions={len(self.markov.transitions)}"
                )
        if self.markov_generate:
            print("[*] Markov generation: enabled (15% of seeds)")
        if self.mc:
            if self.mc_bandit:
                print(f"[*] MC bandit: Thompson sampling over {len(self.mc.arm_alpha)} arms")
            if self.mc_cem:
                print(
                    f"[*] MC CEM: elite_frac={self.mc.elite_frac}, "
                    f"refit_interval={self.mc.refit_interval}"
                )
        if self.stats_file:
            print(f"[*] Stats: {self.stats_file} every {self.stats_interval} iterations")
        if self.minimize_every_execs > 0:
            print(f"[*] Minimize: every {self.minimize_every_execs} execs")
        import datetime

        epoch_start = time.time()
        boot_start = time.monotonic()
        try:
            with open("/proc/uptime") as f:
                boot_start = float(f.read().split()[0])
        except OSError:
            pass
        print(
            f"[*] Epoch start: {epoch_start:.3f} ({datetime.datetime.fromtimestamp(epoch_start).isoformat()})"
        )
        print(f"[*] Boot ticks start: {boot_start:.3f}")
        print("[*] Starting fuzzing...\n")

        # Quick raw-target-speed measurement before the main loop
        try:
            _probe = b"\x00" * 64
            _n = min(100, max(10, int(len(self.corpus) * 0.1 + 1)))
            _t0 = time.perf_counter()
            for _ in range(_n):
                self._run_target(_probe)
            _t1 = time.perf_counter()
            _raw_eps = _n / (_t1 - _t0) if _t1 > _t0 else 0
            print(f"[*] Raw target speed: {_raw_eps:,.0f} eps ({_n} probes)")
        except Exception:
            pass

        i = 0
        try:
            # Run each seed as-is before mutating — catches crashes in the
            # initial corpus and gathers baseline coverage.
            for seed in list(self.corpus):
                returncode, stderr = self._run_target(seed)
                if self._diff_tracker:
                    self._check_differential(seed)
                # Validate AFL shim on first execution
                if not getattr(self, "_shim_checked", False):
                    self._shim_checked = True
                    if "[shim]" in stderr:
                        log.info("AFL shim: %s", stderr.strip())
                        if "area=(nil)" in stderr and self.shm_cov:
                            log.warning(
                                "AFL shim area is NULL — SHM not attached. "
                                "Coverage data will be empty."
                            )
                self.exec_count += 1
                # Mark seed as having been executed (even though not via
                # fuzz_one's mutate path).  This ensures loaded seeds don't
                # all show fuzz_count=0 to auto_minimize_corpus.
                meta = self.seed_meta.get(seed)
                if meta is not None:
                    meta["fuzz_count"] += 1
                    self._cached_total_fuzz += 1
                if self._is_crash(returncode, stderr):
                    self.crash_count += 1
                    self.save_crash(seed, returncode, stderr)
                    self._prune_crash_data()
            # Baseline exec_count after initial seed replay — used for
            # periodic minimization modulus so it fires at clean intervals
            # regardless of initial corpus size.
            _exec_baseline = self.exec_count
            self._exec_baseline = _exec_baseline

            # Calibration pass: bootstrap coverage stats before main loop
            if self._calibrate > 0:
                self._run_calibration(self._calibrate)

            # Initialize GA lifecycle if enabled
            if self._ga_enabled:
                from fuzzer_tool.core.ga import GALifecycle

                self.ga = GALifecycle(
                    pop_size=self._ga_pop_size,
                    elite_fraction=self._ga_elite_frac,
                    crossover_rate=self._ga_crossover_rate,
                    mutation_rate=self._ga_mutation_rate,
                    tournament_size=self._ga_tournament_size,
                    generation_size=self._ga_gen_size,
                    speciation_threshold=self._ga_speciation_threshold,
                )
                self.ga.initialize(self.corpus, self._edge_tracker)

            # Initialize differential fuzzing if enabled
            if self._diff_target:
                from fuzzer_tool.services.differential import DifferentialTracker

                self._diff_tracker = DifferentialTracker()
                print(f"[*] Differential: comparing against {self._diff_target}")
                ga_data = self._state_store.get("ga")
                if self.resume and ga_data is not None:
                    self.ga.from_dict(ga_data)
                    print(f"[*] GA: loaded state from state store (gen={self.ga.generation})")
                print(
                    f"[*] GA: pop_size={self.ga.pop_size}, "
                    f"gen_size={self.ga.generation_size}, "
                    f"elite={self.ga.elite_fraction:.0%}, "
                    f"crossover={self.ga.crossover_rate:.0%}, "
                    f"mutation={self.ga.mutation_rate:.0%}"
                )

            # Initialize QEA lifecycle if enabled
            if self._qea_enabled:
                from fuzzer_tool.core.qea import ALPHA_STRONG, QEALifecycle

                self.qea = QEALifecycle(
                    pop_size=self._ga_pop_size,
                    elite_fraction=self._ga_elite_frac,
                    generation_size=self._ga_gen_size,
                    tournament_size=self._ga_tournament_size,
                    speciation_threshold=self._ga_speciation_threshold,
                    rotation_angle=self._qea_rotation_angle,
                    strong_bias=(
                        ALPHA_STRONG if self._qea_strong_bias is None else self._qea_strong_bias
                    ),
                    elite_reset_every=self._qea_elite_reset,
                )
                self.qea.initialize(self.corpus, self._edge_tracker)
                qea_data = self._state_store.get("qea")
                if self.resume and qea_data is not None:
                    self.qea.from_dict(qea_data)
                    print(f"[*] QEA: loaded state from state store (gen={self.qea.generation})")
                print(
                    f"[*] QEA: pop_size={self.qea.pop_size}, "
                    f"gen_size={self.qea.generation_size}, "
                    f"rotation_angle={self.qea.rotation_angle}, "
                    f"mutation_prob={self.qea.mutation_prob}"
                )

            if self._mcts is not None:
                mcts_data = self._state_store.get("mcts")
                if self.resume and mcts_data is not None:
                    self._mcts.from_dict(mcts_data)
                    print(
                        "[*] MCTS: loaded state from state store "
                        f"(nodes={self._mcts.stats()['tracked_nodes']})"
                    )
                print(f"[*] MCTS seed scheduling: exploration={self._mcts.exploration:.3f}")

            # Print WFC mode status
            if self._wfc_enabled:
                print("[*] WFC: enabled — structural chunk reordering and pixel generation active")

            while not _shutdown:
                if iterations and i >= iterations:
                    break
                if self.continue_until_crash and self.crash_count > 0:
                    break
                # Cycle through targets in multi-target mode
                if self.multi_targets:
                    self._select_next_target()
                # Run any deferred minimization before picking the next seed.
                # This gives freshly-added seeds one full iteration to be selected.
                if self._minimize_pending:
                    self._flush_pending_minimize()
                seed = self._pick_seed()
                # Compute seed-level energy multiplier for mutation budget
                meta = self.seed_meta.get(seed)
                if meta is not None and self._seed_scorer:
                    # Lazy recompute aggregate cache when corpus changed
                    if not self._agg_cache_valid:
                        self._refresh_agg_cache()
                    avg_exec_us = max(
                        1,
                        int(self._cached_total_time / max(1, self._cached_total_fuzz) * 1_000_000),
                    )
                    exec_us = max(
                        1,
                        int(
                            meta.get("total_time", 0)
                            / max(1, meta.get("fuzz_count", 1))
                            * 1_000_000
                        ),
                    )
                    bitmap_size = meta.get("coverage_edges", 0)
                    avg_bitmap_size = max(
                        1,
                        int(self._cached_total_edges / max(1, len(self.seed_meta))),
                    )
                    depth = meta.get("lineage_depth", 0)
                    fuzz_level = meta.get("fuzz_count", 0)
                    n_fuzz = fuzz_level

                    # Honggfuzz power factors (only when --honggfuzz enabled)
                    hf_kwargs: dict = {}
                    if self.honggfuzz:
                        seed_key = self._seed_key(seed)
                        new_edges = 0
                        if seed_key in self._edge_tracker.seed_edges:
                            seed_e = self._edge_tracker.seed_edges[seed_key]
                            others = set()
                            for sk, se in self._edge_tracker.seed_edges.items():
                                if sk != seed_key:
                                    others.update(se)
                            new_edges = len(seed_e - others)
                        time_added = meta.get("added_at", 0.0)
                        now = time.time()
                        child_count = meta.get("child_count", 0)
                        select_count = fuzz_level
                        timed_out = meta.get("timed_out", False)
                        rare_edges = self._edge_tracker.rare_edge_count(seed_key)

                        # Track honggfuzz factor stats
                        if new_edges > 0 and now - time_added < 600:
                            self._hf_novelty_boosts += 1
                        if now - time_added < 60:
                            self._hf_freshness_boosts += 1
                        if child_count > 0:
                            self._hf_fertility_boosts += 1
                        if bitmap_size > 0 and len(seed) > 0:
                            density = (bitmap_size * 100) / len(seed)
                            if density > 50:
                                self._hf_density_boosts += 1
                        if timed_out:
                            self._hf_timeout_penalties += 1

                        hf_kwargs = dict(
                            new_edges=new_edges,
                            time_added=time_added,
                            now=now,
                            input_size=len(seed),
                            select_count=select_count,
                            child_count=child_count,
                            rare_edge_count=rare_edges,
                            timed_out=timed_out,
                            max_cov=max(1, self._edge_tracker.get_cumulative_edge_count()),
                            hw_instructions=self._last_perf_deltas.get("instructions", 0),
                            hw_branches=self._last_perf_deltas.get("branches", 0),
                        )

                    self._last_perf_score = self._seed_scorer.score(
                        exec_us=exec_us,
                        avg_exec_us=avg_exec_us,
                        bitmap_size=bitmap_size,
                        avg_bitmap_size=avg_bitmap_size,
                        handicap=0,
                        depth=depth,
                        fuzz_level=fuzz_level,
                        n_fuzz=n_fuzz,
                        total_execs=max(1, self.exec_count),
                        mean_log_n_fuzz=self._cached_mean_log_n_fuzz,
                        avg_distance=meta.get("avg_distance", -1.0) if self._distance else -1.0,
                        max_distance=(
                            self._dist_max_observed
                            if self._distance and self._dist_max_observed is not None
                            else (self._distance.max_distance if self._distance else 0.0)
                        ),
                        anneal_progress=self._anneal_progress,
                        min_distance=self._dist_min_observed or 0.0,
                        elapsed_sec=time.time() - self.start_time,
                        t_x_minutes=self._seed_scorer.t_x_minutes,
                        **hf_kwargs,
                    )
                else:
                    # Markov-generated or synthetic seed: reset to neutral multiplier
                    self._last_perf_score = 100.0
                if self._diff_tracker:
                    self._check_differential(seed)
                self.fuzz_one(seed)
                # Backpropagate this iteration's discovery up the MCTS path.
                # A no-op unless the mcts arm actually selected this seed —
                # update() ignores an empty path — so it stays correct when
                # Elo hands the pick to another strategy.
                if self._mcts is not None:
                    self._mcts.update(self._last_new_edge_count)
                i += 1
                effective_interval = self._stats_effective_interval()
                if self.exec_count - self._last_stats_exec >= effective_interval:
                    # Sample Shannon entropy for rate-of-change tracking
                    if self._edge_tracker._global_edge_hits:
                        sh = self._edge_tracker.shannon_entropy_global()
                        self._record_entropy_sample(sh)
                    # Feed incremental edge count to Allan variance detector
                    current_edges = self._edge_tracker.get_cumulative_edge_count()
                    delta = current_edges - self._last_allan_edge_count
                    self._allan.update(delta)
                    self._last_allan_edge_count = current_edges
                    # Record coverage snapshot for temporal analysis
                    self._edge_tracker.record_coverage_snapshot(self.exec_count)
                    if not self.quiet_stats:
                        self.print_stats()
                    self._append_coverage_log()
                    self._record_discovery_snapshot()
                    # Stall detection: no new edges in threshold execs
                    execs_since_edge = self.exec_count - self._last_new_edge_exec
                    if (
                        not self._stall_recovery_active
                        and execs_since_edge >= self._stall_threshold
                        and self.exec_count > 0
                    ):
                        self._maybe_trigger_stall_recovery(execs_since_edge)
                    # Memory-based corpus pruning
                    self._check_memory_and_prune()
                    # Periodic GC to return freed memory to OS
                    if i % 500 == 0:
                        import gc

                        gc.collect()
                    self._last_stats_exec = self.exec_count
                    if self.stats_file:
                        self._dump_stats()
                        self._save_state()
                if i % 500 == 0 and self.replay_n > 0:
                    self._run_crash_replays()
                if i % 500 == 0 and (self.asan_target or self.ubsan_target):
                    self._run_sanitizer_replays()
        except (KeyboardInterrupt, SystemExit):
            pass
        except OSError as e:
            log.warning("Fuzzing interrupted by OS error: %s", e)

        # Final coverage snapshot for temporal analysis
        self._edge_tracker.record_coverage_snapshot(self.exec_count)
        self._dump_stats()
        self._dump_coverage_report()
        if self.markov.is_trained():
            self._state_store.set("markov", self.markov.to_dict())
        if self._use_mi and self._mi:
            self._state_store.set("mi", self._mi.to_dict())
        self._state_store.set("crash_mi", self._crash_mi.save())
        self._state_store.set("length_tracker", self._length_tracker.save())
        self._flush_pending_minimize()
        if self.ga:
            self._state_store.set("ga", self.ga.to_dict())
            print(f"[*] GA: saved state (gen={self.ga.generation})")
        if self.qea:
            self._state_store.set("qea", self.qea.to_dict())
            print(f"[*] QEA: saved state (gen={self.qea.generation})")
        if self._mcts is not None:
            # Drop stats for seeds minimization removed, so the persisted
            # state does not grow without bound across resumes.
            if self._lineage is not None:
                self._mcts.prune(set(self._lineage.nodes))
            self._state_store.set("mcts", self._mcts.to_dict())
            print(f"[*] MCTS: saved state ({self._mcts.stats()['tracked_nodes']} nodes)")
        self._save_state()
        if self._ablation_file:
            self._ablation_file.flush()
            self._ablation_file.close()
            self._ablation_file = None
            print(f"[*] Schedule ablation log: {self._ablation_path}")
        if not self.quiet_stats:
            self.print_stats()
        print(
            f"\n\n[*] Fuzzing stopped. {self.crash_count} crashes found "
            f"({len(self.crash_sigs)} unique signatures)."
        )
        if self.crash_sigs:
            print("[*] Crash signatures:")
            for sig, count in sorted(self.crash_sigs.items(), key=lambda x: -x[1]):
                print(f"    {sig} ({count}x)")
            print(f"\n[*] Crash files in: {self.crashes_dir}")
        # Show convergence stats for every active scheduler
        if self.mc and self.mc_bandit:
            print("\n[*] Bandit convergence (Thompson sampling):")
            for name, (a, b) in sorted(
                # bandit_stats() subtracts priors, so never-selected arms show
                # (0, 0) and are omitted below
                self.mc.bandit_stats().items(),
                key=lambda x: -(x[1][0] / max(x[1][0] + x[1][1], 1)),
            ):
                if a + b <= 0:
                    continue
                total = a + b
                pct = a / total * 100 if total else 0
                print(f"    {name:20s}: {a:.1f}/{b:.1f} ({pct:.0f}% success)")
        if self._mopt:
            print("\n[*] MOpt convergence (PSO):")
            for p in self._mopt.particle_stats()[:5]:
                print(
                    f"    {p['name']:<20s}: fitness={p['fitness']:.4f} "
                    f"top={p['top_op']}({p['top_prob']:.1%})"
                )
        if self._replicator:
            print("\n[*] Replicator convergence:")
            for s in self._replicator.operator_stats():
                if s["window_execs"] > 0:
                    rate = s["window_successes"] / s["window_execs"] * 100
                    print(
                        f"    {s['name']:<20s}: pop={s['population']:.4f} "
                        f"({s['window_successes']}/{s['window_execs']} = {rate:.0f}%)"
                    )
        # Seed strategy convergence (only strategies actually used this run)
        seed_rows = self._seed_convergence_rows()
        if seed_rows:
            print("\n[*] Seed strategy convergence:")
            for s, rating, delta, count in seed_rows:
                sign = "+" if delta >= 0 else ""
                print(f"    {s:<20s}: {rating:>7.0f} ({sign}{delta:.0f}, {count} matches)")
        # Operator strategy convergence (only schedulers actually selected)
        op_rows = self._operator_convergence_rows()
        if op_rows:
            print("\n[*] Operator strategy convergence:")
            for s, rating, delta, count in op_rows:
                sign = "+" if delta >= 0 else ""
                print(f"    {s:<20s}: {rating:>7.0f} ({sign}{delta:.0f}, {count} matches)")
        self._print_run_summary()
        epoch_end = time.time()
        boot_end = time.monotonic()
        try:
            with open("/proc/uptime") as f:
                boot_end = float(f.read().split()[0])
        except OSError:
            pass
        print(
            f"\n[*] Epoch end: {epoch_end:.3f} ({datetime.datetime.fromtimestamp(epoch_end).isoformat()})"
        )
        print(f"[*] Boot ticks end: {boot_end:.3f}")
        print()  # blank line before next epoch or shell prompt
