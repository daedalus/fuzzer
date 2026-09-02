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
    import numpy as np  # noqa: F401

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

from fuzzer_tool.adapters.process import (
    _child_pids,
    disable_aslr,
)
from fuzzer_tool.adapters.shm import MAX_COUNT_GROWTH_FACTOR, ShmCoverage
from fuzzer_tool.core.bloom import BloomFilter
from fuzzer_tool.core.byte_entropy import byte_entropy_pct
from fuzzer_tool.core.cost_ledger import cost_samples, seed_exec_us
from fuzzer_tool.core.markov import MarkovChain, MarkovEnsemble
from fuzzer_tool.core.mi import MI_MAX_POSITIONS, MutualInformationTracker
from fuzzer_tool.core.operator_registry import REGISTRY
from fuzzer_tool.core.percolation import CoverageRegime
from fuzzer_tool.core.running_stats import RunningMoments
from fuzzer_tool.core.sanitizer import SanitizerReport
from fuzzer_tool.core.schedulers import (
    CMAESScheduler,
    ContextualLinUCBScheduler,
    CUCBScheduler,
    DUCBScheduler,
    EpsilonGreedyScheduler,
    Exp3Scheduler,
    GPUCBScheduler,
    HierarchicalBanditScheduler,
    MonteCarloScheduler,
    MOptScheduler,
    ReplicatorScheduler,
    SWUCBScheduler,
)
from fuzzer_tool.core.schedules import (
    ENTROPY_RANDOM_PCT,
    ENTROPY_SPARSE_PCT,
    SeedScorer,
    compute_mean_log_n_fuzz,
)
from fuzzer_tool.core.secretary import DEFAULT_EXPLORATION_FRAC, SecretaryStopping
from fuzzer_tool.core.seed_quality import BayesianSeedQuality
from fuzzer_tool.core.shapley import ShapleyAttribution
from fuzzer_tool.core.skipdet import SkipDetector
from fuzzer_tool.core.validity import Validity, ValidityChannel
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
    "cmaes",
    "ducb",
    "swucb",
    "cucb",
    "invasion",
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
    "ecofuzz",
    "katz",
)


_kill_children_enabled = os.environ.get("FUZZER_DISABLE_KILL_CHILDREN", "") not in (
    "1",
    "true",
    "yes",
)

_environ_snapshot: dict[str, str] | None = None
"""os.environ as it stood before the first Fuzzer() in this process touched it.

Fuzzer.__init__ / run() write __AFL_DIST_SHM_ID, __AFL_SHM_ID, AFL_MAP_SIZE,
LD_PRELOAD (ASAN) and UBSAN_OPTIONS directly into the process environment,
because those keys have to be visible to subprocess.Popen()/os.exec* calls
made throughout the run. Only the cmplog shim's LD_PRELOAD edit was ever
restored (see the end of run()) -- the rest leaked into whatever ran next in
the same process: a second target in a multi-target session, a caller
embedding Fuzzer as a library, or the next test in a pytest run. Captured
once per process (not per Fuzzer instance) so a second Fuzzer() built while
the first is still mutating the environment doesn't re-baseline over those
mutations and adopt them as "original".
"""


def _snapshot_environ_once() -> None:
    global _environ_snapshot
    if _environ_snapshot is None:
        _environ_snapshot = dict(os.environ)


def _restore_environ() -> None:
    """Put os.environ back the way ``_snapshot_environ_once`` found it.

    Removes keys this process added and restores keys it changed. Safe to
    call more than once (idempotent) and safe to call from atexit (no
    exceptions escape).
    """
    global _environ_snapshot
    if _environ_snapshot is None:
        return
    with contextlib.suppress(Exception):
        for key in list(os.environ.keys()):
            if key not in _environ_snapshot:
                os.environ.pop(key, None)
        for key, value in _environ_snapshot.items():
            if os.environ.get(key) != value:
                os.environ[key] = value


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
    atexit.register(_restore_environ)
    try:
        signal.signal(signal.SIGTERM, _kill_children)
        signal.signal(signal.SIGINT, _kill_children)
    except (ValueError, OSError):
        return False
    return True


install_cleanup_handlers()

# Native-fault diagnostics. faulthandler installs a *C-level* SIGSEGV/SIGBUS/
# SIGFPE/SIGABRT handler that dumps every thread's Python stack from signal
# context and then re-raises with the default action, so the process still
# dies with the right exit status and a core is still produced.
#
# This replaces a Python-level `signal.signal(SIGSEGV, _handle_sigsegv)` that
# printed a traceback and called sys.exit(1). That could not work and made
# real crashes harder to diagnose, in three ways:
#
#  * Python signal handlers do not run from signal context. The C shim sets a
#    flag and returns, and the flag is only checked between bytecodes -- but a
#    segfault is synchronous, so the faulting instruction re-executes
#    immediately and faults again before any Python runs.
#  * A fault raised inside a native extension (z3, numpy) or on a non-main
#    thread never reaches the main interpreter loop at all, so the handler
#    produced no output whatsoever.
#  * Owning the signal *suppressed* faulthandler, which does work. An
#    intermittent suite crash was silent for exactly this reason until
#    SIGSEGV was handed back -- see
#    docs/handover/suite_segfault_z3_finalization_2026-08-16.md.
#
# Also registers SIGUSR1 for on-demand live traces: `kill -USR1 <fuzzer-pid>`
# dumps every thread's Python stack to stderr without killing the process.
# SIGKILL (kill -9) is uncatchable in-process -- for that, run with
# --stack-heartbeat, whose periodic main-thread stack file survives the kill.
#
# adapters/inprocess.py installs its own SIGSEGV handler around in-process
# target execution and restores it afterwards; that one is scoped to a call
# where a fault is an expected result rather than a bug, and is unaffected.
try:
    import faulthandler as _faulthandler

    if not _faulthandler.is_enabled():
        _faulthandler.enable()
    _faulthandler.register(signal.SIGUSR1)
except (AttributeError, OSError, ValueError):  # pragma: no cover - env-dependent
    pass


def _in_taint(taints, offset: int, length: int) -> bool:
    """True when ``[offset, offset+length)`` lies wholly inside one taint.

    A taint region is a byte range colorization proved the target does not
    read: every byte in it was replaced without moving the execution path.
    An operand occurring entirely inside one is therefore a coincidence of
    the byte values, not a value the comparison consumed -- which is the
    false-positive class colorization exists to remove.

    Partial overlap counts as *not* tainted: if any byte of the occurrence
    is path-relevant, the match is worth keeping.
    """
    if not taints:
        return False
    end = offset + length - 1
    return any(region.start <= offset and end <= region.end for region in taints)


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

# ── Adaptive timeout (--adaptive-timeout) ────────────────────────────────
# suggested_timeout() reads the empirical CDF, so it needs a settled one:
# below this many observations it is tracking the warm-up, not the target.
ADAPTIVE_TIMEOUT_MIN_SAMPLES = 200
# Relative change required before a retune is applied. Without a dead band
# the value chases its own tail -- every retune changes which inputs time
# out, which changes the distribution the next suggestion is drawn from.
ADAPTIVE_TIMEOUT_HYSTERESIS = 0.25
# Minimum execs between retunes, so a drifting target cannot turn this into
# a per-100-exec handshake with the loader.
ADAPTIVE_TIMEOUT_COOLDOWN_EXECS = 1_000
# Absolute floor. Below ~5ms the deadline is measuring scheduler noise; the
# loader clamps at 1ms independently (an all-zero itimer disarms).
ADAPTIVE_TIMEOUT_FLOOR = 0.005
# Ceiling, as a multiple of the timeout the caller asked for. Retuning is
# allowed to loosen -- a target slower than the default produces false
# timeouts, which is a correctness problem, not just a throughput one --
# but not without bound, or one pathological input drags the deadline up
# and every later hang costs that much wall clock.
ADAPTIVE_TIMEOUT_MAX_GROWTH = 10.0
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


def _current_rss_kb() -> int | None:
    """Return the process's *current* resident set size in KiB.

    ``/proc/self/statm`` field 2 is resident pages right now. Deliberately not
    ``getrusage().ru_maxrss``, which is the monotonic high-water mark: a
    threshold check against a peak can only ever latch on.

    Returns:
        Current RSS in KiB, or None if /proc is unavailable or unparseable.
    """
    try:
        with open("/proc/self/statm") as fh:
            resident_pages = int(fh.read().split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return resident_pages * (os.sysconf("SC_PAGE_SIZE") // 1024)


# Symbols that mean the target can populate an edge bitmap: the shim's own
# (linked or preloaded) and clang's sancov callbacks.
_AFL_SYMS = ("__afl_area", "__afl_map_shm", "__sanitizer_cov")


def afl_instrumentation_status(target_path: str) -> str:
    """Classify *target_path*'s edge instrumentation as present/absent/unknown.

    The third state is the point.  ``nm`` reports nothing at all for a
    stripped binary, so a plain boolean cannot tell "this target has no
    instrumentation" from "this target's symbol table was removed" — and a
    stripped, fully-instrumented target is a normal thing to be handed.
    Treating that as absent would fire the no-instrumentation warning on a
    run that is working perfectly, which is the fastest way to teach someone
    to ignore the warning.

    The rule, checked against every target shape in ``targets/``:

    - the static symbol table has entries and one of :data:`_AFL_SYMS` is
      among them (or among the dynamic ones)     -> ``"present"``
    - the static symbol table has entries and none match  -> ``"absent"``
    - the static symbol table is empty, or ``nm`` is unusable -> ``"unknown"``

    The dynamic table alone is not enough to rule instrumentation *out*:
    ``__afl_area`` lives in the static symtab, so a stripped target keeps
    thousands of dynamic symbols and none of the ones looked for here.

    Returns:
        One of ``"present"``, ``"absent"``, ``"unknown"``.
    """
    import subprocess

    def _nm(*flags: str) -> str | None:
        try:
            r = subprocess.run(
                ["nm", *flags, target_path],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return r.stdout

    static = _nm()
    if static is None or not static.strip():
        # Stripped, or no nm on this box. Cannot distinguish; say so.
        return "unknown"
    dynamic = _nm("-D") or ""
    haystack = static + dynamic
    return "present" if any(s in haystack for s in _AFL_SYMS) else "absent"


def _detect_afl(target_path: str) -> bool:
    """True when *target_path* has AFL edge coverage instrumentation.

    Kept as the boolean face of :func:`afl_instrumentation_status` for the
    call sites that only need "should I print [AFL]".  Anything that decides
    whether to *warn* must use the tri-state instead — see the docstring
    there for why ``unknown`` must not collapse into ``False``.
    """
    return afl_instrumentation_status(target_path) == "present"


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

    def _report_instrumentation(self) -> None:
        """Print the target's instrumentation state, and warn if it has none.

        Coverage is on by default, so the common failure is no longer "the
        user forgot -c" but "the target was never built with instrumentation".
        Both produce the same symptom — healthy throughput, an empty bitmap,
        a corpus that never grows — and until this warning existed only the
        first one was ever reported.
        """
        status = afl_instrumentation_status(self.target)
        if status == "present":
            print("[*] AFL instrumentation: detected")
        elif status == "absent":
            self._warn_uninstrumented([self.target])
        # "unknown" (stripped binary, or no nm): say nothing rather than
        # guess. A false alarm here trains people to ignore the real one.

    def _warn_uninstrumented(self, targets: list[str]) -> None:
        """Warn that coverage is on but the target(s) cannot report edges.

        Not fatal: crash and timeout detection still work, so a blind run
        against an uninstrumented binary is a legitimate thing to want. It
        just must not look like a coverage-guided one.
        """
        if not self.use_coverage or getattr(self, "_uninstrumented_warned", False):
            return
        if getattr(self, "ptrace_cov", None) is not None:
            # ptrace derives edges from breakpoints on the binary itself and
            # needs no build-time instrumentation, so the premise of this
            # warning does not hold. Caught by running --no-shm against an
            # uninstrumented target: coverage was working (5 breakpoints,
            # edges accumulating) while this told the user the bitmap would
            # stay empty and offered --no-coverage as the fix.
            return
        if getattr(self, "shm_cov", None) is None:
            # Not on the SHM path at all (in-process modes); those have their
            # own warning in _warn_no_coverage.
            return
        self._uninstrumented_warned = True
        which = targets[0] if len(targets) == 1 else f"{len(targets)} targets"
        msg = (
            f"No edge instrumentation found in {which}: the coverage bitmap "
            "will stay empty, so edge discovery, coverage-guided scheduling "
            "and corpus growth are all inactive. Rebuild with "
            "tools/build_targets.sh, or pass --no-coverage to run blind on "
            "purpose (crash detection is unaffected)."
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
        cmaes=False,
        cmaes_pop_size=8,
        cmaes_generation_size=200,
        cmaes_step_size=0.3,
        cmaes_elite_frac=0.5,
        targets=None,
        anneal_budget=0,
        boltzmann=False,
        ecofuzz=False,
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
        calibrate_stability=0,
        cmplog=None,  # None = auto-detect, True = force on, False = force off
        cmplog_max_tokens=0,
        cmplog_max_pairs=0,
        cmplog_workdir=None,
        cmplog_fifo_sink=False,
        cmplog_fifo_sink_size=None,
        asan_target=None,
        ubsan_target=None,
        max_corpus=0,
        max_corpus_bytes=0,
        minimize_every_execs=0,
        prune_corpus_max_memory=80,
        no_shm=False,
        use_ptrace=False,
        adaptive_havoc=True,
        use_cfg_cache=True,
        adaptive_timeout=False,
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
        invasion=False,
        exp3=False,
        exp3_gamma=0.1,
        eps_greedy=False,
        eps_greedy_epsilon0=1.0,
        eps_greedy_decay=0.9995,
        hierarchical_bandit=False,
        gp_ucb=False,
        gp_length_scale=1.0,
        gp_beta=2.0,
        ducb=False,
        ducb_gamma=0.9999,
        swucb=False,
        swucb_window=4000,
        cucb=False,
        cucb_gamma=0.9995,
        contextual=False,
        contextual_alpha=1.0,
        contextual_lambda=1.0,
        overlap_density=False,
        overlap_density_mode="modifier",
        overlap_min_jaccard=0.25,
        overlap_density_blend=0.5,
        fractal_diversity=False,
        fractal_diversity_depth=3,
        fractal_diversity_bonus=1.3,
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
        qea_correlation=False,
        qea_correlation_delta=0.02,
        qea_correlation_max=2.0,
        qea_correlation_sweeps=3,
        qea_cooling=False,
        qea_cooling_decay=0.98,
        qea_cooling_min_angle=0.005,
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
        colorize=False,
        colorize_max_execs=512,
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
        fluctuation=False,
        fluctuation_beta=1.0,
        fluctuation_window=1000,
        # Appended rather than grouped with the other mutation-targeting
        # flags: this signature is positional, so inserting a parameter
        # mid-list silently shifts every caller argument after it.
        region_profile=False,
        deterministic=True,
        forkserver=True,
        seed_skip_size=0,
        seed_truncate_size=0,
        seed_slide_size=0,
        seed_slide_max_seeds=0,
        perf_novelty=True,
        reject_code=None,
        sharpe_kelly_blend=0.0,
        bootstrap=False,
        bootstrap_k=1,
        # Weizz structure tags (P1 collector + P2 operators). Appended at the
        # end so positional callers of Fuzzer() are not shifted.
        weizz_tags=False,
        weizz_tags_max_len=8192,
        email_on_crash=None,
    ):
        # Snapshot os.environ before anything below (or later in run()) can
        # write __AFL_DIST_SHM_ID / __AFL_SHM_ID / AFL_MAP_SIZE / LD_PRELOAD /
        # UBSAN_OPTIONS into it, so run() can hand the process environment
        # back afterwards. See _restore_environ()/finding #10.
        _snapshot_environ_once()
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
        self._novel_input_count = 0  # execs where record_edges found ≥1 new edge
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
        # Adaptive timeout: retune self.timeout from the live
        # ExecutionTimeTracker rather than leaving it fixed at construction.
        # Opt-in, because suggested_timeout() is derived from one target's
        # observed distribution and is not a safe global -- the default
        # stays exactly where the caller put it.
        self._adaptive_timeout = adaptive_timeout
        self._timeout_initial = timeout
        # (exec_count, old, new) per applied retune; reported at the end.
        self._timeout_retunes: list[tuple[int, float, float]] = []
        self._last_timeout_retune_exec = 0
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
        self._last_corpus_prune_exec = 0
        self._last_bloat_warn_exec = 0
        self._minimize_pending = False
        # Set by run()'s broad handler when the loop dies on an unexpected
        # exception. State is still persisted; this only marks the run as
        # incomplete so the summary does not read like a clean stop.
        self._aborted_by_error = False
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
        # Colorization (--colorize): off by default. Costs executions and
        # buys redqueen precision, so it needs a per-target A/B first.
        self.colorize = colorize
        self.colorize_max_execs = colorize_max_execs
        self._colorize_taint_cache: dict[int, object] = {}
        self._colorize_execs = 0
        # Weizz structure tags (--weizz-tags): off by default. Passive
        # collector consumes existing cmplog pairs; field/chunk operators
        # only fire when a seed carries a StructureMap in seed_meta.
        self.weizz_tags = weizz_tags
        self.weizz_tags_max_len = weizz_tags_max_len
        self._weizz_tags_collected = 0
        # MailConfig | None — novel-crash email notification (see services/sendmail.py)
        self.email_on_crash = email_on_crash
        self.enable_x86_mutator = enable_x86_mutator
        self.enable_arm_mutator = enable_arm_mutator
        self.seed = seed
        random.seed(seed)
        # RandPool holds its OWN np.random.default_rng(seed) Generator, which
        # shares no state with the legacy global np.random.* functions. Nothing
        # in src/ seeded that global, so every np.random draw outside RandPool
        # — qea.py:267,361,364 (observe/mutate amplitudes) and
        # schedulers/monte_carlo.py:778,895 (spectral probe vectors) — ran off
        # OS entropy and made --seed non-reproducible whenever QEA or the
        # Monte-Carlo scheduler was active. Seed it here, next to random.seed,
        # so the three streams start together.
        self._seed_global_numpy(seed)
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
        self._qea_correlation = qea_correlation
        self._qea_correlation_delta = qea_correlation_delta
        self._qea_correlation_max = qea_correlation_max
        self._qea_correlation_sweeps = qea_correlation_sweeps
        self._qea_cooling = qea_cooling
        self._qea_cooling_decay = qea_cooling_decay
        self._qea_cooling_min_angle = qea_cooling_min_angle
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

        # Seed preprocessing: skip/truncate/slide
        self._seed_skip_size = seed_skip_size
        self._seed_truncate_size = seed_truncate_size
        self._seed_slide_size = seed_slide_size
        self._seed_slide_max_seeds = seed_slide_max_seeds

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
        # The most recent execution's own comparison vector, refilled by the
        # drain in fuzz_one and empty whenever cmplog is off.
        self._last_cmp_fired: dict[str, int] = {}
        self._last_cmp_asserted: dict[str, int] = {}
        self._redqueen_index = 0
        self._cmplog_skip_counter = 0  # adaptive cmplog collection skip
        # Tri-state: None = auto-detect, True = forced on, False = forced off.
        # Auto-detect resolves here rather than at the direct_lite decision
        # further down, because that site only runs when self._cmplog is
        # already non-None -- i.e. it could refine how cmplog runs, never
        # whether it runs at all.
        # Seed stability calibration (handover item D). n_runs per accepted
        # seed; 0 disables. Opt-in: see _calibrate_seed_stability.
        self._calibrate_stability = int(calibrate_stability or 0)
        self._unstable_edges: set[int] = set()
        self._stability_calibrations = 0
        self._cmplog_auto = cmplog is None
        # Always detect whether the target is instrumented, regardless of
        # the tri-state. The detection result drives the confirmation
        # message; the tri-state decides whether cmplog runs.
        has_cmplog = _detect_cmplog(self.target)
        if cmplog is None:
            cmplog = has_cmplog
            if cmplog:
                print(
                    "[*] Cmplog: target is instrumented, enabling automatically (--no-cmplog to disable)"
                )
        elif cmplog and has_cmplog:
            # Force-enabled via --cmplog or --hail-mary; confirm detection.
            print("[*] Cmplog: target is instrumented, cmplog enabled")
        elif cmplog and not has_cmplog:
            # Force-enabled but target has no cmplog instrumentation.
            print(
                "[!] Cmplog: --cmplog enabled but target is not instrumented; "
                "shim compilation will be attempted"
            )
        if cmplog:
            from fuzzer_tool.core.cmplog import CmplogCollector

            self._cmplog = CmplogCollector(
                max_tokens=cmplog_max_tokens,
                max_pairs=cmplog_max_pairs,
                workdir=cmplog_workdir,
                fifo_sink=cmplog_fifo_sink,
                fifo_max_buffered=cmplog_fifo_sink_size,
                debug=self.debug,
            )
            if self._cmplog.start():
                from fuzzer_tool.core.elf import detect_cmplog_functions

                funcs = detect_cmplog_functions(self.target)
                funcs_str = ",".join(funcs)
                print(f"[*] Cmplog: comparison tracing enabled ({funcs_str})")
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

        # Gated on the *request* (enable_smt_z3), not on self._smt_solver.
        # It used to test `self._smt_solver is not None`, but the z3-missing
        # branch above has already set that to None -- so on a machine
        # without z3 this whole block was skipped and _enable_smt_z3 stayed
        # True, leaving the flag claiming an SMT path that has no solver
        # *and* no cmplog behind it. The two conditions are independent and
        # both have to clear the flag.
        if enable_smt_z3 and self._cmplog is None:
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
        else:
            # Not merely "don't load": get() lazy-loads on first access, so
            # skipping load() here deferred the read instead of preventing it.
            self._state_store.start_empty()

        self._fluctuation = None
        self._fluctuation_beta = fluctuation_beta
        self._fluctuation_window = fluctuation_window
        if fluctuation:
            from fuzzer_tool.core.fluctuation import WorkFunctional

            self._fluctuation = WorkFunctional(beta=fluctuation_beta, window=fluctuation_window)
            data = self._state_store.get("fluctuation")
            if data is not None:
                self._fluctuation.restore(data)
                print(
                    f"[*] Fluctuation tracker loaded (beta={fluctuation_beta}, "
                    f"samples={sum(len(v) for v in self._fluctuation._states.values())})"
                )

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
        # Performance novelty (per-edge max hit count). Separate from the
        # coverage signal on purpose -- see ShmCoverage._update_max_counts.
        self._perf_novelty = perf_novelty
        self._perf_novelty_hits = 0
        # Zest validity channel: the harness reports parser rejection with
        # an exit code, and coverage reached on accepted inputs gets its own
        # map. Inert without --reject-code -- see core/validity.py.
        self._validity = ValidityChannel(reject_code)
        self._validity_admits = 0
        # Comparison progress (per-callback max asserted count in a single
        # execution). The same shape as the per-edge maxima above, one
        # channel over: an input that satisfies more comparisons of some
        # family than any input before it got further into the parser, and
        # says so even when it flipped no branch the map had not already
        # seen. That is the regime the cmplog-band operators
        # (magic_byte_search, climb_hill, gradient_descent, condstmt_solve)
        # work in, and the only reward they had was the edge that arrives
        # once the comparison is fully solved -- so seven of eight correct
        # bytes paid exactly nothing.
        self._cmp_max_asserted: dict[str, int] = {}
        self._cmp_novelty_hits = 0
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
        # Selections where the operator could actually have fired -- i.e.
        # its own format sniffer matched the input it was handed. Equal to
        # op_counts for every ungated operator; strictly smaller for a
        # sniffer-gated one on a corpus that does not contain its format.
        # This, not op_counts, is the honest denominator for a success rate
        # in that regime -- paired with op_success_applicable below, which
        # is the numerator over the same selections.
        self.op_applicable: dict[str, int] = {}
        self.op_success_applicable: dict[str, int] = {}
        self.op_edges: dict[str, int] = {}
        self._peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self._discovery_execs: array = array("Q")  # exec_count per discovery snapshot
        self._discovery_edges: array = array("Q")  # cumulative edges per snapshot
        self._discovery_timestamps: array = array("d")  # wall-clock timestamp per snapshot
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
        self._favored: set[str] = set()

        # Execution time tracking for adaptive timeout calibration
        from fuzzer_tool.core.execution_time import ExecutionTimeTracker

        self._exec_time_tracker = ExecutionTimeTracker()

        # Calibrated anomaly detection for unusually slow executions.
        # Additive only: never replaces the hard f.timeout hang ceiling.
        from fuzzer_tool.core.exec_time_anomaly import ExecTimeCalibrator

        self._exec_time_anomaly = ExecTimeCalibrator()

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
                "temperature,base_w,burst,penalty,subsumption,diversity,"
                "spatial,mdl,final_w,new_coverage,new_crash\n"
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

        # Seed key: computed on demand from corpus manager.  Caching the
        # full bytes object as a dict key pinned every unique mutation in
        # memory between minimization passes, which is the primary cause of
        # the 10 GB RSS growth during long stalls.

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
        self._apply_seed_transforms()
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

        # SkipDet: initialize the skip detector for deterministic stage
        # gating. Default on to match already-shipped behavior; opt out
        # with --no-deterministic if the exec-budget cost (bitflip 1/1
        # alone is 8*len(seed) execs per favored seed) isn't wanted.
        self._skip_detector = (
            SkipDetector(map_size=getattr(self._edge_tracker, "map_size", 65536))
            if deterministic
            else None
        )
        self._det_execs: int = 0

        self.mc_bandit = mc_bandit
        self._sharpe_kelly_blend = sharpe_kelly_blend
        # Bootstrap percolation corpus minimization
        self._use_bootstrap = bootstrap
        self._bootstrap_k = bootstrap_k
        # ── Vectorized random number pool for mutation hotpath ────────
        # Generates random values in batches (one numpy C-level call per
        # batch) instead of per-call Python-level random() invocations.
        #
        # Constructed here, ahead of every scheduler, because a scheduler
        # following Hard Rule 16 takes it as its `rng`. CMAESScheduler
        # already did -- and read it ~9,000 characters before it was
        # assigned, so `--cma-es` raised AttributeError at construction.
        from fuzzer_tool.core.rand_pool import RandPool

        self._rand_pool = RandPool(seed=seed)

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
        if self.mc is not None and sharpe_kelly_blend > 0:
            self.mc.set_sharpe_kelly_blend(sharpe_kelly_blend)
        self._mopt = None
        if mopt:
            self._mopt = MOptScheduler(n_particles=5, window_size=200)
            log.info("MOpt PSO scheduling enabled (5 particles, window=200)")
        self._use_cmaes = cmaes
        self._cmaes = None
        if cmaes:
            self._cmaes = CMAESScheduler(
                # Without an explicit rng, CMAESScheduler falls back to
                # RandPool(), which seeds from OS entropy -- so --seed did not
                # determine CMA-ES behaviour and a crash found under CMA-ES
                # scheduling could not be replayed. Every other scheduler
                # draws from the module-level `random` and is covered by the
                # global seeding done at startup.
                rng=self._rand_pool,
                pop_size=cmaes_pop_size,
                generation_size=cmaes_generation_size,
                step_size=cmaes_step_size,
                elite_frac=cmaes_elite_frac,
            )
            log.info(
                "CMA-ES scheduling enabled (pop=%d, gen=%d, sigma=%.3f)",
                cmaes_pop_size,
                cmaes_generation_size,
                cmaes_step_size,
            )
        self._use_replicator = replicator
        self._seed_strategy = None
        self._seed_strategy_pool: list[str] = []
        self._seed_strategies_used: set[str] = set()
        self._use_boltzmann = boltzmann
        self._use_ecofuzz = ecofuzz
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

        # Recency-weighted UCB pair (Garivier & Moulines). Both take the
        # shared RandPool per Hard Rule 16 so --seed reproduces the campaign.
        self._use_ducb = ducb
        self._ducb = None
        if ducb:
            self._ducb = DUCBScheduler(gamma=ducb_gamma, rng=self._rand_pool)
            log.info("D-UCB enabled (gamma=%.5f)", ducb_gamma)

        self._use_swucb = swucb
        self._swucb = None
        if swucb:
            self._swucb = SWUCBScheduler(window=swucb_window, rng=self._rand_pool)
            log.info("SW-UCB enabled (window=%d)", swucb_window)

        # Combinatorial UCB: the only scheduler here that models the round's
        # operator stack as one superarm rather than N independent pulls.
        self._use_cucb = cucb
        self._cucb = None
        if cucb:
            self._cucb = CUCBScheduler(gamma=cucb_gamma, rng=self._rand_pool)
            log.info("CUCB enabled (gamma=%.5f)", cucb_gamma)

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

        from fuzzer_tool.core.coverage_regime import (
            CoverageRegimeDetector,
        )
        from fuzzer_tool.core.critical_slowing import (
            CoverageHomogeneityDetector,
            CriticalSlowingDown,
        )

        self._csd = CriticalSlowingDown(window_size=50, rise_threshold=1.5, min_observations=20)

        # Coverage-column homogeneity detector — spatial clustering check
        num_cols = max(1, self.map_size // 8192)
        self._homogeneity = CoverageHomogeneityDetector(
            num_columns=num_cols,
            window_size=10,
            homogeneity_p_threshold=0.01,
        )
        self._homogeneity_col_cumulative: list[int] = [0] * num_cols

        # Coverage regime detector: percolation phase classification
        # (subcritical / critical / supercritical).  Wraps the existing
        # CriticalSlowingDown + CoverageHomogeneityDetector + stall
        # threshold into a single actionable signal for the main loop.
        self._regime = CoverageRegimeDetector(
            csd=self._csd,
            homogeneity=self._homogeneity,
            stall_threshold=self._stall_threshold,
        )

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
        # Executions credited to _cached_total_time.  Not _cached_total_fuzz:
        # the initial seed replay in run() bumps fuzz_count without timing
        # anything, so the two diverge by the corpus size on every campaign.
        self._cached_cost_samples: int = 0
        self._cached_total_edges: int = 0
        self._cached_mean_log_n_fuzz: float = 0.0
        self._agg_cache_valid: bool = False

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
            # _cmaes was absent here while its dispatch branch in
            # operators.py was live: with only --cma-es enabled,
            # _track_op_effect stayed False, `effective` stayed None, and
            # every no-op operator in the round was credited with the
            # round's success exactly like the operator that did the work.
            or self._cmaes
            or self._contextual
            or self._ducb
            or self._swucb
            or self._cucb
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

        # Invasion percolation operator selection (percolation handover
        # Module 4): an additional Elo-arbitrated strategy, not a bandit
        # family of its own -- it reads f.mc's existing bandit_stats() as
        # its resistance signal rather than tracking separate arm state, so
        # it also requires f.mc/mc_bandit to be enabled.
        self._use_invasion = invasion
        if invasion and not (self.mc and self.mc_bandit):
            log.warning(
                "--invasion has no effect without --mc-bandit (invasion_select "
                "reads operator success/failure stats from the MC bandit tracker)"
            )

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

        # Fractal Voronoi corpus-diversity bonus (see
        # core/parallel_fractal_partition.py; applied here within one
        # corpus rather than across parallel workers)
        self._use_fractal_diversity = fractal_diversity
        self._fractal_diversity_depth = fractal_diversity_depth
        self._fractal_diversity_bonus = fractal_diversity_bonus

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

            self._distance = TargetDistance(target, targets, use_cfg_cache=use_cfg_cache)
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

        # K-Scheduler node channel: mutually exclusive with directed mode
        # (both upload __AFL_DIST_SHM_ID; evaluation campaigns are not
        # directed).
        self._katz_channel = None
        if not targets:
            try:
                from fuzzer_tool.services.katz_channel import KatzChannel

                ch = KatzChannel.build(target, use_cfg_cache=use_cfg_cache)
                if ch is not None and ch.upload():
                    self._katz_channel = ch
                    print(
                        f"[*] K-Scheduler node channel: {len(ch.node_of)} probe sites, "
                        f"{ch.n_nodes} ICFG nodes"
                    )
            except Exception as e:  # noqa: BLE001
                log.warning("Katz channel setup failed: %s", e)

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
        if self._cmaes:
            _register_arms(self._cmaes)
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
        if self._ducb:
            _register_arms(self._ducb)
        if self._swucb:
            _register_arms(self._swucb)
        if self._cucb:
            _register_arms(self._cucb)
        if self._contextual:
            _register_arms(self._contextual)
        if self._elo:
            _register_arms(self._elo)
        del _format_priors  # free priors dict after arm registration

        self._persistent_runner = None
        if self.persistent:
            from fuzzer_tool.adapters.persistent_signal import PersistentRunner

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
                    # Matches the artifact FILENAME, not the source file. The
                    # preload shim is still cached as
                    # fuzz_cmplog_shim.<digest>.so; it is built from
                    # afl_shim.c -D__AFL_PRELOAD_ONLY now that cmplog_shim.c
                    # is gone. tracecmp_shim is a name from an older split
                    # that external wrappers may still preload.
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
            # Cmplog: mirror the auto-detect .so branch's env/shim setup
            # (see above). This branch is taken whenever --inprocess is
            # explicit -- including via --hail-mary, which force-enables
            # both --inprocess and --inprocess-direct. Without this block,
            # _CMPLOG_OUT never gets set before InProcessRunner loads the
            # target below, so a compiled-in cmplog shim has nowhere to
            # write and cmplog silently collects nothing even though
            # _detect_cmplog() reports the target as instrumented.
            if self._cmplog is not None:
                has_cmplog = _detect_cmplog(self.target)
                has_tracecmp = _detect_tracecmp_target(self.target)
                if has_cmplog or has_tracecmp:
                    if has_cmplog:
                        print("[*] Cmplog: compiled into target .so (direct_lite compatible)")
                    else:
                        print(
                            "[*] Trace-cmp: compiled into target .so "
                            "(direct_lite compatible, preloading shim)"
                        )
                elif direct_ok:
                    # Not compiled in -- direct ctypes mode needs either an
                    # externally LD_PRELOAD'd shim (LD_PRELOAD is fixed at
                    # process start, so it still applies to a ctypes-loaded
                    # .so) or falls back to the subprocess loader, which
                    # picks up the shim via preload_shims()/_CMPLOG_OUT
                    # like any other cmplog-off-by-default path.
                    ld_preload = os.environ.get("LD_PRELOAD", "")
                    shim_in_preload = (
                        "cmplog_shim" in ld_preload or "tracecmp_shim" in ld_preload
                    )
                    if not shim_in_preload:
                        direct_ok = False
                    else:
                        print("[*] Cmplog: externally LD_PRELOAD'd (direct_lite compatible)")
                # Must happen regardless of execution mode: direct ctypes
                # loads the .so in-process (needs the env var set before
                # CDLL below), subprocess loader inherits os.environ.
                self._cmplog.setup_env_for_run()
                if direct_ok:
                    self._cmplog.preload_shims()
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

        if forkserver:
            self._setup_forkserver()

    def _setup_forkserver(self) -> None:
        """Start the C fuzz_loader for the default (spawn-per-exec) path.

        Replaces `run_target_fast`'s posix_spawn + ELF load + dynamic linker
        + libc init per execution with a fork+exec from an already-loaded
        process. The loader is spawned holding __AFL_SHM_ID / AFL_MAP_SIZE,
        so its children attach to the fuzzer's own coverage segment through
        afl_shim.c's constructor — no bitmap is round-tripped.

        Only claims the exact set of runs `run_target_fast` handles today:
        every other mode either owns the child itself (in-process,
        persistent, network, ptrace) or needs per-execution setup the
        loader's environment is fixed against (cmplog truncates its log per
        run; perf counters must be opened on the child pid we never see;
        file_mode/target_args build their own argv; multi-target needs more
        than one binary).
        """
        if (
            self._inprocess_runner
            or self._persistent_runner
            or self._network_runner
            or self.ptrace_cov
            or self.multi_targets
            or self.file_mode
            or self.target_args
            or self._cmplog
            or self._perf_counters
        ):
            return

        from fuzzer_tool.adapters.forkserver import ForkserverRunner

        env: dict[str, str] = {}
        if self.use_coverage:
            env["AFL_MAP_SIZE"] = str(self.map_size)
        if self.shm_cov:
            env["__AFL_SHM_ID"] = self.shm_cov.env_id

        runner = ForkserverRunner(self.target, timeout=self.timeout, env=env)
        if runner.start():
            self._forkserver = runner
            print(f"[*] Forkserver: fork+exec from a loaded process ({self.target})")
        else:
            log.warning("Forkserver unavailable, falling back to spawn-per-exec")

    def _setup_ptrace(self, target, deep_coverage, max_bps, fallback_hint=False):
        from fuzzer_tool.core.elf import detect_ngram_k

        cov = PtraceCoverage(
            target,
            deep_coverage=deep_coverage,
            max_bps=max_bps,
            ngram_k=detect_ngram_k(target),
        )
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
        """Return content hash for *data*."""
        return self._corpus_manager.seed_key(data)

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

    def _apply_seed_transforms(self) -> None:
        """Apply skip/truncate/slide transforms to the loaded corpus.

        If any transform flag is set, seeds are not used as-is. The order is:
        skip -> truncate -> slide. Sliding replaces each seed with a set of
        fixed-size windows.
        """
        if not self.corpus:
            return
        if not (self._seed_skip_size or self._seed_truncate_size or self._seed_slide_size):
            return

        original = list(self.corpus)
        transformed: list[bytes] = []
        # None means "uncapped" (0, or slide disabled) — every `cap is not
        # None` check below is then simply skipped.
        cap = (
            self._seed_slide_max_seeds
            if self._seed_slide_size and self._seed_slide_max_seeds
            else None
        )

        for seed in original:
            if cap is not None and len(transformed) >= cap:
                break
            if self._seed_skip_size and len(seed) > self._seed_skip_size:
                continue
            if self._seed_slide_size:
                win = max(1, self._seed_slide_size)
                if len(seed) <= win:
                    transformed.append(seed)
                else:
                    # bytes slicing is already a single C-level memcpy, so
                    # there is nothing for memoryview to save here — it just
                    # adds an extra allocation on the way to the same bytes.
                    for i in range(len(seed) - win + 1):
                        if cap is not None and len(transformed) >= cap:
                            break
                        transformed.append(seed[i : i + win])
            else:
                if self._seed_truncate_size and len(seed) > self._seed_truncate_size:
                    seed = seed[: self._seed_truncate_size]
                transformed.append(seed)

        self.corpus = transformed

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

    def _op_probability(self, op: str, available: list[str]) -> float:
        """Return a normalized selection probability for *op*.

        Reads from the active scheduler when possible; falls back to uniform
        over the available operator list so the work functional is always
        defined.
        """
        if available:
            if self.mc and self.mc_bandit:
                stats = self.mc.bandit_stats()
                if op in stats:
                    a, b = stats[op]
                    return max((a + 1.0) / (a + b + 2.0), 1e-12)
            if (
                self._mopt
                and getattr(self, "_meta_strategy", None) == "mopt"
                and op in getattr(self._mopt, "particles", {})
            ):
                return max(1.0 / max(len(available), 1), 1e-12)
            if (
                self._use_elo
                and self._elo
                and self._meta_strategy
                in {
                    *_OPERATOR_STRATEGY_NAMES,
                    *_SEED_STRATEGY_NAMES,
                }
            ):
                try:
                    ranking = self._elo.get_strategy_ranking()
                    top = next((name for name, _ in ranking), None)
                except Exception:
                    top = None
                if top == self._meta_strategy and len(available) > 1:
                    return max(1.0 / len(available), 1e-12)
        return max(1.0 / max(len(available), 1), 1e-12)

    def _record_fluctuation_observation(self, outcome: str, hit_edges: set[int]) -> None:
        """Ingest the current round's operator trajectory into the fluctuation tracker."""
        f = self._fluctuation
        if f is None or not self._last_ops_used:
            return
        available = (
            list(self._operators._available)
            if hasattr(self._operators, "_available")
            else list(self._last_ops_used)
        )
        probs = tuple(self._op_probability(op, available) for op in self._last_ops_used)
        from fuzzer_tool.core.fluctuation import TrajectoryRecord

        record = TrajectoryRecord(
            ops=tuple(self._last_ops_used),
            probs=probs,
            outcome=outcome,
            hit_edges=frozenset(hit_edges),
            new_edges=self._last_new_edge_count,
        )
        f.observe(record)

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

    def _seed_entropy_pct(self, seed: bytes, meta: dict | None) -> float:
        """Byte entropy of ``seed`` on the 0-100 scale, memoised in seed_meta.

        Entropy is a pure function of the seed bytes and seeds are immutable,
        so this is computed once per seed and cached alongside ``input_size``.
        Caching in ``seed_meta`` rather than a companion dict is deliberate:
        ``seed_meta`` is already rebuilt from survivors when the corpus is
        pruned (corpus_manager._maybe_prune), so the cache cannot outlive its
        seed. A separate seed-keyed map would need its own eviction and is
        exactly the shape that has leaked stale entries here before.
        """
        if meta is None:
            return byte_entropy_pct(seed)
        cached = meta.get("input_entropy")
        if cached is None:
            cached = byte_entropy_pct(seed)
            meta["input_entropy"] = cached
        return cached

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

        Uses /proc/meminfo for total RAM and /proc/self/statm for current RSS.
        Only triggers once per 1000 execs to avoid constant polling overhead.

        NOT ``getrusage(RUSAGE_SELF).ru_maxrss``, which this used to read: that
        is the high-water mark and never decreases, so a single transient spike
        past the threshold armed the pruner permanently and every later check
        re-pruned an already-small corpus while printing the stale peak as if it
        were current usage.
        """
        if self.prune_corpus_max_memory <= 0:
            return
        if self.exec_count - self._last_memory_prune_exec < 1000:
            return
        self._last_memory_prune_exec = self.exec_count

        try:
            rss_kb = _current_rss_kb()
            if rss_kb is None:
                return
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

    def _check_corpus_size_and_prune(self):
        """Auto-minimize corpus when it far exceeds the edge-derived target size.

        This runs independently of ``--minimize-every-execs`` so a large initial
        corpus does not stay bloated for the entire campaign. The trigger is
        throttled to avoid thrashing.
        """
        if self.exec_count - self._last_corpus_prune_exec < 1000:
            return
        self._last_corpus_prune_exec = self.exec_count

        if len(self.corpus) <= 1:
            return

        if self.max_corpus > 0:
            target_size = self.max_corpus
        else:
            edges = 0
            if self.shm_cov:
                edges = self.shm_cov.cumulative_edges
            elif self.ptrace_cov:
                edges = self.ptrace_cov.cumulative_edges
            target_size = min(max(edges, 50), 5000)

        if len(self.corpus) > max(target_size * 2, 5000):
            before = len(self.corpus)
            self._auto_minimize_corpus()
            after = len(self.corpus)
            if after < before:
                print(f"\n[*] CORPUS PRUNE: {before} → {after} seeds (target_size={target_size})")

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
        self._cached_cost_samples = sum(cost_samples(m) for m in self.seed_meta.values())
        self._cached_total_edges = sum(m.get("coverage_edges", 0) for m in self.seed_meta.values())
        n_fuzz_vals = [m.get("fuzz_count", 0) for m in self.seed_meta.values()]
        self._cached_mean_log_n_fuzz = compute_mean_log_n_fuzz(n_fuzz_vals)
        self._agg_cache_valid = True

    def mean_exec_time(self) -> float:
        """Corpus-wide mean target time per execution, in seconds.

        Zero until something has been timed.  This is the stand-in the cost
        ledger uses for seeds it has no samples for — a resumed seed or one
        that has never been fuzzed is *unmeasured*, not free, and giving it
        the 1 microsecond floor made it beat every timed seed in the favored
        set (see core/cost_ledger.py).
        """
        if not self._agg_cache_valid:
            self._refresh_agg_cache()
        if self._cached_cost_samples <= 0:
            return 0.0
        return self._cached_total_time / self._cached_cost_samples

    def _cull_queue(self) -> None:
        """Compute AFL-style top_rated / favored minimal-set-cover.

        For each edge, pick the cheapest seed that covers it, then greedily
        build a favored set that covers all edges. Seeds in the favored set
        receive energy bonuses in FAST/COE schedules.
        """
        top_rated: dict[int, tuple[str, float]] = {}
        mean_us = self.mean_exec_time() * 1_000_000
        for key, edges in self._edge_tracker.seed_edges.items():
            m = self.seed_meta.get(key) or {}
            exec_us = seed_exec_us(m, mean_us)
            input_size = max(1, m.get("input_size", 1))
            cost = exec_us * input_size
            for e in edges:
                cur = top_rated.get(e)
                if cur is None or cost < cur[1]:
                    top_rated[e] = (key, cost)

        covered: set[int] = set()
        favored: set[str] = set()
        # Cover the rarest edges first: an edge reached by one seed forces that
        # seed into the favored set, while an edge reached by many is likely to
        # be picked up for free along the way.
        #
        # This used to sort by ``rare_edge_count(e)``, which is keyed by *seed*,
        # not by edge. Passing an edge id looked up an absent seed and returned
        # 0 for every edge, so the key was constant and ``sorted`` -- being
        # stable -- left the edges in dict insertion order. The greedy cover ran
        # in an arbitrary order and the rarity prioritisation this loop exists
        # for never happened. The edge id breaks ties so the favored set is a
        # function of the coverage data alone, not of insertion history.
        et = self._edge_tracker
        for e in sorted(top_rated, key=lambda e: (et.edge_owner_count(e), e)):
            if e in covered:
                continue
            k = top_rated[e][0]
            favored.add(k)
            covered |= self._edge_tracker.seed_edges[k]
        self._favored = favored

    def _maybe_retune_timeout(self) -> None:
        """Feed ``suggested_timeout()`` back into the live timeout.

        The tracker has always computed this; nothing ever consumed it, so
        the timeout stayed at whatever it was constructed with and the
        report printed a suggestion nobody acted on. Two things had to be
        true before it could be applied: fractional timeouts had to survive
        the loader handshake (c99bb27), and the forkserver had to be able to
        accept a new deadline without a re-handshake (the TIMEOUT command).

        The forkserver is updated *first* and the rest only follows if it
        succeeded. Every other consumer reads ``self.timeout`` per exec, so
        they cannot disagree with it; the loader is the one that is told
        once and then believed. Updating ``self.timeout`` on a loader that
        rejected the change would leave the reported deadline and the
        enforced deadline different -- which is the entire defect class the
        E2 timeout work was about, reintroduced from the other end.
        """
        if not self._adaptive_timeout:
            return
        tracker = self._exec_time_tracker
        if tracker.count < ADAPTIVE_TIMEOUT_MIN_SAMPLES:
            return
        if self.exec_count - self._last_timeout_retune_exec < ADAPTIVE_TIMEOUT_COOLDOWN_EXECS:
            return

        proposed = tracker.suggested_timeout()
        ceiling = self._timeout_initial * ADAPTIVE_TIMEOUT_MAX_GROWTH
        proposed = max(ADAPTIVE_TIMEOUT_FLOOR, min(proposed, ceiling))

        current = self.timeout
        if current > 0 and abs(proposed - current) / current < ADAPTIVE_TIMEOUT_HYSTERESIS:
            return

        if self._forkserver is not None and not self._forkserver.set_timeout(proposed):
            # No 'retune' capability (a stale loader binary), or the loader
            # did not answer. Either way the deadline it enforces is not
            # moving, so nothing else may move either. Disable rather than
            # retry every cooldown: the capability will not appear mid-run.
            log.warning(
                "Adaptive timeout: loader would not accept %.3fs; disabling retuning", proposed
            )
            self._adaptive_timeout = False
            return

        applied = self._forkserver.timeout if self._forkserver is not None else proposed
        for runner in (self._inprocess_runner, self._persistent_runner):
            if runner is not None:
                runner.timeout = applied

        self._timeout_retunes.append((self.exec_count, current, applied))
        self._last_timeout_retune_exec = self.exec_count
        self.timeout = applied
        log.info(
            "Adaptive timeout: %.3fs -> %.3fs at exec %d (p99=%.3fs, n=%d)",
            current,
            applied,
            self.exec_count,
            tracker.p99,
            tracker.count,
        )

    def _reset_cmplog(self):
        """Flush cmplog buffer to disk/FIFO before collecting tokens.

        In direct_lite mode with cmplog compiled into the target .so,
        the shim buffers CMP lines in a 256KB internal buffer. This flushes
        that buffer to disk/FIFO so collect_tokens() can read the data.

        Does NOT truncate the file — collect_tokens() handles truncation
        after reading. In fifo_sink mode the flush still happens (the shim
        buffer must be emptied into the pipe); only the file-truncate is
        skipped.
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

    def _record_cmp_progress(self, asserted: dict[str, int]) -> bool:
        """Fold one execution's asserted counts into the per-callback maxima.

        Args:
            asserted: This execution's ``{callback: satisfied count}``.
                Empty whenever cmplog is off, which makes the whole channel
                inert rather than needing a flag of its own.

        Returns:
            True if some callback's maximum grew enough to report.

        The high-water mark is updated on every increase; only increases
        past the growth threshold are *reported*. Separating the two is the
        point: a climb of one extra satisfied comparison per input would
        otherwise report on every step of the climb, and each report admits
        an input to the corpus.
        """
        reported = False
        for name, count in asserted.items():
            prev = self._cmp_max_asserted.get(name, 0)
            if count <= prev:
                continue
            if prev == 0 or count >= max(prev + 1, int(prev * MAX_COUNT_GROWTH_FACTOR)):
                reported = True
            self._cmp_max_asserted[name] = count
        if reported:
            self._cmp_novelty_hits += 1
        return reported

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

        self._cov_before_fuzz = (
            len(self._edge_tracker._global_edge_hits)
            if hasattr(self._edge_tracker, "_global_edge_hits")
            else 0
        )
        # The timing window covers _run_target only. It used to open before
        # _dedup_mutate, which folded Python-side mutation cost into every
        # consumer of t_elapsed, and those consumers all want target time:
        #
        #   * _exec_time_anomaly -> is_slow -> `success` (see below), which
        #     is credited to the operators that ran this iteration. Ops whose
        #     own cost is milliseconds-to-seconds (gradient_descent,
        #     condstmt_solve, path_negate, crc_learn -- see
        #     _cost_adjusted_weight) pushed t_elapsed over the anomaly
        #     threshold by running at all, were credited for it, and were
        #     therefore selected more often. The contaminant was correlated
        #     with the arm being rewarded, so it compounded.
        #   * meta["total_time"] -> exec_us -> Schedule._speed_factor and the
        #     _cull_queue favored-set cost, making the favored set partly a
        #     function of which operators happened to produce each seed.
        #   * _exec_time_tracker -> suggested_timeout(), inflated by mutation.
        #
        # _dedup_mutate also calls mutate() up to EXEC_DEDUP_RETRIES + 1
        # times, so the contamination carried a multiplier driven by bloom
        # saturation rather than by anything the target did.
        mutated = self._dedup_mutate(data)
        t_start = time.monotonic()
        returncode, stderr = self._run_target(mutated)
        t_elapsed = time.monotonic() - t_start
        self.exec_count += 1
        if self._stall_recovery_active:
            self._stall_recovery_execs += 1

        # Per-seed wall-clock cost
        if meta is not None:
            meta["total_time"] = meta.get("total_time", 0.0) + t_elapsed
            meta["cost_samples"] = meta.get("cost_samples", 0) + 1
            self._cached_total_time += t_elapsed
            self._cached_cost_samples += 1

        # Record execution time for adaptive timeout calibration
        self._exec_time_tracker.record(t_elapsed)

        # Feed the anomaly calibrator for slow-but-completed detection.
        self._exec_time_anomaly.observe(t_elapsed)

        if self.mc:
            self.mc.execs_since_refit += 1

        # Flush tracecmp buffer before collecting tokens (direct_lite mode)
        self._reset_cmplog()

        # Drain the comparison counters here, on the execution boundary, and
        # NOT on the token-collection schedule below.
        #
        # collect_counts() was only ever reached through collect_tokens(),
        # which throttles itself to every 5th and then every 20th iteration
        # once the pair pool saturates. That is right for tokens -- parsing
        # the record stream costs 14-23ms -- and wrong for the counters,
        # which are a handful of short lines read from a saved offset. On
        # the throttled schedule a delta is the sum over up to twenty
        # executions, so it says nothing about which input produced it; on
        # this schedule it is the executed input's own comparison vector.
        #
        # The reset above is what makes the shim dump: __tracecmp_flush and
        # __cmplog_reset both call __afl_cmp_dump_counts, and in subprocess
        # mode the exiting child has already dumped from __afl_cmplog_fini.
        #
        # Caveat: trimming re-executes the target on admission iterations,
        # so those vectors carry the trim runs too. Those iterations found
        # coverage by definition, which is the stronger signal anyway.
        self._last_cmp_fired = {}
        self._last_cmp_asserted = {}
        if self._cmplog:
            self._last_cmp_fired, self._last_cmp_asserted = self._cmplog.collect_counts()

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
            # In direct_lite mode the compiled-in shim keeps the cmplog file
            # open with O_APPEND. collect_tokens() truncates the file
            # externally, but the shim's internal file offset is not reset by
            # that truncation. Call __cmplog_reset() so the next execution
            # writes at offset 0 instead of a stale position, which would
            # create a sparse file and inflate RSS.
            runner = self._inprocess_runner
            if runner and runner.direct_lite and runner._lib:
                try:
                    if hasattr(runner._lib, "__cmplog_reset"):
                        runner._lib.__cmplog_reset()
                except (AttributeError, OSError):
                    pass
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
            # Colorization taints for this seed, if the pass is enabled. Bytes
            # inside a taint can be replaced without changing the execution
            # path, so an operand found there is coincidence, not
            # input-to-state. See _colorize_seed().
            taints = self._colorize_seed(mutated)
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
                        if _in_taint(taints, idx, len(op_a)):
                            # Every byte of this occurrence can be replaced
                            # without changing the path, so the target never
                            # read it: a coincidental match, not the operand
                            # the comparison actually consumed.
                            pos = idx + 1
                            continue
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
            # Same cadence as the other periodic bookkeeping; the method's
            # own cooldown and hysteresis decide whether anything happens.
            self._maybe_retune_timeout()

        # Selections split by regime. A sniffer-gated operator has two of
        # them: it mutates a file of its own format, or -- on input that is
        # not that format -- synthesises a fresh one from scratch
        # (_op_png_chunk_mutate's parse_png_chunks() else
        # _generate_random_png() branch, and the same shape in every other
        # format op). Pooling the two makes the reported rate a function of
        # how much of the corpus happens to be that format, which is the
        # distortion the bootstrap trickle and the live-format short
        # circuit both feed.
        #
        # setdefault, so the key exists even at zero: absent has to keep
        # meaning "unknown" (a state file predating this) rather than
        # "never", or the report cannot tell them apart and falls back to
        # the raw count -- printing exactly the inflated rate this removes.
        applicable_now = getattr(self, "_last_ops_applicable", set())
        for op in set(self._last_ops_used):
            self.op_counts[op] = self.op_counts.get(op, 0) + 1
            self.op_applicable.setdefault(op, 0)
            if op in applicable_now:
                self.op_applicable[op] += 1

        # Track cmplog as its own operator
        if cmplog_found:
            self.op_counts["cmplog"] = self.op_counts.get("cmplog", 0) + 1

        # Track SMT solver as its own operator
        if smt_found:
            self.op_counts["smt_solver"] = self.op_counts.get("smt_solver", 0) + 1

        # -1 is the cross-backend timeout sentinel. stderr is not part of the
        # contract: forkserver reports loader hangs as (-1, "") after its
        # restart retry, so keying on the stderr text missed every one.
        is_timeout = returncode == -1
        if is_timeout:
            self.timeout_count += 1
            # Mark the parent seed as timeout-causing for power schedule
            parent_meta = self.seed_meta.get(self._last_parent_seed)
            if parent_meta is not None:
                parent_meta["timed_out"] = True
            self._corpus_manager.save_timeout(mutated)

        is_crash = self._is_crash(returncode, stderr)
        is_interesting = self._is_interesting(returncode, stderr)
        is_slow = False
        if not is_timeout and not is_crash:
            thresh = self._exec_time_anomaly.threshold()
            if thresh is not None and t_elapsed > thresh:
                is_slow = True
        # Check new coverage (per-target SHM in multi-target mode).
        # Use is_new_coverage_with_edges() on SHM to get both the boolean
        # and the edge set in one buffer scan, avoiding redundant scans.
        self._current_edges_cache = None  # will be set below if SHM scanned
        # Set from whichever ShmCoverage was actually scanned this iteration.
        # Only the sparse SHM path maintains per-edge maxima; the ptrace
        # bitmap has no counts to take a maximum of, so it stays 0 there.
        scanned_shm = None
        if self.multi_targets:
            active_shm = self._target_shm_covs.get(self.target)
            if active_shm:
                has_new, edge_ids = active_shm.is_new_coverage_with_edges()
                self._current_edges_cache = edge_ids
                has_new_coverage = has_new
                scanned_shm = active_shm
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
            scanned_shm = self.shm_cov
        else:
            has_new_coverage = bool(self.ptrace_cov and self.ptrace_cov.is_new_coverage())

        # Performance novelty: an edge whose trip count grew substantially
        # past anything seen before. The hit-count buckets saturate (129 and
        # 10^6 are the same bucket), so this is the only signal that stays
        # live once a loop is merely being spun harder -- which is the
        # algorithmic-complexity bug class the timing channel is actually
        # good for. Suppressed on timeout and crash: a partial execution's
        # counts are truncated, not extreme.
        new_max_edges = 0
        if self._perf_novelty and scanned_shm is not None and not is_timeout and not is_crash:
            new_max_edges = scanned_shm.new_max_edges
        is_new_max = new_max_edges > 0
        if is_new_max:
            self._perf_novelty_hits += 1

        # Comparison progress: a callback family this input satisfied more
        # times, in one execution, than any input before it. Suppressed on
        # timeout and crash for the same reason as above -- a truncated
        # execution's counts are short, not extreme, and the crash handler's
        # dump would be attributed to the wrong boundary.
        #
        # Growth rather than strict `>`, exactly as _update_max_counts
        # argues: each report both rewards operators and admits an input, so
        # the number of times one callback can report over a whole campaign
        # has to be bounded. A first-ever assert reports unconditionally --
        # unlike a first-seen edge it is not already covered by another
        # signal, and there are only twenty-seven callbacks, so the
        # unbounded-looking case is bounded at twenty-seven.
        is_cmp_progress = False
        if not is_timeout and not is_crash:
            is_cmp_progress = self._record_cmp_progress(self._last_cmp_asserted)

        # Zest validity channel: coverage reached while the target ACCEPTED
        # the input, tracked in its own map. An input that is valid and
        # covers something no valid input covered before is worth keeping
        # even when the main map has seen those edges already -- reached
        # from the parser's error path, they lead nowhere; reached from an
        # accepted input, they are the semantic stages behind the syntax
        # check. Inert unless --reject-code gave the harness a way to say
        # "rejected".
        is_new_valid_coverage = False
        validity = Validity.UNKNOWN
        if self._validity.enabled and not is_timeout and not is_crash:
            validity = self._validity.classify(returncode)
            valid_edges = (
                self._current_edges_cache
                if self._current_edges_cache is not None
                else self._get_current_edge_set()
            )
            is_new_valid_coverage = self._validity.record(validity, valid_edges)
            if is_new_valid_coverage:
                self._validity_admits += 1

        # Region liveness (item 4, handover_skittercreek_tailslayer_port.md):
        # fold this exec's coverage diff into the per-region
        # LiveBitMaskEstimator for whichever byte the mutation touched.
        # Deliberately unconditional on has_new_coverage above -- that flag
        # only means "globally new edge", which is rare; the liveness
        # estimator needs the far more common "no new edges, but still an
        # observation" samples to ever reach convergence at all. Cheap and
        # skipped outright when there's no edge data or no known parent
        # baseline to diff against.
        _liveness_parent = getattr(self, "_last_parent_seed", None)
        _liveness_offset = getattr(self, "_last_mutation_offset", None)
        if (
            self._current_edges_cache is not None
            and _liveness_parent is not None
            and _liveness_offset is not None
        ):
            parent_key = self._seed_key(_liveness_parent)
            baseline_edges = self._edge_tracker.seed_edges.get(parent_key)
            if baseline_edges:
                newly_dead = self._operators.record_coverage_diff(
                    _liveness_parent,
                    _liveness_offset,
                    baseline_edges,
                    self._current_edges_cache,
                )
                if newly_dead is not None and self._format_learner:
                    region_offset, region_width = newly_dead
                    self._format_learner.record_liveness(
                        region_offset, region_width, confirmed_dead=True
                    )

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

        # Weizz structure tags: once-per-lineage passive collection after a
        # coverage gain, gated by --weizz-tags and max_len. Uses existing
        # cmplog pairs (+ optional colorize taints); no second tracer.
        if (
            has_new_coverage
            and self.weizz_tags
            and self._cmplog is not None
            and len(mutated) <= self.weizz_tags_max_len
        ):
            self._maybe_collect_weizz_tags(mutated)

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
                f"{ps['temperature']},{ps['base_w']},{ps['burst']},{ps['penalty']},"
                f"{ps['subsumption']},{ps['diversity']},{ps['spatial']},"
                f"{ps['mdl']},{ps['final_w']},"
                f"{1 if has_new_coverage else 0},{1 if is_crash else 0}\n"
            )
            if self.exec_count % 100 == 0:
                self._ablation_file.flush()

        # K-Scheduler bitmap sampling runs EVERY exec (beta needs R_i over
        # all mutations); per-seed mask attribution only for corpus-worthy
        # inputs, keyed like EdgeTracker.
        if getattr(self, "_katz_channel", None) is not None:
            bits = self._katz_channel.sample()
            if bits is not None and bits.any():
                katz_key = self._seed_key(data) if has_new_coverage else None
                self._katz_channel.record(bits, seed_key=katz_key)

        # Record edges for per-seed tracking
        if has_new_coverage:
            seed_key = self._seed_key(data)
            # Prefer sparse edge set with counts (SHM), fall back to byte bitmap (ptrace)
            # Must read from `scanned_shm` (the segment actually scanned above —
            # the per-target one in multi-target mode), not unconditionally
            # from `self.shm_cov`, which in multi-target mode is a separate,
            # unscanned shared segment.
            if scanned_shm is not None and not self.ptrace_cov:
                hit_counts = scanned_shm.get_edge_counts()
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
                if scanned_shm is not None:
                    stack_depth = scanned_shm.read_stack_depth()
                    path_hash = scanned_shm.read_path_hash()
                # Fallback: compute path hash from edge IDs in Python
                if path_hash == 0 and hit_edges and not isinstance(hit_edges, bytes):
                    path_hash = (
                        scanned_shm.compute_path_hash_from_edges(hit_edges)
                        if scanned_shm is not None
                        else 0
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
                    self._novel_input_count += 1
                    self._saturation = None  # invalidate cached saturation
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

        # is_cmp_progress joins the disjunction rather than replacing any
        # part of it: it is a weaker event than an edge (a comparison can be
        # satisfied more often without the branch it guards ever flipping),
        # but it arrives during exactly the stretches where the edge signal
        # is silent, which is when the cmplog-band operators are doing their
        # work and getting paid nothing for it.
        success = bool(
            is_crash
            or is_interesting
            or has_new_coverage
            or is_slow
            or is_new_max
            or is_cmp_progress
            or is_new_valid_coverage
        )

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
                # Numerator for the mutate-regime rate. It has to be
                # restricted to the same selections as op_applicable or the
                # two do not divide: a format op that synthesised a file
                # from scratch and found an edge that way is a success on a
                # selection op_applicable never counted, which produced
                # successes against a zero denominator in the first cut of
                # this change.
                if op in applicable_now:
                    self.op_success_applicable[op] = self.op_success_applicable.get(op, 0) + 1
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
            self._cmaes,
            self._ducb,
            self._swucb,
            self._cucb,
        ):
            if scheduler is None:
                continue
            for op, ok, w in op_rewards:
                scheduler.record(op, ok, weight=w)

        # CUCB batches the round rather than updating per operator, so the
        # superarm is only complete once the loop above has run.
        if self._cucb:
            self._cucb.settle_round()

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
            self._record_fluctuation_observation("crash", self._get_current_edge_set())
            return True

        # is_new_max admits too. Without admission the signal cannot compound:
        # amplifying a loop is incremental, and the input that first doubled a
        # trip count is the only useful parent for the one that doubles it
        # again. Bounded by the growth factor -- an edge can report at most
        # log_1.5(2^24) < 40 times over a whole run -- so this cannot run away
        # the way PerfFuzz's strict `>` would on a counted loop.
        # is_cmp_progress admits for the reason is_new_max does: the signal
        # cannot compound without it. Getting one comparison further into a
        # length-prefixed header is only useful if the input that managed it
        # becomes the parent for the input that gets one further still --
        # and during a magic-bytes plateau, which is precisely when this
        # fires, no other criterion is admitting anything at all.
        if (
            is_interesting
            or has_new_coverage
            or is_new_max
            or is_cmp_progress
            or is_new_valid_coverage
        ):
            _corpus_len_before = len(self.corpus)
            self.save_to_corpus(mutated, parent=data)
            # Validity is a property of this execution, so it is recorded
            # here rather than reconstructed later: the seed picker reads it
            # off the metadata and nothing re-runs the input to ask again.
            if validity is not Validity.UNKNOWN:
                meta = self.seed_meta.get(mutated)
                if meta is not None:
                    meta["valid"] = validity is Validity.VALID
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
            self._record_fluctuation_observation("success", self._get_current_edge_set())
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
                self._record_fluctuation_observation("success", self._get_current_edge_set())
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

        self._record_fluctuation_observation("boring", self._get_current_edge_set())
        return False

    def _record_discovery_snapshot(self):
        return self._stats.record_discovery_snapshot()

    def _run_calibration(self, max_execs: int = 1000):
        return self._stats.run_calibration(max_execs)

    def _calibrate_seed_stability(self, data: bytes, n_runs: int = 3) -> set[int]:
        """Re-run *data* and mask edges that don't reproduce.

        AFL-style per-seed stability calibration. An identical input run
        several times should produce an identical edge set; any edge that
        appears in some runs and not others is nondeterministic — ASLR-,
        time-, thread-, or uninitialized-memory-dependent. Left alone those
        edges read as an endless supply of new coverage and permanently
        absorb mutation energy, because every re-execution "discovers" them
        again.

        Returns the set of unstable edge ids found (already masked). Empty
        set means the seed was stable, calibration was disabled, or there
        was no SHM coverage to read.

        Deliberately *not* using ``read_path_hash()`` alone as the verdict.
        The hash is order- and multiplicity-sensitive, so it diverges when
        the same edges fire in a different order or a different number of
        times — which is the common case for a loop whose trip count depends
        on nothing but scheduling, and which is not what we want to mask.
        The hash is used only as a cheap screen: identical hashes across all
        runs means definitely stable, skip the set comparison entirely.
        Divergence sends us to the per-edge set-diff, which is what actually
        decides.

        Cost is ``n_runs`` extra executions per accepted seed. That is a
        real throughput tax on a corpus that accepts often, which is why
        this is opt-in via ``--calibrate-stability`` rather than on by
        default: no A/B against a real target has been run, and the repo's
        standing rule is that an unmeasured throughput change does not
        become the default.
        """
        shm = self.shm_cov
        if shm is None or n_runs < 2:
            return set()

        edge_sets: list[set[int]] = []
        hashes: set[int] = set()
        for _ in range(n_runs):
            try:
                self._run_target(data)
            except Exception:
                # A seed that will not re-run tells us nothing about
                # stability; leave it unmasked rather than guessing.
                return set()
            edge_sets.append(shm.get_edge_ids())
            hashes.add(shm.read_path_hash())

        if not edge_sets:
            return set()

        # Cheap screen: one distinct hash across every run means the same
        # edges fired the same number of times in the same order.
        if len(hashes) == 1:
            self._stability_calibrations += 1
            return set()

        stable = set.intersection(*edge_sets)
        seen = set.union(*edge_sets)
        unstable = seen - stable

        self._stability_calibrations += 1
        if not unstable:
            # Hashes diverged but the edge *sets* agree: ordering or trip
            # counts moved, not which code ran. Not an unstable-edge case.
            return set()

        newly = shm.mask_edges(unstable)
        self._unstable_edges |= unstable
        if newly:
            log.info(
                "Stability calibration: %d unstable edge(s) masked (%d total) after %d runs",
                newly,
                len(self._unstable_edges),
                n_runs,
            )
        return unstable

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

    def _colorize_seed(self, data: bytes):
        """Colorization taints for ``data``, or ``None`` when disabled.

        AFL++/Redqueen colorization replaces every byte it can while holding
        the execution path fixed. What survives is the set of bytes the
        target actually reads, and what gets replaced is a *taint region* --
        provably path-irrelevant.

        The redqueen match loop needs this because it accepts every literal
        occurrence of a comparison operand as an input-to-state candidate,
        filtered only by ``len(op_a) >= 2``. A two-byte operand hits a 4 KiB
        input roughly sixteen times by chance, each coincidence becomes an
        entry in a list capped at 50, and the operator then substitutes at a
        random one. Colorization is what tells the two apart.

        Off by default (``--colorize``). It costs real executions -- up to
        ``2 * len(data)``, bounded here to keep a large seed from stalling
        the loop -- and buys precision, not throughput, so the trade needs
        measuring per target before it becomes a default. Results are cached
        per seed; a seed is colorized at most once.

        Returns None when disabled, unavailable, or the budget was spent
        without a usable answer. Callers treat None as "no filtering",
        which is the pre-existing behaviour.
        """
        if not getattr(self, "colorize", False):
            return None
        if not data or self.shm_cov is None:
            return None

        cache = self._colorize_taint_cache
        key = hash(data)
        if key in cache:
            return cache[key]

        def exec_fn(candidate: bytes) -> int:
            """Path checksum for one execution. 0 means 'unknown'."""
            try:
                self._runner.run_target(candidate)
            except Exception:
                return 0
            path_hash = self.shm_cov.read_path_hash()
            if path_hash == 0:
                # Shim without the rolling hash: fall back to the edge set,
                # which is coarser (order- and multiplicity-insensitive) but
                # still separates different paths.
                edges = self._get_current_edge_set()
                path_hash = self.shm_cov.compute_path_hash_from_edges(edges) if edges else 0
            return path_hash

        try:
            from fuzzer_tool.core.colorization import colorize

            result = colorize(
                bytes(data),
                exec_fn,
                use_type_aware=True,
                max_execs=min(2 * len(data), self.colorize_max_execs),
            )
        except Exception:
            log.debug("colorization failed for a seed; continuing unfiltered", exc_info=True)
            cache[key] = None
            return None

        self.exec_count += result.exec_count
        self._colorize_execs += result.exec_count
        taints = result.taints or None

        # Bounded cache: colorization is per-seed and the corpus grows.
        if len(cache) > 512:
            cache.clear()
        cache[key] = taints
        return taints

    def _maybe_collect_weizz_tags(self, data: bytes) -> None:
        """Passive Weizz structure-tag collection for one coverage-gaining seed.

        Once-per-lineage: if ``seed_meta`` already carries a non-dirty RLE
        map for this exact byte string, skip. Size-gated by
        ``weizz_tags_max_len``. Does not run the differential path (that
        stays available via ``get_deps`` for callers that need causality).
        """
        if not data or self._cmplog is None:
            return
        meta = self.seed_meta.get(data)
        if meta is not None:
            existing = meta.get("weizz_tags_rle")
            if existing and not meta.get("weizz_tags_dirty", False):
                return
        try:
            from fuzzer_tool.core.weizz_tags import (
                TagCollectorConfig,
                attach_tags_to_meta,
                collect_structure_map,
            )

            taints = None
            if self.colorize:
                taints = self._colorize_seed(data)
            smap = collect_structure_map(
                data,
                self._cmplog,
                colorization_result=taints,
                config=TagCollectorConfig(max_input_len=self.weizz_tags_max_len),
            )
            if smap is None or smap.ntypes == 0:
                return
            if meta is None:
                meta = {}
                self.seed_meta[data] = meta
            attach_tags_to_meta(meta, smap)
            self._weizz_tags_collected += 1
        except Exception:  # noqa: BLE001 — never take down the fuzz loop
            log.debug("weizz tag collection failed", exc_info=True)

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

    @staticmethod
    def _seed_global_numpy(seed: int | None) -> None:
        """Seed the legacy global ``np.random`` state.

        Separate from ``RandPool``, which owns an independent
        ``default_rng`` Generator. ``np.random.seed`` accepts only
        ``[0, 2**32)``, so a wider seed is folded rather than raising;
        ``None`` reseeds from OS entropy, matching ``random.seed(None)``.

        Args:
            seed: The run seed, or None for an unseeded run.
        """
        if not _HAS_NUMPY:
            return
        np.random.seed(None if seed is None else seed & SEED_MASK_32)

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

        All three streams are reseeded together, matching how ``__init__``
        seeds them: ``random`` drives the non-hotpath choices, ``RandPool``
        owns its own ``default_rng`` Generator and backs the mutation
        hotpath, and the global ``np.random`` state backs QEA and the
        Monte-Carlo scheduler. ``RandPool`` is NOT backed by global
        ``np.random`` — an earlier version of this docstring said it was,
        which is why the global went unseeded. ``RandPool.reseed`` also drops
        the pre-fetched pool, which would otherwise keep dispensing
        old-stream values for another ``_POOL_ENTRIES`` draws.

        Returns:
            The seed that was applied.
        """
        self._stall_reseed_count += 1
        new_seed = self._derive_stall_seed()
        random.seed(new_seed)
        self._seed_global_numpy(new_seed)
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

        # What the edge signal cannot distinguish: a stall the campaign is
        # still driving into a comparison it never passes, versus a stall
        # where it stopped reaching that comparison at all. Both are "no new
        # edges for N execs"; the first wants budget on the wall, the second
        # wants the parent seeds back.
        if self._cmplog is not None:
            walls = self._cmplog.wall_summary()
            if walls:
                reason += f" + {walls}"

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
                if self._forkserver:
                    self._forkserver.update_shm_after_resize(self.shm_cov.env_id, new_size)
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
        # cmaes was missing from this ballot while operators.py::select_op
        # listed it: a cmaes-vs-other match was recorded when cmaes was the
        # selected strategy, but never when the other one was, so its rating
        # moved on only half its games.
        if self._cmaes:
            all_strategies.append("cmaes")
        if self._contextual:
            all_strategies.append("contextual")
        if self._ducb:
            all_strategies.append("ducb")
        if self._swucb:
            all_strategies.append("swucb")
        if self._cucb:
            all_strategies.append("cucb")
        if self._use_invasion and self.mc and self.mc_bandit:
            all_strategies.append("invasion")
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
        # Was _hierarchical_bandit, an attribute that has never existed --
        # the constructor stores _use_hierarchical -- so the banner silently
        # omitted the hierarchical bandit on every run it was enabled.
        if getattr(self, "_use_hierarchical", False):
            ops.append("hierarchical")
        if getattr(self, "_gp_ucb", False):
            ops.append("gp_ucb")
        if getattr(self, "_cmaes", False):
            ops.append("cmaes")
        if getattr(self, "_contextual", False):
            ops.append("contextual")
        if getattr(self, "_ducb", False):
            ops.append("ducb")
        if getattr(self, "_swucb", False):
            ops.append("swucb")
        if getattr(self, "_cucb", False):
            ops.append("cucb")
        if getattr(self, "_use_invasion", False) and self.mc_bandit:
            ops.append("invasion")
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
        if getattr(self, "_use_ecofuzz", False):
            seeds.append("ecofuzz")
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

    def _calibrate_seed_baselines(self) -> None:
        """Execute every corpus seed verbatim once, before the fuzz loop.

        fuzz_one() runs ONLY mutated input: _dedup_mutate() transforms even
        seed iterations, so the pristine bytes a corpus file contains are
        never executed. The tracker's edge universe is therefore built
        exclusively from mutants that happen to stay format-valid — for
        structured targets that is a tiny sliver of what the seeds really
        reach (observed on png_read.so: 11 valid PNGs produced shm=6 edges
        after 150 execs, while one direct execution of a single seed
        records ~330; the campaign then plateaus immediately and Good-
        Turing reads the starved universe as 100% saturated).

        This pass gives every seed one unmutated execution through the
        normal pipeline (_run_target + is_new_coverage_with_edges +
        record_edges), so scheduling, corpus admission, GT estimation and
        cmplog token collection all start from the coverage the seeds
        genuinely carry. Crashes/timeouts during calibration are counted,
        not fatal — a hostile corpus must not kill startup.
        """
        if not self.use_coverage or not self.shm_cov or self.multi_targets:
            return
        if not self.corpus:
            return
        baseline_edges = 0
        t0 = time.monotonic()
        for seed in list(self.corpus):
            returncode, stderr = self._run_target(seed)
            if self._is_crash(returncode, stderr):
                self.crash_count += 1
                continue
            if returncode == -1:  # timeout sentinel
                self.timeout_count += 1
                self._corpus_manager.save_timeout(seed)
                continue
            has_new, edge_ids = self.shm_cov.is_new_coverage_with_edges()
            if not edge_ids:
                continue
            hit_counts = self.shm_cov.get_edge_counts()
            stack_depth = self.shm_cov.read_stack_depth()
            path_hash = self.shm_cov.read_path_hash()
            if path_hash == 0:
                path_hash = self.shm_cov.compute_path_hash_from_edges(edge_ids)
            new = self._edge_tracker.record_edges(
                self._seed_key(seed),
                edge_ids,
                target_name=os.path.basename(self.target),
                hit_counts=hit_counts,
                stack_depth=stack_depth,
                path_hash=path_hash,
                hw_instructions=self._last_perf_deltas.get("instructions", 0),
                hw_branches=self._last_perf_deltas.get("branches", 0),
                hw_branch_misses=self._last_perf_deltas.get("branch_misses", 0),
            )
            baseline_edges += len(new)
        if baseline_edges or len(self.corpus):
            print(
                f"[*] Seed calibration: {len(self.corpus)} seeds -> "
                f"{baseline_edges} baseline edges "
                f"({time.monotonic() - t0:.2f}s)"
            )
        self._report_comparison_reach(len(self.corpus))

    def _report_comparison_reach(self, n_execs: int) -> None:
        """Say whether the comparison instrumentation reached the target.

        "cmplog is on but doing nothing" is currently silent in every form
        it takes, and it takes several: an -O2 build without -fno-builtin
        (measured on cmplog_exercise.c: 20 call sites, 4 records), a preload
        that lost the symbol-lookup race to the executable's own weak sancov
        stubs, a target that never reaches its parser on the seeds it was
        given. All of them look identical from the outside -- the campaign
        runs, the token pool just stays empty -- and a user reads that as
        "this target has no interesting comparisons".

        The counters answer it directly, and calibration is where the
        question is cheap: every seed has just been executed exactly once,
        so the totals are over a known number of executions of unmutated
        input, before the fuzz loop can muddy them.

        Printed like the distance line, and warned about only in the case
        with no benign reading. Zero fires across a whole seed pass is that
        case: a target worth pointing a fuzzer at compares *something*.
        """
        if self._cmplog is None or n_execs <= 0:
            return
        self._reset_cmplog()
        self._cmplog.collect_counts()
        (l1_fired, _), (l2_fired, _) = self._cmplog.layer_totals()
        total = l1_fired + l2_fired
        if total == 0:
            print(
                "[!] Comparison instrumentation: no comparisons observed in "
                f"{n_execs} seed executions — cmplog is enabled but not "
                "reaching the target (check -fno-builtin-* on the target "
                "build, and that the shim is linked in rather than preloaded)"
            )
            return
        print(
            f"[*] Comparison instrumentation: {total} comparisons over "
            f"{n_execs} seed executions (libc {l1_fired}, trace-cmp {l2_fired})"
        )
        if l1_fired == 0:
            print(
                "[*]   Comparisons are inlined: the libc layer sees nothing, "
                "so trace-cmp records the post-expansion (0, 1) pairs rather "
                "than operands. Rebuild with -fno-builtin-memcmp "
                "(and -strcmp, -strncmp) to recover them."
            )

    def _print_enabled_features(self) -> None:
        """Print every enabled feature, grouped by category."""
        groups: dict[str, list[str]] = {
            "Scheduling": [],
            "Seed selection": [],
            "Mutation": [],
            "Analysis": [],
            "Generation": [],
            "Execution": [],
            "Output": [],
        }

        sched_pol = getattr(self, "_power_schedule", "base")
        if sched_pol != "base":
            groups["Scheduling"].append(f"power-schedule={sched_pol}")

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
            ops.append("eps-greedy")
        if getattr(self, "_use_hierarchical", False):
            ops.append("hierarchical")
        if getattr(self, "_gp_ucb", False):
            ops.append("gp-ucb")
        if getattr(self, "_ducb", False):
            ops.append("ducb")
        if getattr(self, "_swucb", False):
            ops.append("swucb")
        if getattr(self, "_cucb", False):
            ops.append("cucb")
        if getattr(self, "_use_contextual", False):
            ops.append("contextual")
        if getattr(self, "_use_invasion", False):
            ops.append("invasion")
        if getattr(self, "_use_shapley", False):
            ops.append("shapley")
        if getattr(self, "_cmaes", False):
            groups["Scheduling"].append("cma-es")
        if ops:
            groups["Scheduling"].extend(ops)

        if getattr(self, "_use_elo", False):
            groups["Scheduling"].append("elo")

        if self.ga:
            groups["Seed selection"].append("ga")
        if self.qea:
            groups["Seed selection"].append("qea")
        if getattr(self, "_use_bayesian", False):
            groups["Seed selection"].append("bayesian")
        if getattr(self, "_use_boltzmann", False):
            groups["Seed selection"].append("boltzmann")
        if self.markov_generate:
            groups["Seed selection"].append("markov-gen")
        if getattr(self, "_distance", None) is not None:
            groups["Seed selection"].append("aflgo")

        if self.markov_trained:
            groups["Mutation"].append("markov")
        if getattr(self, "_adaptive_havoc", False):
            groups["Mutation"].append("adaptive-havoc")
        if self.enable_x86_mutator:
            groups["Mutation"].append("x86-mutator")
        if self.enable_arm_mutator:
            groups["Mutation"].append("arm-mutator")
        if self.enable_regex_bomb:
            groups["Mutation"].append("regex-bomb")
        if self.weizz_tags:
            groups["Mutation"].append("weizz-tags")
        if self.colorize:
            groups["Mutation"].append("colorize")

        if getattr(self, "_use_mi", False):
            groups["Analysis"].append("mi-guided")
        if getattr(self, "_use_renyi_weight", False):
            groups["Analysis"].append("renyi")
        if getattr(self, "_use_transfer_entropy", False):
            groups["Analysis"].append("transfer-entropy")
        if getattr(self, "_use_sensitivity", False):
            groups["Analysis"].append("sensitivity")
        if getattr(self, "_use_lineage", False):
            groups["Analysis"].append("lineage")
        if getattr(self, "_use_lineage_backtrack", False):
            groups["Analysis"].append("lineage-backtrack")
        if getattr(self, "_use_overlap_density", False):
            groups["Analysis"].append("overlap-density")
        if getattr(self, "_use_region_profile", False):
            groups["Analysis"].append("region-profile")
        if getattr(self, "_calibrate", 0) > 0:
            groups["Analysis"].append(f"calibrate={self._calibrate}")

        if getattr(self, "_wfc_enabled", False):
            groups["Generation"].append("wfc")
        if getattr(self, "_use_mcts", False):
            groups["Generation"].append("mcts")
        if getattr(self, "_corpus_boost", 0) > 0:
            groups["Generation"].append(f"corpus-boost={self._corpus_boost}")
        if getattr(self, "_use_bootstrap", False):
            groups["Generation"].append("bootstrap")

        if self.persistent:
            groups["Execution"].append("persistent")
        if self._inprocess_runner:
            groups["Execution"].append("inprocess")
        if self._adaptive_timeout:
            groups["Execution"].append("adaptive-timeout")
        if self.honggfuzz:
            groups["Execution"].append("honggfuzz")
        if self.hw_perf:
            groups["Execution"].append("hw-perf")
        if self._diff_target:
            groups["Execution"].append("differential")

        if self.debug:
            groups["Output"].append("debug")
        if self._tracer is not None:
            groups["Output"].append("trace-crashes")
        if self._format_learner is not None:
            groups["Output"].append("learn-format")

        print("[*] Enabled features:")
        for group, feats in groups.items():
            if feats:
                print(f"    {group}: {', '.join(feats)}")

    def run(self, iterations=0):
        self._start_stack_heartbeat()
        if self.multi_targets:
            print(f"[*] Multi-target: {len(self.multi_targets)} targets, shared corpus")
            uninstrumented = []
            for i, t in enumerate(self.multi_targets):
                status = afl_instrumentation_status(t)
                tag = {"present": " [AFL]", "absent": " [no-AFL]", "unknown": " [AFL?]"}[status]
                if status == "absent":
                    uninstrumented.append(t)
                dist = _detect_distance(t)
                if dist:
                    tag += " [DIST]"
                print(f"  [{i}] {t}{tag}")
            if uninstrumented:
                self._warn_uninstrumented(uninstrumented)
        else:
            print(f"[*] Target: {self.target}")
            self._report_instrumentation()
            if _detect_distance(self.target):
                print("[*] Distance instrumentation: detected")
                if self._distance is None:
                    print(
                        "[*]   Directed mode idle: pass --target-functions "
                        "(function, address, or file.c:line) to engage the "
                        "distance channel (dist: stats + aflgo schedule/elo arm)"
                    )
        from fuzzer_tool.core.elf import detect_ngram_k

        print(f"[*] Ngram: k={detect_ngram_k(self.target)}")
        if self._validity.enabled:
            print(f"[*] Validity channel: reject-code {self._validity.reject_code}")
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
        print(f"[*] Edge bitmap: {self.map_size:,} entries (auto-sized)")
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
        self._print_enabled_features()
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
                # Restore and announce here, not inside the differential
                # block below: nesting it there meant `--ga --resume` without
                # a differential target silently restarted GA at generation 0,
                # while `--differential-target` without `--ga` reached the
                # banner with self.ga still None and died on .pop_size.
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

            # Initialize differential fuzzing if enabled
            if self._diff_target:
                from fuzzer_tool.services.differential import DifferentialTracker

                self._diff_tracker = DifferentialTracker()
                print(f"[*] Differential: comparing against {self._diff_target}")

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
                    use_correlation=self._qea_correlation,
                    correlation_delta=self._qea_correlation_delta,
                    correlation_max=self._qea_correlation_max,
                    correlation_sweeps=self._qea_correlation_sweeps,
                    use_cooling=self._qea_cooling,
                    cooling_decay=self._qea_cooling_decay,
                    cooling_min_angle=self._qea_cooling_min_angle,
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
                    + (
                        f", correlation=on (delta={self.qea.correlation_delta}, "
                        f"max={self.qea.correlation_max}, sweeps={self.qea.correlation_sweeps})"
                        if self.qea.use_correlation
                        else ""
                    )
                )

            if self._cmaes:
                cmaes_data = self._state_store.get("cmaes")
                if self.resume and cmaes_data is not None:
                    self._cmaes.from_dict(cmaes_data)
                    print(
                        f"[*] CMA-ES: loaded state from state store "
                        f"(gen={self._cmaes.convergence_stats()['generation']})"
                    )
                stats = self._cmaes.convergence_stats()
                print(
                    f"[*] CMA-ES: pop={self._cmaes.pop_size}, "
                    f"gen={self._cmaes.generation_size}, "
                    f"sigma={stats['sigma']:.3f}, "
                    f"top={stats['top_op']}({stats['top_prob']:.1%})"
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

            self._calibrate_seed_baselines()

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
                    # Denominator is the number of timed executions, not
                    # fuzz_count: the seed replay above bumps fuzz_count with
                    # no time credited, and a resumed seed used to carry a
                    # restored count against a zero numerator.
                    avg_exec_us = max(1, int(self.mean_exec_time() * 1_000_000))
                    exec_us = max(1, int(seed_exec_us(meta, avg_exec_us)))
                    bitmap_size = meta.get("coverage_edges", 0)
                    avg_bitmap_size = max(
                        1,
                        int(self._cached_total_edges / max(1, len(self.seed_meta))),
                    )
                    depth = meta.get("lineage_depth", 0)
                    fuzz_level = meta.get("fuzz_count", 0)
                    n_fuzz = fuzz_level
                    seed_key = self._seed_key(seed)

                    # Honggfuzz power factors (only when --honggfuzz enabled)
                    hf_kwargs: dict = {}
                    if self.honggfuzz:
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
                        input_entropy = self._seed_entropy_pct(seed, meta)
                        if input_entropy > ENTROPY_RANDOM_PCT or input_entropy < ENTROPY_SPARSE_PCT:
                            self._hf_entropy_penalties += 1

                        hf_kwargs = dict(
                            new_edges=new_edges,
                            time_added=time_added,
                            now=now,
                            input_size=len(seed),
                            input_entropy=input_entropy,
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
                        favored=(seed_key in self._favored),
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
                        katz_energy=(
                            self._katz_channel.seed_energy(seed_key)
                            if getattr(self, "_katz_channel", None) is not None
                            else 0.0
                        ),
                        **hf_kwargs,
                    )
                else:
                    # Markov-generated or synthetic seed: reset to neutral multiplier
                    self._last_perf_score = 100.0
                if self._diff_tracker:
                    self._check_differential(seed)
                # Deterministic-stage mutations are now drawn inline by
                # OperatorEngine.mutate() (see maybe_deterministic_mutation),
                # so they get the same execution/coverage/corpus-save path
                # as every other mutation instead of a separate blocking
                # loop that has to duplicate it.
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
                    # Update favored set / top_rated cull queue every interval
                    if self._edge_tracker.seed_edges:
                        self._cull_queue()
                    # Sample Shannon entropy for rate-of-change tracking
                    if self._edge_tracker._global_edge_hits:
                        sh = self._edge_tracker.shannon_entropy_global()
                        self._record_entropy_sample(sh)
                    # Feed incremental edge count to Allan variance detector
                    current_edges = self._edge_tracker.get_cumulative_edge_count()
                    delta = current_edges - self._last_allan_edge_count
                    self._allan.update(delta)
                    self._last_allan_edge_count = current_edges
                    # Feed per-column edge counts to CoverageHomogeneityDetector
                    if self.shm_cov and hasattr(self, "_homogeneity"):
                        log.debug("homogeneity: shm_cov present, observing col counts")
                        hit_counts = self.shm_cov.get_edge_counts()
                        n_cols = self._homogeneity.num_columns
                        col_totals_this_tick = [0] * n_cols
                        for edge_id, _count in hit_counts.items():
                            col_totals_this_tick[edge_id % n_cols] += 1
                        col_deltas = [
                            col_totals_this_tick[i] - self._homogeneity_col_cumulative[i]
                            for i in range(n_cols)
                        ]
                        self._homogeneity_col_cumulative = col_totals_this_tick
                        log.debug("homogeneity: col_deltas=%s", col_deltas)
                        self._homogeneity.observe(col_deltas)
                        try:
                            result = self._homogeneity.is_homogeneous()
                            log.debug("homogeneity: result=%s", result)
                            if not result["homogeneous"] and result["total_edges"] > 0:
                                log.info(
                                    "Coverage clustered: χ²=%.2f, p=%.4f, V=%.3f, edges=%d",
                                    result["chi2"],
                                    result["p_value"],
                                    result["cramers_v"],
                                    result["total_edges"],
                                )
                        except Exception as ex:
                            log.debug("Coverage homogeneity check failed: %s", ex)
                    # Capture the homogeneity result for the regime detector.
                    # `result` is only defined inside the shm_cov branch above;
                    # fall back to None when the detector wasn't initialised
                    # or shm_cov isn't available.
                    homogeneity_result = (
                        result if (self.shm_cov and hasattr(self, "_homogeneity")) else None
                    )

                    # Record coverage snapshot for temporal analysis
                    self._edge_tracker.record_coverage_snapshot(self.exec_count)
                    if not self.quiet_stats:
                        self.print_stats()
                    self._append_coverage_log()
                    self._record_discovery_snapshot()
                    # Feed the regime detector with all observed signals.
                    # The CriticalSlowingDown detector is already fed from
                    # _print_stats_dr_str (stats.py:583); we only read its
                    # state here.  The homogeneity detector is fed above.
                    execs_since_edge = self.exec_count - self._last_new_edge_exec
                    self._regime.observe(
                        discovery_rate=self._stats.discovery_rate(),
                        allan_delta=delta,
                        homogeneity_result=homogeneity_result,
                        execs_since_edge=execs_since_edge,
                        exec_count=self.exec_count,
                    )
                    # Regime-driven strategy adjustment
                    if self._regime.actionable:
                        regime = self._regime.regime
                        reason = self._regime.reason
                        log.info("REGIME: %s — %s", regime.value, reason)
                        if regime is CoverageRegime.SUBCRITICAL:
                            # Subcritical: escalate mutation diversity, force stall recovery
                            if not self._stall_recovery_active:
                                self._maybe_trigger_stall_recovery(execs_since_edge)
                            # Bump havoc energy if the operator engine exposes that knob
                            ops = getattr(self, "_operators", None)
                            if ops is not None and hasattr(ops, "_havoc_energy_scale"):
                                ops._havoc_energy_scale = min(
                                    getattr(ops, "_havoc_energy_scale", 1.0) * 1.5, 5.0
                                )
                        elif regime is CoverageRegime.CRITICAL:
                            # CSD: near a coverage jump — preserve current strategy
                            log.info("REGIME: critical — preserving strategy (%s)", reason)
                        elif regime is CoverageRegime.SUPERCRITICAL:
                            # Healthy: clear any emergency state
                            if self._stall_recovery_active:
                                self._stall_recovery_active = False
                                log.info("REGIME: supercritical — clearing stall recovery")
                        self._regime.acknowledge()
                    # Stall detection: no new edges in threshold execs
                    if (
                        not self._stall_recovery_active
                        and execs_since_edge >= self._stall_threshold
                        and self.exec_count > 0
                    ):
                        self._maybe_trigger_stall_recovery(execs_since_edge)
                    # Memory-based corpus pruning
                    self._check_memory_and_prune()
                    # Corpus-size-based pruning: minimize when corpus is
                    # significantly larger than the edge-derived target size,
                    # even if --minimize-every-execs is not set.
                    self._check_corpus_size_and_prune()
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
        except Exception:
            # Everything below this try is end-of-run persistence: _dump_stats,
            # every _state_store.set, both _save_state, and the ablation fd
            # close. None of it is in a `finally`, so before this handler any
            # exception the loop did not name — a ValueError out of a scheduler,
            # a KeyError out of a mutator — propagated straight past all of it
            # and discarded the whole campaign: Markov model, Elo ratings,
            # crash-MI counters, GA/QEA/CMA-ES/MCTS generations. Hours of work
            # for a bug in one mutation. Catch broadly so the state lands, and
            # log the traceback so the underlying defect stays loud rather than
            # being swallowed (Hard Rule 20).
            log.exception("Fuzzing aborted by unexpected error — persisting state before exit")
            self._aborted_by_error = True

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
        if self._cmaes:
            self._state_store.set("cmaes", self._cmaes.to_dict())
            print(f"[*] CMA-ES: saved state (gen={self._cmaes.generation})")
        if self._mcts is not None:
            # Drop stats for seeds minimization removed, so the persisted
            # state does not grow without bound across resumes.
            if self._lineage is not None:
                self._mcts.prune(set(self._lineage.nodes))
            self._state_store.set("mcts", self._mcts.to_dict())
            print(f"[*] MCTS: saved state ({self._mcts.stats()['tracked_nodes']} nodes)")
        if self._fluctuation is not None:
            self._state_store.set("fluctuation", self._fluctuation.snapshot())
            samples = sum(len(v) for v in self._fluctuation._states.values())
            print(f"[*] Fluctuation: saved state (samples={samples})")
        self._save_state()
        if self._cmplog is not None:
            # Releases this run's .cmplog/.counts/.sites files. Nothing else
            # called stop(), so every run left its trio behind under a fresh
            # uuid and the cache directory grew for the life of the machine.
            with contextlib.suppress(Exception):
                self._cmplog.stop()
        if self._ablation_file:
            self._ablation_file.flush()
            self._ablation_file.close()
            self._ablation_file = None
            print(f"[*] Schedule ablation log: {self._ablation_path}")
        if not self.quiet_stats:
            self.print_stats()
        stop_word = "aborted by an unexpected error" if self._aborted_by_error else "stopped"
        print(
            f"\n\n[*] Fuzzing {stop_word}. {self.crash_count} crashes found "
            f"({len(self.crash_sigs)} unique signatures)."
        )
        if self._aborted_by_error:
            print("[!] State was persisted; see the log for the traceback.")
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
        if self._cmaes:
            print("\n[*] CMA-ES convergence:")
            stats = self._cmaes.convergence_stats()
            print(
                f"    generation={stats['generation']} sigma={stats['sigma']:.4f} "
                f"top={stats['top_op']}({stats['top_prob']:.1%}) "
                f"discoveries={stats['total_discoveries']}/{stats['total_execs']}"
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
        # Hand os.environ back the way we found it. setup_env_for_run() puts
        # the cmplog shim on the process-global LD_PRELOAD, and that shim
        # conflicts with the ASAN runtime, so anything exec'd after this run
        # -- the next epoch in a multi-target session, a replay, a caller
        # embedding Fuzzer -- would silently find no crashes.
        if self._cmplog is not None:
            self._cmplog.restore_env()
        # cmplog's restore_env() above only ever undid its own LD_PRELOAD
        # edit. __AFL_DIST_SHM_ID, __AFL_SHM_ID, AFL_MAP_SIZE, the ASAN
        # LD_PRELOAD injection and UBSAN_OPTIONS were never restored, so
        # they leaked into whatever ran next in this process -- see
        # finding #10. _restore_environ() puts back everything this Fuzzer
        # (or an earlier one in the same process) changed.
        _restore_environ()
