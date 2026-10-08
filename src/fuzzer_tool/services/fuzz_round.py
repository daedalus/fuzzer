"""One fuzz iteration: mutate, execute, observe, credit, admit.

Extracted from ``Fuzzer.fuzz_one``. One ``FuzzRound`` per iteration, so
per-round outcome flags live on the instance and cannot leak into the next
round. Long-lived state stays on the Fuzzer (``self._f``).

    run()
     ├─ _begin           reset the Fuzzer's per-round fields
 ├─ _search_fixpoint --i2s-fixpoint, seed's first round only
     ├─ _execute         mutate, timed target run, cmp counters
     ├─ _mine_cmplog     tokens, learners, dictionary cap, redqueen/SMT
     ├─ _periodic        RSS / EPS / crash-rate bookkeeping
     ├─ _count_ops       selection counts
     ├─ _classify        timeout / crash / interesting / slow
     ├─ _scan_coverage   new edges + novelty confirmation
     ├─ _observe         side signals (perf, cmp progress, validity, liveness)
     ├─ _credit_seed     seed-arena posteriors
     ├─ _feed_models     cmplog / dict / weizz / MI / katz / tang
     ├─ _record_edges    per-seed edges, op attribution
     ├─ _learn_format    format-learner transition
     ├─ _track_edges     lifetimes, length, distance, anneal
     ├─ _judge           success
     ├─ _credit_ops      bandits, Elo, arenas, attribution
     └─ _on_crash | _admit | _on_boring
"""

import logging
import os
import resource
from typing import TYPE_CHECKING

from fuzzer_tool.core.analyzers.analyzer_distance import _NO_VALUE_DISTANCE
from fuzzer_tool.core.analyzers.analyzer_pll import Series as PLLSeries
from fuzzer_tool.core.analyzers.analyzer_recurrence import Novelty
from fuzzer_tool.core.cadence import due
from fuzzer_tool.core.clock import clock_of
from fuzzer_tool.core.cmplog import CmplogRecords
from fuzzer_tool.core.one_fifth import Outcome as FifthOutcome
from fuzzer_tool.core.ro_rd import classify_operator_name
from fuzzer_tool.core.schedulers.pos_base import Outcome
from fuzzer_tool.core.secretary import SecretaryStopping
from fuzzer_tool.core.validity import Validity
from fuzzer_tool.services.runner import TargetRunner, ptrace_available

if TYPE_CHECKING:
    from fuzzer_tool.services.fuzzer import Fuzzer

log = logging.getLogger(__name__)

# ── Memory bounds ────────────────────────────────────────────────────
CRASH_RATE_HISTORY_MAX = 500  # max entries in _crash_rate_execs/_crash_rate_counts
SEED_SECRETARY_MAX = 500  # max per-seed SecretaryStopping entries
MATCH_CAP = 50  # redqueen matches kept per seed


def _apply_reward_shape(
    op_rewards: list[tuple[str, bool, float]], shape: float | None
) -> list[tuple[str, bool, float]]:
    """Scale the successful rounds' reward weights by *shape*.

    ``shape is None`` means "the class partition has nothing to say about this
    round" and the list is returned unchanged -- see
    :meth:`Fuzzer._credit_reward_shape` for when that is the case.

    Only the ``ok`` entries move. A failure already carries weight 0.0 (the
    reward loop passes ``surprisal_weight if ok else 0.0``), so scaling it would
    be arithmetic on a zero, but keeping the branch explicit means a future
    non-zero failure weight does not get shaped by a factor that describes a
    discovery the failure did not make.

    The shaping is applied *after* the [0, 1] cap the loop already imposes, so
    the result cannot climb back above the cap: scaling a capped weight is a
    discount, never a promotion.
    """
    if shape is None:
        return op_rewards
    return [(op, ok, w * shape if ok else w) for op, ok, w in op_rewards]


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


class FuzzRound:
    """One ``fuzz_one`` iteration over parent seed *data*.

    Example::

        admitted = FuzzRound(fuzzer, seed).run()
    """

    __slots__ = (
        "_f",
        "_data",
        "_meta",
        "_mutated",
        "_returncode",
        "_stderr",
        "_t_elapsed",
        "_collect_now",
        "_cmplog_found",
        "_smt_found",
        "_applicable_now",
        "_is_timeout",
        "_is_crash",
        "_is_interesting",
        "_is_slow",
        "_scanned_shm",
        "_has_new_coverage",
        "_is_new_max",
        "_is_cmp_progress",
        "_is_new_valid_coverage",
        "_ltl_new",
        "_validity",
        "_success",
        "_effective",
        "_surprisal_weight",
        "_distance",
    )

    def __init__(self, fuzzer: "Fuzzer", data: bytes):
        self._f = fuzzer
        self._data = data
        self._meta = None
        self._mutated = b""
        self._returncode = 0
        self._stderr = ""
        self._t_elapsed = 0.0
        self._collect_now = False
        self._cmplog_found = False
        self._smt_found = False
        self._applicable_now: set = set()
        self._is_timeout = False
        self._is_crash = False
        self._is_interesting = False
        self._is_slow = False
        self._scanned_shm = None
        self._has_new_coverage = False
        self._is_new_max = False
        self._is_cmp_progress = False
        self._is_new_valid_coverage = False
        self._ltl_new = False
        self._validity = Validity.UNKNOWN
        self._success = False
        self._effective = None
        self._surprisal_weight = 0.0
        self._distance: float | None = None

    def run(self) -> bool:
        """Run the round; True when the mutant crashed or was admitted."""
        self._begin()
        self._search_fixpoint()
        self._generalize()
        self._execute()
        self._mine_cmplog()
        self._periodic()
        self._count_ops()
        self._classify()
        self._ltl_observe()
        self._scan_coverage()
        self._observe()
        self._gate_explore()
        self._credit_seed()
        self._feed_models()
        self._record_edges()
        self._learn_format()
        self._track_edges()
        self._judge()
        self._credit_ops()
        self._push_recurrence()

        if self._is_crash:
            self._queue_variant()
            return self._on_crash()
        if self._admits():
            return self._admit()
        return self._on_boring()

    # ── Setup and execution ──────────────────────────────────────────

    def _search_fixpoint(self) -> None:
        # --i2s-fixpoint: once per seed, on its first round (_begin counted
        # it), so initial, admitted and resumed seeds all get one search.
        # Before _execute: the search's runs must not overwrite this round's.
        fixpoint = getattr(self._f, "_i2s_fixpoint", None)
        if fixpoint is None or self._meta is None or self._meta["fuzz_count"] != 1:
            return
        fixpoint.search(self._data)

    def _generalize(self) -> None:
        # --grimoire: the stage generalizes each tracked seed once. Not gated
        # on fuzz_count: other paths bump it before a first round, which
        # would leave those seeds ungeneralized forever. Before _execute for
        # the same reason as _search_fixpoint: the stage's runs must not
        # overwrite this round's coverage.
        stage = getattr(self._f, "_grimoire", None)
        if stage is None or self._meta is None:
            return
        stage.generalize(self._data)

    def _begin(self) -> None:
        f = self._f
        data = self._data
        # Invalidate Elo K-factor cache at the start of each iteration
        # so record_strategy_match calls recompute K from the current
        # prediction errors if record_match hasn't been called yet.
        if f._use_elo and f._elo:
            f._elo._eff_k_cache = None
        f._last_parent_seed = data
        if f._op_strata is not None:
            f._op_strata.set_stratum(f._strata_stratum(data))
        f._last_new_edge_count = 0  # reset; set when record_edges finds new edges
        # The ids themselves, not just how many: the shaped reward needs the
        # identities to ask the canonical partition how many classes they are.
        # Reset every round so a round with no discovery cannot be shaped by the
        # previous round's edges.
        f._last_new_edge_ids = []
        # This round's full hit-edge trace, for _continuum_reward_shape's
        # neighbourhood. Reset for the same reason: a round with no
        # discovery must not be priced against a stale trace.
        f._last_trace_edges = ()
        self._meta = f.seed_meta.get(data)
        if self._meta is not None:
            self._meta["fuzz_count"] += 1
            f._cached_total_fuzz += 1

        f._cov_before_fuzz = (
            len(f._edge_tracker._global_edge_hits)
            if hasattr(f._edge_tracker, "_global_edge_hits")
            else 0
        )

    def _execute(self) -> None:
        f = self._f
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
        self._mutated = f._dedup_mutate(self._data)
        self._gate_records()
        self._ltl_clear()
        clock = clock_of(f)
        t_start = clock.monotonic()
        self._returncode, self._stderr = f._run_target(self._mutated)
        # One tick per execution: under --clock virtual this is the whole
        # of time's progress, so every exec reads as VIRTUAL_EXEC_S.
        clock.tick()
        t_elapsed = clock.monotonic() - t_start
        self._t_elapsed = t_elapsed
        f.exec_count += 1

        # A timeout may be a crash that reports late (fork-wrapped main);
        # re-run once at a longer deadline. Outside t_elapsed: the re-run
        # is the fuzzer's cost, not the target's speed.
        if self._returncode == -1:
            confirmed = f._confirm_hang(self._mutated)
            if confirmed is not None:
                self._returncode, self._stderr = confirmed
        # Effector map: read the trace hash while it is still this
        # execution's. One ctypes word read, and only when the mutant just
        # executed came from the byteflip 8/8 pass.
        f._note_det_effector()
        if f._stall_recovery_active:
            f._stall_recovery_execs += 1

        # Per-seed wall-clock cost
        meta = self._meta
        if meta is not None:
            meta["total_time"] = meta.get("total_time", 0.0) + t_elapsed
            meta["cost_samples"] = meta.get("cost_samples", 0) + 1
            f._cached_total_time += t_elapsed
            f._cached_cost_samples += 1

        # Record execution time for adaptive timeout calibration
        f._exec_time_tracker.record(t_elapsed)
        # PLL observation (--pll): getattr since __new__-built test fuzzers skip wire_all.
        pll = getattr(f, "_pll", None)
        if pll is not None:
            pll.push(PLLSeries.EXEC_TIME, t_elapsed)

        # Feed the anomaly calibrator for slow-but-completed detection.
        f._exec_time_anomaly.observe(t_elapsed)

        if f.mc:
            f.mc.execs_since_refit += 1

        # Flush tracecmp buffer before collecting tokens (direct_lite mode)
        f._reset_cmplog()

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
        f._last_cmp_fired = {}
        f._last_cmp_asserted = {}
        if f._cmplog:
            f._last_cmp_fired, f._last_cmp_asserted = f._cmplog.collect_counts()

    # ── Cmplog mining ────────────────────────────────────────────────

    def _mine_cmplog(self) -> None:
        if not self._f._cmplog:
            return
        self._collect_tokens()
        self._feed_learners()
        self._cap_dictionary()
        self._find_matches()

    def _collect_tokens(self) -> None:
        f = self._f
        # Collect cmplog tokens — periodic sampling once pool is saturated.
        # collect_tokens() reads + parses the whole cmplog file (~14-23ms
        # with 5000 pairs); running it every iteration when the pool is
        # already saturated destroys throughput.  Adaptive: eager while
        # building the pool, then sample every N iterations.
        self._collect_now = self._collect_due()
        f._cmplog_skip_counter += 1
        if self._collect_now:
            f._cmplog_skip_counter = 0
            new_tokens = f._cmplog.collect_tokens()
        else:
            new_tokens = []
        self._cmplog_found = bool(new_tokens)
        self._rewind_shim()
        f._add_dict_tokens(new_tokens)

    def _collect_due(self) -> bool:
        # Whether this round's _collect_tokens will drain the records.
        f = self._f
        pr_count = len(f._cmplog.pairs)
        interval = 1 if pr_count < 500 else (5 if pr_count < 2000 else 20)
        return f._cmplog_skip_counter + 1 >= interval

    def _gate_records(self) -> None:
        # Records nobody parses cost the target 3.1x on ffmpeg: emit them
        # only on rounds that collect.
        f = self._f
        if not f._cmplog:
            return
        f._cmplog_records(CmplogRecords.ON if self._collect_due() else CmplogRecords.OFF)

    def _rewind_shim(self) -> None:
        # In direct_lite mode the compiled-in shim keeps the cmplog file
        # open with O_APPEND. collect_tokens() truncates the file
        # externally, but the shim's internal file offset is not reset by
        # that truncation. Call __cmplog_reset() so the next execution
        # writes at offset 0 instead of a stale position, which would
        # create a sparse file and inflate RSS.
        # Only after a collect: the shim truncates, and on skipped rounds the
        # records are still waiting to be read.
        if self._collect_now:
            self._f._rewind_cmplog_shim()

    def _feed_learners(self) -> None:
        f = self._f
        data = self._data
        # Feed checksum learner: format-aware pairs from current input
        # + cmplog heuristic pairs (gated on _collect_now to avoid
        # repeated work when the pool is saturated).
        if f.checksum_learner and self._collect_now:
            fmt_pairs = f.checksum_learner.extract_format_pairs(data)
            if fmt_pairs:
                f.checksum_learner.add_pairs(fmt_pairs)
            cmp_pairs = f.checksum_learner.extract_cmplog_pairs(data)
            if cmp_pairs:
                f.checksum_learner.add_pairs(cmp_pairs)

        # Feed PRNG state learner: same cmplog pool, mirror-image
        # extraction (operands absent from the input rather than
        # present in it -- see core/prng_state_learner.py). Same
        # _collect_now gate as checksum_learner, for the same reason.
        if f.prng_state_learner and self._collect_now:
            f.prng_state_learner.observe_execution(data)

    def _cap_dictionary(self) -> None:
        f = self._f
        # Dynamic cap: scale with recent throughput.
        # High EPS → larger dictionary (more mutations explore more).
        # Low EPS → smaller dictionary (reduce overhead).
        # Window: last 500 iterations. Range: [64, 1024].
        window = 500
        if f.exec_count > 0 and due(f.exec_count, 100, "fuzzer.dict_eps"):
            elapsed = clock_of(f).time() - f.start_time
            eps = (f.exec_count - f._resume_baseline_exec) / elapsed if elapsed > 0 else 0
            f._dict_eps_window.append(eps)
            if len(f._dict_eps_window) > 10:
                f._dict_eps_window.pop(0)

        if not f._dict_eps_window or f.exec_count - f._dict_last_prune < window:
            return
        # Use Kalman-filtered EPS if available, fall back to window avg.
        if hasattr(f, "_eps_filtered") and f._eps_filtered is not None and f._eps_filtered > 0:
            avg_eps = f._eps_filtered
        else:
            avg_eps = sum(f._dict_eps_window) / len(f._dict_eps_window)
        # Map EPS to cap: 10 eps → 128, 30 eps → 256, 100+ eps → 1024
        dyn_cap = max(64, min(1024, int(avg_eps * 8)))
        if len(f.dictionary) > dyn_cap:
            keep = max(dyn_cap // 2, 32)
            f.dictionary = f.dictionary[-keep:]
            f._dict_set = set(f.dictionary)
            f._dict_last_prune = f.exec_count

    def _find_matches(self) -> None:
        # Record redqueen matches: (offset, operand_a, operand_b)
        # for input-to-state matching during mutation.
        # Only scan new pairs (not yet seen) to avoid O(5000) per iteration.
        meta = self._meta
        matches = list(meta.get("redqueen_matches", [])) if meta is not None else []
        seen = {(m[1], m[2]) for m in matches}  # dedup by (A, B)
        self._redqueen_scan(matches, seen)
        self._smt_sample(matches, seen)
        self._concolic(matches)
        self._path_negate(matches)
        if meta is not None:
            meta["redqueen_matches"] = matches[:MATCH_CAP]
            # Keep legacy field for state compat
            meta["redqueen_offsets"] = [m[0] for m in meta["redqueen_matches"]]

    def _redqueen_scan(self, matches: list, seen: set) -> None:
        f = self._f
        # Colorization taints for this seed, if the pass is enabled. Bytes
        # inside a taint can be replaced without changing the execution
        # path, so an operand found there is coincidence, not
        # input-to-state. See _colorize_seed().
        taints = f._colorize_seed(self._mutated)
        _pending = f._cmplog.pending_new_pairs() if self._meta is not None else []
        if not _pending:
            return
        _consumed = 0
        for op_a, op_b in _pending:
            _consumed += 1
            if len(op_a) < 2 or (op_a, op_b) in seen:
                continue

            # Pass 1: find op_a literally in mutated input (original redqueen)
            matched = self._scan_operand(op_a, op_b, matches, seen, taints)
            if len(matches) >= MATCH_CAP:
                break
            if matched:
                continue

            # Pass 2: try finding op_b instead (reverse direction)
            # swap: replace op_b with op_a
            if len(op_b) >= 2:
                self._scan_operand(op_b, op_a, matches, seen, None)
            if len(matches) >= MATCH_CAP:
                break

        # Only what the loop actually reached. Breaking at the match
        # cap above leaves the rest queued for the next iteration
        # instead of dropping it on the floor.
        f._cmplog.consume_new_pairs(_consumed)

    def _scan_operand(self, needle: bytes, repl: bytes, matches: list, seen: set, taints) -> bool:
        """Append every untainted ``needle`` occurrence; True if any was found."""
        mutated = self._mutated
        pos = 0
        matched = False
        while pos <= len(mutated) - len(needle):
            idx = mutated.find(needle, pos)
            if idx == -1:
                break
            if _in_taint(taints, idx, len(needle)):
                # Every byte of this occurrence can be replaced
                # without changing the path, so the target never
                # read it: a coincidental match, not the operand
                # the comparison actually consumed.
                pos = idx + 1
                continue
            matches.append((idx, needle, repl))
            seen.add((needle, repl))
            matched = True
            pos = idx + 1
            if len(matches) >= MATCH_CAP:
                break
        return matched

    def _smt_budget(self) -> int:
        # Tune sample budget: if solve rate is sustained >50% we can
        # invest more; if <10% we're mostly wasting time on that input.
        solver = self._f._smt_solver
        _smt_q = solver.queries_attempted
        if _smt_q <= 50:
            return 5
        _rate = solver.queries_solved / _smt_q
        if _rate < 0.1:
            return 2
        if _rate > 0.5:
            return 10
        return 5

    def _smt_sample(self, matches: list, seen: set) -> None:
        # SMT sampling pass: runs every iteration regardless of redqueen gate.
        # Adaptive sample size based on historical solve rate.
        f = self._f
        solver = f._smt_solver
        if solver is None or not f._cmplog.pairs:
            return
        solver.reset_batch()
        _budget = self._smt_budget()
        sample = f._cmplog.pairs[:]
        f._rng.shuffle(sample)
        smt_counter = 0
        for op_a, op_b in sample:
            if smt_counter >= _budget:
                break
            if (op_a, op_b) in seen:
                continue
            smt_counter += 1
            pc = f._cmplog.pair_pc(op_a, op_b)
            result = solver.solve_cmplog_pair(op_a, op_b, pc=pc)
            if result is None:
                continue
            solved = result["solved_bytes"]
            # Per-pair flag: a later miss must not erase an earlier hit.
            found = False
            for candidate in (op_a, op_b):
                if len(candidate) < 2:
                    continue
                found = self._scan_operand(candidate, solved, matches, seen, None)
                if found or len(matches) >= MATCH_CAP:
                    break
            self._smt_found = self._smt_found or found
            if len(matches) >= MATCH_CAP:
                break

    def _concolic(self, matches: list) -> None:
        # Concolic mode: after accumulating trace entries, solve and inject
        solver = self._f._smt_solver
        if (
            solver is None
            or solver.mod_solving_mode != "concolic"
            or solver.concolic_trace is None
            or not solver.concolic_trace.has_entries()
        ):
            return
        mutated = self._mutated
        concolic_result = solver.solve_concolic(mutated)
        if concolic_result is None or concolic_result == mutated:
            solver.queries_failed += 1
            return
        solver.queries_solved += 1
        solver.batch_solved += 1
        # Inject the concolic solution as a replacement mutation
        matches.append((0, mutated, concolic_result))
        self._smt_found = True

    def _path_negate(self, matches: list) -> None:
        # Path negation: solve for an input that takes the opposite side
        # of a branch this run actually took. Unlike the concolic block
        # above — which pins every byte to a literal, giving a fully
        # determined system that reproduces the observed operands — this
        # leaves the operand window symbolic and asserts the *negated*
        # predicate, so z3 searches for a value reaching the sibling
        # branch rather than replaying one already seen.
        f = self._f
        if f._path_solver is None or f._cmplog is None:
            return
        from fuzzer_tool.core.path_constraints import records_from_collector

        records = records_from_collector(f._cmplog)
        if not records:
            return
        mutated = self._mutated
        negated = f._path_solver.solve_first(records, mutated)
        if negated is not None and negated != mutated:
            matches.append((0, mutated, negated))
            self._smt_found = True

    # ── Bookkeeping and classification ───────────────────────────────

    def _periodic(self) -> None:
        f = self._f
        if not due(f.exec_count, 100, "fuzzer.rss_eps"):
            return
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if rss > f._peak_rss:
            f._peak_rss = rss
        elapsed = clock_of(f).time() - f.start_time
        eps = (f.exec_count - f._resume_baseline_exec) / elapsed if elapsed > 0 else 0
        if eps > f._peak_eps:
            f._peak_eps = eps
        f._crash_rate_execs.append(f.exec_count)
        f._crash_rate_counts.append(f.crash_count)
        if len(f._crash_rate_execs) > CRASH_RATE_HISTORY_MAX:
            del f._crash_rate_execs[:250]
            del f._crash_rate_counts[:250]
        # Same cadence as the other periodic bookkeeping; the method's
        # own cooldown and hysteresis decide whether anything happens.
        f._maybe_retune_timeout()

    def _count_ops(self) -> None:
        f = self._f
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
        self._applicable_now = getattr(f, "_last_ops_applicable", set())
        for op in set(f._last_ops_used):
            f.op_counts[op] = f.op_counts.get(op, 0) + 1
            f.op_applicable.setdefault(op, 0)
            if op in self._applicable_now:
                f.op_applicable[op] += 1

        # Track cmplog as its own operator
        if self._cmplog_found:
            f.op_counts["cmplog"] = f.op_counts.get("cmplog", 0) + 1

        # Track SMT solver as its own operator
        if self._smt_found:
            f.op_counts["smt_solver"] = f.op_counts.get("smt_solver", 0) + 1

    def _classify(self) -> None:
        f = self._f
        self._is_crash = f._is_crash(self._returncode, self._stderr)
        # -1 is the cross-backend timeout sentinel. stderr is not part of the
        # contract: forkserver reports loader hangs as (-1, "") after its
        # restart retry, so keying on the stderr text missed every one.
        # A timeout carrying a sanitizer report is the crash, not a hang.
        self._is_timeout = self._returncode == -1 and not self._is_crash
        if self._is_timeout:
            f.timeout_count += 1
            # Mark the parent seed as timeout-causing for power schedule
            parent_meta = f.seed_meta.get(f._last_parent_seed)
            if parent_meta is not None:
                parent_meta["timed_out"] = True
            f._corpus_manager.save_timeout(self._mutated)

        self._is_interesting = f._is_interesting(self._returncode, self._stderr)
        if not self._is_timeout and not self._is_crash:
            thresh = f._exec_time_anomaly.threshold()
            if thresh is not None and self._t_elapsed > thresh:
                self._is_slow = True

    # ── LTL (--ltl) ──────────────────────────────────────────────────

    def _ltl_clear(self) -> None:
        # Empty the event file so it holds exactly this execution's trace.
        ch = getattr(self._f, "ltl", None)
        if ch is not None:
            ch.clear()

    def _ltl_observe(self) -> None:
        # Run the monitor over this execution's events. A fired transition
        # no run has fired before admits the mutant; a violation is a crash
        # (unless the target already crashed: its own report keeps the
        # signature). A timeout's trace is partial and may be doubled by the
        # hang-confirm re-run, so it is skipped.
        ch = getattr(self._f, "ltl", None)
        if ch is None or self._is_timeout:
            return

        obs = ch.collect(self._mutated)
        self._ltl_new = obs.novel
        if obs.violation is None or self._is_crash:
            return

        self._is_crash = True
        self._stderr += f"\n{obs.marker}\n"

    # ── Coverage ─────────────────────────────────────────────────────

    def _edges_now(self):
        """This execution's edges: the scan cache, else a fresh read."""
        f = self._f
        if f._current_edges_cache is not None:
            return f._current_edges_cache
        return f._get_current_edge_set()

    def _push_recurrence(self) -> None:
        # --recurrence: one hash(seed, path) symbol per exec. Path hash is the
        # shim header's rolling hash; 0 off the SHM path (seed channel only).
        rec = getattr(self._f, "_recurrence", None)
        if rec is None:
            return

        shm = self._scanned_shm
        path = shm.read_path_hash() if shm is not None else 0
        novelty = Novelty.NEW if self._has_new_coverage else Novelty.NONE
        rec.push(hash((self._data, path)), novelty)

    def _scan_coverage(self) -> None:
        f = self._f
        # Check new coverage (per-target SHM in multi-target mode).
        # Use is_new_coverage_with_edges() on SHM to get both the boolean
        # and the edge set in one buffer scan, avoiding redundant scans.
        f._current_edges_cache = None  # will be set below if SHM scanned
        # Set from whichever ShmCoverage was actually scanned this iteration.
        # Only the sparse SHM path maintains per-edge maxima; the ptrace
        # bitmap has no counts to take a maximum of, so it stays 0 there.
        active_shm = f._target_shm_covs.get(f.target) if f.multi_targets else f.shm_cov
        if active_shm:
            has_new, edge_ids = active_shm.is_new_coverage_with_edges()
            f._current_edges_cache = edge_ids
            has_new_coverage = has_new
            self._scanned_shm = active_shm
        elif f.multi_targets:
            has_new_coverage = self._poll_multi()
        else:
            has_new_coverage = self._poll_single()

        self._has_new_coverage, f._current_edges_cache = f._confirm_new_coverage(
            self._mutated,
            self._scanned_shm,
            has_new_coverage,
            f._current_edges_cache,
            skip=self._is_crash or self._is_timeout,
        )

    def _poll_multi(self) -> bool:
        # Multi-target without a per-target SHM: every backend, shm included.
        # bool(): `x and x.f()` yields None (not False) when x is
        # None, and that None propagates into `success`, which
        # MonteCarloScheduler.record() feeds to float().
        f = self._f
        return bool(
            (f.ptrace_cov and f.ptrace_cov.is_new_coverage())
            or (f.shm_cov and f.shm_cov.is_new_coverage())
            or (f.pt_cov and f.pt_cov.is_new_coverage())
            or (f.branch_cov and f.branch_cov.is_new_coverage())
        )

    def _poll_single(self) -> bool:
        # No SHM map: the non-SHM backends only.
        f = self._f
        return bool(
            (f.ptrace_cov and f.ptrace_cov.is_new_coverage())
            or (f.pt_cov and f.pt_cov.is_new_coverage())
            or (f.branch_cov and f.branch_cov.is_new_coverage())
        )

    def _gate_explore(self) -> None:
        # Crash exploration (--crash-explore): the only signal is a crash on
        # a new crash path. Everything else reads as boring.
        explorer = self._f._crash_explorer
        if explorer is None:
            return
        self._is_crash = self._is_crash and explorer.observe(self._edges_now())
        self._is_interesting = self._has_new_coverage = self._is_new_max = False
        self._is_cmp_progress = self._is_new_valid_coverage = self._is_slow = False

    # ── Side signals ─────────────────────────────────────────────────

    def _observe(self) -> None:
        self._observe_counts()
        self._cmp_progress()
        self._valid_coverage()
        self._region_liveness()

    def _observe_counts(self) -> None:
        f = self._f
        scanned_shm = self._scanned_shm
        clean = not self._is_crash and not self._is_timeout
        # Sampled per-execution hit mass for the stall reason's effective-edge
        # trend.  Every executed input, not only admitted ones -- a stall is
        # precisely a run of inputs that are never admitted.  Crashes and
        # timeouts are skipped: their counts are truncated executions.
        if f._exec_perplexity.due() and scanned_shm is not None and clean:
            f._exec_perplexity.observe(scanned_shm.get_edge_counts())

        # Power Doppler slow-time sample: this mutant's hit counts, filed
        # under its parent seed. Truncated executions skipped as above.
        if f._doppler is not None and scanned_shm is not None and clean:
            f._doppler.observe(
                f._doppler_key(f._seed_key(self._data)), scanned_shm.get_edge_counts()
            )

        # Performance novelty: an edge whose trip count grew substantially
        # past anything seen before. The hit-count buckets saturate (129 and
        # 10^6 are the same bucket), so this is the only signal that stays
        # live once a loop is merely being spun harder -- which is the
        # algorithmic-complexity bug class the timing channel is actually
        # good for. Suppressed on timeout and crash: a partial execution's
        # counts are truncated, not extreme.
        new_max_edges = 0
        if (
            f._perf_novelty
            and scanned_shm is not None
            and not self._is_timeout
            and not self._is_crash
        ):
            new_max_edges = scanned_shm.new_max_edges
        self._is_new_max = new_max_edges > 0
        if self._is_new_max:
            f._perf_novelty_hits += 1

    def _cmp_progress(self) -> None:
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
        if self._is_timeout or self._is_crash:
            return
        f = self._f
        # Per-PC-site asserted counts are the fine-grained axis; folding
        # by callback merges progress at one site with stagnation at
        # another (measured memcmp (4,3) vs sites (3,3)+(1,0)).  Feed
        # BOTH axes rather than preferring one: the shim's site table is
        # fixed-size and never evicts, so once it saturates the per-site
        # view is a subset and the sites it refused would silently lose
        # their progress signal.  Keys cannot collide -- sites are
        # (callback, pc) tuples, callbacks are strings.
        site_asserted = getattr(f._cmplog, "last_site_asserted", None) if f._cmplog else None
        self._is_cmp_progress = f._record_cmp_progress(f._last_cmp_asserted, site_asserted or {})

    def _valid_coverage(self) -> None:
        # Zest validity channel: coverage reached while the target ACCEPTED
        # the input, tracked in its own map. An input that is valid and
        # covers something no valid input covered before is worth keeping
        # even when the main map has seen those edges already -- reached
        # from the parser's error path, they lead nowhere; reached from an
        # accepted input, they are the semantic stages behind the syntax
        # check. Inert unless --reject-code gave the harness a way to say
        # "rejected".
        f = self._f
        if not f._validity.enabled or self._is_timeout or self._is_crash:
            return
        self._validity = f._validity.classify(self._returncode)
        f._reject_stats.record(f._last_ops_used, self._data, self._mutated, self._validity)
        self._is_new_valid_coverage = f._validity.record(self._validity, self._edges_now())
        if self._is_new_valid_coverage:
            f._validity_admits += 1

    def _region_liveness(self) -> None:
        # Region liveness (item 4, handover_skittercreek_tailslayer_port.md):
        # fold this exec's coverage diff into the per-region
        # LiveBitMaskEstimator for whichever byte the mutation touched.
        # Deliberately unconditional on has_new_coverage above -- that flag
        # only means "globally new edge", which is rare; the liveness
        # estimator needs the far more common "no new edges, but still an
        # observation" samples to ever reach convergence at all. Cheap and
        # skipped outright when there's no edge data or no known parent
        # baseline to diff against.
        f = self._f
        _liveness_parent = getattr(f, "_last_parent_seed", None)
        # --joint-liveness: a two-region probe has no single offset to credit
        # (the mutation loop publishes None for it), so its diff goes to the
        # pair ledger instead of a region's estimator.
        _joint_probe = getattr(f, "_last_joint_probe", None)
        if _joint_probe is not None:
            f._last_joint_probe = None
            if f._current_edges_cache is None or _liveness_parent is None:
                return
            baseline_edges = f._edge_tracker.seed_edges.get(f._seed_key(_liveness_parent))
            if baseline_edges:
                f._operators.record_joint_coverage_diff(
                    _joint_probe[0], _joint_probe[1], baseline_edges, f._current_edges_cache
                )
            return
        _liveness_offset = getattr(f, "_last_mutation_offset", None)
        if f._current_edges_cache is None or _liveness_parent is None or _liveness_offset is None:
            return
        parent_key = f._seed_key(_liveness_parent)
        baseline_edges = f._edge_tracker.seed_edges.get(parent_key)
        if not baseline_edges:
            return
        newly_dead = f._operators.record_coverage_diff(
            _liveness_parent,
            _liveness_offset,
            baseline_edges,
            f._current_edges_cache,
        )
        if newly_dead is not None and f._format_learner:
            region_offset, region_width = newly_dead
            f._format_learner.record_liveness(
                region_offset, region_width, confirmed_dead=True, input_bytes=self._mutated
            )

    def _credit_seed(self) -> None:
        # Bayesian seed quality feedback: record whether this parent seed
        # produced new coverage (Thompson sampling posterior update).
        #
        # The outcome weight is proportional to discovery rarity, per
        # seed_quality.record_outcome's contract.  When the F0 estimator is
        # wired (edge_tracker.enable_f0) it supplies that rarity signal
        # directly: in an unsaturated corpus a new edge is common, so the
        # weight is small; as the estimate saturates toward the exact count
        # each remaining discovery is rare, so the weight rises toward 1.
        # The weight is bounded above by 1.0, so the F0 signal never inflates
        # a posterior beyond the default -- it only ever re-weights.
        f = self._f
        if not (
            f._seed_quality
            or f._seed_canary
            or f._seed_round_robin
            or f._seed_drr
            or f._seed_os_arms
        ):
            return
        data = self._data
        found = bool(self._has_new_coverage)
        parent_key = f._seed_key(data)
        weight = 1.0
        f0_est = f._edge_tracker.estimate_distinct_edges_f0()
        if f0_est is not None:
            observed = f._edge_tracker.get_cumulative_edge_count()
            # Fraction of the estimated total that has been discovered.
            # Near 1 = saturated -> each remaining discovery is rare ->
            # weight rises toward 1.  Near 0 = lots undiscovered ->
            # discoveries are common -> weight shrinks.  Bounded in
            # (0, 1], so the F0 signal never inflates a posterior
            # beyond the default -- it only ever re-weights.
            weight = observed / max(1.0, f0_est)
        if f._seed_quality:
            f._seed_quality.init_seed(parent_key)
            f._seed_quality.record_outcome(parent_key, discovered=found, weight=weight)
        # Same off-policy signal, fed to the seed-arena canary floor
        # (see core/schedulers/seed_canary.py) regardless of which seed
        # strategy actually picked this parent, and independent of
        # whether --bayesian is on -- canary does not need
        # BayesianSeedQuality enabled to track its own posterior.
        if f._seed_canary:
            f._seed_canary.record(parent_key, success=found, weight=weight)
        # Elo-compatibility signal only -- round-robin's own selection
        # ignores it entirely (deterministic cycling), same as its
        # operator-side counterpart's record().
        if f._seed_round_robin:
            f._seed_round_robin.record(parent_key, success=found, weight=weight)
        if f._seed_drr:
            f._seed_drr.record(parent_key, success=found, weight=weight)
        f._record_seed_os_arms(data, found, weight)

    def _feed_models(self) -> None:
        f = self._f
        mutated = self._mutated
        # Credit the cmplog operands this gain is attributable to: the
        # input-to-state matches found in the input, which are the operands
        # the mutators actually had to work with. Crediting the whole
        # resident pool instead (what this did before) cannot rank anything
        # and froze the pool at the first gain -- see mark_coverage_gain.
        if self._has_new_coverage and f._cmplog:
            meta = self._meta
            _credit = meta.get("redqueen_matches", ()) if meta is not None else ()
            f._cmplog.mark_coverage_gain(
                pairs=[(m[1], m[2]) for m in _credit],
                tokens=[m[1] for m in _credit],
            )

        # Dirichlet token posterior: credit the tokens this round inserted
        if f._dict_picker is not None:
            if self._has_new_coverage:
                f._dict_picker.reward(f._dict_scratch_idx)
            else:
                f._dict_picker.clear()

        self._feed_weizz()

        # Record crash MI: I(byte_position; crash_outcome)
        if f._crash_mi:
            f._crash_mi.record(mutated, self._is_crash)

        # Write ablation log row: signal data + outcome
        if f._ablation_file and hasattr(f, "_last_pick_signals"):
            f._write_ablation_row(self._has_new_coverage, self._is_crash)

        self._feed_katz()
        self._feed_entropic()
        self._feed_good_turing()
        self._feed_op_good_turing()
        self._feed_pos_good_turing()

        # Tang low-rank refit. Gated on the interval inside maybe_refit, and
        # placed here rather than on the pick path because the SVD plus the
        # matrix build is 100ms+ at ffmpeg scale -- three orders of magnitude
        # past the per-pick budget the seed-picker profiling established.
        if f._tang is not None:
            f._tang.maybe_refit(f._edge_tracker, f.exec_count)

    def _feed_weizz(self) -> None:
        # Weizz structure tags: once-per-lineage passive collection after a
        # coverage gain, gated by --weizz-tags and max_len. Uses existing
        # cmplog pairs (+ optional colorize taints); no second tracer.
        f = self._f
        mutated = self._mutated
        if (
            self._has_new_coverage
            and f.weizz_tags
            and f._cmplog is not None
            and len(mutated) <= f.weizz_tags_max_len
        ):
            f._maybe_collect_weizz_tags(mutated)

    def _feed_katz(self) -> None:
        # K-Scheduler bitmap sampling runs EVERY exec (beta needs R_i over
        # all mutations); per-seed mask attribution only for corpus-worthy
        # inputs, keyed like EdgeTracker.
        f = self._f
        if getattr(f, "_katz_channel", None) is None:
            return
        katz_key = f._seed_key(self._data) if self._has_new_coverage else None
        f._katz_channel.observe(seed_key=katz_key)

    def _feed_entropic(self) -> None:
        # Every exec, crashes and timeouts included: each mutant spends the
        # parent's budget whatever it hit (libFuzzer NumExecutedMutations).
        strategy = getattr(self._f, "_entropic_seed", None)
        if strategy is None:
            return
        edges = self._edges_now()
        strategy.observe(self._data, edges if isinstance(edges, set | frozenset) else ())

    def _feed_good_turing(self) -> None:
        # Every exec, like entropic: the sample is the parent's mutants.
        strategy = getattr(self._f, "_good_turing_seed", None)
        if strategy is None:
            return
        edges = self._edges_now()
        strategy.observe(self._data, edges if isinstance(edges, set | frozenset) else ())

    def _feed_op_good_turing(self) -> None:
        # Every exec: each distinct operator in the stack is credited with
        # the mutant's edges (see op_good_turing.py for the smear caveat).
        strategy = getattr(self._f, "_op_good_turing", None)
        if strategy is None:
            return
        edges = self._edges_now()
        strategy.observe(
            self._f._last_ops_used, edges if isinstance(edges, set | frozenset) else ()
        )

    def _feed_pos_good_turing(self) -> None:
        # Every exec: edges are stashed here and credited to the mutated bins
        # at settle (see pos_good_turing.py).
        strategy = getattr(self._f, "_pos_good_turing", None)
        if strategy is None:
            return
        edges = self._edges_now()
        strategy.observe(edges if isinstance(edges, set | frozenset) else ())

    # ── Per-seed edges ───────────────────────────────────────────────

    def _record_edges(self) -> None:
        # Record edges for per-seed tracking
        if not self._has_new_coverage:
            return
        f = self._f
        seed_key = f._seed_key(self._data)
        hit_edges, hit_counts = self._hit_edges()
        self._observe_occupation(hit_counts)
        if not hit_edges:
            return
        new = self._commit_edges(seed_key, hit_edges, hit_counts)
        f._strata_observe(self._data, hit_edges)
        if new:
            self._on_new_edges(new, hit_edges)
        meta = self._meta
        if meta is not None and new:
            meta["coverage_edges"] += len(new)
            f._cached_total_edges += len(new)
            meta["momentum"] = 0.8 * meta["momentum"] + 0.2 * 1.0
        elif meta is not None:
            meta["momentum"] = 0.8 * meta["momentum"]
        self._observe_secretary(seed_key, new)

    def _hit_edges(self):
        # Prefer sparse edge set with counts (SHM), fall back to byte bitmap (ptrace)
        # Must read from `scanned_shm` (the segment actually scanned above —
        # the per-target one in multi-target mode), not unconditionally
        # from `self.shm_cov`, which in multi-target mode is a separate,
        # unscanned shared segment.
        f = self._f
        scanned_shm = self._scanned_shm
        if scanned_shm is not None and not f.ptrace_cov:
            hit_counts = f._only_confirmed(scanned_shm.get_edge_counts())
            return set(hit_counts.keys()), hit_counts
        return self._edges_now(), None

    def _observe_occupation(self, hit_counts) -> None:
        f = self._f
        if getattr(f, "_occupation_rarity", None) is None or not hit_counts:
            return
        # Finite-time occupation (Du, Sec. 3): the run's own edge-visit
        # counts as a longitudinal statistical support, distinct from
        # EdgeTracker's horizontal (across-seeds) hit frequencies.
        from fuzzer_tool.core.analyzers.analyzer_occupation import OccupationMeasure

        occ = OccupationMeasure.from_counts(hit_counts)
        f._last_occupation = occ.sparse_snapshot(f._occupation_max_edges)
        f._occupation_rarity.observe(occ)

    def _commit_edges(self, seed_key, hit_edges, hit_counts):
        f = self._f
        scanned_shm = self._scanned_shm
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
        return f._edge_tracker.record_edges(
            seed_key,
            hit_edges,
            target_name=os.path.basename(f.target) if f.multi_targets else "",
            hit_counts=hit_counts,
            stack_depth=stack_depth,
            path_hash=path_hash,
            hw_instructions=f._last_perf_deltas.get("instructions", 0),
            hw_branches=f._last_perf_deltas.get("branches", 0),
            hw_branch_misses=f._last_perf_deltas.get("branch_misses", 0),
        )

    def _on_new_edges(self, new, hit_edges) -> None:
        f = self._f
        f._max_edge_gap = max(f._max_edge_gap, f.exec_count - f._last_new_edge_exec)
        f._last_new_edge_exec = f.exec_count
        f._exec_perplexity.note_new_edge()
        f._last_new_edge_count = len(new)
        f._last_new_edge_ids = list(new)
        f._last_trace_edges = hit_edges
        f._novel_input_count += 1
        f._saturation = None  # invalidate cached saturation
        self._attribute_edges(new)
        # The fold follows the tracker: refit on the arms' own cadence
        # whenever new edges (hence new seed rows) have appeared.
        if f._matrix_substrate is not None:
            f._matrix_substrate.maybe_refit(f._edge_tracker, f.exec_count)
        # The saliency net rides the same discovery hook and the same RefitCadence
        # policy (executions, min seeds before stamping, skip if nothing grew).
        saliency = getattr(f, "_pos_saliency", None)
        if saliency is not None:
            saliency.maybe_refit(f.exec_count)
        # Separate counter for cmplog-involved edge discoveries
        # (cumulative with the op attribution above — cmplog is a
        #  signal source, not a mutation op, so it can overlap).
        if self._cmplog_found:
            f.op_edges["cmplog"] = f.op_edges.get("cmplog", 0.0) + len(new)
        if self._smt_found:
            f.op_edges["smt_solver"] = f.op_edges.get("smt_solver", 0.0) + len(new)
        f._stall_note_coverage(len(new))

    def _attribute_edges(self, new) -> None:
        # Attribute new edges to the operators that ran this iteration.
        # Proportional split: edges ÷ unique ops in _last_ops_used.
        f = self._f
        unique_ops = list(dict.fromkeys(f._last_ops_used))
        if not unique_ops:
            return
        share = len(new) / len(unique_ops)
        for op in unique_ops:
            f.op_edges[op] = f.op_edges.get(op, 0.0) + share
            orientation = classify_operator_name(op).value
            f._ro_rd_edge_counts[orientation] = f._ro_rd_edge_counts.get(orientation, 0.0) + share
        # op_tang: attribute the actual new edge ids (not
        # just a scalar share) to every contributing op --
        # duplication across ops is fine, Tang's math
        # doesn't require exclusive ownership. Kept at this
        # call site (rather than the shared op_rewards loop
        # below) because Tang needs the edge identities
        # themselves, which only exist here; op_katz only
        # needs a success flag, so it's fed from the
        # canonical op_rewards loop instead, alongside
        # replicator/exp3/etc, so it gets the same
        # effective-op-filtered success signal they do
        # rather than a locally-reinvented one.
        if f._op_tang is not None:
            for op in unique_ops:
                f._op_tang.observe_new_edges(op, new)
            f._op_tang.maybe_refit(f.exec_count)
        # op_credit needs the edge identities too: it stores what an
        # operator found and settles the credit against the canonical
        # classes at read time, so a class that splits later is right.
        if f._op_credit is not None:
            for op in unique_ops:
                f._op_credit.observe_new_edges(op, new)

    def _observe_secretary(self, seed_key, new) -> None:
        # Secretary-problem: track seed discovery rate for optimal stopping
        f = self._f
        if not f._secretary or not seed_key:
            return
        if seed_key not in f._seed_secretary:
            f._seed_secretary[seed_key] = SecretaryStopping(
                window_size=f._secretary_window,
                exploration_frac=f._secretary_exploration,
            )
            if len(f._seed_secretary) > SEED_SECRETARY_MAX:
                # Evict oldest 100 entries (dict preserves insertion order)
                for k in list(f._seed_secretary)[:100]:
                    del f._seed_secretary[k]
        meta = self._meta
        fuzz_count = max(meta["fuzz_count"], 1) if meta else 1
        discovery_rate = len(new) / fuzz_count
        f._seed_secretary[seed_key].observe(discovery_rate)

    def _learn_format(self) -> None:
        # Format learner. Runs after the record_edges block above on purpose:
        # coverage_after is len(_global_edge_hits), which only record_edges
        # grows, and _cov_before_fuzz was sampled from the same dict at the
        # top of the round. Recorded before record_edges (where this block
        # used to sit) the two were always equal, so every TimelineEntry
        # carried delta == 0 and the learner's delta statistics never saw a
        # discovery. Only record when coverage actually changes.
        f = self._f
        if not f._format_learner:
            return
        if not (f._last_ops_used and self._has_new_coverage):
            f._prev_edge_set = self._edges_now()
            return
        current_edges = self._edges_now()
        new_edges = set()
        lost_edges = set()
        if hasattr(f, "_prev_edge_set"):
            new_edges = current_edges - f._prev_edge_set
            lost_edges = f._prev_edge_set - current_edges
        f._prev_edge_set = current_edges

        cov_after = (
            len(f._edge_tracker._global_edge_hits)
            if hasattr(f._edge_tracker, "_global_edge_hits")
            else 0
        )
        mutated = self._mutated
        parent_meta = f.seed_meta.get(f._last_parent_seed)
        stride = parent_meta.get("record_stride") if parent_meta else None
        # Set per-format-cluster (not globally): different formats in a
        # multi-format target can have different record strides, so this
        # is routed by `mutated`'s own signature rather than compared
        # against whatever the primary cluster's stride happens to be.
        if stride is not None:
            f._format_learner.set_record_stride(stride, input_bytes=mutated)
        f._format_learner.record_transition(
            input_bytes=mutated,
            mutation_op=f._last_ops_used[0] if f._last_ops_used else "unknown",
            mutation_offset=f._last_mutation_offset,
            mutation_width=len(mutated),
            coverage_before=f._cov_before_fuzz,
            coverage_after=cov_after,
            new_edges=new_edges,
            lost_edges=lost_edges,
        )

    def _track_edges(self) -> None:
        f = self._f
        # Update edge lifetime tracking for every execution
        if f._inprocess_runner or f.ptrace_cov or f.shm_cov:
            current_edges = self._edges_now()
            if current_edges:
                f._edge_tracker.record_edge_lifetimes(current_edges, f.exec_count)

        # Track input-length → edge discovery correlation. The edges this
        # round DISCOVERED (_on_new_edges sets them), not every edge the input
        # hit: LengthEdgeTracker.record credits each edge passed to that
        # length. Passing the whole trace scored a length by how much code
        # its inputs execute -- long inputs win by construction -- and on
        # FFmpeg, where one trace is thousands of context-sensitive edges,
        # it overflowed the 200-edge per-length cap on every call and paid
        # a sort of the whole bucket each time (~2ms per admitted input).
        if self._has_new_coverage and f._length_tracker and f._last_new_edge_ids:
            f._length_tracker.record(len(self._mutated), set(f._last_new_edge_ids))

        self._update_distance()

        # Update annealing progress for directed mode
        if f._distance and f.exec_count > 0:
            # Anneal over first 20% of max_len-scaled iterations
            anneal_target = max(5000, f.max_len * 10)
            f._anneal_progress = min(1.0, f.exec_count / anneal_target)

    def _update_distance(self) -> None:
        # Compute directed distance for targeted fuzzing.  Prefer the
        # runtime average from the SHM tail (AFLGo channel, exact per-BB
        # distances accumulated in the target) when the target carries
        # the distance table; otherwise, on new coverage, the ptrace block
        # addresses. Edge ids are hashes, never blocks: no other source.
        # The value is the MUTANT's: held on the round and attached to the
        # mutant's seed_meta on admission (_tag_distance), never the parent's.
        f = self._f
        if not f._distance:
            return
        runtime_avg = f._read_runtime_avg_distance()
        if runtime_avg is not None:
            self._distance = runtime_avg
            self._note_distance(runtime_avg)
            return
        if not self._has_new_coverage:
            return
        avg_dist = f._block_distance()
        if avg_dist is None:
            return
        self._distance = avg_dist
        if avg_dist < _NO_VALUE_DISTANCE:  # exclude the no-valued-blocks sentinel
            self._note_distance(avg_dist)

    def _tag_distance(self) -> None:
        # Admitted mutant inherits this round's measured distance; aflgo/go
        # rank it by that. A duplicate re-measures the same bytes: same value.
        if self._distance is None:
            return
        meta = self._f.seed_meta.get(self._mutated)
        if meta is None:
            return
        meta["avg_distance"] = self._distance

    def _note_distance(self, value: float) -> None:
        f = self._f
        f._dist_last_value = value
        if f._dist_min_observed is None or value < f._dist_min_observed:
            f._dist_min_observed = value
        if f._dist_max_observed is None or value > f._dist_max_observed:
            f._dist_max_observed = value

    # ── Operator credit ──────────────────────────────────────────────

    def _judge(self) -> None:
        # is_cmp_progress joins the disjunction rather than replacing any
        # part of it: it is a weaker event than an edge (a comparison can be
        # satisfied more often without the branch it guards ever flipping),
        # but it arrives during exactly the stretches where the edge signal
        # is silent, which is when the cmplog-band operators are doing their
        # work and getting paid nothing for it.
        self._success = bool(
            self._is_crash
            or self._is_interesting
            or self._has_new_coverage
            or self._is_slow
            or self._is_new_max
            or self._is_cmp_progress
            or self._is_new_valid_coverage
        )

    def _op_success(self, op: str) -> bool:
        return self._success and (self._effective is None or op in self._effective)

    def _credit_ops(self) -> None:
        f = self._f
        # Gravity splice fit: every splice round is one PPML observation,
        # zero-yield rounds included -- they are most of the signal. Runs
        # after record_edges, which is what sets _last_new_edge_count.
        if f._gravity is not None:
            f._gravity.observe(f._last_new_edge_count if self._has_new_coverage else 0)

        # Per-operator credit. An operator that was selected but left the
        # buffer unchanged cannot have caused this round's outcome, so it
        # must not be recorded as a success -- but it must still be recorded
        # as a failure, or an operator that no-ops forever would never be
        # deprioritised and would keep consuming selection slots.
        self._effective = f._last_ops_effective if f._track_op_effect else None
        self._surprisal_weight = self._surprisal()
        self._count_successes()
        op_rewards = self._op_rewards()
        self._record_slopt(op_rewards)
        self._record_one_fifth()
        self._record_mc(op_rewards)
        self._record_particles(op_rewards)
        self._record_schedulers(op_rewards)
        self._record_contextual(op_rewards)
        self._maybe_chi2()
        self._record_elo()
        self._record_arenas()
        self._record_attribution()

    def _surprisal(self) -> float:
        # Surprisal-weighted reward: discoveries in sparse regions of the
        # coverage bitmap carry more information than discoveries near
        # already-saturated areas. Weight = 1 - density so rare discoveries
        # (low density) get higher credit; saturated regions (high density)
        # get lower credit.
        f = self._f
        if self._success and f._edge_tracker and f._edge_tracker.map_size:
            density = f._edge_tracker.bitmap_density()
            return max(0.05, 1.0 - density)
        return 1.0 if self._success else 0.0

    def _count_successes(self) -> None:
        f = self._f
        if self._success and f._last_havoc_subops:
            # Havoc's inner branches, credited on the same signal the outer
            # bandits use. Trials are counted at application time, so a
            # branch whose guard fails accrues trials without hits and
            # decays -- the same treatment no-op operators get above.
            f._operators.credit_havoc_subops(f._last_havoc_subops)

        if not self._success:
            return
        effective = self._effective
        # Same rule as the bandits: no-op operators didn't earn this.
        for op in effective if effective is not None else set(f._last_ops_used):
            f.op_success[op] = f.op_success.get(op, 0) + 1
            # Numerator for the mutate-regime rate. It has to be
            # restricted to the same selections as op_applicable or the
            # two do not divide: a format op that synthesised a file
            # from scratch and found an edge that way is a success on a
            # selection op_applicable never counted, which produced
            # successes against a zero denominator in the first cut of
            # this change.
            if op in self._applicable_now:
                f.op_success_applicable[op] = f.op_success_applicable.get(op, 0) + 1
        if self._cmplog_found:
            f.op_success["cmplog"] = f.op_success.get("cmplog", 0) + 1
        if self._smt_found:
            f.op_success["smt_solver"] = f.op_success.get("smt_solver", 0) + 1

    def _op_rewards(self) -> list[tuple[str, bool, float]]:
        f = self._f
        # Per-operator outcome and reward, computed once. Every scheduler
        # below scored the same operator identically, so this was seven
        # copies of the same dedup-and-weight loop.
        # Bounded to [0, 1]: the contract every consumer below documents --
        # KL-UCB's Bernoulli divergence and Exp3's exponent assume it, the
        # UCB widths take b=1.0 as the range, and the Beta posteriors (mc,
        # hierarchical) add the weight as pseudo-successes, so a weight of
        # 15 was fifteen discoveries.
        op_rewards = []
        for op in dict.fromkeys(f._last_ops_used):
            ok = self._op_success(op)
            w = f._cost_adjusted_weight(op, self._surprisal_weight if ok else 0.0)
            op_rewards.append((op, ok, min(1.0, w)))

        # Class-deduplicated shaping, applied once to the finished list because
        # the partition does not depend on which operator ran. Applying it here
        # rather than inside each scheduler is the whole point: every consumer of
        # op_rewards below sees the same shaped number, so a paired run moves one
        # variable. Kept as a pure transform (`_apply_reward_shape`) so it can be
        # driven by a test without a live campaign.
        op_rewards = _apply_reward_shape(op_rewards, f._credit_reward_shape())
        return _apply_reward_shape(op_rewards, f._continuum_reward_shape())

    def _record_slopt(self, op_rewards) -> None:
        # SLOPT: credit the batch exponent drawn for this round's operator
        # with that operator's outcome. The scheme applies one operator per
        # round, so the round's reward is the arm's reward.
        f = self._f
        if f._slopt is None or f._last_slopt_arm is None:
            return
        s_op, s_len, s_exp = f._last_slopt_arm
        for op, ok, w in op_rewards:
            if op == s_op:
                f._slopt.record(s_op, s_len, s_exp, ok, weight=w)
                break

    def _record_one_fifth(self) -> None:
        # 1/5 rule: a round "beats its parent" when it finds new coverage.
        f = self._f
        if f._one_fifth is None:
            return
        f._one_fifth.record(FifthOutcome.HIT if self._has_new_coverage else FifthOutcome.MISS)

    def _record_mc(self, op_rewards) -> None:
        f = self._f
        if not (f.mc and f.mc_bandit):
            return
        for op, ok, w in op_rewards:
            f.mc.record(op, ok, weight=w)
            f.mc.record_brier(op, ok, weight=w)
            # Secretary-problem: track operator quality for optimal stopping
            if not f._secretary:
                continue
            if op not in f._op_secretary:
                f._op_secretary[op] = SecretaryStopping(
                    window_size=f._secretary_window,
                    exploration_frac=f._secretary_exploration,
                    min_observations=50,
                )
            a = f.mc.arm_alpha.get(op, 1.0)
            b = f.mc.arm_beta.get(op, 1.0)
            f._op_secretary[op].observe(a / (a + b))

    def _particle_draws(self, op_rewards):
        """Yield ``(op, ok, w, particle_id)`` once per op, first particle wins."""
        f = self._f
        rewards_by_op = {op: (ok, w) for op, ok, w in op_rewards}
        seen = set()
        for op, pid in zip(f._last_ops_used, f._last_mopt_particles, strict=False):
            if op in seen or op not in rewards_by_op:
                continue
            ok, w = rewards_by_op[op]
            seen.add(op)
            yield op, ok, w, pid

    def _record_particles(self, op_rewards) -> None:
        # On-policy learners: their update is only valid for operators they
        # drew themselves, so they learn from the rounds they selected and
        # from nothing else. Everything else below learns from every round,
        # which is sound for them -- a sample mean or Beta posterior does not
        # care who pulled the arm -- and measurably helps: fed only their own
        # draws, a 4-scheduler Elo portfolio lost ~8% of discoveries on a
        # 150-arm synthetic campaign.
        #
        # - MOpt credits the particle that drew each op; another scheduler's
        #   draw has no particle, and record(particle_id=None) spreads it over
        #   every particle. This was already gated under Elo; with Elo off it
        #   still recorded, whichever scheduler had precedence.
        # - Exp3's estimate r / p_i is unbiased only when p_i is the
        #   probability Exp3 itself drew i with. Fed another scheduler's draw
        #   it divided by a _last_probs left over from the last round Exp3
        #   selected -- possibly thousands of rounds stale.
        # - CMA-ES closes a generation after generation_size records. Counting
        #   every scheduler's records closed it after ~generation_size / N of
        #   its own evaluations, and credited its current candidate whenever
        #   another scheduler happened to pick the op that candidate had last
        #   drawn.
        f = self._f
        selector = f._op_selector
        if f._mopt and selector == "mopt":
            # MOpt is separate: it needs the particle each operator was drawn
            # from, so it pairs each op with its first particle rather than
            # iterating the deduped list.
            for op, ok, w, pid in self._particle_draws(op_rewards):
                f._mopt.record(op, ok, particle_id=pid, weight=w)

        if f._op_firefly and selector == "op_firefly":
            # Same reasoning as MOpt just above: firefly needs the id of
            # the firefly that drew each op, not a broadcast record() over
            # every firefly, or fitness-proportional selection could never
            # differentiate them. It reuses the same _last_mopt_particles
            # list select_op() populated with firefly ids for this round
            # (see operators.py's dispatch chain).
            for op, ok, w, fid in self._particle_draws(op_rewards):
                f._op_firefly.record(op, ok, firefly_id=fid, weight=w)

    def _record_schedulers(self, op_rewards) -> None:
        f = self._f
        selector = f._op_selector
        # Schedulers sharing the record(op, success, weight=...) signature.
        for scheduler in (
            f._replicator if f._use_replicator else None,
            f._exp3 if selector == "exp3" else None,
            f._exp4 if selector == "exp4" else None,
            f._eps_greedy,
            f._hierarchical,
            f._gp_ucb,
            f._bo_gp_ucb,
            f._cmaes if selector == "cmaes" else None,
            f._ducb,
            f._swucb,
            f._kl_ducb,
            f._kl_swucb,
            f._cucb,
            f._cusum_ucb,
            f._fewa,
            f._fpl,
            # On-policy, like exp3/cmaes above: the importance weight is only
            # unbiased against the distribution that produced the draw, so a
            # round another scheduler selected must not reach it.
            f._corral if selector == "corral" else None,
            f._tsallis if selector == "tsallis" else None,
            f._kalman_ts,
            f._gamma_poisson,
            f._ids,
            f._phe,
            f._exp3_ix if selector == "exp3_ix" else None,
            f._regret_matching if selector == "regret_matching" else None,
            f._automaton if selector == "automaton" else None,
            f._ant_colony,
            f._gradient,
            f._whittle,
            f._successive_elim,
            f._consolidated_v1,
            f._consolidated_v2,
            f._moss,
            f._las_vegas,
            f._bayes_ucb,
            f._canary,
            f._op_katz,
            f._op_kuramoto,
            f._op_tang,
            f._op_kruskal_count,
            f._op_credit,
            f._op_tpe,
            f._op_strata,
            f._op_stride,
            f._op_p2c,
            f._op_good_turing,
            f._softmax,
            f._topk,
        ):
            if scheduler is None:
                continue
            for op, ok, w in op_rewards:
                scheduler.record(op, ok, weight=w)

        # CUCB batches the round rather than updating per operator, so the
        # superarm is only complete once the loop above has run.
        if f._cucb:
            f._cucb.settle_round()

    def _record_contextual(self, op_rewards) -> None:
        f = self._f
        if f._contextual:
            # LinUCB takes a feature vector rather than a success flag.
            for op, ok, w in op_rewards:
                f._contextual.record(op, f._operators._context_vector(op), w if ok else 0.0)

        if not f._c2ucb:
            return
        # Stage every operator's outcome+context into the open round;
        # settle_round() computes the actual per-arm credit once the
        # whole round's membership is known (see op_c2ucb.py's module
        # docstring for why this can't happen per-record like
        # ContextualLinUCBScheduler above).
        for op, ok, w in op_rewards:
            f._c2ucb.record(op, f._operators._context_vector(op), ok, weight=w)
        # When _track_op_effect is on, `ok` above is already per-op
        # attributed truth (see _op_success), not a broadcast outcome --
        # bypass C2UCB's own inclusion-contrast entirely and hand it
        # that truth directly. This is the documented difference
        # between C2UCB actually working and merely running; see
        # "Context dilution" in op_c2ucb.py.
        c2ucb_credits = (
            {op: (w if ok else 0.0) for op, ok, w in op_rewards} if f._track_op_effect else None
        )
        f._c2ucb.settle_round(credits=c2ucb_credits)

    def _maybe_chi2(self) -> None:
        # Chi-squared operator heterogeneity test
        f = self._f
        if (
            f._chi2_operator_interval > 0
            and f.exec_count > 0
            and f.exec_count % f._chi2_operator_interval == 0
        ):
            try:
                f._run_chi2_operator_test()
            except Exception as ex:
                log.debug("Chi-squared operator test failed: %s", ex)

    def _record_elo(self) -> None:
        # Elo: record matches between operators that were used
        # `>= 1`, not `>= 2`: a SLOPT round applies one operator 2**t times,
        # so its deduplicated set always has one member. The old guard
        # skipped every such round, and record_round's cross-round path is
        # the one that can still score it (see there).
        f = self._f
        if not (f._use_elo and f._elo and f._last_ops_used):
            return
        unique_ops = list(dict.fromkeys(f._last_ops_used))  # preserve order, dedup
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
        winners = set(f._last_ops_effective) if self._success else set()
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
        if winners and f._last_new_edge_count:
            # Split across winners, not all selected operators: a no-op
            # operator contributed no edges, and including it in the
            # denominator diluted everyone else's share.
            raw_share = f._last_new_edge_count / len(winners)
            edge_counts = {op: f._cost_adjusted_weight(op, raw_share) for op in winners}
        f._elo.record_round(unique_ops, winners, edge_counts=edge_counts, crash=self._is_crash)
        # Apply periodic decay
        f._elo_decay_counter += 1
        if f._elo_decay_counter >= f._elo_decay_interval:
            f._elo_decay_counter = 0
            f._elo.apply_decay()
            if f._use_canary and f._canary:
                f._check_canary_inspection()

    def _record_arenas(self) -> None:
        f = self._f
        score = self._surprisal_weight if self._success else 0.0
        # Meta-elo: record operator strategy-level match. Only when the current
        # strategy is a real selectable scheduler (random_stall is excluded, so
        # stall recovery never accrues phantom matches)
        if f._use_elo and f._elo and f._meta_strategy:
            f._record_operator_strategy_matches(score)

        # Meta-elo: record seed strategy-level match
        if f._use_elo and f._elo and f._seed_strategy:
            f._record_seed_strategy_matches(score)

        # Position arena matches and burn-front credit
        f._settle_positions(Outcome.GAIN if self._success else Outcome.MISS, self._surprisal_weight)
        f._settle_targets(self._success, self._surprisal_weight)

    def _record_attribution(self) -> None:
        f = self._f
        if f._use_shapley and f._shapley:
            new_edges = f._get_current_edge_set()
            effective = self._effective
            if new_edges:
                f._shapley.record(
                    set(effective) if effective is not None else set(f._last_ops_used),
                    len(new_edges),
                    new_edges,
                )
            elif f.exec_count > 0 and f._last_ops_used:
                # Even with no edges, record a zero to track operator impact
                f._shapley.record(set(f._last_ops_used), 0, set())

        if f._use_mi and f._mi:
            current_edges = f._get_current_edge_set()
            if current_edges:
                f._mi.record(self._data, current_edges, f.map_size)

        if f._use_transfer_entropy and f._te:
            self._record_te()

    def _record_te(self) -> None:
        f = self._f
        current_edges = f._get_current_edge_set()
        if not current_edges:
            return
        data = self._data
        f._te_input_history.append(data[:64] if len(data) > 64 else data)
        f._te_edge_history.append(current_edges)
        if len(f._te_input_history) > f._te_history_max:
            f._te_input_history = f._te_input_history[-f._te_history_max :]
            f._te_edge_history = f._te_edge_history[-f._te_history_max :]
        # Update byte→edge causal map periodically
        if len(f._te_input_history) % 100 == 0 and len(f._te_input_history) > 50:
            f._update_te_causal_map()
            # Same cadence: feed the causal-sector graph from the TE
            # edge history already being maintained above, rather
            # than recomputing full pairwise TE every iteration.
            if getattr(f, "_causal_sector", None) is not None:
                f._update_causal_sector()

    # ── Exits ────────────────────────────────────────────────────────

    def _on_crash(self) -> bool:
        f = self._f
        mutated = self._mutated
        f.crash_count += 1
        self._triage_crash()
        crash_name = f.save_crash(mutated, self._returncode, self._stderr)
        f._prune_crash_data()
        # Generate GDB/strace trace report if enabled
        if f._tracer and crash_name:
            report = f._tracer.trace(mutated, self._returncode)
            f._tracer.save_report(report, str(f.crashes_dir), crash_name)
        if f.mc and f.mc_cem:
            f.mc.add_elite(mutated, 3, temperature=f._temperature)
            f.mc.maybe_refit()
        # Schedule crash replay for reproducibility check.
        #
        # The key is the signature save_crash() counted this crash under,
        # published on the fuzzer rather than re-derived here. The old
        # `self.crash_sigs.get(crash_name, crash_name)` looked a FILENAME
        # up in a signature-keyed dict: it always missed, so the key
        # became the filename, and every downstream consumer that keys by
        # signature (_prune_crash_data, the reproducibility report) was
        # working against a different key space (finding #22).
        sig = f._last_crash_signature
        if f.replay_n > 0 and sig and sig not in f._crash_replays:
            f._crash_replays[sig] = []
        # Schedule sanitizer replay: re-run crash on ASAN/UBSAN targets
        if (f.asan_target or f.ubsan_target) and sig and sig not in f._crash_sanitizer_replays:
            f._crash_sanitizer_replays[sig] = {
                "data": mutated,
                "asan": None,
                "ubsan": None,
            }
        f._record_fluctuation_observation("crash", f._get_current_edge_set())
        f._maybe_periodic_minimize()
        return True

    def _queue_variant(self) -> None:
        # Crash exploration: a surviving variant is also a seed, so the next
        # mutants start from it.
        f = self._f
        if f._crash_explorer is None:
            return
        before = len(f.corpus)
        f.save_to_corpus(self._mutated, parent=self._data)
        self._tag_distance()
        f._record_lineage_insert(self._mutated, self._data, before)

    def _triage_crash(self) -> None:
        # direct_lite crashes carry no fault address (no ptrace, and the
        # guarded call reports only the signal). Re-run the input once
        # through the ptrace-attached loader to capture si_addr + regs.
        f = self._f
        if not (
            f._last_fault_addr is None
            and f._triage_ok is not False
            and f._inprocess_runner is not None
            and f._inprocess_runner.direct_lite
            and str(f.target).lower().endswith((".so", ".dylib", ".dll"))
        ):
            return
        if f._triage_ok is None:
            f._triage_ok = ptrace_available()
        if f._triage_ok:
            try:
                TargetRunner(f)._run_triage_ptrace(self._mutated)
            except Exception as e:
                log.debug("crash triage failed: %s", e)

    def _admits(self) -> bool:
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
        return bool(
            self._is_interesting
            or self._has_new_coverage
            or self._is_new_max
            or self._is_cmp_progress
            or self._is_new_valid_coverage
            or self._ltl_new
        )

    def _admit(self) -> bool:
        f = self._f
        data = self._data
        mutated = self._mutated
        _corpus_len_before = len(f.corpus)
        f.save_to_corpus(mutated, parent=data)
        self._tag_distance()
        # Validity is a property of this execution, so it is recorded
        # here rather than reconstructed later: the seed picker reads it
        # off the metadata and nothing re-runs the input to ask again.
        if self._validity is not Validity.UNKNOWN:
            meta = f.seed_meta.get(mutated)
            if meta is not None:
                meta["valid"] = self._validity is Validity.VALID
        f._record_lineage_insert(mutated, data, _corpus_len_before)
        self._note_novelty()
        f._record_entropy_gradient_credit(mutated, data, _corpus_len_before)
        self._feed_population()
        self._analyze_sensitivity()
        self._probe_uninit()
        # Coverage-guided trimming: try to minimize inputs that hit new edges
        if self._has_new_coverage and len(mutated) > 10:
            f._trim_new_coverage(mutated, data)
        if f.mc and f.mc_cem:
            f.mc.add_elite(mutated, 2, temperature=f._temperature)
            f.mc.maybe_refit()
        # Periodic minimization based on edge stats
        f._maybe_periodic_minimize(dedup=True)
        f._record_fluctuation_observation("success", f._get_current_edge_set())
        return True

    def _note_novelty(self) -> None:
        # --grimoire: the edges this input was admitted for are what
        # generalization must preserve (consumed on its first round).
        stage = getattr(self._f, "_grimoire", None)
        if stage is not None and self._has_new_coverage:
            stage.note(self._mutated, self._f._last_new_edge_ids)

    def _probe_uninit(self) -> None:
        # --uninit-probe: new coverage is where a fresh decoder path may emit
        # heap bytes it never wrote. Three extra execs per admission.
        probe = self._f._uninit_probe
        if probe is not None:
            probe.check(self._mutated)

    def _last_seed_edges(self) -> int:
        tracker = self._f._edge_tracker
        if not hasattr(tracker, "_last_seed_key"):
            return 0
        return len(tracker.seed_edges.get(tracker._last_seed_key, set()))

    def _feed_population(self) -> None:
        f = self._f
        mutated = self._mutated
        # GA: add new-coverage individual to population
        if f.ga and self._has_new_coverage:
            edge_count = self._last_seed_edges()
            ind = f.ga.on_fuzz_result(mutated, True, edge_count, f._edge_tracker)
            if ind is not None:
                f.ga.add_to_population(ind)
        # QEA: amplitude rotation feedback + new individual on coverage
        if f.qea:
            edge_count = self._last_seed_edges()
            qea_ind = f.qea.on_fuzz_result(
                mutated, self._has_new_coverage, edge_count, f._edge_tracker
            )
            if qea_ind is not None:
                f.qea.add_to_population(qea_ind)

    def _analyze_sensitivity(self) -> None:
        # Analyze byte sensitivity for seeds that found new coverage (optional)
        f = self._f
        if not (self._has_new_coverage and f.shm_cov and f._use_sensitivity):
            return
        try:
            edges = f.shm_cov.get_edge_ids()
            if edges:

                def _exec_fn(data):
                    rc, _ = f._run_target(data)
                    if f.shm_cov:
                        return f.shm_cov.get_edge_ids()
                    return set()

                f._sensitivity.analyze_seed(self._mutated, edges, _exec_fn)
        except Exception:
            pass

    def _on_boring(self) -> bool:
        f = self._f
        mutated = self._mutated
        # ── Metropolis acceptance for non-improving / non-crashing inputs ──
        if f._metropolis and f._anneal_budget > 0 and not self._is_timeout and self._metropolis():
            return True

        # Periodic minimization (also for non-interesting iterations)
        f._maybe_periodic_minimize()

        # GA: trigger generation boundary for non-coverage iterations
        if f.ga:
            f.ga.on_fuzz_result(mutated, False, 0, f._edge_tracker)

        # QEA: trigger generation boundary and rotation for non-coverage
        if f.qea:
            f.qea.on_fuzz_result(mutated, False, 0, f._edge_tracker)

        f._record_fluctuation_observation("boring", f._get_current_edge_set())
        return False

    def _metropolis(self) -> bool:
        f = self._f
        data = self._data
        mutated = self._mutated
        mutant_edges = f._get_current_edge_set()
        p_accept = f._metropolis_accept_p(data, mutant_edges)
        if f._rng.random() >= p_accept:
            return False
        _corpus_len_before = len(f.corpus)
        f.save_to_corpus(mutated, parent=data)
        self._tag_distance()
        f._record_lineage_insert(mutated, data, _corpus_len_before)
        f._record_entropy_gradient_credit(mutated, data, _corpus_len_before)
        if f.mc and f.mc_cem:
            f.mc.add_elite(mutated, 1, temperature=f._temperature)
            f.mc.maybe_refit()
        f._record_fluctuation_observation("success", mutant_edges)
        f._maybe_periodic_minimize()
        return True
