"""Parallel fuzzing: fork N workers sharing corpus/crashes directories."""

import multiprocessing
import os
import signal
import time
from pathlib import Path


def _worker_main(
    worker_id: int,
    result_queue: multiprocessing.Queue,
    target: str,
    corpus_dir: str,
    crashes_dir: str,
    max_len: int,
    timeout: float,
    mutations_per_input: int,
    use_coverage: bool,
    deep_coverage: bool,
    max_bps: int,
    dictionary: list[bytes],
    file_mode: bool,
    target_args: list[str],
    markov_order: int,
    markov_generate: bool,
    mc_bandit: bool,
    mc_cem: bool,
    mc_elite_frac: float,
    mc_refit_interval: int,
    stats_file: str | None,
    stats_interval: int,
    coverage_report: str | None,
    iterations: int,
    sync_interval: int,
    stop_event: multiprocessing.Event,
    rng_seed: int = 42,
    secretary: bool = False,
    secretary_window: int = 500,
    secretary_exploration: float = 0.368,
    overlap_density: bool = False,
    overlap_density_mode: str = "modifier",
    overlap_min_jaccard: float = 0.25,
    overlap_density_blend: float = 0.5,
    exp3: bool = False,
    exp3_gamma: float = 0.1,
    eps_greedy: bool = False,
    eps_greedy_epsilon0: float = 1.0,
    eps_greedy_decay: float = 0.9995,
    hierarchical_bandit: bool = False,
    gp_ucb: bool = False,
    ducb: bool = False,
    swucb: bool = False,
    cucb: bool = False,
    gp_length_scale: float = 1.0,
    gp_beta: float = 2.0,
    asan_target: str | None = None,
    ubsan_target: str | None = None,
    chi2_operator_interval: int = 0,
    markov_blend: float = 0.0,
    mc_decay_interval: int = 100,
    pairwise_blend: float = 0.0,
    lineage: bool = False,
    lineage_backtrack: bool = False,
    contextual: bool = False,
    contextual_alpha: float = 1.0,
    contextual_lambda: float = 1.0,
    resize_map_on_stall: bool = True,
    n_workers: int = 1,
    fractal_partition: bool = False,
    fractal_partition_depth: int = 3,
    fractal_diversity: bool = False,
    fractal_diversity_depth: int = 3,
    fractal_diversity_bonus: float = 1.3,
):
    """Entry point for each fuzzing worker process."""
    from fuzzer_tool.services.fuzzer import Fuzzer

    prefix = f"[worker-{worker_id}]"

    worker_corpus = Path(corpus_dir) / f".w{worker_id}"
    worker_corpus.mkdir(parents=True, exist_ok=True)

    worker_stats = None
    if stats_file:
        p = Path(stats_file)
        worker_stats = str(p.with_name(f"{p.stem}_w{worker_id}{p.suffix}"))

    fuzzer = Fuzzer(
        target=target,
        corpus_dir=str(worker_corpus),
        crashes_dir=crashes_dir,
        max_len=max_len,
        timeout=timeout,
        mutations_per_input=mutations_per_input,
        use_coverage=use_coverage,
        deep_coverage=deep_coverage,
        max_bps=max_bps,
        dictionary=dictionary,
        file_mode=file_mode,
        target_args=target_args,
        markov_order=markov_order,
        markov_generate=markov_generate,
        markov_blend=markov_blend,
        mc_bandit=mc_bandit,
        mc_cem=mc_cem,
        mc_elite_frac=mc_elite_frac,
        mc_refit_interval=mc_refit_interval,
        mc_decay_interval=mc_decay_interval,
        pairwise_blend=pairwise_blend,
        stats_file=worker_stats,
        stats_interval=stats_interval,
        coverage_report=coverage_report,
        seed=rng_seed + worker_id,
        secretary=secretary,
        secretary_window=secretary_window,
        secretary_exploration=secretary_exploration,
        overlap_density=overlap_density,
        overlap_density_mode=overlap_density_mode,
        overlap_min_jaccard=overlap_min_jaccard,
        overlap_density_blend=overlap_density_blend,
        exp3=exp3,
        exp3_gamma=exp3_gamma,
        eps_greedy=eps_greedy,
        eps_greedy_epsilon0=eps_greedy_epsilon0,
        eps_greedy_decay=eps_greedy_decay,
        hierarchical_bandit=hierarchical_bandit,
        gp_ucb=gp_ucb,
        ducb=ducb,
        swucb=swucb,
        cucb=cucb,
        gp_length_scale=gp_length_scale,
        gp_beta=gp_beta,
        asan_target=asan_target,
        ubsan_target=ubsan_target,
        chi2_operator_interval=chi2_operator_interval,
        lineage=lineage,
        lineage_backtrack=lineage_backtrack,
        contextual=contextual,
        contextual_alpha=contextual_alpha,
        contextual_lambda=contextual_lambda,
        resize_map_on_stall=resize_map_on_stall,
        fractal_diversity=fractal_diversity,
        fractal_diversity_depth=fractal_diversity_depth,
        fractal_diversity_bonus=fractal_diversity_bonus,
    )

    print(f"{prefix} Started (target={target})")

    i = 0
    last_sync = time.time()
    try:
        while not stop_event.is_set():
            fuzzer._flush_pending_minimize()  # deferred minimize from save_to_corpus in sync
            if iterations and i >= iterations:
                break

            now = time.time()
            if now - last_sync >= sync_interval:
                _sync_corpus_in(
                    Path(corpus_dir),
                    fuzzer,
                    self_dir=worker_corpus,
                    worker_id=worker_id if fractal_partition else None,
                    n_workers=n_workers if fractal_partition else None,
                    fractal_depth=fractal_partition_depth,
                )
                last_sync = now

            seed_data = fuzzer._pick_seed()
            fuzzer.fuzz_one(seed_data)
            i += 1

            if i % 100 == 0:
                elapsed = time.time() - fuzzer.start_time
                eps = fuzzer.exec_count / elapsed if elapsed > 0 else 0
                print(
                    f"\r{prefix} execs: {fuzzer.exec_count} | corpus: {len(fuzzer.corpus)} | "
                    f"crashes: {fuzzer.crash_count} | eps: {eps:.0f}",
                    end="",
                    flush=True,
                )
    except KeyboardInterrupt:
        pass

    fuzzer._dump_stats()
    fuzzer._dump_coverage_report()

    elapsed = time.time() - fuzzer.start_time
    eps = fuzzer.exec_count / elapsed if elapsed > 0 else 0
    print(
        f"\n{prefix} Done. execs={fuzzer.exec_count} crashes={fuzzer.crash_count} "
        f"eps={eps:.0f} corpus={len(fuzzer.corpus)}"
    )
    result_queue.put(
        {
            "worker_id": worker_id,
            "exec_count": fuzzer.exec_count,
            "crash_count": fuzzer.crash_count,
            "corpus_size": len(fuzzer.corpus),
            "timeout_count": fuzzer.timeout_count,
            "edges": fuzzer.shm_cov.cumulative_edges if fuzzer.shm_cov else 0,
            "elapsed": elapsed,
        }
    )


# Per-sibling set of seed FILENAMES already offered to this worker.
#
# This was an integer cursor into a `sorted()` listing. Seed filenames are
# `id_<hash>`, so their sort order is effectively random with respect to
# creation order: a seed written later can sort before the cursor, and every
# such seed was skipped permanently. Keying on the name has no such failure
# mode, and the name is exactly what identifies the entry.
_sync_seen: dict[str, set[str]] = {}

# Bound the bookkeeping. Reached only on a corpus far larger than anything
# these workers hold in memory; dropping the set costs a re-offer, which
# `seen_hashes` and `save_to_corpus` both dedup.
_SYNC_SEEN_MAX = 200_000


def _sync_corpus_in(
    parent_dir: Path,
    fuzzer,
    max_new: int = 50,
    self_dir: Path | None = None,
    *,
    worker_id: int | None = None,
    n_workers: int | None = None,
    fractal_depth: int = 3,
):
    """Pull new corpus entries from sibling worker dirs.

    Seeds live at ``<worker>/seeds/<hh>/id_<hash>`` (plus the
    ``irreplaceable/`` and ``crashing/`` subtrees), written there by
    ``adapters.filesystem.save_to_corpus``. This used to list the worker
    directory NON-recursively and take the files, so it transferred zero
    seeds and imported the one top-level file that does exist —
    ``state.pkl.gz`` — as a garbage seed into every sibling.

    ``pruned/`` is excluded to match ``load_corpus``: those entries were
    deliberately dropped by the sibling, and re-importing them would undo
    its minimization. Delta records under ``deltas/`` are also skipped —
    a delta names a parent hash this worker may not hold, so it is not
    self-contained enough to transfer.

    Args:
        parent_dir: The shared corpus directory holding the ``.wN`` dirs.
        fuzzer: The importing worker's Fuzzer.
        max_new: Cap on seeds imported per call.
        self_dir: This worker's own corpus dir, skipped if given.
        worker_id: This worker's index. When given together with
            ``n_workers``, sync is restricted to fractal Voronoi
            partitioning (Approach C, ``core/parallel_fractal_partition.py``):
            a sibling's seed is only imported if this worker owns its root
            cell, or the seed crosses a fractal boundary. ``None`` (the
            default) keeps the original fully-shared behavior.
        n_workers: Total worker count for the same partitioning. Ignored
            unless ``worker_id`` is also given.
        fractal_depth: Fractal layer depth for the partition, only used
            when partitioning is active.
    """
    from fuzzer_tool.adapters.filesystem import hash_data

    partition = None
    if worker_id is not None and n_workers:
        from fuzzer_tool.core.parallel_fractal_partition import accept_for_worker

        partition = (worker_id, n_workers, fractal_depth)

    added = 0
    own = Path(self_dir).name if self_dir is not None else None

    for sibling_dir in sorted(parent_dir.iterdir()):
        if added >= max_new:
            break
        if not sibling_dir.is_dir() or not sibling_dir.name.startswith(".w"):
            continue
        if own is not None and sibling_dir.name == own:
            continue

        seeds_root = sibling_dir / "seeds"
        if not seeds_root.is_dir():
            continue

        key = str(sibling_dir)
        seen_names = _sync_seen.setdefault(key, set())
        if len(seen_names) > _SYNC_SEEN_MAX:
            seen_names.clear()

        for entry in sorted(seeds_root.rglob("id_*")):
            if added >= max_new:
                break
            if "pruned" in entry.parts or not entry.is_file():
                continue
            name = entry.name
            if name in seen_names:
                continue

            # The filename IS the content hash, so a seed this worker
            # already holds can be skipped without reading it.
            name_hash = name[3:]
            if name_hash in fuzzer.seen_hashes:
                seen_names.add(name)
                continue

            try:
                data = entry.read_bytes()
            except OSError:
                continue

            # A sibling's write is not atomic, so a truncated read is
            # possible. Verify against the name before importing, and leave
            # the entry unmarked so the next round retries it.
            if hash_data(data) != name_hash:
                continue

            if partition is not None:
                wid, n, depth = partition
                if not accept_for_worker(data, wid, n, depth):
                    seen_names.add(name)  # not ours; do not keep re-checking it
                    continue

            seen_names.add(name)
            fuzzer.save_to_corpus(data)
            added += 1


def run_parallel(
    target: str,
    jobs: int,
    corpus_dir: str,
    crashes_dir: str,
    max_len: int = 4096,
    timeout: float = 5,
    mutations_per_input: int = 8,
    use_coverage: bool = False,
    deep_coverage: bool = False,
    max_bps: int = 50000,
    dictionary: list[bytes] | None = None,
    file_mode: bool = False,
    target_args: list[str] | None = None,
    markov_order: int = 0,
    markov_generate: bool = False,
    mc_bandit: bool = False,
    mc_cem: bool = False,
    mc_elite_frac: float = 0.1,
    mc_refit_interval: int = 1000,
    stats_file: str | None = None,
    stats_interval: int = 1000,
    coverage_report: str | None = None,
    iterations: int = 0,
    sync_interval: int = 30,
    seed: int = 42,
    secretary: bool = False,
    secretary_window: int = 500,
    secretary_exploration: float = 0.368,
    overlap_density: bool = False,
    overlap_density_mode: str = "modifier",
    overlap_min_jaccard: float = 0.25,
    overlap_density_blend: float = 0.5,
    exp3: bool = False,
    exp3_gamma: float = 0.1,
    eps_greedy: bool = False,
    eps_greedy_epsilon0: float = 1.0,
    eps_greedy_decay: float = 0.9995,
    hierarchical_bandit: bool = False,
    gp_ucb: bool = False,
    ducb: bool = False,
    swucb: bool = False,
    cucb: bool = False,
    gp_length_scale: float = 1.0,
    gp_beta: float = 2.0,
    asan_target: str | None = None,
    ubsan_target: str | None = None,
    chi2_operator_interval: int = 0,
    markov_blend: float = 0.0,
    mc_decay_interval: int = 100,
    pairwise_blend: float = 0.0,
    lineage: bool = False,
    lineage_backtrack: bool = False,
    contextual: bool = False,
    contextual_alpha: float = 1.0,
    contextual_lambda: float = 1.0,
    resize_map_on_stall: bool = True,
    fractal_partition: bool = False,
    fractal_partition_depth: int = 3,
    fractal_diversity: bool = False,
    fractal_diversity_depth: int = 3,
    fractal_diversity_bonus: float = 1.3,
):
    """Launch N parallel fuzzer workers sharing the same corpus directory.

    Each worker writes to its own corpus subdirectory (.w0, .w1, ...) and
    periodically pulls new entries from siblings. Crashes go to the shared
    crashes directory.
    """
    target_name = os.path.basename(os.path.abspath(target))
    print(f"[*] Parallel fuzzing: {jobs} workers on {target_name}")
    print(f"[*] Corpus: {corpus_dir}")
    print(f"[*] Crashes: {crashes_dir}")
    print(f"[*] Sync interval: {sync_interval}s")

    Path(corpus_dir).mkdir(parents=True, exist_ok=True)
    Path(crashes_dir).mkdir(parents=True, exist_ok=True)

    stop_event = multiprocessing.Event()
    result_queue = multiprocessing.Queue()

    def _signal_handler(sig, frame):
        print(f"\n[*] Received signal {sig}, stopping workers...")
        stop_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    processes = []
    restart_counts = [0] * jobs
    # Live aggregated stats
    worker_stats: dict[int, dict] = {}
    worker_kwargs = dict(
        result_queue=result_queue,
        target=target,
        corpus_dir=corpus_dir,
        crashes_dir=crashes_dir,
        max_len=max_len,
        timeout=timeout,
        mutations_per_input=mutations_per_input,
        use_coverage=use_coverage,
        deep_coverage=deep_coverage,
        max_bps=max_bps,
        dictionary=dictionary or [],
        file_mode=file_mode,
        target_args=target_args or [],
        markov_order=markov_order,
        markov_generate=markov_generate,
        markov_blend=markov_blend,
        mc_bandit=mc_bandit,
        mc_cem=mc_cem,
        mc_elite_frac=mc_elite_frac,
        mc_refit_interval=mc_refit_interval,
        mc_decay_interval=mc_decay_interval,
        pairwise_blend=pairwise_blend,
        stats_file=stats_file,
        stats_interval=stats_interval,
        coverage_report=coverage_report,
        iterations=iterations,
        sync_interval=sync_interval,
        stop_event=stop_event,
        rng_seed=seed,
        secretary=secretary,
        secretary_window=secretary_window,
        secretary_exploration=secretary_exploration,
        overlap_density=overlap_density,
        overlap_density_mode=overlap_density_mode,
        overlap_min_jaccard=overlap_min_jaccard,
        overlap_density_blend=overlap_density_blend,
        exp3=exp3,
        exp3_gamma=exp3_gamma,
        eps_greedy=eps_greedy,
        eps_greedy_epsilon0=eps_greedy_epsilon0,
        eps_greedy_decay=eps_greedy_decay,
        hierarchical_bandit=hierarchical_bandit,
        gp_ucb=gp_ucb,
        ducb=ducb,
        swucb=swucb,
        cucb=cucb,
        gp_length_scale=gp_length_scale,
        gp_beta=gp_beta,
        asan_target=asan_target,
        ubsan_target=ubsan_target,
        chi2_operator_interval=chi2_operator_interval,
        lineage=lineage,
        lineage_backtrack=lineage_backtrack,
        contextual=contextual,
        contextual_alpha=contextual_alpha,
        contextual_lambda=contextual_lambda,
        resize_map_on_stall=resize_map_on_stall,
        n_workers=jobs,
        fractal_partition=fractal_partition,
        fractal_partition_depth=fractal_partition_depth,
        fractal_diversity=fractal_diversity,
        fractal_diversity_depth=fractal_diversity_depth,
        fractal_diversity_bonus=fractal_diversity_bonus,
    )

    def _spawn_worker(worker_id: int, rng_seed: int) -> multiprocessing.Process:
        p = multiprocessing.Process(
            target=_worker_main,
            kwargs={**worker_kwargs, "worker_id": worker_id, "rng_seed": rng_seed},
            daemon=True,
        )
        p.start()
        return p

    for worker_id in range(jobs):
        processes.append(_spawn_worker(worker_id, seed + worker_id))

    try:
        while not stop_event.is_set():
            # Restart dead workers
            for i, p in enumerate(processes):
                if not p.is_alive() and not stop_event.is_set():
                    exitcode = p.exitcode
                    # Restart on abnormal exit: signal-killed (< 0) or
                    # unhandled exception (> 0, not clean exit code 0)
                    abnormal = exitcode is not None and exitcode != 0
                    if abnormal:
                        restart_counts[i] += 1
                        # Fresh seed: base + worker_id + restart_count * jobs
                        new_seed = seed + i + restart_counts[i] * jobs
                        print(
                            f"\n[!] Worker-{i} died (exitcode={exitcode}), restarting (attempt {restart_counts[i]})..."
                        )
                        processes[i] = _spawn_worker(i, new_seed)

            # Drain result_queue for live aggregated stats
            while not result_queue.empty():
                try:
                    result = result_queue.get_nowait()
                    worker_stats[result["worker_id"]] = result
                except Exception:
                    break

            alive = any(p.is_alive() for p in processes)
            if not alive:
                break
            # Show per-worker stats
            if worker_stats:
                parts = []
                for wid in sorted(worker_stats):
                    w = worker_stats[wid]
                    eps = w["exec_count"] / w["elapsed"] if w["elapsed"] > 0 else 0
                    parts.append(f"W{wid}: {eps:.0f} eps")
                total_eps = sum(
                    w["exec_count"] / w["elapsed"]
                    for w in worker_stats.values()
                    if w["elapsed"] > 0
                )
                print(f"\r[*] {' | '.join(parts)} | total: {total_eps:.0f} eps", end="", flush=True)
            stop_event.wait(timeout=2)
    except KeyboardInterrupt:
        stop_event.set()

    for p in processes:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()

    # Final summary from aggregated stats (drain any remaining)
    while not result_queue.empty():
        try:
            result = result_queue.get_nowait()
            wid = result["worker_id"]
            worker_stats[wid] = result
        except Exception:
            break

    total_execs = sum(r["exec_count"] for r in worker_stats.values())
    total_crashes = sum(r["crash_count"] for r in worker_stats.values())
    total_corpus = sum(r["corpus_size"] for r in worker_stats.values())
    total_timeouts = sum(r["timeout_count"] for r in worker_stats.values())

    print(f"\n[*] All {jobs} workers stopped.")
    print(
        f"[*] Total: {total_execs} execs, {total_crashes} crashes, "
        f"{total_corpus} corpus, {total_timeouts} timeouts"
    )
