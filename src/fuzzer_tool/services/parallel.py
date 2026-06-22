"""Parallel fuzzing: fork N workers sharing corpus/crashes directories."""

import multiprocessing
import os
import signal
import time
from pathlib import Path


def _worker_main(
    worker_id: int,
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
    seed: int = 42,
):
    """Entry point for each fuzzing worker process."""
    from fuzzer_tool.services.fuzzer import Fuzzer

    # Each worker prints with its ID for distinguishability
    prefix = f"[worker-{worker_id}]"

    # Resolve corpus/crashes with worker suffix for write isolation during sync
    # Workers write to a per-worker subdir, then merge into the shared dir
    worker_corpus = Path(corpus_dir) / f".w{worker_id}"
    worker_crashes = Path(crashes_dir)
    worker_corpus.mkdir(parents=True, exist_ok=True)

    # If stats_file is set, make it per-worker
    worker_stats = None
    if stats_file:
        p = Path(stats_file)
        worker_stats = str(p.with_name(f"{p.stem}_w{worker_id}{p.suffix}"))

    fuzzer = Fuzzer(
        target=target,
        corpus_dir=str(worker_corpus),
        crashes_dir=str(worker_crashes),
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
        mc_bandit=mc_bandit,
        mc_cem=mc_cem,
        mc_elite_frac=mc_elite_frac,
        mc_refit_interval=mc_refit_interval,
        stats_file=worker_stats,
        stats_interval=stats_interval,
        coverage_report=coverage_report,
        seed=seed + worker_id,
    )

    print(f"{prefix} Started (target={target})")

    i = 0
    last_sync = time.time()
    try:
        while not stop_event.is_set():
            if iterations and i >= iterations:
                break

            # Periodic corpus sync: pick new seeds from the shared parent dir
            now = time.time()
            if now - last_sync >= sync_interval:
                _sync_corpus_in(worker_corpus, Path(corpus_dir), fuzzer)
                last_sync = now

            seed = fuzzer._pick_seed()
            fuzzer.fuzz_one(seed)
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
    return {
        "worker_id": worker_id,
        "exec_count": fuzzer.exec_count,
        "crash_count": fuzzer.crash_count,
        "corpus_size": len(fuzzer.corpus),
        "timeout_count": fuzzer.timeout_count,
    }


def _sync_corpus_in(worker_corpus: Path, shared_corpus: Path, fuzzer, max_new: int = 50):
    """Pull new corpus entries from the shared directory into the worker."""
    from fuzzer_tool.adapters.filesystem import hash_data

    if not shared_corpus.exists():
        return

    added = 0
    for f in sorted(shared_corpus.iterdir()):
        if not f.is_file() or f.suffix in (".txt", ".log"):
            continue
        if added >= max_new:
            break
        data = f.read_bytes()
        h = hash_data(data)
        if h not in fuzzer.seen_hashes:
            fuzzer.save_to_corpus(data)
            added += 1

    # Also scan parent for entries from other workers
    if shared_corpus.parent.exists():
        for f in sorted(shared_corpus.parent.iterdir()):
            if not f.is_dir() or f.name.startswith(".w") or f == shared_corpus:
                continue
            if added >= max_new:
                break
            for entry in sorted(f.iterdir()):
                if not entry.is_file() or entry.suffix in (".txt", ".log"):
                    continue
                if added >= max_new:
                    break
                data = entry.read_bytes()
                h = hash_data(data)
                if h not in fuzzer.seen_hashes:
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
):
    """Launch N parallel fuzzer workers sharing the same corpus directory.

    Each worker writes to its own corpus subdirectory (.w0, .w1, ...) and
    periodically pulls new entries from siblings and the shared parent.
    Crashes are written directly to the shared crashes directory.

    Args:
        target: Path to target binary.
        jobs: Number of parallel workers.
        corpus_dir: Shared corpus directory.
        crashes_dir: Shared crashes directory.
        sync_interval: Seconds between corpus syncs.
        **kwargs: All other Fuzzer parameters forwarded to each worker.
    """
    target_name = os.path.basename(os.path.abspath(target))
    print(f"[*] Parallel fuzzing: {jobs} workers on {target_name}")
    print(f"[*] Corpus: {corpus_dir}")
    print(f"[*] Crashes: {crashes_dir}")
    print(f"[*] Sync interval: {sync_interval}s")

    # Ensure shared dirs exist
    Path(corpus_dir).mkdir(parents=True, exist_ok=True)
    Path(crashes_dir).mkdir(parents=True, exist_ok=True)

    stop_event = multiprocessing.Event()

    def _signal_handler(sig, frame):
        print(f"\n[*] Received signal {sig}, stopping workers...")
        stop_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    processes = []
    for worker_id in range(jobs):
        p = multiprocessing.Process(
            target=_worker_main,
            kwargs=dict(
                worker_id=worker_id,
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
                mc_bandit=mc_bandit,
                mc_cem=mc_cem,
                mc_elite_frac=mc_elite_frac,
                mc_refit_interval=mc_refit_interval,
                stats_file=stats_file,
                stats_interval=stats_interval,
                coverage_report=coverage_report,
                iterations=iterations,
                sync_interval=sync_interval,
                stop_event=stop_event,
                seed=seed,
            ),
            daemon=True,
        )
        processes.append(p)
        p.start()

    # Wait for all workers or stop event
    try:
        while not stop_event.is_set():
            alive = any(p.is_alive() for p in processes)
            if not alive:
                break
            stop_event.wait(timeout=1)
    except KeyboardInterrupt:
        stop_event.set()

    # Give workers time to finish gracefully
    for p in processes:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()

    # Final summary
    print(f"\n[*] All {jobs} workers stopped.")
    total_execs = 0
    total_crashes = 0
    for p in processes:
        if hasattr(p, "_result"):
            total_execs += p._result.get("exec_count", 0)
            total_crashes += p._result.get("crash_count", 0)
    print(f"[*] Total: {total_execs} execs, {total_crashes} crashes")
