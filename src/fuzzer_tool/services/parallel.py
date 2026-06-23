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
        mc_bandit=mc_bandit,
        mc_cem=mc_cem,
        mc_elite_frac=mc_elite_frac,
        mc_refit_interval=mc_refit_interval,
        stats_file=worker_stats,
        stats_interval=stats_interval,
        coverage_report=coverage_report,
        seed=rng_seed + worker_id,
    )

    print(f"{prefix} Started (target={target})")

    i = 0
    last_sync = time.time()
    try:
        while not stop_event.is_set():
            if iterations and i >= iterations:
                break

            now = time.time()
            if now - last_sync >= sync_interval:
                _sync_corpus_in(Path(corpus_dir), fuzzer)
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
    result_queue.put({
        "worker_id": worker_id,
        "exec_count": fuzzer.exec_count,
        "crash_count": fuzzer.crash_count,
        "corpus_size": len(fuzzer.corpus),
        "timeout_count": fuzzer.timeout_count,
    })


def _sync_corpus_in(parent_dir: Path, fuzzer, max_new: int = 50):
    """Pull new corpus entries from sibling worker dirs."""
    from fuzzer_tool.adapters.filesystem import hash_data

    added = 0
    for sibling_dir in sorted(parent_dir.iterdir()):
        if not sibling_dir.is_dir() or not sibling_dir.name.startswith(".w"):
            continue
        if added >= max_new:
            break
        for entry in sorted(sibling_dir.iterdir()):
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
    for worker_id in range(jobs):
        p = multiprocessing.Process(
            target=_worker_main,
            kwargs=dict(
                worker_id=worker_id,
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
                rng_seed=seed,
            ),
            daemon=True,
        )
        processes.append(p)
        p.start()

    try:
        while not stop_event.is_set():
            alive = any(p.is_alive() for p in processes)
            if not alive:
                break
            stop_event.wait(timeout=1)
    except KeyboardInterrupt:
        stop_event.set()

    for p in processes:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()

    # Collect results from queue
    total_execs = 0
    total_crashes = 0
    while not result_queue.empty():
        try:
            result = result_queue.get_nowait()
            total_execs += result["exec_count"]
            total_crashes += result["crash_count"]
        except Exception:
            break

    print(f"\n[*] All {jobs} workers stopped.")
    print(f"[*] Total: {total_execs} execs, {total_crashes} crashes")
