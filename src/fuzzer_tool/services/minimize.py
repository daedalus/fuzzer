"""Corpus minimizer: prune redundant corpus entries while preserving coverage.

Two modes:
  1. With SHM coverage (-c): greedy set-cover over edge maps. Requires target
     to be AFL-instrumented and __AFL_SHM_ID set.
  2. Without coverage: content-hash dedup (kept if unique hash),
     with optional Hamming-based fuzzy dedup for near-duplicates.
"""

import contextlib
import ctypes
import gzip
import hashlib
import os
import pickle
import resource
import shutil
import sys
import tempfile
import time
from enum import Enum
from pathlib import Path

import numpy as np

from fuzzer_tool.adapters import libc_mem, libc_shm
from fuzzer_tool.adapters.filesystem import hash_data
from fuzzer_tool.adapters.process import disable_aslr
from fuzzer_tool.adapters.shm import ShmCoverage
from fuzzer_tool.core.count_items import _TAG_BITS, CountItems
from fuzzer_tool.core.count_items import count_items as _count_items
from fuzzer_tool.core.edge_components import components_report
from fuzzer_tool.core.edge_hist import hist_report
from fuzzer_tool.core.rand_pool import RandPool, get_default_rand_pool
from fuzzer_tool.core.run_mode import Switch
from fuzzer_tool.core.sdr import distinct_reps, reps_report
from fuzzer_tool.core.set_cover import min_cover


def _maxrss_kb() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def _fmt_kb(kb: int) -> str:
    return f"{kb // 1024}MB" if kb >= 1024 else f"{kb}KB"


class ExecMode(Enum):
    """How coverage-mode replay runs the target (same backends as ``fuzz``)."""

    SPAWN = "spawn"  # one process per seed (default)
    LOADER = "loader"  # --inprocess: subprocess loader, crash-isolated
    DIRECT_LITE = "direct_lite"  # --inprocess-direct: ctypes call, shim in the .so
    DIRECT = "direct"  # --inprocess --inprocess-direct: ctypes call


DEFAULT_INPROCESS_FUNC = "LLVMFuzzerTestOneInput"
_SO_SUFFIXES = (".so", ".dylib", ".dll")


class PruneMode(Enum):
    """Coverage-mode pruning criterion."""

    SET_COVER = "set_cover"
    RATE_DISTORTION = "rate_distortion"
    # Set-cover plus backups for the seeds whose removal loses most edges.
    MINIMAX_ROBUST = "minimax_robust"


class Twins(Enum):
    """What to do with seeds whose edge sets are identical."""

    COLLAPSE = "collapse"  # keep the smallest: twins add nothing to a cover
    KEEP = "keep"  # minimax-robust: a twin is a free backup


# Twin policy per prune mode; anything absent collapses.
_TWINS = {PruneMode.MINIMAX_ROBUST: Twins.KEEP}


class _EdgeBook:
    """Per-seed edge ids as sorted uint32 arrays, twins collapsed or kept.

    An array is ~1/15 the bytes of a set[int]. Under COLLAPSE the
    coverage bytes are the dedup key and the stored array is a view on
    them, so each unique coverage is held once.

        add a {1,2} 10B; add b {1,2} 4B; add c {1,3}  ->  edges: b, c
    """

    def __init__(self, twins: Twins, dtype: type = np.uint32) -> None:
        self._dtype = dtype
        self.edges: dict[str, np.ndarray] = {}
        self.sizes: dict[str, int] = {}
        self._by_sig: dict[bytes, str] = {}
        self.add = self._add_collapse if twins is Twins.COLLAPSE else self._add_keep

    def _add_keep(self, key: str, ids: np.ndarray, size: int) -> None:
        self.edges[key] = ids.copy()
        self.sizes[key] = size

    def _add_collapse(self, key: str, ids: np.ndarray, size: int) -> None:
        sig = ids.tobytes()
        kept = self._by_sig.get(sig)
        if kept is not None and self.sizes[kept] <= size:
            return

        # Smaller twin replaces the kept one.
        if kept is not None:
            del self.edges[kept], self.sizes[kept]
        self._by_sig[sig] = key
        self.edges[key] = np.frombuffer(sig, dtype=self._dtype)
        self.sizes[key] = size


def _as_sets(edges: dict[str, np.ndarray]) -> dict[str, set[int]]:
    """Set view for the rate-distortion pruners, which use set algebra."""
    return {k: set(e.tolist()) for k, e in edges.items()}


def _prune(book: _EdgeBook, prune: PruneMode, target_frac: float, map_entries: int) -> list[str]:
    """Keys of the seeds *prune* keeps from *book*."""
    n = len(book.edges)
    if prune is PruneMode.MINIMAX_ROBUST:
        from fuzzer_tool.core.rate_distortion import RateDistortionCorpus

        rd = RateDistortionCorpus(map_size=map_entries)
        kept, frac = rd.minimax_robust_pruning(_as_sets(book.edges), target_fraction=target_frac)
        print(f"[*] Minimax-robust: kept {len(kept)}/{n} files ({frac:.1%} coverage)")
        return kept

    if prune is PruneMode.RATE_DISTORTION:
        print("[*] Using rate-distortion optimal pruning...")
        from fuzzer_tool.core.rate_distortion import RateDistortionCorpus

        rd = RateDistortionCorpus(map_size=map_entries)
        kept, frac = rd.optimal_pruning(_as_sets(book.edges), target_fraction=target_frac)
        print(f"[*] Rate-distortion: kept {len(kept)}/{n} files ({frac:.1%} coverage)")
        return kept

    # Forced seeds + dominance + lazy greedy, smaller file on ties
    # (core/set_cover.py). The previous version scored candidates with
    # numpy popcounts over uint8 bitmap views, the right shape for AFL's
    # byte bitmap and the wrong one for this shim's entry table.
    return min_cover(book.edges, book.sizes)


def _read_shm_edges(shm_id: str, size: int = 65536) -> bytearray:
    """Attach a SysV segment and read ``size`` raw bytes out of it.

    Returns all zeros if the segment cannot be attached. Callers must treat
    that as "no coverage information", not "this input covers nothing".

    NO LONGER ON THE CMIN PATH. _minimize_with_coverage used to read the
    coverage segment through this, treating it as AFL's byte-per-edge bitmap;
    this shim writes {edge_id, count} entries behind a 32-byte header, so the
    byte indices it yielded were not edge ids. That path now asks
    ShmCoverage for edge ids instead.

    Retained because tests/test_regression_shmat_restype.py exercises it as
    one of the three shmat call sites named in
    docs/bugreport_2026-08-21_merged.md -- shmat bound with a default c_int
    restype truncated the returned address to 32 bits and string_at() then
    read an unmapped page. The sentinel and restype behaviour is also covered
    directly against libc_shm in that file, so if this helper is dropped the
    regression stays guarded; it is kept rather than deleted so that removing
    it is a deliberate decision and not collateral of a sizing fix.
    """
    ptr = libc_shm.shmat(int(shm_id))
    if ptr is None:
        return bytearray(size)
    try:
        data = ctypes.string_at(ptr, size)
    finally:
        libc_shm.shmdt(ptr)
    return bytearray(data)


def _open_runner(target: str, mode: ExecMode, func: str, timeout: float, shm: ShmCoverage):
    """InProcessRunner on ``shm``'s segment, or None when it cannot start.

    A missing entry point or a failed dlopen is a failed measurement; the
    caller keeps the whole corpus rather than read it as "covers nothing".
    """
    from fuzzer_tool.adapters.inprocess import InProcessRunner

    # fuzz probes the entry when the default name is kept on a .so.
    if func == DEFAULT_INPROCESS_FUNC and target.lower().endswith(_SO_SUFFIXES):
        from fuzzer_tool.services.fuzzer import Fuzzer

        func = Fuzzer._probe_so_function(target)

    try:
        return InProcessRunner(
            target=target,
            function_name=func,
            timeout=timeout,
            shm_size=shm.num_entries,
            direct=mode is ExecMode.DIRECT,
            direct_lite=mode is ExecMode.DIRECT_LITE,
            coverage_env_id=shm.env_id,
            cov=True,
        )
    except (AttributeError, OSError, ImportError) as e:
        print(f"[-] In-process runner failed for {func}: {e}", file=sys.stderr)
        return None


def _discover_corpus_files(corpus_path: Path) -> list[Path]:
    """Find corpus files under either the sharded or a flat layout.

    save_to_corpus writes seeds/<hh>/id_<hash>, but this module used a flat
    iterdir() on the directory it was handed. Pointed at a real corpus it
    therefore found nothing, printed "Corpus is empty" and exited 0 -- so the
    coverage path below was unreachable in normal use, which is what kept the
    all-zero-bitmap corpus wipe hidden.

    Accepts both spellings so a directory of loose files still works:
      - <dir>/seeds/**   canonical, what the fuzzer writes
      - <dir>/**         the dir is itself a seeds root, or flat

    pruned/ is excluded at every level: those entries were already removed
    from the live corpus, and re-minimizing would resurrect them.
    crashing/ and irreplaceable/ are excluded too: their contents are marked
    never-prune and must not be treated as minimization candidates.

    The layout itself lives in adapters.filesystem.discover_seed_files, which
    is also what root_cause uses; the exclusions below are this module's, the
    walk is not.
    """
    from fuzzer_tool.adapters.filesystem import discover_seed_files

    return discover_seed_files(
        corpus_path,
        include_pruned=False,
        include_crashing=False,
        include_irreplaceable=False,
    )


def _pick_seeds(files: list[Path], max_seeds: int, rng: RandPool) -> list[Path]:
    """Uniform k-of-n subset of *files*, in disk order; all when uncapped.

    Disk order is seeds/<hh>/ by hash prefix, so a head slice only ever
    saw low-prefix seeds.
    """
    n = len(files)
    if max_seeds <= 0 or n <= max_seeds:
        return files

    return [files[i] for i in sorted(rng.sample(n, max_seeds))]


def minimize_corpus(
    target: str,
    corpus_dir: str,
    timeout: float = 5.0,
    file_mode: bool = False,
    target_args: list[str] | None = None,
    use_coverage: bool = False,
    output_dir: str | None = None,
    rate_distortion: bool = False,
    target_frac: float = 0.95,
    fuzzy_dedup: int = 0,
    prune: PruneMode = PruneMode.SET_COVER,
    exec_mode: ExecMode = ExecMode.SPAWN,
    inprocess_func: str = DEFAULT_INPROCESS_FUNC,
    max_seeds: int = 0,
    batch_mode: bool = False,
    rng: RandPool | None = None,
    edge_hist: Switch = Switch.OFF,
    edge_components: Switch = Switch.OFF,
    count_items: CountItems = CountItems.OFF,
    hall: Switch = Switch.OFF,
) -> tuple[int, int]:
    """Minimize a corpus by removing redundant inputs.

    With -c/--coverage: replays each file, reads SHM edge bitmap, then
    set-cover keeps few, small files that cover all edges.
    Without -c: content-hash dedup (keeps first occurrence of each hash).

    Args:
        target: Path to the target binary.
        corpus_dir: Path to the corpus directory.
        timeout: Execution timeout in seconds.
        file_mode: Write input to temp file instead of stdin.
        target_args: Target arguments ({file} placeholder).
        use_coverage: Enable SHM coverage (passed to env).
        output_dir: Output directory for minimized corpus. If None, overwrites in-place.
        rate_distortion: Legacy alias for ``prune=PruneMode.RATE_DISTORTION``.
        target_frac: Target coverage fraction for rate-distortion (default: 0.95).
        fuzzy_dedup: Maximum Hamming distance for near-duplicate detection.
            0 disables fuzzy dedup. Only used without coverage mode.
            e.g. fuzzy_dedup=3 removes seeds that differ by <=3 bytes.
        prune: Coverage-mode pruning criterion.
        exec_mode: Coverage-mode replay backend (spawn, loader, direct_lite, direct).
        inprocess_func: Entry point for the in-process backends.
        max_seeds: Max seeds to process (0=all), drawn uniformly at random.
            Allows batch minification.
        rng: Sampler for ``max_seeds`` (default: shared RandPool).
        edge_hist: ON prints seeds-per-edge-count histograms (replayed, kept);
            coverage mode only.
        count_items: BUCKET or THERMO covers (edge, hit-count) items instead of
            bare edges (see core/count_items.py); coverage mode only.
        edge_components: ON prints connected components of the seed Gram
            matrix after stripping near-universal edges; coverage mode only.
        hall: ON adds a distinct seed per rare edge to the kept set (Hall
            matching, core/sdr.py) and prints the report; coverage mode only.

    Returns:
        Tuple of (files_kept, files_removed).
    """
    corpus_path = Path(corpus_dir)
    if not corpus_path.is_dir():
        print(f"[-] Corpus directory not found: {corpus_dir}", file=sys.stderr)
        return 0, 0

    all_corpus_files = _discover_corpus_files(corpus_path)
    state_file = corpus_path / ".minimize_batch_state.pkl.gz"
    processed_hashes = set()
    if batch_mode and state_file.exists():
        try:
            with gzip.open(state_file, "rb") as f:
                state = pickle.load(f)
            processed_hashes = set(state.get("processed", set()))
        except Exception:
            processed_hashes = set()

    if batch_mode:
        filtered = []
        for f in all_corpus_files:
            try:
                if hash_data(f.read_bytes()) not in processed_hashes:
                    filtered.append(f)
            except Exception:
                continue
        corpus_files = filtered
    else:
        corpus_files = all_corpus_files

    if not corpus_files:
        print("[-] Corpus is empty or all files already processed in batch mode", file=sys.stderr)
        return 0, 0

    corpus_files = _pick_seeds(corpus_files, max_seeds, rng or get_default_rand_pool())

    print(f"[*] Corpus: {len(corpus_files)} files in {corpus_dir} (total: {len(all_corpus_files)})")

    if use_coverage:
        kept, removed = _minimize_with_coverage(
            corpus_files,
            target,
            timeout,
            file_mode,
            target_args,
            output_dir,
            corpus_path,
            prune=PruneMode.RATE_DISTORTION if rate_distortion else prune,
            target_frac=target_frac,
            exec_mode=exec_mode,
            inprocess_func=inprocess_func,
            batch_mode=batch_mode,
            all_corpus_files=all_corpus_files,
            edge_hist=edge_hist,
            edge_components=edge_components,
            count_items=count_items,
            hall=hall,
        )
    else:
        kept, removed = _minimize_by_hash(corpus_files, output_dir, corpus_path, fuzzy_dedup)

    print(f"[+] Minimized: {len(corpus_files)} -> {kept} files ({removed} removed)")
    return kept, removed


def _edge_ids(arr: np.ndarray) -> np.ndarray:
    """Distinct edge ids of a book entry; count items carry the id above the tag bits."""
    if arr.dtype == np.uint64:
        return np.unique(arr >> np.uint64(_TAG_BITS))
    return arr


def _observer(shm, mode: CountItems):
    """Callable returning ``(edge ids, cover items)`` for the last replay.

    Chosen once so the replay loop does not branch on the mode.
    """
    if mode is CountItems.OFF:

        def ids_only():
            ids = shm.edge_id_array()
            return ids, ids

        return ids_only

    def with_counts():
        ids, counts = shm.get_edge_arrays()
        return np.unique(ids).astype(np.uint32, copy=False), _count_items(ids, counts, mode)

    return with_counts


def _spawn_replay(target, timeout, file_mode, tmp_dir, target_args, env):
    """Replay callable that spawns the target once per input."""
    from fuzzer_tool.adapters.process import run_target_file, run_target_stdin

    if file_mode:
        return lambda data: run_target_file(
            target, data, timeout, str(tmp_dir), target_args or [], env=env
        )
    return lambda data: run_target_stdin(target, data, timeout, env=env)


def _runner_replay(runner):
    """Replay callable on an InProcessRunner, from a clean shim state per input."""

    def replay(data):
        runner.reset_exec_state()
        return runner.run_one(data)

    return replay


def _minimize_with_coverage(
    corpus_files: list[Path],
    target: str,
    timeout: float,
    file_mode: bool,
    target_args: list[str] | None,
    output_dir: str | None,
    corpus_path: Path,
    prune: PruneMode = PruneMode.SET_COVER,
    target_frac: float = 0.95,
    exec_mode: ExecMode = ExecMode.SPAWN,
    inprocess_func: str = DEFAULT_INPROCESS_FUNC,
    batch_mode: bool = False,
    all_corpus_files: list[Path] | None = None,
    edge_hist: Switch = Switch.OFF,
    edge_components: Switch = Switch.OFF,
    count_items: CountItems = CountItems.OFF,
    hall: Switch = Switch.OFF,
) -> tuple[int, int]:
    """Greedy set-cover, rate-distortion or minimax-robust pruning over SHM edge bitmaps."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="cmin_"))

    # Caller-context edge ids hash a raw return address; with ASLR on, every
    # replay reports fresh ids and set-cover runs on noise. Same as fuzz.
    disable_aslr()

    # One segment, reset between files, owned by ShmCoverage.
    #
    # This path used to hand-roll shmget/shmat/string_at with a single
    # `edge_map_size = 65536` used for three incompatible purposes: the
    # AFL_MAP_SIZE the child is told (which the shim reads as a count of
    # ENTRIES), the byte size handed to shmget, and the byte count read back.
    # 65,536 entries needs 32 + 65_536 * 8 = 524,320 bytes, so the shim wrote
    # 448 KiB past the end of a 64 KiB segment and every child died on
    # SIGSEGV. The blackout guard below then refused to prune, which is why
    # this failed safe rather than deleting corpora -- cmin simply never
    # worked on an instrumented target.
    #
    # The read was wrong independently of the size. It treated the segment as
    # AFL's byte-per-edge bitmap and took nonzero byte INDICES as edge ids,
    # but this shim writes {edge_id, count} entries behind a header, so the
    # "edges" were byte offsets into a hash table plus the header. Asking
    # ShmCoverage for edge ids fixes the sizing and the decoding together,
    # and keeps the layout in the one module that owns it.
    shm = ShmCoverage()
    env_base = os.environ.copy()
    env_base["__AFL_SHM_ID"] = shm.env_id
    env_base["AFL_MAP_SIZE"] = str(shm.num_entries)

    book = _EdgeBook(
        _TWINS.get(prune, Twins.COLLAPSE), np.uint32 if count_items is CountItems.OFF else np.uint64
    )
    observe = _observer(shm, count_items)
    runner = None
    peak_rss_kb = _maxrss_kb()
    try:
        # One replay callable, chosen once: the loop below never branches on mode.
        if exec_mode is ExecMode.SPAWN:
            replay = _spawn_replay(target, timeout, file_mode, tmp_dir, target_args, env_base)
        else:
            runner = _open_runner(target, exec_mode, inprocess_func, timeout, shm)
            if runner is None:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                return len(corpus_files), 0
            replay = _runner_replay(runner)

        start_time = time.time()
        unique_edges = set()
        replayed_counts: list[int] = []
        for i, fpath in enumerate(corpus_files):
            if not fpath.exists():
                print(f"[!] Skipping missing corpus file: {fpath}")
                continue
            data = fpath.read_bytes()
            shm.reset_edge_map()
            replay(data)
            ids, items = observe()
            book.add(str(fpath), items, len(data))
            unique_edges.update(ids.tolist())
            replayed_counts.append(ids.size)
            peak_rss_kb = max(peak_rss_kb, _maxrss_kb())
            # Reclaim heap memory periodically like in fuzz mode
            if (i + 1) % 100 == 0:
                with contextlib.suppress(Exception):
                    freed = libc_mem.trim_heap() >> 10
                    if freed > 0:
                        freed_mb = freed // 1024
                        if freed_mb >= 1:
                            print(f"\r[*] Trimmed {freed_mb}MB heap", end=" ", flush=True)
                        else:
                            print(f"\r[*] Trimmed {freed}KB heap", end=" ", flush=True)

            if (i + 1) % 10 == 0 or (i + 1) == len(corpus_files):
                elapsed = time.time() - start_time
                files_done = i + 1
                total_files = len(corpus_files)
                live_kb = None
                trim_str = ""
                with contextlib.suppress(Exception):
                    freed = libc_mem.trim_heap() >> 10
                    if freed > 0:
                        if freed >= 1024:
                            trim_str = f" | trim: {freed // 1024}MB"
                        else:
                            trim_str = f" | trim: {freed}KB"
                    live_kb = libc_mem.current_rss() >> 10
                if elapsed > 0 and files_done > 0:
                    rate = files_done / elapsed
                    remaining = total_files - files_done
                    eta = remaining / rate if rate > 0 else 0
                    rss_str = f" | rss: {_fmt_kb(peak_rss_kb)}"
                    if live_kb is not None:
                        rss_str += f" | live: {_fmt_kb(live_kb)}"
                    print(
                        f"\r[*] Replayed {files_done}/{total_files} ({files_done / total_files * 100:.1f}%) "
                        f"{rate:.1f} seeds/s ETA: {eta:.1f}s | elapsed: {elapsed:.1f}s | edges: {len(unique_edges)}{rss_str}{trim_str}",
                        end="",
                        flush=True,
                    )
                else:
                    rss_str = f" | rss: {_fmt_kb(peak_rss_kb)}"
                    if live_kb is not None:
                        rss_str += f" | live: {_fmt_kb(live_kb)}"
                    print(
                        f"\r[*] Replayed {files_done}/{total_files} | elapsed: {elapsed:.1f}s | edges: {len(unique_edges)}{rss_str}{trim_str}...",
                        end="",
                        flush=True,
                    )
    finally:
        map_entries = shm.num_entries
        if runner is not None:
            runner.stop()
        shm.cleanup()

    print()

    shutil.rmtree(tmp_dir, ignore_errors=True)

    # Refuse to prune on a total coverage blackout. Both set-cover and
    # rate-distortion select files by the edges they contribute, so an
    # all-empty edge set means nothing contributes anything and *every* file
    # looks redundant -- the corpus is wiped rather than minimized.
    #
    # A blackout means the measurement failed, not that the seeds are
    # worthless: an uninstrumented target, a failed attach, or a segment the
    # child never wrote. Callers must read "no edges" as "no coverage
    # information", never as "covers nothing".
    if not unique_edges:
        print(
            "[-] No edges recorded for any corpus file -- refusing to prune.\n"
            "    Every file would look redundant and the whole corpus would be "
            "deleted.\n"
            "    Usually this means the target is not instrumented (rebuild with "
            "tools/build_targets.sh),\n"
            "    or the target never wrote the SHM segment.",
            file=sys.stderr,
        )
        return len(corpus_files), 0

    print(f"[*] Total unique edges: {len(unique_edges)}")
    print(f"[*] Coverage twins collapsed: {len(corpus_files) - len(book.edges)}")

    covered_files = _prune(book, prune, target_frac, map_entries)
    if hall is Switch.ON:
        covered_files = _add_reps(book, covered_files)

    if edge_hist is Switch.ON:
        _print_edge_hist(replayed_counts, unique_edges, book, covered_files)

    if edge_components is Switch.ON:
        print(components_report({k: _edge_ids(e) for k, e in book.edges.items()}))

    return _commit_results(
        corpus_files,
        covered_files,
        output_dir,
        corpus_path,
        batch_mode=batch_mode,
        all_corpus_files=all_corpus_files,
    )


def _add_reps(book: _EdgeBook, kept: list[str]) -> list[str]:
    """*kept* plus a distinct seed for each rare edge, cover seeds tried first."""
    reps = distinct_reps({k: _edge_ids(e) for k, e in book.edges.items()}, prefer=kept)
    have = set(kept)
    extra = [k for k in reps.owner.values() if k not in have]
    print(reps_report(reps, added=len(extra)))
    return kept + extra


def _print_edge_hist(
    replayed_counts: list[int],
    unique_edges: set[int],
    book: _EdgeBook,
    kept_keys: list[str],
) -> None:
    """Edge histograms before (every replayed seed) and after pruning.

    "kept" counts come from the book, so twins collapsed before pruning are
    in "replayed" only.
    """
    print(hist_report(replayed_counts, len(unique_edges), "minimize: replayed"))

    kept = [_edge_ids(book.edges[k]) for k in kept_keys]
    kept_unique = len(np.unique(np.concatenate(kept))) if kept else 0
    print(hist_report([e.size for e in kept], kept_unique, "minimize: kept"))


def _minimize_by_hash(
    corpus_files: list[Path],
    output_dir: str | None,
    corpus_path: Path,
    fuzzy_dedup: int = 0,
) -> tuple[int, int]:
    """Content-hash dedup: keep first occurrence of each SHA-256.

    When fuzzy_dedup > 0, also removes entries that are within Hamming
    distance of an already-kept entry (near-duplicate detection).
    """
    from fuzzer_tool.core.similarity import hamming_distance

    seen_hashes: set[str] = set()
    kept_files: list[str] = []
    kept_data: list[bytes] = []

    for fpath in corpus_files:
        if not fpath.exists():
            print(f"[!] Skipping missing corpus file: {fpath}")
            continue
        data = fpath.read_bytes()
        h = hashlib.sha256(data).hexdigest()[:16]
        if h in seen_hashes:
            continue

        # Fuzzy dedup: skip if within Hamming distance of any kept entry
        if fuzzy_dedup > 0 and kept_data:
            is_near_dup = False
            for kept in kept_data:
                if len(kept) == len(data):
                    try:
                        if hamming_distance(data, kept) <= fuzzy_dedup:
                            is_near_dup = True
                            break
                    except ValueError:
                        pass
            if is_near_dup:
                continue

        seen_hashes.add(h)
        kept_files.append(str(fpath))
        if fuzzy_dedup > 0:
            kept_data.append(data)

    return _commit_results(corpus_files, kept_files, output_dir, corpus_path)


def _commit_results(
    corpus_files: list[Path],
    kept: list[str],
    output_dir: str | None,
    corpus_path: Path,
    batch_mode: bool = False,
    all_corpus_files: list[Path] | None = None,
) -> tuple[int, int]:
    """Write minimized corpus to output dir or prune in-place.

    When no output_dir is specified, removed files are moved to a
    ``pruned/`` subfolder inside the corpus directory instead of being
    deleted.  This preserves coverage-redundant inputs for later
    analysis while keeping the active corpus lean.
    """
    kept_set = set(kept)
    out_path = Path(output_dir) if output_dir else corpus_path

    state_file = corpus_path / ".minimize_batch_state.pkl.gz"
    state = {"processed": set(), "survivors": set()}
    if batch_mode and state_file.exists():
        try:
            with gzip.open(state_file, "rb") as f:
                state = pickle.load(f)
            if "processed" not in state:
                state["processed"] = set()
            else:
                state["processed"] = set(state["processed"])
            if "survivors" not in state:
                state["survivors"] = set()
            else:
                state["survivors"] = set(state["survivors"])
        except Exception:
            state = {"processed": set(), "survivors": set()}

    if output_dir:
        out_path.mkdir(parents=True, exist_ok=True)
        for fpath_str in kept:
            fpath = Path(fpath_str)
            if not fpath.exists():
                continue
            shutil.copy2(fpath, out_path / fpath.name)
            meta = fpath.with_suffix(".txt")
            if meta.exists():
                shutil.copy2(meta, out_path / meta.name)
    else:
        pruned_dir = corpus_path / "pruned"
        pruned_dir.mkdir(parents=True, exist_ok=True)
        for fpath in corpus_files:
            if str(fpath) not in kept_set:
                if not fpath.exists():
                    continue
                # Preserve directory structure like the seed corpus (seeds/xx/id_*)
                rel = (
                    fpath.relative_to(corpus_path)
                    if str(fpath).startswith(str(corpus_path))
                    else Path(fpath.name)
                )
                dest_dir = pruned_dir / rel.parent
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest = dest_dir / rel.name
                shutil.move(str(fpath), str(dest))
                meta = fpath.with_suffix(".txt")
                if meta.exists():
                    meta_rel = (
                        meta.relative_to(corpus_path)
                        if str(meta).startswith(str(corpus_path))
                        else Path(meta.name)
                    )
                    meta_dest_dir = pruned_dir / meta_rel.parent
                    meta_dest_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(meta), str(meta_dest_dir / meta_rel.name))
            elif batch_mode:
                # In batch mode, keep survivors in place and record them
                pass

    if batch_mode:
        processed = state.get("processed", set())
        survivors = state.get("survivors", set())
        for fpath in corpus_files:
            fpath_p = Path(fpath)
            if not fpath_p.exists():
                continue
            try:
                h = hash_data(fpath_p.read_bytes())
            except Exception:
                continue
            processed.add(h)
            if str(fpath) in kept_set:
                survivors.add(h)
        try:
            with gzip.open(state_file, "wb") as f:
                pickle.dump({"processed": processed, "survivors": survivors}, f)
        except Exception:
            pass

    removed = len(corpus_files) - len(kept)
    return len(kept), removed


def main():
    """CLI entry point for fuzzer-tool minimize."""
    import argparse

    parser = argparse.ArgumentParser(description="Minimize a corpus by removing redundant inputs")
    parser.add_argument("target", help="Path to target binary")
    parser.add_argument("-d", "--corpus", required=True, help="Corpus directory")
    parser.add_argument("-t", "--timeout", type=float, default=5, help="Timeout in seconds")
    parser.add_argument(
        "-F", "--file-mode", action="store_true", help="Write input to temp file instead of stdin"
    )
    parser.add_argument(
        "-A",
        "--target-args",
        nargs=argparse.REMAINDER,
        help="Target arguments ({file} placeholder)",
    )
    parser.add_argument("-c", "--coverage", action="store_true", help="Enable SHM coverage")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output directory for minimized corpus (default: overwrite in-place)",
    )
    parser.add_argument(
        "--fuzzy-dedup",
        type=int,
        default=0,
        help="Maximum Hamming distance for near-duplicate detection (0=disabled)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.target):
        print(f"[-] Target not found: {args.target}", file=sys.stderr)
        sys.exit(1)

    kept, removed = minimize_corpus(
        target=args.target,
        corpus_dir=args.corpus,
        timeout=args.timeout,
        file_mode=args.file_mode,
        target_args=args.target_args,
        use_coverage=args.coverage,
        output_dir=args.output,
        fuzzy_dedup=args.fuzzy_dedup,
    )

    if removed == 0:
        print("[*] Corpus already minimal")
