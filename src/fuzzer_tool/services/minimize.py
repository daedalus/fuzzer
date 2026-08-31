"""Corpus minimizer: prune redundant corpus entries while preserving coverage.

Two modes:
  1. With SHM coverage (-c): greedy set-cover over edge maps. Requires target
     to be AFL-instrumented and __AFL_SHM_ID set.
  2. Without coverage: content-hash dedup (kept if unique hash),
     with optional Hamming-based fuzzy dedup for near-duplicates.
"""

import ctypes
import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from fuzzer_tool.adapters import libc_shm


def _read_shm_edges(shm_id: str, size: int = 65536) -> bytearray:
    """Read edge bitmap from AFL SHM segment.

    Returns an all-zero bitmap if the segment cannot be attached.  Callers must
    treat an all-zero result as "no coverage information", not as "this input
    covers nothing" -- see ``_minimize_with_coverage``.
    """
    ptr = libc_shm.shmat(int(shm_id))
    if ptr is None:
        return bytearray(size)
    try:
        data = ctypes.string_at(ptr, size)
    finally:
        libc_shm.shmdt(ptr)
    return bytearray(data)


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
    is also what root_cause and the parallel worker sync use; the exclusions
    below are this module's, the walk is not.
    """
    from fuzzer_tool.adapters.filesystem import discover_seed_files

    return discover_seed_files(
        corpus_path,
        include_pruned=False,
        include_crashing=False,
        include_irreplaceable=False,
    )


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
) -> tuple[int, int]:
    """Minimize a corpus by removing redundant inputs.

    With -c/--coverage: replays each file, reads SHM edge bitmap, then
    greedy set-cover keeps minimum files that cover all edges.
    Without -c: content-hash dedup (keeps first occurrence of each hash).

    Args:
        target: Path to the target binary.
        corpus_dir: Path to the corpus directory.
        timeout: Execution timeout in seconds.
        file_mode: Write input to temp file instead of stdin.
        target_args: Target arguments ({file} placeholder).
        use_coverage: Enable SHM coverage (passed to env).
        output_dir: Output directory for minimized corpus. If None, overwrites in-place.
        rate_distortion: Use rate-distortion optimal pruning instead of greedy set-cover.
        target_frac: Target coverage fraction for rate-distortion (default: 0.95).
        fuzzy_dedup: Maximum Hamming distance for near-duplicate detection.
            0 disables fuzzy dedup. Only used without coverage mode.
            e.g. fuzzy_dedup=3 removes seeds that differ by <=3 bytes.

    Returns:
        Tuple of (files_kept, files_removed).
    """
    corpus_path = Path(corpus_dir)
    if not corpus_path.is_dir():
        print(f"[-] Corpus directory not found: {corpus_dir}", file=sys.stderr)
        return 0, 0

    corpus_files = _discover_corpus_files(corpus_path)
    if not corpus_files:
        print("[-] Corpus is empty", file=sys.stderr)
        return 0, 0

    print(f"[*] Corpus: {len(corpus_files)} files in {corpus_dir}")

    if use_coverage:
        kept, removed = _minimize_with_coverage(
            corpus_files,
            target,
            timeout,
            file_mode,
            target_args,
            output_dir,
            corpus_path,
            rate_distortion=rate_distortion,
            target_frac=target_frac,
        )
    else:
        kept, removed = _minimize_by_hash(corpus_files, output_dir, corpus_path, fuzzy_dedup)

    print(f"[+] Minimized: {len(corpus_files)} -> {kept} files ({removed} removed)")
    return kept, removed


def _minimize_with_coverage(
    corpus_files: list[Path],
    target: str,
    timeout: float,
    file_mode: bool,
    target_args: list[str] | None,
    output_dir: str | None,
    corpus_path: Path,
    rate_distortion: bool = False,
    target_frac: float = 0.95,
) -> tuple[int, int]:
    """Greedy set-cover or rate-distortion optimal pruning over SHM edge bitmaps."""
    from fuzzer_tool.adapters.process import run_target_file, run_target_stdin

    tmp_dir = Path(tempfile.mkdtemp(prefix="cmin_"))
    edge_map_size = 65536
    file_edges: dict[str, bytearray] = {}

    for i, fpath in enumerate(corpus_files):
        data = fpath.read_bytes()
        env = os.environ.copy()
        env["AFL_MAP_SIZE"] = str(edge_map_size)

        # Create a unique SHM segment for this run
        shmid = libc_shm.shmget(edge_map_size)
        if shmid is None:
            file_edges[str(fpath)] = bytearray(edge_map_size)
            continue
        env["__AFL_SHM_ID"] = str(shmid)

        if file_mode:
            run_target_file(target, data, timeout, str(tmp_dir), target_args or [], env=env)
        else:
            run_target_stdin(target, data, timeout, env=env)

        # Read the edge bitmap from SHM.  shmat() is bound with
        # restype=c_void_p; attaching with the default c_int restype truncated
        # this address to 32 bits and string_at() then read an unmapped page.
        ptr = libc_shm.shmat(shmid)
        if ptr is not None:
            try:
                file_edges[str(fpath)] = bytearray(ctypes.string_at(ptr, edge_map_size))
            finally:
                libc_shm.shmdt(ptr)
        else:
            file_edges[str(fpath)] = bytearray(edge_map_size)
        libc_shm.shmctl_rmid(shmid)

        if (i + 1) % 10 == 0 or (i + 1) == len(corpus_files):
            print(f"\r[*] Replayed {i + 1}/{len(corpus_files)}...", end="", flush=True)

    print()

    shutil.rmtree(tmp_dir, ignore_errors=True)

    # Refuse to prune on a total coverage blackout. Both set-cover and
    # rate-distortion select files by the edges they contribute, so an
    # all-zero bitmap set means nothing contributes anything and *every* file
    # looks redundant -- the corpus is wiped rather than minimized.
    #
    # A blackout means the measurement failed, not that the seeds are
    # worthless: an uninstrumented target, a failed shmat, or a segment the
    # child never wrote. _read_shm_bitmap's docstring already says callers must
    # read all-zero as "no coverage information" rather than "covers nothing";
    # this is that check. Deleting a corpus on a broken measurement is the
    # worst available outcome, so bail out and name the likely cause.
    if not any(any(bm) for bm in file_edges.values()):
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

    # Convert to sets for rate-distortion module
    seed_edges = {}
    for fpath, bm in file_edges.items():
        seed_edges[fpath] = {j for j in range(edge_map_size) if bm[j]}

    if rate_distortion:
        print("[*] Using rate-distortion optimal pruning...")
        from fuzzer_tool.core.rate_distortion import RateDistortionCorpus

        rd = RateDistortionCorpus(map_size=edge_map_size)
        covered_files, actual_frac = rd.optimal_pruning(seed_edges, target_fraction=target_frac)
        print(
            f"[*] Rate-distortion: kept {len(covered_files)}/{len(corpus_files)} "
            f"files ({actual_frac:.1%} coverage)"
        )
    else:
        # Greedy set cover (bitmaps as numpy uint8 views: count_new = popcount
        # of (edges & ~covered) — ~100x faster than a 65K-wide Python scan).
        total_coverage = np.zeros(edge_map_size, dtype=np.uint8)
        covered_files: list[str] = []
        remaining = list(file_edges.keys())

        while remaining:
            best_file = None
            best_new_edges = 0
            for fpath in remaining:
                edges = np.frombuffer(file_edges[fpath], dtype=np.uint8)
                new = int(np.count_nonzero(edges & ~total_coverage))
                if new > best_new_edges:
                    best_new_edges = new
                    best_file = fpath

            if best_file is None or best_new_edges == 0:
                break

            covered_files.append(best_file)
            total_coverage |= np.frombuffer(file_edges[best_file], dtype=np.uint8)
            remaining.remove(best_file)

    return _commit_results(corpus_files, covered_files, output_dir, corpus_path)


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
) -> tuple[int, int]:
    """Write minimized corpus to output dir or prune in-place.

    When no output_dir is specified, removed files are moved to a
    ``pruned/`` subfolder inside the corpus directory instead of being
    deleted.  This preserves coverage-redundant inputs for later
    analysis while keeping the active corpus lean.
    """
    kept_set = set(kept)
    out_path = Path(output_dir) if output_dir else corpus_path

    if output_dir:
        out_path.mkdir(parents=True, exist_ok=True)
        for fpath_str in kept:
            fpath = Path(fpath_str)
            shutil.copy2(fpath, out_path / fpath.name)
            meta = fpath.with_suffix(".txt")
            if meta.exists():
                shutil.copy2(meta, out_path / meta.name)
    else:
        pruned_dir = corpus_path / "pruned"
        pruned_dir.mkdir(parents=True, exist_ok=True)
        for fpath in corpus_files:
            if str(fpath) not in kept_set:
                dest = pruned_dir / fpath.name
                shutil.move(str(fpath), str(dest))
                meta = fpath.with_suffix(".txt")
                if meta.exists():
                    shutil.move(str(meta), str(pruned_dir / meta.name))

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
