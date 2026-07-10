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


def _read_shm_edges(shm_id: str, size: int = 65536) -> bytearray:
    """Read edge bitmap from AFL SHM segment."""
    import ctypes

    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    shmid = int(shm_id)
    ptr = libc.shmat(shmid, None, 0)
    if ptr == -1:
        return bytearray(size)
    data = ctypes.string_at(ptr, size)
    libc.shmdt(ptr)
    return bytearray(data)


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

    corpus_files = sorted(
        f for f in corpus_path.iterdir() if f.is_file() and f.suffix not in (".txt", ".log")
    )
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
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        shmid = libc.shmget(0, edge_map_size, 0o600 | 0o2000)  # IPC_PRIVATE
        if shmid < 0:
            file_edges[str(fpath)] = bytearray(edge_map_size)
            continue
        env["__AFL_SHM_ID"] = str(shmid)

        if file_mode:
            run_target_file(target, data, timeout, str(tmp_dir), target_args or [], env=env)
        else:
            run_target_stdin(target, data, timeout, env=env)

        # Read the edge bitmap from SHM
        ptr = libc.shmat(shmid, None, 0)
        if ptr != -1:
            file_edges[str(fpath)] = bytearray(ctypes.string_at(ptr, edge_map_size))
            libc.shmdt(ptr)
        else:
            file_edges[str(fpath)] = bytearray(edge_map_size)
        libc.shmctl(shmid, 0, None)  # IPC_RMID

        if (i + 1) % 10 == 0 or (i + 1) == len(corpus_files):
            print(f"\r[*] Replayed {i + 1}/{len(corpus_files)}...", end="", flush=True)

    print()

    shutil.rmtree(tmp_dir, ignore_errors=True)

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
        # Greedy set cover
        total_coverage = bytearray(edge_map_size)
        covered_files: list[str] = []
        remaining = list(file_edges.keys())

        while remaining:
            best_file = None
            best_new_edges = 0
            for fpath in remaining:
                edges = file_edges[fpath]
                new = sum(1 for j in range(edge_map_size) if edges[j] and not total_coverage[j])
                if new > best_new_edges:
                    best_new_edges = new
                    best_file = fpath

            if best_file is None or best_new_edges == 0:
                break

            covered_files.append(best_file)
            for j in range(edge_map_size):
                if file_edges[best_file][j]:
                    total_coverage[j] = 1
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
