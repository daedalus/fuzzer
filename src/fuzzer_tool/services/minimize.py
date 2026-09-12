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


from fuzzer_tool.adapters import libc_shm
from fuzzer_tool.adapters.shm import ShmCoverage


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

    seed_edges: dict[str, set[int]] = {}
    try:
        for i, fpath in enumerate(corpus_files):
            data = fpath.read_bytes()
            shm.reset_edge_map()

            if file_mode:
                run_target_file(
                    target, data, timeout, str(tmp_dir), target_args or [], env=env_base
                )
            else:
                run_target_stdin(target, data, timeout, env=env_base)

            seed_edges[str(fpath)] = shm.get_edge_ids()

            if (i + 1) % 10 == 0 or (i + 1) == len(corpus_files):
                print(f"\r[*] Replayed {i + 1}/{len(corpus_files)}...", end="", flush=True)
    finally:
        map_entries = shm.num_entries
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
    if not any(seed_edges.values()):
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

    if rate_distortion:
        print("[*] Using rate-distortion optimal pruning...")
        from fuzzer_tool.core.rate_distortion import RateDistortionCorpus

        rd = RateDistortionCorpus(map_size=map_entries)
        covered_files, actual_frac = rd.optimal_pruning(seed_edges, target_fraction=target_frac)
        print(
            f"[*] Rate-distortion: kept {len(covered_files)}/{len(corpus_files)} "
            f"files ({actual_frac:.1%} coverage)"
        )
    else:
        # Greedy set cover over edge-id sets. The previous version scored
        # candidates with numpy popcounts over uint8 bitmap views, which was
        # the right shape for AFL's byte bitmap and the wrong one for this
        # shim's entry table; set difference is both correct here and cheaper
        # than it looks, since the sets hold only edges actually hit.
        covered: set[int] = set()
        covered_files: list[str] = []
        remaining = list(seed_edges.keys())

        while remaining:
            best_file = None
            best_new_edges = 0
            for fpath in remaining:
                new = len(seed_edges[fpath] - covered)
                if new > best_new_edges:
                    best_new_edges = new
                    best_file = fpath

            if best_file is None or best_new_edges == 0:
                break

            covered_files.append(best_file)
            covered |= seed_edges[best_file]
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
