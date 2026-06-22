"""Corpus minimizer: prune redundant corpus entries while preserving coverage."""

import os
import shutil
import sys
from pathlib import Path


def minimize_corpus(
    target: str,
    corpus_dir: str,
    timeout: float = 5.0,
    file_mode: bool = False,
    target_args: list[str] | None = None,
    use_coverage: bool = False,
    output_dir: str | None = None,
) -> tuple[int, int]:
    """Minimize a corpus by removing inputs that don't contribute unique coverage.

    Replays each corpus file, collects edge coverage for each, then removes
    inputs whose coverage is fully covered by other inputs.

    Args:
        target: Path to the target binary.
        corpus_dir: Path to the corpus directory.
        timeout: Execution timeout in seconds.
        file_mode: Write input to temp file instead of stdin.
        target_args: Target arguments ({file} placeholder).
        use_coverage: Enable SHM coverage (passed to env).
        output_dir: Output directory for minimized corpus. If None, overwrites in-place.

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

    from fuzzer_tool.adapters.process import run_target_file, run_target_stdin

    tmp_dir = Path("/tmp") / f"cmin_{os.getpid()}"
    if file_mode:
        tmp_dir.mkdir(parents=True, exist_ok=True)

    edge_map_size = 65536
    import os as _os

    file_edges: dict[str, bytearray] = {}
    file_data: dict[str, bytes] = {}

    for i, fpath in enumerate(corpus_files):
        data = fpath.read_bytes()
        file_data[str(fpath)] = data

        edges = bytearray(edge_map_size)
        env = _os.environ.copy()
        if use_coverage:
            env["AFL_MAP_SIZE"] = str(edge_map_size)

        if file_mode:
            returncode, _ = run_target_file(
                target,
                data,
                timeout,
                str(tmp_dir),
                target_args or [],
                env=env,
            )
        else:
            returncode, _ = run_target_stdin(
                target,
                data,
                timeout,
                env=env,
            )

        file_edges[str(fpath)] = edges

        if (i + 1) % 10 == 0 or (i + 1) == len(corpus_files):
            print(f"\r[*] Replayed {i + 1}/{len(corpus_files)}...", end="", flush=True)

    print()

    # Greedy set cover: iteratively pick the file that covers the most new edges
    total_coverage = bytearray(edge_map_size)
    covered_files: list[str] = []

    remaining = list(file_edges.keys())
    while remaining:
        best_file = None
        best_new_edges = 0
        for fpath in remaining:
            edges = file_edges[fpath]
            new = sum(1 for i in range(edge_map_size) if edges[i] and not total_coverage[i])
            if new > best_new_edges:
                best_new_edges = new
                best_file = fpath

        if best_file is None or best_new_edges == 0:
            break

        covered_files.append(best_file)
        for i in range(edge_map_size):
            if file_edges[best_file][i]:
                total_coverage[i] = 1
        remaining.remove(best_file)

    kept = len(covered_files)
    removed = len(corpus_files) - kept
    print(f"[+] Minimized: {len(corpus_files)} -> {kept} files ({removed} removed)")

    out_path = Path(output_dir) if output_dir else corpus_path
    if output_dir:
        out_path.mkdir(parents=True, exist_ok=True)
        for fpath in covered_files:
            dest = out_path / Path(fpath).name
            shutil.copy2(fpath, dest)
            # Also copy metadata if exists
            meta = Path(fpath).with_suffix(".txt")
            if meta.exists():
                shutil.copy2(meta, out_path / meta.name)
    else:
        # In-place: remove files not in covered set
        covered_set = set(covered_files)
        for fpath in corpus_files:
            if str(fpath) not in covered_set:
                fpath.unlink(missing_ok=True)
                meta = fpath.with_suffix(".txt")
                if meta.exists():
                    meta.unlink(missing_ok=True)

    # Clean up temp dir
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return kept, removed


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
    )

    if removed == 0:
        print("[*] Corpus already minimal")
