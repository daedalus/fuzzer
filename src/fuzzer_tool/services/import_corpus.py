"""Import corpus from AFL/libFuzzer output directories.

Supports importing seeds from:
- AFL output: queue/id:* files and crashes/crash-* files
- libFuzzer output: corpus/ directory files
- Honggfuzz output: findings/ directory files
"""

import shutil
import string
import sys
from collections import Counter
from pathlib import Path

from fuzzer_tool.adapters.filesystem import hash_data

# Bytes considered part of a token: AFL++ autotokens groups printable
# ASCII, treating anything else as a separator. `_` and `-` are kept in
# rather than splitting tokens like "IHDR" from a hyphenated identifier,
# matching how AFL++'s afl-analyze token-extraction pass behaves.
_TOKEN_CHARS = frozenset((string.ascii_letters + string.digits + "_-.").encode())


def _write_seed(dest_corpus: Path, data: bytes, h: str) -> Path:
    """Write *data* into the standard `seeds/<h[:2]>/id_<h>` layout.

    Matches the main fuzzer's save_to_corpus layout so imported seeds land
    in the same place the fuzzer reads/writes (not the corpus root).
    """
    sub = dest_corpus / "seeds" / h[:2]
    sub.mkdir(parents=True, exist_ok=True)
    dest = sub / f"id_{h}"
    dest.write_bytes(data)
    return dest


def _existing_hashes(dest_corpus: Path) -> set[str]:
    """Hash every seed already stored in *dest_corpus* (seeds/ + legacy root)."""
    seen: set[str] = set()
    if not dest_corpus.is_dir():
        return seen
    for f in dest_corpus.rglob("id_*"):
        if f.is_file():
            try:
                seen.add(hash_data(f.read_bytes()))
            except OSError:
                continue
    return seen


def import_from_afl(
    afl_out_dir: str, target_corpus: str, target_crashes: str | None = None
) -> tuple[int, int]:
    """Import seeds from an AFL output directory.

    Args:
        afl_out_dir: Path to AFL output directory (contains queue/ and crashes/).
        target_corpus: Destination corpus directory.
        target_crashes: Destination crashes directory (optional).

    Returns:
        Tuple of (seeds_imported, crashes_imported).
    """
    afl_path = Path(afl_out_dir)
    if not afl_path.is_dir():
        print(f"[-] AFL output directory not found: {afl_out_dir}", file=sys.stderr)
        return 0, 0

    corpus_out = Path(target_corpus)
    corpus_out.mkdir(parents=True, exist_ok=True)
    seen_hashes = _existing_hashes(corpus_out)

    seeds_imported = 0
    crashes_imported = 0

    # Import from queue/
    queue_dir = afl_path / "queue"
    if queue_dir.is_dir():
        for f in sorted(queue_dir.iterdir()):
            if not f.is_file():
                continue
            data = f.read_bytes()
            h = hash_data(data)
            if h not in seen_hashes:
                seen_hashes.add(h)
                _write_seed(corpus_out, data, h)
                seeds_imported += 1

    # Import from crashes/
    crash_dir = afl_path / "crashes"
    if crash_dir.is_dir() and target_crashes:
        crash_out = Path(target_crashes)
        crash_out.mkdir(parents=True, exist_ok=True)
        for f in sorted(crash_dir.iterdir()):
            if not f.is_file() or f.suffix == ".txt":
                continue
            data = f.read_bytes()
            h = hash_data(data)
            dest = crash_out / f"imported_{h}.bin"
            if not dest.exists():
                dest.write_bytes(data)
                crashes_imported += 1
                # Copy metadata if exists
                meta = f.with_suffix(".txt")
                if meta.exists():
                    shutil.copy2(meta, crash_out / f"imported_{h}.txt")

    return seeds_imported, crashes_imported


def import_from_libfuzzer(corpus_dir: str, target_corpus: str) -> int:
    """Import seeds from a libFuzzer corpus directory.

    Args:
        corpus_dir: Path to libFuzzer corpus directory.
        target_corpus: Destination corpus directory.

    Returns:
        Number of seeds imported.
    """
    src = Path(corpus_dir)
    if not src.is_dir():
        print(f"[-] libFuzzer corpus not found: {corpus_dir}", file=sys.stderr)
        return 0

    dest = Path(target_corpus)
    dest.mkdir(parents=True, exist_ok=True)
    seen_hashes = _existing_hashes(dest)

    imported = 0
    for f in sorted(src.iterdir()):
        if not f.is_file():
            continue
        data = f.read_bytes()
        if not data:
            continue
        h = hash_data(data)
        if h not in seen_hashes:
            seen_hashes.add(h)
            _write_seed(dest, data, h)
            imported += 1

    return imported


def import_from_honggfuzz(
    findings_dir: str, target_corpus: str, target_crashes: str | None = None
) -> tuple[int, int]:
    """Import seeds from a honggfuzz findings directory.

    Args:
        findings_dir: Path to honggfuzz findings/ directory.
        target_corpus: Destination corpus directory.
        target_crashes: Destination crashes directory (optional).

    Returns:
        Tuple of (seeds_imported, crashes_imported).
    """
    src = Path(findings_dir)
    if not src.is_dir():
        print(f"[-] honggfuzz findings not found: {findings_dir}", file=sys.stderr)
        return 0, 0

    dest = Path(target_corpus)
    dest.mkdir(parents=True, exist_ok=True)
    seen_hashes = _existing_hashes(dest)

    imported = 0
    for f in sorted(src.iterdir()):
        if not f.is_file():
            continue
        data = f.read_bytes()
        if not data:
            continue
        h = hash_data(data)
        if h not in seen_hashes:
            seen_hashes.add(h)
            _write_seed(dest, data, h)
            imported += 1

    return imported, 0


def extract_tokens(data: bytes, min_len: int = 3, max_len: int = 32) -> set[bytes]:
    """Extract candidate dictionary tokens from *data*.

    A token is a maximal run of bytes in ``_TOKEN_CHARS`` (printable ASCII
    identifier/keyword characters) whose length is within
    ``[min_len, max_len]``. This mirrors AFL++'s autotokens pass: no format
    knowledge is needed, just enough of a heuristic to pull out
    magic strings, tag names, and keywords that a byte-level mutator would
    otherwise rarely stumble onto by chance.

    Args:
        data: Raw seed bytes to scan.
        min_len: Shortest token kept (shorter runs are mutation noise, not
            meaningful tokens -- e.g. single letters).
        max_len: Longest token kept (longer runs are usually encoded binary
            data that happens to fall in the ASCII range, not a keyword).

    Returns:
        Set of unique token byte strings found in *data*.
    """
    tokens: set[bytes] = set()
    start = -1
    for i, b in enumerate(data):
        if b in _TOKEN_CHARS:
            if start == -1:
                start = i
            continue
        if start != -1:
            if min_len <= i - start <= max_len:
                tokens.add(data[start:i])
            start = -1
    if start != -1 and min_len <= len(data) - start <= max_len:
        tokens.add(data[start:])
    return tokens


def build_autotoken_dictionary(
    corpus_dir: str, min_len: int = 3, max_len: int = 32, max_tokens: int = 200
) -> list[bytes]:
    """Tokenize every seed under *corpus_dir* into a whole-token dictionary.

    Tokens are ranked by how many distinct seeds contain them (not raw byte
    count), so a token repeated many times within one file doesn't crowd
    out tokens that are genuinely common across the corpus.

    Args:
        corpus_dir: Corpus directory (canonical ``seeds/<hh>/id_*`` layout,
            or a bare directory of seed files).
        min_len: Shortest token kept, see :func:`extract_tokens`.
        max_len: Longest token kept, see :func:`extract_tokens`.
        max_tokens: Cap on the number of tokens returned (most-common
            first) -- a dictionary with thousands of entries slows every
            havoc stage down for diminishing returns.

    Returns:
        Token byte strings, most-seeds-containing-it first.
    """
    from fuzzer_tool.adapters.filesystem import discover_seed_files

    doc_freq: Counter[bytes] = Counter()
    for path in discover_seed_files(corpus_dir):
        try:
            data = path.read_bytes()
        except OSError:
            continue
        doc_freq.update(extract_tokens(data, min_len, max_len))

    return [tok for tok, _count in doc_freq.most_common(max_tokens)]


def _escape_token(token: bytes) -> str:
    """Render *token* as an AFL dictionary-format quoted string."""
    out = []
    for b in token:
        if b in (0x22, 0x5C):  # '"' or '\\'
            out.append(f"\\{chr(b)}")
        elif 0x20 <= b < 0x7F:
            out.append(chr(b))
        else:
            out.append(f"\\x{b:02x}")
    return "".join(out)


def write_dictionary(tokens: list[bytes], path: str) -> None:
    """Write *tokens* to *path* in the standard AFL dictionary format.

    Args:
        tokens: Token byte strings, e.g. from :func:`build_autotoken_dictionary`.
        path: Destination ``.dict`` file path.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write("# Auto-generated dictionary (fuzzer-tool autotokens)\n")
        for i, tok in enumerate(tokens):
            f.write(f'auto_{i:04d}="{_escape_token(tok)}"\n')


def _detect_source_format(source: Path) -> str:
    """Infer the corpus format from a source directory's layout.

    Checked most-specific first, since an AFL output directory can also
    contain plain files at the top level and would otherwise look like a
    libFuzzer corpus.

    Args:
        source: Directory to inspect.

    Returns:
        One of ``"afl"``, ``"libfuzzer"``, ``"honggfuzz"``. Falls back to
        ``"afl"`` when the layout matches nothing, preserving the historical
        default.
    """
    if (source / "cases_honggfuzz").exists() or any(source.glob("SIG*.fuzz")):
        return "honggfuzz"
    # AFL writes queue/ (and usually crashes/ + hangs/ or a fuzzer_stats file).
    if (source / "queue").is_dir() or (source / "fuzzer_stats").is_file():
        return "afl"
    # A libFuzzer corpus is a flat directory of inputs with none of the above.
    if source.is_dir() and not any(
        (source / d).exists() for d in ("queue", "crashes", "findings", "hangs")
    ):
        return "libfuzzer"
    return "afl"


def main():
    """CLI entry point for fuzzer-tool import."""
    import argparse

    parser = argparse.ArgumentParser(description="Import corpus from AFL/libFuzzer/honggfuzz")
    parser.add_argument(
        "source_dir", help="Source directory (AFL output, libFuzzer corpus, or honggfuzz findings)"
    )
    parser.add_argument("-d", "--corpus", required=True, help="Destination corpus directory")
    parser.add_argument(
        "-o", "--crashes", default=None, help="Destination crashes directory (for AFL)"
    )
    parser.add_argument(
        "--format",
        choices=["afl", "libfuzzer", "honggfuzz"],
        default=None,
        help="Source format (default: auto-detect from directory layout)",
    )
    parser.add_argument(
        "--autotokens",
        default=None,
        metavar="FILE",
        help="Also tokenize the destination corpus into a whole-token "
        "AFL-format dictionary written to FILE",
    )
    args = parser.parse_args()

    # `args.format == "afl" or (args.format == "afl" and ...)` was a tautology:
    # the second disjunct can only be true when the first already is, and "afl"
    # is the DEFAULT. So every invocation without an explicit --format took the
    # AFL branch, and a libFuzzer corpus imported 0 seeds while printing a
    # success line. Auto-detect only when the user did not choose a format,
    # which argparse cannot distinguish from an explicit --format afl unless the
    # default is None.
    fmt = args.format
    if fmt is None:
        fmt = _detect_source_format(Path(args.source_dir))

    if fmt == "libfuzzer":
        imported = import_from_libfuzzer(args.source_dir, args.corpus)
        print(f"[+] Imported {imported} seeds from libFuzzer corpus")
    elif fmt == "honggfuzz":
        imported, _ = import_from_honggfuzz(args.source_dir, args.corpus, args.crashes)
        print(f"[+] Imported {imported} seeds from honggfuzz")
    else:
        seeds, crashes = import_from_afl(args.source_dir, args.corpus, args.crashes)
        print(f"[+] Imported {seeds} seeds, {crashes} crashes from AFL output")

    if args.autotokens:
        tokens = build_autotoken_dictionary(args.corpus)
        write_dictionary(tokens, args.autotokens)
        print(f"[+] Wrote {len(tokens)} autotokens to {args.autotokens}")
