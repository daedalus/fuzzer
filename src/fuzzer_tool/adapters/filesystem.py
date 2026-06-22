"""Filesystem operations for corpus and crash management."""

import hashlib
import time
from pathlib import Path

from fuzzer_tool.core.bloom import BloomFilter
from fuzzer_tool.core.sanitizer import SanitizerReport


def hash_data(data: bytes) -> str:
    """Compute SHA-256 hash prefix for deduplication.

    Args:
        data: Raw bytes to hash.

    Returns:
        16-character hex digest.
    """
    return hashlib.sha256(data).hexdigest()[:16]


def load_corpus(corpus_dir: Path, bloom: BloomFilter | None = None) -> tuple[list[bytes], set[str]]:
    """Load existing corpus from directory.

    Args:
        corpus_dir: Path to corpus directory.
        bloom: Optional bloom filter to populate for fast dedup.

    Returns:
        Tuple of (corpus list, seen hashes set).
    """
    corpus: list[bytes] = []
    seen: set[str] = set()
    if corpus_dir.exists():
        for f in corpus_dir.iterdir():
            if f.is_file():
                data = f.read_bytes()
                h = hash_data(data)
                if h not in seen:
                    seen.add(h)
                    if bloom is not None:
                        bloom.add(h)
                    corpus.append(data)
    if not corpus:
        corpus.append(b"AAAAAAAA")
    return corpus, seen


def save_to_corpus(
    data: bytes, corpus_dir: Path, seen_hashes: set[str], bloom: BloomFilter | None = None
) -> bool:
    """Save input to corpus if not already seen.

    Uses bloom filter as fast pre-check when available. False positives
    (bloom says "seen" but set says "new") fall through to the authoritative set.

    Args:
        data: Input bytes to save.
        corpus_dir: Path to corpus directory.
        seen_hashes: Set of already-seen hashes.
        bloom: Optional bloom filter for fast pre-check.

    Returns:
        True if saved (new), False if duplicate.
    """
    h = hash_data(data)
    if bloom is not None:
        # bloom.query=False → definitely not in filter (new)
        if not bloom.query(h):
            bloom.add(h)
        # bloom.query=True → maybe in filter; check authoritative set
        elif h in seen_hashes:
            return False  # confirmed duplicate
        else:
            bloom.add(h)  # false positive, still new
    else:
        if h in seen_hashes:
            return False
    seen_hashes.add(h)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    corpus_file = corpus_dir / f"id_{h}"
    corpus_file.write_bytes(data)
    return True


def save_crash(
    data: bytes,
    returncode: int,
    stderr: str,
    crashes_dir: Path,
    crash_hashes: set[str],
    crash_sigs: dict[str, int],
) -> bool:
    """Save crash input with metadata.

    Args:
        data: Crashing input bytes.
        returncode: Process return code.
        stderr: Standard error output.
        crashes_dir: Path to crashes directory.
        crash_hashes: Set of already-seen crash hashes.
        crash_sigs: Dict of signature -> count.

    Returns:
        True if saved (new crash), False if duplicate.
    """
    h = hash_data(data)
    if h in crash_hashes:
        return False

    report = SanitizerReport.parse(stderr)
    sig = report.signature if report and report.is_valid() else f"signal:{abs(returncode)}"
    crash_hashes.add(h)
    crash_sigs[sig] = crash_sigs.get(sig, 0) + 1

    ts = int(time.time())
    crash_file = crashes_dir / f"crash_{ts}_{h}"
    crash_file.write_bytes(data)

    meta = crash_file.with_suffix(".txt")
    lines = [f"returncode: {returncode}"]
    if report and report.is_valid():
        lines.extend(
            [
                f"sanitizer: {report.sanitizer}",
                f"error: {report.error_type}",
                f"fault_addr: {report.fault_addr}",
                f"signature: {sig}",
                f"seen: {crash_sigs[sig]}x",
                "",
                "=== stack trace ===",
            ]
        )
        for i, frame in enumerate(report.frames[:12]):
            lines.append(f"  #{i} {frame}")
        lines.extend(["", "=== raw stderr ===", report.raw])
    else:
        lines.extend(["", "=== stderr ===", stderr])
    meta.write_text("\n".join(lines))
    return True
