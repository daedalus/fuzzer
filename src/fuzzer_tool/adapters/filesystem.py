"""Filesystem operations for corpus and crash management."""

import hashlib
import os
import time
from pathlib import Path

from fuzzer_tool.core.bloom import BloomFilter
from fuzzer_tool.core.crash_metadata import CrashMetadata
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
    metadata: CrashMetadata | None = None,
) -> bool:
    """Save crash input with enriched triage metadata.

    Deduplicates by crash signature. Generates:
    - .bin — crash input bytes
    - .txt — enriched sidecar with all context
    - .sh — self-contained reproducer script
    - .hex — hexdump of input

    Args:
        data: Crashing input bytes.
        returncode: Process return code.
        stderr: Standard error output.
        crashes_dir: Path to crashes directory.
        crash_hashes: Set of already-seen crash hashes.
        crash_sigs: Dict of signature -> count.
        metadata: Optional pre-built CrashMetadata from the fuzzer.

    Returns:
        True if saved (new crash), False if duplicate.
    """
    h = hash_data(data)
    if h in crash_hashes:
        return False

    report = SanitizerReport.parse(stderr)
    sig = report.signature if report and report.is_valid() else f"signal:{abs(returncode)}"

    # Deduplicate by signature: skip if this crash signature was already seen
    if sig in crash_sigs:
        crash_hashes.add(h)
        crash_sigs[sig] += 1
        return False

    crash_hashes.add(h)
    crash_sigs[sig] = 1

    # Build CrashMetadata if not provided
    if metadata is None:
        metadata = CrashMetadata()

    metadata.build_cluster_id(sig)

    # Derive error short name for filename
    if report and report.is_valid():
        error_short = report.error_type.replace("-", "")[:20]
        sanitizer_short = report.sanitizer.replace("Sanitizer", "")[:4].lower()
    else:
        error_short = f"signal{abs(returncode)}"
        sanitizer_short = "sig"

    # Fill timestamp if not set
    if not metadata.timestamp:
        metadata.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if not metadata.fuzzer_pid:
        metadata.fuzzer_pid = os.getpid()

    ts = int(time.time())
    base_name = f"crash_{ts}_{metadata.cluster_id}_{sanitizer_short}_{error_short}"

    # Write crash input
    crash_file = crashes_dir / f"{base_name}.bin"
    crash_file.write_bytes(data)

    # Build and write enriched sidecar
    if report:
        metadata.sanitizer = report.sanitizer
        metadata.error_type = report.error_type
        metadata.fault_addr = report.fault_addr
        metadata.frames = report.frames
        metadata.access_type = report.access_type
        metadata.access_size = report.access_size
        metadata.shadow_info = report.shadow_info
        metadata.alloc_frames = report.alloc_frames
        metadata.dealloc_frames = report.dealloc_frames
        metadata.exploitability = report.exploitability
    else:
        metadata.returncode = returncode

    sidecar = crashes_dir / f"{base_name}.txt"
    sidecar.write_text(metadata.format_sidecar())

    # Write reproducer script
    script = crashes_dir / f"{base_name}.sh"
    script.write_text(metadata.format_reproducer(data, metadata.target or "./target"))
    script.chmod(0o755)

    # Write hexdump
    hexdump_file = crashes_dir / f"{base_name}.hex"
    metadata.build_hexdump(data)
    metadata.build_text_repr(data)
    hexdump_file.write_text(metadata.input_hexdump + "\n\n" + metadata.input_text_repr + "\n")

    return True
