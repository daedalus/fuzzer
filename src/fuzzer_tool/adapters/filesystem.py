"""Filesystem operations for corpus and crash management.

Supports delta-encoded corpus storage with periodic full snapshots:
- Mutations are typically small edits to a parent, so storing
  (parent_hash, patch) instead of full bytes saves disk and preserves
  lineage. Falls back to full storage when diff > 25% of input.
- Every SNAPSHOT_INTERVAL generations, writes a full snapshot instead
  of a delta. This caps worst-case reconstruction cost (like git's
  loose/packed object split) and prevents unbounded chain depth.
"""

import hashlib
import json
import os
import time
from pathlib import Path

from fuzzer_tool.core.bloom import BloomFilter
from fuzzer_tool.core.crash_metadata import CrashMetadata
from fuzzer_tool.core.sanitizer import SanitizerReport
from fuzzer_tool.core.similarity import crash_signature_similarity

SNAPSHOT_INTERVAL = 20

# ── Memory bounds ────────────────────────────────────────────────────
SEEN_HASHES_MAX = 200_000  # max unique seed hashes retained


def compute_delta(parent: bytes, child: bytes) -> list[list[int]] | None:
    """Compute a compact byte-level diff between parent and child.

    Returns a list of [offset, new_byte] pairs for bytes that differ,
    or None if the diff isn't worth storing (> 25% of child size, or
    different lengths).

    The delta format is deliberately simple: just positions and new values.
    Parent bytes at those positions are overwritten; everything else is
    inherited. This is cheaper than a full diff algorithm and works well
    for fuzzer mutations (bit flips, byte replacements, small insertions
    that happen to preserve length).
    """
    if len(parent) != len(child):
        return None

    diff = []
    for i in range(len(parent)):
        if parent[i] != child[i]:
            diff.append([i, child[i]])

    # Not worth delta-encoding if diff covers > 25% of the input
    if len(diff) > len(child) // 4:
        return None

    return diff


def apply_delta(parent: bytes, diff: list[list[int]]) -> bytes:
    """Reconstruct child bytes from parent and delta.

    Args:
        parent: Full parent input bytes.
        diff: List of [offset, new_byte] pairs from compute_delta.

    Returns:
        Reconstructed child bytes.
    """
    child = bytearray(parent)
    for offset, new_byte in diff:
        child[offset] = new_byte
    return bytes(child)


def compute_delta_v2(parent: bytes, child: bytes) -> list[list] | None:
    """Compute a delta that handles length-changing mutations.

    Uses Levenshtein alignment to produce an edit script:
      [0, offset, byte]  -- substitution at offset
      [1, offset, byte]  -- insert byte before offset
      [2, offset, 0]     -- delete byte at offset

    Falls back to None if the edit script is > 25% of child size.
    This extends delta-encoding to splice, block_insert, block_delete,
    and any havoc chain that changes length.

    Args:
        parent: Original bytes.
        child: Mutated bytes.

    Returns:
        Edit script as list of [op, offset, byte_or_0], or None.
    """
    from fuzzer_tool.core.similarity import levenshtein_align

    script = levenshtein_align(parent, child)

    # Count non-match ops
    ops = [(op, pos, data) for op, pos, data in script if op != "match"]

    # Not worth delta-encoding if edits cover > 50% of child size.
    # v2 handles length-changing mutations which tend to have fewer edit ops
    # than the positional diff, so we use a more generous threshold.
    # Empty parent always gets delta-encoded (pure insertion).
    if parent and len(ops) > len(child) // 2:
        return None

    # Convert to compact format
    result = []
    for op, pos, data in ops:
        if op == "replace":
            result.append([0, pos, data[0]])
        elif op == "insert":
            result.append([1, pos, data[0]])
        elif op == "delete":
            result.append([2, pos, 0])

    return result


def apply_delta_v2(parent: bytes, diff: list[list]) -> bytes:
    """Reconstruct child from parent using v2 edit script.

    Args:
        parent: Full parent input bytes.
        diff: Edit script from compute_delta_v2.

    Returns:
        Reconstructed child bytes.
    """
    result = bytearray(parent)
    # Process in reverse order to keep offsets valid
    for op, offset, byte_val in reversed(diff):
        if op == 0:  # substitute
            if offset < len(result):
                result[offset] = byte_val
        elif op == 1:  # insert
            result[offset:offset] = bytes([byte_val])
        elif op == 2 and offset < len(result):  # delete
            del result[offset]
    return bytes(result)


def hash_data(data: bytes) -> str:
    """Compute fast hash for deduplication (xxhash, ~20x faster than SHA-256).

    Falls back to SHA-256 if xxhash is not installed.
    For crash filenames where collision resistance matters, use hash_data_crypto().

    Args:
        data: Raw bytes to hash.

    Returns:
        16-character hex digest.
    """
    try:
        import xxhash

        return xxhash.xxh64(data).hexdigest()[:16]
    except ImportError:
        return hashlib.sha256(data).hexdigest()[:16]


def hash_data_crypto(data: bytes) -> str:
    """Compute SHA-256 hash for crash filenames (collision-resistant).

    Used where cryptographic hash properties matter (crash filenames,
    reproducibility). For corpus dedup, use hash_data() instead.
    """
    return hashlib.sha256(data).hexdigest()[:16]


def load_corpus(corpus_dir: Path, bloom: BloomFilter | None = None) -> tuple[list[bytes], set[str]]:
    """Load existing corpus from corpus_dir/seeds/.

    Handles both full files (id_*.*) and delta-encoded files (delta_*.json).
    Delta files are reconstructed from their parent chain.

    Args:
        corpus_dir: Path to corpus directory (seeds live in seeds/ subdir).
        bloom: Optional bloom filter to populate for fast dedup.

    Returns:
        Tuple of (corpus list, seen hashes set).
    """
    corpus: list[bytes] = []
    seen: set[str] = set()
    seeds_dir = corpus_dir / "seeds"
    deltas_dir = corpus_dir / "deltas"

    # First pass: load all full files and build hash lookup for delta reconstruction
    full_files: dict[str, bytes] = {}
    delta_files: list[tuple[str, Path]] = []

    if seeds_dir.exists():
        for f in seeds_dir.iterdir():
            if not f.is_file():
                continue
            # Full file: id_*, legacy names, etc.
            data = f.read_bytes()
            h = hash_data(data)
            full_files[h] = data

    if deltas_dir.exists():
        for f in deltas_dir.iterdir():
            if not f.is_file():
                continue
            if f.suffix == ".json" and f.name.startswith("delta_"):
                h = f.name[6:-5]  # strip "delta_" prefix and ".json" suffix
                delta_files.append((h, f))

    # Load full files
    for h, data in full_files.items():
        if h not in seen:
            seen.add(h)
            if bloom is not None:
                bloom.add(h)
            corpus.append(data)

    # Reconstruct delta chains via topological resolution.
    # Each delta depends on its parent; resolve in order from full snapshots.
    if delta_files:
        resolved: dict[str, bytes] = dict(full_files)
        remaining = dict(delta_files)

        # Resolve in passes: each pass resolves deltas whose parent is already resolved.
        # Caps at SNAPSHOT_INTERVAL passes since chains can't be deeper than that.
        for _ in range(SNAPSHOT_INTERVAL + 1):
            if not remaining:
                break
            still_remaining = {}
            for h, f in remaining.items():
                try:
                    delta = json.loads(f.read_text())
                    parent_hash = delta["parent"]
                    if parent_hash in resolved:
                        version = delta.get("v", 1)
                        if version == 2:
                            reconstructed = apply_delta_v2(resolved[parent_hash], delta["diff"])
                        else:
                            reconstructed = apply_delta(resolved[parent_hash], delta["diff"])
                        resolved[h] = reconstructed
                    else:
                        still_remaining[h] = f
                except (json.JSONDecodeError, KeyError, IndexError):
                    pass  # corrupt delta — skip
            remaining = still_remaining

        for h, _ in delta_files:
            if h in resolved and h not in seen:
                seen.add(h)
                if bloom is not None:
                    bloom.add(h)
                corpus.append(resolved[h])

    if not corpus:
        corpus.append(b"AAAAAAAA")
    return corpus, seen


def save_to_corpus(
    data: bytes,
    corpus_dir: Path,
    seen_hashes: set[str],
    bloom: BloomFilter | None = None,
    parent: bytes | None = None,
    lineage_depth: int = 0,
) -> bool:
    """Save input to corpus if not already seen.

    Uses bloom filter as fast pre-check when available. False positives
    (bloom says "seen" but set says "new") fall through to the authoritative set.

    When parent is provided and the diff is compact (< 25% of child size),
    stores a delta file instead of the full input. Every SNAPSHOT_INTERVAL
    generations, forces a full snapshot to cap chain depth.

    Args:
        data: Input bytes to save.
        corpus_dir: Path to corpus directory.
        seen_hashes: Set of already-seen hashes.
        bloom: Optional bloom filter for fast pre-check.
        parent: Parent input bytes (for delta encoding).
        lineage_depth: Number of delta hops from the nearest full snapshot.

    Returns:
        True if saved (new), False if duplicate.
    """
    h = hash_data(data)
    if bloom is not None:
        if not bloom.query(h):
            bloom.add(h)
        elif h in seen_hashes:
            return False
        else:
            bloom.add(h)
    else:
        if h in seen_hashes:
            return False
    seen_hashes.add(h)
    # Cap seen_hashes to bound memory; bloom filter handles fast dedup
    if len(seen_hashes) > SEEN_HASHES_MAX:
        seen_hashes.clear()
    seeds_dir = corpus_dir / "seeds"
    seeds_dir.mkdir(parents=True, exist_ok=True)
    deltas_dir = corpus_dir / "deltas"

    # Force full snapshot at interval to cap chain depth.
    # v1 delta handles same-length mutations; v2 handles length-changing ones.
    use_delta = parent is not None and lineage_depth < SNAPSHOT_INTERVAL

    delta = None
    if use_delta:
        diff = compute_delta(parent, data)
        if diff is not None:
            parent_hash = hash_data(parent)
            delta = {"parent": parent_hash, "diff": diff, "v": 1}
        elif len(data) <= 512:
            # Try v2 for length-changing mutations on small inputs only.
            # v2 uses levenshtein_align which is O(n*m) — skip for large inputs.
            diff_v2 = compute_delta_v2(parent, data)
            if diff_v2 is not None:
                parent_hash = hash_data(parent)
                delta = {"parent": parent_hash, "diff": diff_v2, "v": 2}

    if delta is not None:
        deltas_dir.mkdir(parents=True, exist_ok=True)
        delta_file = deltas_dir / f"delta_{h}.json"
        delta_file.write_text(json.dumps(delta, separators=(",", ":")))
    else:
        corpus_file = seeds_dir / f"id_{h}"
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
        Base name of saved files (e.g. "crash_1234567890_abc12345_sig_signal6"),
        or False if duplicate.
    """
    h = hash_data(data)
    if h in crash_hashes:
        return False

    report = SanitizerReport.parse(stderr)
    sig = report.signature if report and report.is_valid() else f"signal:{abs(returncode)}"

    # Deduplicate by signature: skip if this crash signature was already seen.
    # Uses Levenshtein similarity for fuzzy matching — crashes at the same
    # function with different instruction offsets or inlined frames are grouped.
    # Only fuzzy-match sanitizer signatures (contain @); exact-match signal fallbacks.
    if sig in crash_sigs:
        crash_hashes.add(h)
        crash_sigs[sig] += 1
        return False

    if "@" in sig:
        for existing_sig in crash_sigs:
            if crash_signature_similarity(sig, existing_sig) >= 0.8:
                crash_hashes.add(h)
                crash_sigs[existing_sig] += 1
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

    return base_name
