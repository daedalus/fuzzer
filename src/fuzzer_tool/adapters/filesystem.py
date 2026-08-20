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


# ── Canonical corpus layout ──────────────────────────────────────────
# save_to_corpus() writes full snapshots under seeds/<hh>/ and delta records
# under deltas/ — deltas is a *sibling* of seeds, not a child of it. Pruned
# entries are retained on disk for rehydration but are deliberately not loaded
# back into the live corpus.
_CORPUS_FULL_ROOTS = ("seeds",)  # recursively scanned; pruned/ skipped inside
_CORPUS_DELTA_ROOTS = ("deltas", "seeds")  # seeds/ kept for legacy layouts
_REHYDRATE_FULL_ROOTS = ("seeds", "seeds/pruned", "seeds/irreplaceable")
_REHYDRATE_DELTA_ROOTS = ("deltas", "deltas/pruned")


def rehydrate_by_hash(h: str, corpus_dir: Path, _depth: int = 0) -> bytes | None:
    """Reconstruct seed bytes for a 16-hex content hash from disk.

    Checks full seed files (``seeds/``, ``seeds/pruned/``,
    ``seeds/irreplaceable/``) and delta records (``deltas/``,
    ``deltas/pruned/``), reconstructing delta chains recursively via
    ``apply_delta`` / ``apply_delta_v2``. Returns None when the hash is
    not present anywhere (or the delta chain is unresolvable/cyclic).
    """
    if _depth > SNAPSHOT_INTERVAL + 2:
        return None
    corpus_dir = Path(corpus_dir)
    for base in _REHYDRATE_FULL_ROOTS:
        full = corpus_dir / base / h[:2] / f"id_{h}"
        if full.is_file():
            return full.read_bytes()
    for base in _REHYDRATE_DELTA_ROOTS:
        delta_file = corpus_dir / base / h[:2] / f"delta_{h}.json"
        if not delta_file.is_file():
            continue
        try:
            rec = json.loads(delta_file.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        parent = rehydrate_by_hash(str(rec.get("parent", "")), corpus_dir, _depth + 1)
        diff = rec.get("diff")
        if parent is None or not isinstance(diff, list):
            continue
        if rec.get("v") == 2:
            return apply_delta_v2(parent, diff)
        return apply_delta(parent, diff)
    return None


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


def load_corpus(
    corpus_dir: Path,
    bloom: BloomFilter | None = None,
    add_default: bool = True,
    load_irreplaceable: bool = True,
) -> tuple[list[bytes], set[str], set[str]]:
    """Load existing corpus from all subdirectories of corpus_dir (except pruned/).

    Handles both full files (id_*.*) and delta-encoded files (delta_*.json).
    Delta files are reconstructed from their parent chain. Irreplaceable
    seeds are loaded from corpus/seeds/irreplaceable/ and tracked separately so
    they can be excluded from corpus pruning.

    Args:
        corpus_dir: Path to corpus directory. All subdirectories except
            pruned/ are scanned for seeds and delta files.
        bloom: Optional bloom filter to populate for fast dedup.
        add_default: If True and corpus is empty, add b"AAAAAAAA" as a
            synthetic default seed. Set False for commands that need to
            reflect the actual on-disk corpus state (e.g. sweep).
        load_irreplaceable: If True, seeds from corpus/seeds/irreplaceable/ are
            tracked as irreplaceable and excluded from corpus pruning.

    Returns:
        Tuple of (corpus list, seen hashes set, irreplaceable hashes set).
    """
    corpus: list[bytes] = []
    seen: set[str] = set()
    irreplaceable_hashes: set[str] = set()

    if not corpus_dir.exists():
        if add_default:
            return [b"AAAAAAAA"], set(), set()
        return [], set(), set()

    # First pass: load all full files and build hash lookup for delta reconstruction
    full_files: dict[str, bytes] = {}
    delta_files: list[tuple[str, Path]] = []

    def _load_full_from_dir(base_dir: Path, mark_irreplaceable: bool = False) -> None:
        """Read full files from base_dir and its two-digit subdirectories.

        Skips delta_*.json files (handled separately) and pruned/ subdirectories.
        Normalizes filenames to id_{hash} if they don't already match, so that
        pruning (which globs for id_*) can find them.
        """
        if not base_dir.exists():
            return
        for f in base_dir.iterdir():
            if not f.is_file():
                continue
            if f.is_symlink():
                continue
            try:
                if f.resolve().parent != base_dir.resolve():
                    continue
            except OSError:
                continue
            if f.suffix == ".json" and f.name.startswith("delta_"):
                continue  # handled as delta, not full file
            data = f.read_bytes()
            h = hash_data(data)
            # Normalize filename to id_{hash} so pruning can find it.
            expected = f"id_{h}"
            if f.name != expected:
                dest = f.with_name(expected)
                if not dest.exists():
                    f.rename(dest)
                else:
                    f.unlink()  # duplicate of an already-loaded seed
                f = dest
            full_files[h] = data
            if mark_irreplaceable:
                irreplaceable_hashes.add(h)
        for sub in base_dir.iterdir():
            if not sub.is_dir():
                continue
            if sub.name == "pruned":
                continue
            # Propagate mark_irreplaceable: if the subdirectory itself is
            # named "irreplaceable", mark its contents regardless of the
            # parent's mark_irreplaceable value. This handles the
            # corpus/seeds/irreplaceable/ layout.
            sub_mark = mark_irreplaceable or sub.name == "irreplaceable"
            _load_full_from_dir(sub, mark_irreplaceable=sub_mark)

    def _collect_deltas_from_dir(base_dir: Path) -> None:
        """Recursively collect delta_*.json files from base_dir."""
        if not base_dir.is_dir():
            return
        for f in base_dir.iterdir():
            if f.is_file() and f.suffix == ".json" and f.name.startswith("delta_"):
                h = f.name[6:-5]
                delta_files.append((h, f))
            elif f.is_dir() and f.name != "pruned":
                _collect_deltas_from_dir(f)

    # Scan the canonical corpus layout only. Anything else under corpus_dir
    # (cmplog workdirs, scratch files) is not corpus data and must not be
    # loaded as seeds. rehydrate_by_hash() is the authority on where entries
    # live; if a root is added there it must be added here too, or a hash
    # that rehydrates individually will be missing from a bulk load.
    for rel in _CORPUS_FULL_ROOTS:
        base = corpus_dir / rel
        if base.is_dir():
            _load_full_from_dir(base, mark_irreplaceable=False)

    # Deltas live in corpus/deltas/, a *sibling* of seeds/ (see save_to_corpus),
    # so scanning seeds/ alone silently drops every delta-encoded entry.
    for rel in _CORPUS_DELTA_ROOTS:
        base = corpus_dir / rel
        if base.is_dir():
            _collect_deltas_from_dir(base)

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

    if not corpus and add_default:
        corpus.append(b"AAAAAAAA")
    return corpus, seen, irreplaceable_hashes


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
        # Store seeds in two-digit hash subdirectories to avoid too many
        # files in a single directory.
        sub_dir = seeds_dir / h[:2]
        sub_dir.mkdir(parents=True, exist_ok=True)
        corpus_file = sub_dir / f"id_{h}"
        corpus_file.write_bytes(data)
    return True


def save_irreplaceable(
    data: bytes,
    corpus_dir: Path,
    seen_hashes: set[str],
    irreplaceable_hashes: set[str],
    bloom: BloomFilter | None = None,
) -> bool:
    """Save input to corpus/seeds/irreplaceable/ and mark as irreplaceable.

    Irreplaceable seeds are never pruned by auto_minimize_corpus().
    Saved to corpus/seeds/irreplaceable/ using the same two-digit hash subdirectory layout.

    Args:
        data: Input bytes to save.
        corpus_dir: Path to corpus directory.
        seen_hashes: Set of already-seen hashes (updated with new hash).
        irreplaceable_hashes: Set of irreplaceable hashes (updated with new hash).
        bloom: Optional bloom filter for fast pre-check.

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
    irreplaceable_hashes.add(h)
    if len(seen_hashes) > SEEN_HASHES_MAX:
        seen_hashes.clear()

    irep_dir = corpus_dir / "seeds" / "irreplaceable"
    irep_dir.mkdir(parents=True, exist_ok=True)
    sub_dir = irep_dir / h[:2]
    sub_dir.mkdir(parents=True, exist_ok=True)
    corpus_file = sub_dir / f"id_{h}"
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
    crash_blocklist: set[str] | None = None,
    crash_allowlist: set[str] | None = None,
    crash_min_sizes: dict[str, int] | None = None,
    fault_addr: int | None = None,
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
        crash_blocklist: Set of stack hashes to skip (known crashes).
        crash_allowlist: Set of stack hashes that override blocklist.
        crash_min_sizes: Dict of stack_hash -> minimum trigger size.
        fault_addr: Optional faulting memory address (si_addr) captured by the
            ptrace runner; folded into the fallback signal signature so
            same-signal crashes at different addresses dedup separately.

    Returns:
        Base name of saved files (e.g. "crash_1234567890_abc12345_sig_signal6"),
        or False if duplicate or filtered.
    """
    h = hash_data(data)
    if h in crash_hashes:
        return False

    report = SanitizerReport.parse(stderr)
    if report and report.is_valid():
        sig = report.signature
    elif fault_addr is not None:
        # Distinguish NULL-deref / wild-pointer / stack-overflow crashes that
        # all share the same signal number but fault at different addresses.
        sig = f"signal:{abs(returncode)}@{fault_addr:#x}"
    else:
        sig = f"signal:{abs(returncode)}"

    # Stack hash for blocklist/allowlist filtering
    stack_h = report.stack_hash() if report else ""

    # Blocklist check: skip crashes with known stack hashes unless allowlisted
    if (
        crash_blocklist
        and stack_h
        and stack_h in crash_blocklist
        and stack_h not in (crash_allowlist or set())
    ):
        crash_hashes.add(h)
        crash_sigs[sig] = crash_sigs.get(sig, 0) + 1
        return False

    # Deduplicate by signature: skip if this crash signature was already seen.
    # Uses Levenshtein similarity for fuzzy matching — crashes at the same
    # function with different instruction offsets or inlined frames are grouped.
    # Only fuzzy-match sanitizer signatures: normalize_frame() strips 0x-addresses
    # and numbers, which is noise for ASAN sigs but IS the distinguishing signal
    # for address-bearing fallback sigs like "signal:11@0xdead0000".
    if sig in crash_sigs:
        crash_hashes.add(h)
        crash_sigs[sig] += 1
        return False

    if report and report.is_valid() and "@" in sig:
        for existing_sig in crash_sigs:
            if crash_signature_similarity(sig, existing_sig) >= 0.8:
                crash_hashes.add(h)
                crash_sigs[existing_sig] += 1
                return False

    crash_hashes.add(h)
    crash_sigs[sig] = 1

    # Smaller crash replacement: if a crash with the same stack hash exists
    # and the new trigger is smaller, remove the old one.
    if crash_min_sizes is not None and stack_h:
        old_min = crash_min_sizes.get(stack_h)
        if old_min is not None and len(data) >= old_min:
            # New trigger is not smaller — skip replacement
            pass
        elif old_min is not None:
            # New trigger is smaller — find and remove the old crash files
            for f in crashes_dir.iterdir():
                if f.is_file() and f.suffix in (".bin", ".txt", ".sh", ".hex"):
                    try:
                        old_data = f.read_bytes() if f.suffix == ".bin" else None
                        if old_data and hash_data(old_data) != h:
                            # Check if this old crash has the same stack hash
                            old_report = SanitizerReport.parse(
                                (crashes_dir / f"{f.stem}.txt").read_text()
                                if (crashes_dir / f"{f.stem}.txt").exists()
                                else ""
                            )
                            if old_report and old_report.stack_hash() == stack_h:
                                # Remove old crash files
                                for ext in (".bin", ".txt", ".sh", ".hex"):
                                    old_file = crashes_dir / f"{f.stem}{ext}"
                                    if old_file.exists():
                                        old_file.unlink()
                                break
                    except Exception:
                        continue
        crash_min_sizes[stack_h] = len(data)

    # Build CrashMetadata if not provided
    if metadata is None:
        metadata = CrashMetadata()

    # Store raw stderr (contains ASAN file:line diagnostics)
    if not metadata.raw_stderr:
        metadata.raw_stderr = stderr

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
