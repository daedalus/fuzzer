"""Crash metadata collection for enriched triage output."""

from __future__ import annotations

import hashlib
import os
from collections import Counter
from dataclasses import dataclass, field

from fuzzer_tool.core.similarity import (
    hamming_similarity,
    levenshtein_diff_offsets,
    levenshtein_similarity,
    normalize_frame,
    normalized_frame_similarity,
)

# Crash clustering threshold (A2 of the seventeen-source survey).
_CRASH_CLUSTER_THRESHOLD: float = float(
    os.environ.get("FUZZER_CRASH_CLUSTER_THRESHOLD", "0.7") or "0.7"
)


def configure_crash_cluster(threshold: float = 0.7) -> None:
    """Set the default crash-clustering similarity threshold."""
    global _CRASH_CLUSTER_THRESHOLD
    _CRASH_CLUSTER_THRESHOLD = float(threshold)


@dataclass
class CrashMetadata:
    """All context needed for rich crash triage output.

    Collected from sanitizer reports, fuzzer state, and input analysis.
    """

    # Sanitizer report fields
    sanitizer: str = ""
    error_type: str = ""
    fault_addr: str = ""

    # GDB crash replay text (backtrace/registers/fault) embedded in the
    # sidecar; empty when gdb is unavailable or the target isn't traceable.
    gdb_replay: str = ""
    frames: list[str] = field(default_factory=list)
    access_size: int | None = None
    access_type: str | None = None  # "READ" / "WRITE" / "FREE"
    shadow_info: str = ""
    alloc_frames: list[str] | None = None
    dealloc_frames: list[str] | None = None

    # Exploitability
    exploitability: str = "UNKNOWN"

    # Cluster ID
    cluster_id: str = ""

    # Execution metadata
    timestamp: str = ""
    fuzzer_pid: int = 0
    exec_count: int = 0
    corpus_size: int = 0
    parent_seed_hash: str = ""
    mutation_ops: list[str] = field(default_factory=list)
    parent_sites: list[int] = field(default_factory=list)
    target: str = ""
    target_sha256: str = ""
    elapsed: str = ""

    # Input analysis
    input_hexdump: str = ""
    input_text_repr: str = ""
    nearest_corpus_file: str = ""
    nearest_similarity: float = 0.0
    diff_bytes: list[int] = field(default_factory=list)

    # Register state (ptrace)
    rip: int = 0
    rsp: int = 0
    rbp: int = 0
    instruction_bytes: str = ""

    # Raw stderr from the target (contains ASAN file:line info)
    raw_stderr: str = ""

    # Return code for non-sanitizer crashes
    returncode: int | None = None

    def build_cluster_id(self, signature: str) -> str:
        """Build 8-char cluster ID from crash signature."""
        self.cluster_id = hashlib.sha256(signature.encode()).hexdigest()[:8]
        return self.cluster_id

    def build_hexdump(self, data: bytes) -> str:
        """Build hexdump -C style output of the crash input (capped at 512 bytes)."""
        capped = data[:512]
        truncated = len(data) > 512
        lines = []
        for offset in range(0, len(capped), 16):
            chunk = capped[offset : offset + 16]
            hex_part = " ".join(f"{b:02x}" for b in chunk)
            ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
            lines.append(f"{offset:08x}  {hex_part:<48s}  |{ascii_part}|")
        if truncated:
            lines.append(f"... ({len(data) - 512} more bytes truncated)")
        self.input_hexdump = "\n".join(lines)
        return self.input_hexdump

    def build_text_repr(self, data: bytes) -> str:
        """Build escaped text representation of the input."""
        parts = []
        for b in data:
            if 32 <= b < 127:
                parts.append(chr(b))
            elif b == 9:
                parts.append("\\t")
            elif b == 10:
                parts.append("\\n")
            elif b == 13:
                parts.append("\\r")
            else:
                parts.append(f"\\x{b:02x}")
        self.input_text_repr = "".join(parts)
        return self.input_text_repr

    def format_sidecar(self) -> str:
        """Format the complete .txt sidecar content."""
        lines = []

        # Header
        lines.append("# Crash Report")
        lines.append(f"timestamp:     {self.timestamp}")
        lines.append(f"fuzzer_pid:    {self.fuzzer_pid}")
        lines.append(f"exec_count:    {self.exec_count}")
        lines.append(f"corpus_size:   {self.corpus_size}")
        lines.append(f"elapsed:       {self.elapsed}")
        lines.append(f"target:        {self.target}")
        lines.append(f"target_sha256: {self.target_sha256}")
        lines.append("")

        # Sanitizer info
        if self.sanitizer:
            lines.append(f"sanitizer:     {self.sanitizer}")
            lines.append(f"error_type:    {self.error_type}")
            lines.append(f"fault_addr:    {self.fault_addr}")
            if self.access_type and self.access_size is not None:
                lines.append(f"access:        {self.access_type} of size {self.access_size}")
            if self.shadow_info:
                lines.append(f"shadow:        {self.shadow_info}")
            lines.append(f"exploitability: {self.exploitability}")
            lines.append(f"cluster_id:    {self.cluster_id}")
        else:
            if self.returncode is not None:
                lines.append(f"returncode:    {self.returncode}")
            else:
                lines.append("returncode:    signal (see raw stderr)")
            if self.error_type:
                lines.append(f"error_type:    {self.error_type}")
            if self.fault_addr:
                lines.append(f"fault_addr:    {self.fault_addr}")
        lines.append("")

        # Mutation info
        if self.parent_seed_hash:
            lines.append(f"parent_seed:   {self.parent_seed_hash}")
        if self.mutation_ops:
            lines.append(f"mutation_ops:  {', '.join(self.mutation_ops)}")
        if self.parent_sites:
            lines.append(f"mutation_sites: {', '.join(str(s) for s in self.parent_sites)}")
        lines.append("")

        # Stack trace
        if self.frames:
            lines.append("=== stack trace ===")
            for i, frame in enumerate(self.frames[:16]):
                lines.append(f"  #{i} {frame}")
            lines.append("")

        # Allocation/deallocation stacks
        if self.alloc_frames:
            lines.append("=== allocated by ===")
            for i, frame in enumerate(self.alloc_frames[:8]):
                lines.append(f"  #{i} {frame}")
            lines.append("")

        if self.dealloc_frames:
            lines.append("=== freed by ===")
            for i, frame in enumerate(self.dealloc_frames[:8]):
                lines.append(f"  #{i} {frame}")
            lines.append("")

        # Register state. Gate on ANY register being nonzero (not just RIP):
        # a NULL-jump crash has rip == 0 (the faulting address IS 0) but a
        # meaningful rsp/rbp that would otherwise be dropped from the sidecar.
        if self.rip or self.rsp or self.rbp:
            lines.append("=== registers ===")
            lines.append(f"  RIP: {self.rip:#x}")
            lines.append(f"  RSP: {self.rsp:#x}")
            lines.append(f"  RBP: {self.rbp:#x}")
            if self.instruction_bytes:
                lines.append(f"  instruction: {self.instruction_bytes}")
            lines.append("")

        # GDB crash replay (backtrace/registers/fault) — part of the report
        if self.gdb_replay:
            lines.append(self.gdb_replay)
            lines.append("")

        # Nearest corpus
        if self.nearest_corpus_file:
            lines.append(
                f"nearest_corpus: {self.nearest_corpus_file} (similarity: {self.nearest_similarity:.2f})"
            )
            if self.diff_bytes:
                offsets = ", ".join(f"0x{o:02x}" for o in self.diff_bytes[:20])
                lines.append(
                    f"diff_bytes: {len(self.diff_bytes)} bytes differ at offsets [{offsets}]"
                )
            lines.append("")

        # Raw stderr (ASAN diagnostics with file:line)
        if self.raw_stderr:
            lines.append("=== raw stderr ===")
            lines.append(self.raw_stderr)
            lines.append("")

        # Input hexdump
        if self.input_hexdump:
            lines.append("=== input hexdump ===")
            lines.append(self.input_hexdump)
            lines.append("")

        # Input text
        if self.input_text_repr:
            lines.append("=== input text ===")
            lines.append(self.input_text_repr)
            lines.append("")

        return "\n".join(lines)

    def format_reproducer(self, data: bytes, target: str) -> str:
        """Generate a self-contained reproducer shell script."""
        import base64

        b64 = base64.b64encode(data).decode()
        sig = f"{self.error_type} @ {self.frames[0]}" if self.frames else "crash"
        lines = [
            "#!/bin/bash",
            f"# Reproducer: {sig}",
            f"# Generated: {self.timestamp}",
            f"# Input SHA256: {hashlib.sha256(data).hexdigest()[:16]}",
            f"# Target: {target}",
            f"# Exploitability: {self.exploitability}",
            "",
            "set -e",
            "",
        ]
        # Use printf for inputs > 128KB to avoid shell arg length limits
        if len(b64) > 128 * 1024:
            lines.extend(
                [
                    "B64_DATA=$(cat <<'ENDOFB64'",
                    b64,
                    "ENDOFB64",
                    ")",
                    "printf '%s' \"$B64_DATA\" | base64 -d | \\",
                ]
            )
        else:
            lines.append(f"printf '%s' '{b64}' | base64 -d | \\")
        lines.extend(
            [
                "  ASAN_OPTIONS=abort_on_error=1:symbolize=1:detect_leaks=0 \\",
                f"  {target}",
                "",
            ]
        )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize the metadata to a JSON-compatible dict.

        Mirrors ``format_sidecar()`` field-for-field (the ``.txt`` sidecar
        is the human-readable rendering; this dict is the machine-readable
        equivalent written as ``.json`` next to the ``.bin`` so dashboards
        and downstream tools can ingest a crash without re-parsing the txt).

        The list fields are kept as lists (not stringified), the
        ``returncode`` stays an int-or-None, and the empty-string defaults
        are kept as strings -- the on-disk sidecar preserves the same
        shape whether the value was filled or left at default.
        """
        return {
            "timestamp": self.timestamp,
            "fuzzer_pid": self.fuzzer_pid,
            "exec_count": self.exec_count,
            "corpus_size": self.corpus_size,
            "elapsed": self.elapsed,
            "target": self.target,
            "target_sha256": self.target_sha256,
            "sanitizer": self.sanitizer,
            "error_type": self.error_type,
            "fault_addr": self.fault_addr,
            "access_type": self.access_type,
            "access_size": self.access_size,
            "shadow_info": self.shadow_info,
            "exploitability": self.exploitability,
            "cluster_id": self.cluster_id,
            "returncode": self.returncode,
            "parent_seed_hash": self.parent_seed_hash,
            "mutation_ops": list(self.mutation_ops),
            "parent_sites": list(self.parent_sites),
            "frames": list(self.frames),
            "alloc_frames": list(self.alloc_frames) if self.alloc_frames is not None else None,
            "dealloc_frames": list(self.dealloc_frames)
            if self.dealloc_frames is not None
            else None,
            "registers": {
                "rip": self.rip,
                "rsp": self.rsp,
                "rbp": self.rbp,
                "instruction_bytes": self.instruction_bytes,
            },
            "gdb_replay": self.gdb_replay,
            "nearest_corpus_file": self.nearest_corpus_file,
            "nearest_similarity": self.nearest_similarity,
            "diff_bytes": list(self.diff_bytes),
            "raw_stderr": self.raw_stderr,
            "input_hexdump": self.input_hexdump,
            "input_text_repr": self.input_text_repr,
        }


def find_nearest_corpus(
    crash_data: bytes, corpus: list[bytes], max_check: int = 100
) -> tuple[str, float, list[int], str]:
    """Find the corpus entry most similar to the crash input.

    Architecture: cheap 4-gram Jaccard scan to find the candidate, then
    one expensive Levenshtein alignment against the winner to produce the
    actual diff. Neither technique does both jobs.

    Jaccard picks the target (O(n) per entry, length-agnostic);
    Levenshtein describes what actually changed (exact edit positions).

    Returns:
        Tuple of (nearest_label, similarity, diff_byte_offsets, edit_summary).
    """
    if not corpus:
        return "", 0.0, [], ""

    # Phase 1: cheap Jaccard scan to find nearest candidate
    best_jaccard = 0.0
    best_idx = 0
    checked = corpus[:max_check]

    for idx, seed in enumerate(checked):
        crash_4grams = set()
        for i in range(max(0, len(crash_data) - 3)):
            crash_4grams.add(crash_data[i : i + 4])
        seed_4grams = set()
        for i in range(max(0, len(seed) - 3)):
            seed_4grams.add(seed[i : i + 4])
        if not crash_4grams and not seed_4grams:
            jaccard = 1.0
        elif not crash_4grams or not seed_4grams:
            jaccard = 0.0
        else:
            intersection = len(crash_4grams & seed_4grams)
            union = len(crash_4grams | seed_4grams)
            jaccard = intersection / union if union else 0.0

        if jaccard > best_jaccard:
            best_jaccard = jaccard
            best_idx = idx

    # Phase 2: one Levenshtein alignment against the winner for the diff
    nearest = checked[best_idx]
    diff = levenshtein_diff_offsets(crash_data, nearest)

    # Combined similarity score for the metadata
    if len(crash_data) == len(nearest):
        byte_sim = hamming_similarity(crash_data, nearest)
    else:
        byte_sim = levenshtein_similarity(crash_data, nearest)
    sim = 0.4 * best_jaccard + 0.6 * byte_sim

    label = f"seed_{best_idx}"
    from fuzzer_tool.core.similarity import edit_script_summary

    edit_summary = edit_script_summary(nearest, crash_data)
    return label, sim, diff[:30], edit_summary


def cluster_crashes(
    signatures: list[str],
    frame_lists: list[list[str]] | None = None,
    threshold: float | None = None,
) -> list[list[int]]:
    """Cluster crash signatures by Levenshtein similarity (single linkage).

    When ``frame_lists`` is provided, pairs where both sides have frames use
    order-aware frame-sequence Levenshtein (which distinguishes A->B->C from
    C->B->A); every other pair falls back to the signature-string metric.

    The pair loop is pruned with two *exact* lower bounds on edit distance, so
    the result is identical to comparing all ``n*(n-1)/2`` pairs -- only the
    cost differs. An earlier version sparsified with MinHash LSH instead and
    lost about 80% of the pairs it should have merged, because MinHash
    approximates Jaccard while the threshold here is on Levenshtein: a pair can
    sit at Levenshtein 0.8 with a Jaccard far below the band threshold and never
    become a candidate. No banding parameter fixes that, so the bounds below are
    used instead -- they discard only pairs that provably cannot clear the
    threshold.

    Args:
        signatures: List of crash signature strings.
        frame_lists: Optional list of frame lists (in call order) per crash.
        threshold: Minimum similarity to group into the same cluster.
            Defaults to the configured value (0.7).

    Returns:
        List of clusters, where each cluster is a list of indices into
        the signatures list.
    """
    if not signatures:
        return []

    if threshold is None:
        threshold = _CRASH_CLUSTER_THRESHOLD

    n = len(signatures)
    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path halving
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px == py:
            return
        if rank[px] < rank[py]:
            parent[px] = py
        elif rank[px] > rank[py]:
            parent[py] = px
        else:
            parent[py] = px
            rank[px] += 1

    # Both metrics are 1 - dist/max_len, so the pass is driven by the slack the
    # threshold leaves: a pair can only clear it if
    # dist <= (1 - threshold) * max_len. Two sound lower bounds on dist follow,
    # and both are far cheaper than the alignment they replace:
    #   * dist >= |len_a - len_b|, so the shorter side must be at least
    #     threshold * len of the longer one -- a window over length-sorted
    #     order, advanced with a monotone pointer.
    #   * dist >= max_len - |bag_a & bag_b|, since only equal elements can be
    #     matched at zero cost, so the multiset intersection must reach
    #     threshold * max_len.
    # The third saving is not a bound: this is single linkage, so a pair already
    # in the same component contributes nothing and is skipped outright.
    #
    # normalize_frame is hoisted out of the pair loop here. It is applied once
    # per input rather than twice per pair, which is exact --
    # crash_signature_similarity is by definition
    # levenshtein_similarity(normalize_frame(x).encode(), ...) and
    # frame_sequence_similarity already truncates at 8 frames.
    framed = frame_lists is not None and len(frame_lists) > 0
    n_framed = len(frame_lists) if framed else 0

    tok_keys: list[list[str]] = []
    if framed:
        tok_keys = [[normalize_frame(f) for f in frames[:8]] for frames in frame_lists]
    sig_keys = [normalize_frame(sig).encode() for sig in signatures]

    def _pass(idxs, keys, sim_fn, skip_both) -> None:
        if len(idxs) < 2:
            return
        order = sorted(idxs, key=lambda i: len(keys[i]))
        bags = {i: Counter(keys[i]) for i in order}
        lo = 0
        for b in range(len(order)):
            j = order[b]
            len_j = len(keys[j])
            # Lengths are non-decreasing along `order`, so this pointer only
            # moves forward: anything shorter than threshold * len_j can never
            # close the length gap, for this j or any later one.
            while lo < b and len(keys[order[lo]]) < threshold * len_j:
                lo += 1
            bag_j = bags[j]
            for a in range(lo, b):
                i = order[a]
                if skip_both is not None and i in skip_both and j in skip_both:
                    continue
                if find(i) == find(j):
                    continue
                max_len = len_j or len(keys[i])
                if max_len and sum((bags[i] & bag_j).values()) < threshold * max_len:
                    continue
                if sim_fn(i, j) >= threshold:
                    union(i, j)

    if framed:
        _pass(
            [i for i in range(n) if i < n_framed],
            tok_keys,
            lambda i, j: normalized_frame_similarity(tok_keys[i], tok_keys[j]),
            None,
        )
    # Every pair the frame pass did not own falls back to the signature metric,
    # matching the original `frame_lists and i < len and j < len` dispatch.
    _pass(
        list(range(n)),
        sig_keys,
        lambda i, j: levenshtein_similarity(sig_keys[i], sig_keys[j]),
        {i for i in range(n) if i < n_framed} if framed else None,
    )

    clusters_map: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        clusters_map.setdefault(root, []).append(i)

    return list(clusters_map.values())
