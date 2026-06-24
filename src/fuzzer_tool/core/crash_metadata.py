"""Crash metadata collection for enriched triage output."""

import hashlib
from dataclasses import dataclass, field


@dataclass
class CrashMetadata:
    """All context needed for rich crash triage output.

    Collected from sanitizer reports, fuzzer state, and input analysis.
    """

    # Sanitizer report fields
    sanitizer: str = ""
    error_type: str = ""
    fault_addr: str = ""
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
        lines.append("")

        # Mutation info
        if self.parent_seed_hash:
            lines.append(f"parent_seed:   {self.parent_seed_hash}")
        if self.mutation_ops:
            lines.append(f"mutation_ops:  {', '.join(self.mutation_ops)}")
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

        # Register state
        if self.rip:
            lines.append("=== registers ===")
            lines.append(f"  RIP: {self.rip:#x}")
            lines.append(f"  RSP: {self.rsp:#x}")
            lines.append(f"  RBP: {self.rbp:#x}")
            if self.instruction_bytes:
                lines.append(f"  instruction: {self.instruction_bytes}")
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


def find_nearest_corpus(
    crash_data: bytes, corpus: list[bytes], max_check: int = 100
) -> tuple[str, float, list[int]]:
    """Find the corpus entry most similar to the crash input.

    Uses 4-gram Jaccard similarity on up to max_check corpus entries.

    Returns:
        Tuple of (nearest_label, similarity, diff_byte_offsets).
    """
    if not corpus:
        return "", 0.0, []

    crash_4grams = set()
    for i in range(max(0, len(crash_data) - 3)):
        crash_4grams.add(crash_data[i : i + 4])

    best_sim = 0.0
    best_idx = 0
    checked = corpus[:max_check]

    for idx, seed in enumerate(checked):
        seed_4grams = set()
        for i in range(max(0, len(seed) - 3)):
            seed_4grams.add(seed[i : i + 4])
        if not crash_4grams and not seed_4grams:
            sim = 1.0
        elif not crash_4grams or not seed_4grams:
            sim = 0.0
        else:
            intersection = len(crash_4grams & seed_4grams)
            union = len(crash_4grams | seed_4grams)
            sim = intersection / union if union else 0.0
        if sim > best_sim:
            best_sim = sim
            best_idx = idx

    # Compute diff bytes
    nearest = checked[best_idx]
    min_len = min(len(crash_data), len(nearest))
    diff = [i for i in range(min_len) if crash_data[i] != nearest[i]]
    diff += list(range(min_len, max(len(crash_data), len(nearest))))

    label = f"seed_{best_idx}"
    return label, best_sim, diff[:30]
