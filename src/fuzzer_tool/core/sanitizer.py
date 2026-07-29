"""Sanitizer output parsing for crash detection."""

import re

ASAN_ERROR_TYPES = (
    r"heap-buffer-overflow|stack-buffer-overflow|heap-use-after-free"
    r"|global-buffer-overflow|stack-buffer-underflow|heap-buffer-overflow-"
    r"|dynamic-stack-buffer-overflow|stack-use-after-return|stack-use-after-scope"
    r"|allocation-size-too-big|double-free|invalid-malloc-size"
    r"|attempting-free-on-non-deallocated-memory"
    r"|negative-size-param|heap-use-after-scope"
    r"|calloc-overflow|negative-array-size|negative-memset-size"
    r"|memcpy-param-overlap|strncat-param-overlap"
    r"|initialization-order-fiasco|odr-violation"
    r"|alloc-dealloc-mismatch|new-delete-type-mismatch"
)

UBSAN_ERROR_TYPES = (
    r"undefined|shift-exponent|signed-integer-overflow"
    r"|null-pointer-use|integer-divide-by-zero"
    r"|pointer-overflow|misaligned-pointer-use"
    r"|function-pointer-mismatch|bool-constant-evaluation"
    r"|builtin|array-bounds|float-cast-overflow"
    r"|implicit-signed-integer-truncation|implicit-integer-sign-change"
    r"|pointer-subtract-overflow|builtin-unreachable"
    r"|nonnull-attribute-violation|return-nonnull-attribute"
    r"|vla-bound-not-positive|unsigned-integer-overflow"
)

SANITIZER_PATTERNS = [
    (rf"AddressSanitizer:\s*({ASAN_ERROR_TYPES})", "ASAN"),
    (r"MemorySanitizer:\s*(use-of-uninitialized-value)", "MSAN"),
    (r"ThreadSanitizer:\s*(data-race|heap-use-after-race|lock-order-inversion)", "TSAN"),
    (r"LeakSanitizer:\s*(leak)", "LSAN"),
    (rf"UndefinedBehaviorSanitizer:\s*({UBSAN_ERROR_TYPES})", "UBSAN"),
]

# Combined regex for the main error line (used by SanitizerReport.parse)
SANITIZER_ERROR_LINE_RE = re.compile(
    r"(AddressSanitizer|MemorySanitizer|ThreadSanitizer|LeakSanitizer|UndefinedBehaviorSanitizer)"
    r":\s*(\S+)",
    re.IGNORECASE,
)

SANITIZER_STACK_FRAME_RE = re.compile(r"#\d+\s+0x[0-9a-f]+\s+in\s+(\S+)\s+.*")
SANITIZER_FAULT_ADDR_RE = re.compile(
    r"(?:Address|Memory)Sanitizer.*(?:on|at) address\s+(0x[0-9a-f]+)",
    re.IGNORECASE,
)

# Enriched ASAN output patterns
SANITIZER_ACCESS_RE = re.compile(
    r"(READ|WRITE|FREE)\s+of\s+size\s+(\d+)",
    re.IGNORECASE,
)
SANITIZER_SHADOW_RE = re.compile(
    r"(0x[0-9a-f]+,\s*(?:heap-.*|stack-.*|global-.*|freed|allocated|addressable|partial)\b[^\n]*)",
    re.IGNORECASE,
)
SANITIZER_ALLOC_RE = re.compile(
    r"allocated by thread (?:T\d+ )?(?:here|C\d+)\s*:?\s*\n(.*?)(?=\n\n|SUMMARY|\Z)",
    re.DOTALL | re.IGNORECASE,
)
SANITIZER_DEALLOC_RE = re.compile(
    r"freed by thread (?:T\d+ )?(?:here|C\d+)\s*:?\s*\n(.*?)(?=\n\n|SUMMARY|\Z)",
    re.DOTALL | re.IGNORECASE,
)

# PC extraction: "#N 0xADDR in func"
_SANITIZER_PC_RE = re.compile(r"#\d+\s+(0x[0-9a-fA-F]+)")

# Single-frame mask to prevent false uniqueness on 1-frame crashes
_SINGLE_FRAME_MASK = 0xDEAD

# Number of frames to hash
_NUM_FRAMES_NORMAL = 7
_NUM_FRAMES_SANITIZER = 14

# ── Exploitability ───────────────────────────────────────────────

ASAN_EXPLOITABILITY = {
    # WRITE variants → CRITICAL
    "heap-buffer-overflow": "CRITICAL",
    "stack-buffer-overflow": "CRITICAL",
    "global-buffer-overflow": "CRITICAL",
    "heap-use-after-free": "CRITICAL",
    "double-free": "CRITICAL",
    "heap-buffer-overflow-": "CRITICAL",
    "dynamic-stack-buffer-overflow": "CRITICAL",
    "alloc-dealloc-mismatch": "CRITICAL",
    "new-delete-type-mismatch": "CRITICAL",
    "calloc-overflow": "CRITICAL",
    "negative-array-size": "CRITICAL",
    "negative-memset-size": "CRITICAL",
    "memcpy-param-overlap": "CRITICAL",
    "strncat-param-overlap": "CRITICAL",
    # READ variants → MEDIUM-HIGH
    "stack-buffer-underflow": "HIGH",
    "stack-use-after-return": "HIGH",
    "stack-use-after-scope": "HIGH",
    "heap-use-after-scope": "MEDIUM",
    "allocation-size-too-big": "MEDIUM",
    "invalid-malloc-size": "MEDIUM",
    "attempting-free-on-non-deallocated-memory": "MEDIUM",
    "negative-size-param": "MEDIUM",
    "initialization-order-fiasco": "LOW",
    "odr-violation": "LOW",
}

UBSAN_EXPLOITABILITY = {
    "null-pointer-use": "HIGH",
    "function-pointer-mismatch": "HIGH",
    "pointer-overflow": "HIGH",
    "builtin": "HIGH",
    "array-bounds": "HIGH",
    "nonnull-attribute-violation": "HIGH",
    "return-nonnull-attribute": "HIGH",
    "signed-integer-overflow": "MEDIUM",
    "shift-exponent": "MEDIUM",
    "float-cast-overflow": "MEDIUM",
    "implicit-signed-integer-truncation": "MEDIUM",
    "implicit-integer-sign-change": "MEDIUM",
    "pointer-subtract-overflow": "MEDIUM",
    "vla-bound-not-positive": "MEDIUM",
    "misaligned-pointer-use": "LOW",
    "unsigned-integer-overflow": "LOW",
    "bool-constant-evaluation": "LOW",
    "builtin-unreachable": "LOW",
    "integer-divide-by-zero": "LOW",
}

# READ access downgrades CRITICAL to HIGH for heap overflow types
_ASAN_READ_DOWNGRADE = {
    "heap-buffer-overflow",
    "heap-buffer-overflow-",
    "global-buffer-overflow",
}


class SanitizerReport:
    """Parsed sanitizer output from a crashed process.

    Attributes:
        sanitizer: Sanitizer name (ASAN, MSAN, etc.).
        error_type: Specific error type (heap-buffer-overflow, etc.).
        fault_addr: Fault address string.
        frames: List of stack frame function names.
        raw: Raw stderr output.
        signature: Unique crash signature string.
        access_type: "READ", "WRITE", or "FREE" if detected.
        access_size: Memory access size in bytes if detected.
        shadow_info: Shadow memory description string.
        alloc_frames: Stack frames from allocation site.
        dealloc_frames: Stack frames from deallocation site.
        exploitability: Estimated exploitability (CRITICAL/HIGH/MEDIUM/LOW).
    """

    __slots__ = (
        "sanitizer",
        "error_type",
        "fault_addr",
        "frames",
        "raw",
        "signature",
        "access_type",
        "access_size",
        "shadow_info",
        "alloc_frames",
        "dealloc_frames",
        "exploitability",
    )

    def __init__(
        self,
        sanitizer: str,
        error_type: str,
        fault_addr: str,
        frames: list[str],
        raw: str,
    ):
        self.sanitizer = sanitizer
        self.error_type = error_type
        self.fault_addr = fault_addr
        self.frames = frames
        self.raw = raw
        self.signature = self._build_signature()

        # Enriched fields
        self.access_type: str | None = None
        self.access_size: int | None = None
        self.shadow_info: str = ""
        self.alloc_frames: list[str] | None = None
        self.dealloc_frames: list[str] | None = None
        self.exploitability: str = "UNKNOWN"
        self._parse_enriched_fields()

    def _parse_enriched_fields(self):
        """Parse additional fields from the raw stderr."""
        if not self.raw:
            return

        # Access type and size
        m = SANITIZER_ACCESS_RE.search(self.raw)
        if m:
            self.access_type = m.group(1).upper()
            self.access_size = int(m.group(2))

        # Shadow memory info
        m = SANITIZER_SHADOW_RE.search(self.raw)
        if m:
            self.shadow_info = m.group(1).strip()

        # Allocation stack
        m = SANITIZER_ALLOC_RE.search(self.raw)
        if m:
            self.alloc_frames = SANITIZER_STACK_FRAME_RE.findall(m.group(1))

        # Deallocation stack
        m = SANITIZER_DEALLOC_RE.search(self.raw)
        if m:
            self.dealloc_frames = SANITIZER_STACK_FRAME_RE.findall(m.group(1))

        # Exploitability
        if self.sanitizer == "AddressSanitizer":
            base = ASAN_EXPLOITABILITY.get(self.error_type, "MEDIUM")
            if self.access_type == "READ" and self.error_type in _ASAN_READ_DOWNGRADE:
                self.exploitability = "HIGH"
            else:
                self.exploitability = base
        elif self.sanitizer == "MemorySanitizer" or self.sanitizer == "ThreadSanitizer":
            self.exploitability = "MEDIUM"
        elif self.sanitizer == "UndefinedBehaviorSanitizer":
            self.exploitability = UBSAN_EXPLOITABILITY.get(self.error_type, "MEDIUM")
        elif self.sanitizer == "LeakSanitizer":
            self.exploitability = "LOW"

    def _build_signature(self) -> str:
        key = f"{self.sanitizer}:{self.error_type}"
        for f in self.frames[:6]:
            key += f"@{f}"
        return key

    @classmethod
    def parse(cls, stderr: str) -> "SanitizerReport | None":
        """Parse sanitizer output from stderr.

        Args:
            stderr: Standard error output from the target process.

        Returns:
            Parsed report, or None if no sanitizer output found.
        """
        m = SANITIZER_ERROR_LINE_RE.search(stderr)
        if not m:
            return None
        sanitizer = m.group(1)
        error_type = m.group(2).strip()

        fault_addr = ""
        addr_m = SANITIZER_FAULT_ADDR_RE.search(stderr)
        if addr_m:
            fault_addr = addr_m.group(1)

        frames = SANITIZER_STACK_FRAME_RE.findall(stderr)
        return cls(sanitizer, error_type, fault_addr, frames, stderr)

    def is_valid(self) -> bool:
        return bool(self.sanitizer and self.error_type)

    def stack_hash(self) -> str:
        """Compute a stack hash from the crash's PC addresses."""
        pcs = _SANITIZER_PC_RE.findall(self.raw)
        if not pcs:
            return ""

        num_frames = _NUM_FRAMES_SANITIZER if self.sanitizer else _NUM_FRAMES_NORMAL
        h = 0
        count = 0
        for pc_str in pcs[:num_frames]:
            pc = int(pc_str, 16)
            h ^= pc & 0xFFF
            count += 1

        if count == 1:
            h ^= _SINGLE_FRAME_MASK

        return f"{h:016x}"
