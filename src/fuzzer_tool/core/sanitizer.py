"""Sanitizer output parsing for crash detection."""

import re

SANITIZER_PATTERNS = [
    (
        r"AddressSanitizer:\s*(heap-buffer-overflow|stack-buffer-overflow|heap-use-after-free"
        r"|global-buffer-overflow|stack-buffer-underflow|heap-buffer-overflow-|"
        r"dynamic-stack-buffer-overflow|stack-use-after-return|stack-use-after-scope"
        r"|allocation-size-too-big|double-free|invalid-malloc-size"
        r"|attempting-free-on-non-deallocated-memory|"
        r"negative-size-param|heap-use-after-scope)",
        "ASAN",
    ),
    (r"MemorySanitizer:\s*(use-of-uninitialized-value)", "MSAN"),
    (r"ThreadSanitizer:\s*(data-race|heap-use-after-race|lock-order-inversion)", "TSAN"),
    (r"LeakSanitizer:\s*(leak)", "LSAN"),
    (
        r"UndefinedBehaviorSanitizer:\s*(undefined|shift-exponent|signed-integer-overflow"
        r"|null-pointer-use|integer-divide-by-zero)",
        "UBSAN",
    ),
]

SANITIZER_ERROR_RE = re.compile(
    r"(AddressSanitizer|MemorySanitizer|ThreadSanitizer|LeakSanitizer|UndefinedBehaviorSanitizer)"
    r":\s*(\S+)",
    re.IGNORECASE,
)
SANITIZER_STACK_FRAME_RE = re.compile(r"#\d+\s+0x[0-9a-f]+\s+in\s+(\S+)\s+.*")
SANITIZER_FAULT_ADDR_RE = re.compile(
    r"(?:Address|Memory)Sanitizer.*(?:on|at) address\s+(0x[0-9a-f]+)",
    re.IGNORECASE,
)

# New patterns for enriched ASAN output
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

# Exploitability lookup
ASAN_EXPLOITABILITY = {
    # WRITE variants → CRITICAL
    "heap-buffer-overflow": "CRITICAL",
    "stack-buffer-overflow": "CRITICAL",
    "global-buffer-overflow": "CRITICAL",
    "heap-use-after-free": "CRITICAL",
    "double-free": "CRITICAL",
    "heap-buffer-overflow-": "CRITICAL",
    "dynamic-stack-buffer-overflow": "CRITICAL",
    # READ variants → MEDIUM-HIGH
    "stack-buffer-underflow": "HIGH",
    "stack-use-after-return": "HIGH",
    "stack-use-after-scope": "HIGH",
    "heap-use-after-scope": "MEDIUM",
    "allocation-size-too-big": "MEDIUM",
    "invalid-malloc-size": "MEDIUM",
    "attempting-free-on-non-deallocated-memory": "MEDIUM",
    "negative-size-param": "MEDIUM",
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
            self.exploitability = ASAN_EXPLOITABILITY.get(self.error_type, "MEDIUM")
        elif self.sanitizer == "MemorySanitizer" or self.sanitizer == "ThreadSanitizer":
            self.exploitability = "MEDIUM"
        elif self.sanitizer == "UndefinedBehaviorSanitizer" or self.sanitizer == "LeakSanitizer":
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
        m = SANITIZER_ERROR_RE.search(stderr)
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
        """Check if the report has valid sanitizer and error type.

        Returns:
            True if both sanitizer and error_type are non-empty.
        """
        return bool(self.sanitizer and self.error_type)
