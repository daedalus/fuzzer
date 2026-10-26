"""Sanitizer output parsing for crash detection."""

import re
from array import array

from fuzzer_tool.core.hashing import xxh64

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
    (
        r"ThreadSanitizer:\s*("
        # TSAN spells these with spaces, not hyphens — matching on
        # "data-race" never fired and the generic fallback truncated the
        # error type to just "data".
        r"data race"
        r"|heap-use-after-free"
        r"|lock-order-inversion"
        r"|thread leak"
        r"|signal-unsafe call inside of a signal"
        r"|unlock of an unlocked mutex"
        r"|double lock of a mutex"
        r"|destroy of a locked mutex"
        r"|race on a library object"
        r")",
        "TSAN",
    ),
    (r"LeakSanitizer:\s*(detected memory leaks|leak)", "LSAN"),
    (rf"UndefinedBehaviorSanitizer:\s*({UBSAN_ERROR_TYPES})", "UBSAN"),
]

# Report header, as the runtime prints it: "==42==ERROR: AddressSanitizer: x",
# "WARNING: ThreadSanitizer: x", "SUMMARY: UndefinedBehaviorSanitizer: x".
# Anchored so a target echoing "AddressSanitizer: x" mid-line is no report.
SANITIZER_ERROR_LINE_RE = re.compile(
    r"(?:==\d+==|^)(?:ERROR|WARNING|SUMMARY):\s*"
    r"(AddressSanitizer|MemorySanitizer|ThreadSanitizer|LeakSanitizer|UndefinedBehaviorSanitizer)"
    r":\s*(\S+)[^\n]*",
    re.IGNORECASE | re.MULTILINE,
)

# UBSAN diagnostic line: "/src/a.c:3:5: runtime error: signed integer overflow: ..."
_UBSAN_RUNTIME_RE = re.compile(r"^(\S+:\d+:\d+): runtime error: ([^\n]*)", re.MULTILINE)
_UBSAN_RUNTIME_MARK = ": runtime error: "
_UBSAN = "UndefinedBehaviorSanitizer"

# "runtime error:" text -> UBSAN check name; first match wins, so specific
# phrases precede the generic ones they contain.
_UBSAN_RUNTIME_TYPES = (
    ("null pointer passed as argument", "nonnull-attribute-violation"),
    ("null pointer returned from function", "return-nonnull-attribute"),
    ("signed integer overflow", "signed-integer-overflow"),
    ("unsigned integer overflow", "unsigned-integer-overflow"),
    ("division by zero", "integer-divide-by-zero"),
    ("shift exponent", "shift-exponent"),
    ("out of bounds for type", "array-bounds"),
    ("misaligned address", "misaligned-pointer-use"),
    ("pointer index expression", "pointer-overflow"),
    ("null pointer", "null-pointer-use"),
    ("outside the range of representable values", "float-cast-overflow"),
    ("through pointer to incorrect function type", "function-pointer-mismatch"),
    ("unreachable program point", "builtin-unreachable"),
    ("variable length array bound", "vla-bound-not-positive"),
)
_UBSAN_GENERIC_TYPE = "undefined"

# ASAN headers that spell the bug as a sentence; the names are its SUMMARY's.
# Both used to parse as "attempting".
_ASAN_HEADER_TYPES = (
    ("attempting double-free", "double-free"),
    ("attempting free on address which was not malloc()-ed", "bad-free"),
)

# Stack frame line: "#0 0x4011d6 in foo a.c:1" or unsymbolized "#0 0x4011d6  (/bin/t+0x11d6)".
_FRAME_LINE_RE = re.compile(
    r"^\s*#(\d+)\s+(0x[0-9a-fA-F]+)\s+(?:in\s+(\S+)|\(([^)\s]+)\))", re.MULTILINE
)
SANITIZER_FAULT_ADDR_RE = re.compile(
    r"(?:Address|Memory)Sanitizer.*(?:on|at) (?:unknown )?address\s+(0x[0-9a-f]+)",
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

# Stack hash: first N crash-stack PCs, page offset only (survives ASLR).
_NUM_FRAMES_SANITIZER = 14
_PAGE_OFFSET_MASK = 0xFFF

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
    "bad-free": "MEDIUM",
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


def _crash_stack(text: str, start: int = 0) -> list[tuple[int, str]]:
    """``(pc, frame)`` of the first stack at or after *start*.

    Stops at the next ``#0``: the allocation / free stacks that follow
    (e.g. "allocated by thread T0 here:") belong to other sites. A frame is
    its function name, else "module+offset" when unsymbolized.
    """
    out: list[tuple[int, str]] = []
    for idx, pc, name, module in _FRAME_LINE_RE.findall(text, start):
        if idx == "0" and out:
            break
        out.append((int(pc, 16), name or module.rsplit("/", 1)[-1]))
    return out


def _ubsan_type(message: str) -> str:
    """Check name for a "runtime error:" message, e.g. "signed-integer-overflow"."""
    for phrase, name in _UBSAN_RUNTIME_TYPES:
        if phrase in message:
            return name
    return _UBSAN_GENERIC_TYPE


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
        "location",
        "pcs",
    )

    def __init__(
        self,
        sanitizer: str,
        error_type: str,
        fault_addr: str,
        frames: list[str],
        raw: str,
        location: str = "",
        pcs: tuple[int, ...] = (),
    ):
        self.sanitizer = sanitizer
        self.error_type = error_type
        self.fault_addr = fault_addr
        self.frames = frames
        self.raw = raw
        self.location = location
        self.pcs = pcs
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
            self.alloc_frames = [f for _, f in _crash_stack(m.group(1))]

        # Deallocation stack
        m = SANITIZER_DEALLOC_RE.search(self.raw)
        if m:
            self.dealloc_frames = [f for _, f in _crash_stack(m.group(1))]

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

        # No stack (UBSAN's default): the source location tells sites apart.
        if not self.frames and self.location:
            key += f"@{self.location}"
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
        u = _UBSAN_RUNTIME_RE.search(stderr) if _UBSAN_RUNTIME_MARK in stderr else None
        if not m and not u:
            return None

        # UBSAN's "runtime error:" line names the check and the site; its
        # SUMMARY only says "undefined-behavior".
        if u and (not m or u.start() < m.start() or m.group(1).lower() == _UBSAN.lower()):
            stack = _crash_stack(stderr, u.start())
            return cls(
                _UBSAN,
                _ubsan_type(u.group(2)),
                "",
                [f for _, f in stack],
                stderr,
                location=u.group(1),
                pcs=tuple(pc for pc, _ in stack),
            )

        header = m.group(0)
        error_type = m.group(2).strip()

        # The generic line regex captures a single \S+ token, which truncates
        # the multi-word error types some sanitizers emit (TSAN's "data race",
        # "thread leak", ...). Prefer a specific pattern when one matches the
        # header line so the error type — and therefore the crash signature
        # used for dedup — is the full string.
        for pattern, _tag in SANITIZER_PATTERNS:
            spec = re.search(pattern, header, re.IGNORECASE)
            if spec:
                error_type = spec.group(1).strip()
                break
        error_type = next((t for phrase, t in _ASAN_HEADER_TYPES if phrase in header), error_type)

        addr_m = SANITIZER_FAULT_ADDR_RE.search(header)
        fault_addr = addr_m.group(1) if addr_m else ""

        stack = _crash_stack(stderr, m.start())
        return cls(
            m.group(1),
            error_type,
            fault_addr,
            [f for _, f in stack],
            stderr,
            pcs=tuple(pc for pc, _ in stack),
        )

    def to_dict(self) -> dict:
        """Serialize report to a JSON-compatible dict."""
        return {
            "sanitizer": self.sanitizer,
            "error_type": self.error_type,
            "fault_addr": self.fault_addr,
            "access_type": self.access_type,
            "access_size": self.access_size,
            "frames": self.frames,
            "signature": self.signature,
            "exploitability": self.exploitability,
            "alloc_frames": self.alloc_frames,
            "dealloc_frames": self.dealloc_frames,
        }

    def is_valid(self) -> bool:
        return bool(self.sanitizer and self.error_type)

    def stack_hash(self) -> str:
        """Ordered hash of the crash stack's PC page offsets.

        Order-sensitive (XOR folding let [A, B, B] collide with [A, C, C])
        and ASLR-stable: only the low 12 bits of each PC are hashed.
        """
        pcs = self.pcs or tuple(pc for pc, _ in _crash_stack(self.raw))
        if not pcs:
            return ""

        offsets = array("H", (pc & _PAGE_OFFSET_MASK for pc in pcs[:_NUM_FRAMES_SANITIZER]))
        return f"{xxh64(offsets.tobytes()):016x}"
