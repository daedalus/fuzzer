"""Sanitizer output parsing for crash detection."""

import re
from array import array
from typing import NamedTuple

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
    ("hard rss limit exhausted", "rss-limit-exhausted"),
    ("out of memory:", "out-of-memory"),
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

# Frame with its source site: "#1 0x4011d6 in foo /src/a.c:12:5". Groups:
# func, path, line, col (all optional) | module for "(/bin/t+0x11d6)".
_FRAME_LOC_RE = re.compile(
    r"^[ \t]*#(\d+)[ \t]+0x[0-9a-fA-F]+[ \t]+"
    r"(?:in[ \t]+(\S+)(?:[ \t]+(\S+?):(\d+)(?::(\d+))?)?|\(([^)\s]+)\))",
    re.MULTILINE,
)

# Signature frames: the first N target frames of the crash stack.
_SIG_FRAMES = 3

# Sanitizer / libc frames: they name the runtime, not the bug site
# ("#0 in free" of a bad-free). Exact names; prefixes below.
_RUNTIME_FUNCS = frozenset((
    "free", "malloc", "calloc", "realloc", "reallocarray", "memalign", "posix_memalign",
    "aligned_alloc", "memcpy", "memmove", "memset", "memcmp", "strlen", "strnlen", "strcpy",
    "strncpy", "strcmp", "strncmp", "strcat", "strncat", "strchr", "strrchr", "strstr",
    "abort", "raise", "_start", "operator",
))  # fmt: skip
_RUNTIME_PREFIXES = (
    "__interceptor_", "___interceptor_", "__asan", "__msan", "__tsan", "__ubsan",
    "__lsan", "__sanitizer", "__afl_", "__libc_", "__GI_", "__pthread_", "_dl_",
)  # fmt: skip
_RUNTIME_MODULES = (
    "libc.so", "libc-", "ld-linux", "libasan", "libubsan", "libmsan", "libtsan",
    "liblsan", "libclang_rt", "libgcc_s", "libpthread", "libstdc++", "libc++",
)  # fmt: skip

# Harness entry: frames below it are the fuzzer's (ctypes, libffi, loader).
_HARNESS_FUNCS = frozenset(("fuzz_shm_run", "LLVMFuzzerTestOneInput"))


def is_runtime_module(module: str) -> bool:
    """True for libc / sanitizer runtime objects, e.g. "/lib/libc.so.6"."""
    return module.rsplit("/", 1)[-1].startswith(_RUNTIME_MODULES)


def _is_runtime(frame: str) -> bool:
    if frame in _RUNTIME_FUNCS or frame.startswith(_RUNTIME_PREFIXES):
        return True
    return "+0x" in frame and is_runtime_module(frame)


def signature_frames(frames: list[str]) -> list[str]:
    """First ``_SIG_FRAMES`` target frames: runtime skipped, harness ends it.

    ``[free, json_value_free_ex, fuzz_shm_run, __afl_guard_run, libffi...]``
    -> ``[json_value_free_ex]``. A crash in the harness itself keeps it; an
    all-runtime stack keeps its innermost frame.
    """
    out: list[str] = []
    for f in frames:
        if f in _HARNESS_FUNCS and out:
            break
        if _is_runtime(f):
            continue
        out.append(f)
        if len(out) == _SIG_FRAMES:
            break
    return out or frames[:1]


# Stack hash format; bump on any change to stack_hash(). Blocklists carry it
# as their header so a list from another format warns instead of matching
# nothing.
STACK_HASH_VERSION = 2
STACK_HASH_TAG = f"# stack-hash v{STACK_HASH_VERSION}"

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
    "rss-limit-exhausted": "LOW",
    "out-of-memory": "LOW",
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

# READ access downgrades CRITICAL to HIGH: an out-of-bounds or stale read
# leaks; the write corrupts.
_ASAN_READ_DOWNGRADE = {
    "heap-buffer-overflow",
    "heap-buffer-overflow-",
    "global-buffer-overflow",
    "stack-buffer-overflow",
    "dynamic-stack-buffer-overflow",
    "heap-use-after-free",
}

# ── Signal crashes (no report) ───────────────────────────────────

_NULL_PAGE = 0x10000  # Linux vm.mmap_min_addr default: never mapped
_STACK_REACH = 0x10000  # a fault this close to rsp: stack exhaustion

# glibc malloc's abort messages (__libc_message), at line start:
# "free(): invalid pointer", "double free or corruption (out)", ...
_GLIBC_HEAP_RE = re.compile(
    r"^(?:free\(\)|malloc\(\)|realloc\(\)|munmap_chunk\(\)|malloc_consolidate\(\)"
    r"|double free or corruption|corrupted size vs\. prev_size|corrupted double-linked list)",
    re.MULTILINE,
)
_SIGSEGV, _SIGBUS, _SIGABRT, _SIGFPE = 11, 7, 6, 8


def signal_exploitability(signo: int, fault: int | None, rip: int, rsp: int, stderr: str) -> str:
    """Tier of a crash without a sanitizer report, from what the fault shows.

    Heap-corruption abort HIGH; other aborts and SIGFPE LOW; then by fault
    address: null page LOW, ``rip == fault`` (jumped there) CRITICAL, near
    ``rsp`` (stack exhaustion) LOW, else MEDIUM: read vs write is unknown.
    """
    if _GLIBC_HEAP_RE.search(stderr):
        return "HIGH"
    if signo in (_SIGABRT, _SIGFPE):
        return "LOW"
    if fault is None:
        return "UNKNOWN"

    if fault < _NULL_PAGE:
        return "LOW"
    if signo in (_SIGSEGV, _SIGBUS) and fault == rip:
        return "CRITICAL"
    if rsp and abs(fault - rsp) < _STACK_REACH:
        return "LOW"
    return "MEDIUM"


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


class Frame(NamedTuple):
    """One stack frame with its source site; ``path`` empty when unsymbolized."""

    func: str
    path: str
    line: int
    col: int
    module: str


def crash_frames(text: str, start: int = 0) -> list[Frame]:
    """Frames of the first stack at or after *start*, with file:line:col.

    Same stop rule as ``_crash_stack``: the next ``#0`` starts another stack.
    """
    out: list[Frame] = []
    for idx, func, path, line, col, module in _FRAME_LOC_RE.findall(text, start):
        if idx == "0" and out:
            break
        mod = module.rsplit("/", 1)[-1]
        out.append(Frame(func, path, int(line or 0), int(col or 0), mod))
    return out


def _site_frame(site: str) -> Frame:
    """UBSAN "a.c:12:5" site as a one-frame stack."""
    path, line, col = site.rsplit(":", 2)
    return Frame("", path, int(line), int(col), "")


def _use_ubsan(m: re.Match | None, u: re.Match | None) -> bool:
    """UBSAN's "runtime error:" line names the check and the site; its
    SUMMARY only says "undefined-behavior"."""
    if not u:
        return False
    return not m or u.start() < m.start() or m.group(1).lower() == _UBSAN.lower()


def report_stacks(stderr: str) -> dict[str, list[Frame]]:
    """``crash`` / ``free`` / ``alloc`` stacks of a sanitizer report, with sites.

    Empty when *stderr* holds no report. A UBSAN report without a stack gets
    its ``runtime error:`` site as the one crash frame.
    """
    m = SANITIZER_ERROR_LINE_RE.search(stderr)
    u = _UBSAN_RUNTIME_RE.search(stderr) if _UBSAN_RUNTIME_MARK in stderr else None
    if not m and not u:
        return {}

    ubsan = _use_ubsan(m, u)
    crash = crash_frames(stderr, u.start() if ubsan else m.start())
    if ubsan and not crash:
        crash = [_site_frame(u.group(1))]
    stacks = {"crash": crash}

    # Free before alloc: the order ASAN prints them in.
    for key, rx in (("free", SANITIZER_DEALLOC_RE), ("alloc", SANITIZER_ALLOC_RE)):
        hit = rx.search(stderr)
        if hit:
            stacks[key] = crash_frames(hit.group(1))
    return stacks


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
        for f in signature_frames(self.frames):
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

        if _use_ubsan(m, u):
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
