"""Dynamic shim factory for fuzzer coverage collection.

For direct ctypes mode, provides a minimal C shim that supplies the
undefined __sanitizer_cov_8bit_counters_init symbol, then reads the
coverage bitmap directly from the target .so's sancov counters section
by parsing the ELF at runtime.

For subprocess mode, builds a self-contained C executable that links
the sanitizer runtime and writes coverage to a file.
"""

import contextlib
import ctypes
import hashlib
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass

from fuzzer_tool.core.elf import find_load_segment, parse_sancov_offsets

log = logging.getLogger(__name__)

_shim_cache: dict[str, str] = {}

# Minimal C shim that only provides __sanitizer_cov_8bit_counters_init
_MINIMAL_SHIM_SRC = """\
void __sanitizer_cov_8bit_counters_init(void *start, void *stop) {
    (void)start; (void)stop;
}
"""


@dataclass
class ShimResult:
    """Result of building a coverage shim."""

    shim_path: str | None = None
    coverage_type: str = "none"  # "sancov_counters", "inline_8bit", "none"
    bitmap_size: int = 0
    needs_preload: bool = False
    compile_error: str | None = None
    # For direct mode: ELF-parsed counter offsets
    _elf_offsets: tuple[int, int] | None = None  # (start_offset, stop_offset)


def _cache_key(target: str, mode: str) -> str:
    raw = f"{target}:{mode}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _find_compiler() -> str:
    for cc in ("clang", "gcc", "cc"):
        try:
            subprocess.run([cc, "--version"], capture_output=True, timeout=5)
            return cc
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    raise RuntimeError("No C compiler found (tried clang, gcc, cc)")


def _compile_source(source: str, output: str, compiler: str | None = None) -> bool:
    compiler = compiler or _find_compiler()
    fd, src_path = tempfile.mkstemp(suffix=".c", prefix="fuzz_shim_")
    os.write(fd, source.encode())
    os.close(fd)
    try:
        result = subprocess.run(
            [compiler, "-shared", "-fPIC", "-O2", "-o", output, src_path],
            capture_output=True,
            timeout=30,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    finally:
        with contextlib.suppress(OSError):
            os.unlink(src_path)


def _inspect_target(target: str) -> dict:
    """Inspect a target binary to determine its coverage type."""
    info = {
        "is_shared_lib": False,
        "coverage_type": "none",
        "has_sancov_counters": False,
        "has_undefined_sancov_init": False,
        "has_asan": False,
    }
    if not os.path.isfile(target):
        return info
    tl = target.lower()
    if tl.endswith((".so", ".dylib", ".dll")):
        info["is_shared_lib"] = True
    try:
        r = subprocess.run(["nm", "-D", target], capture_output=True, timeout=10)
        if r.returncode == 0:
            for line in r.stdout.decode(errors="replace").splitlines():
                parts = line.split()
                if len(parts) < 3:
                    continue
                _, stype, sname = parts[0], parts[1], parts[2]
                if sname == "__sanitizer_cov_8bit_counters_init" and stype == "U":
                    info["has_undefined_sancov_init"] = True
                if "__start___sancov_cntrs" in sname:
                    info["has_sancov_counters"] = True
                # ASAN detection: look for __asan_init or __asan_register_globals
                if "__asan_init" in sname or "__asan_register_globals" in sname:
                    info["has_asan"] = True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback: try nm without -D for static symbols in executables
    if not info["has_sancov_counters"] and not info["has_undefined_sancov_init"]:
        try:
            r = subprocess.run(["nm", target], capture_output=True, timeout=10)
            if r.returncode == 0:
                for line in r.stdout.decode(errors="replace").splitlines():
                    parts = line.split()
                    if len(parts) < 3:
                        continue
                    _, stype, sname = parts[0], parts[1], parts[2]
                    if sname == "__sanitizer_cov_8bit_counters_init" and stype == "U":
                        info["has_undefined_sancov_init"] = True
                    if "__start___sancov_cntrs" in sname:
                        info["has_sancov_counters"] = True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    if info["has_sancov_counters"] or info["has_undefined_sancov_init"]:
        info["coverage_type"] = "inline_8bit"
    return info


def build_minimal_shim() -> str | None:
    """Build a minimal shim that only provides __sanitizer_cov_8bit_counters_init."""
    key = _cache_key("minimal_shim", "noop")
    if key in _shim_cache and os.path.exists(_shim_cache[key]):
        return _shim_cache[key]
    so_path = os.path.join(tempfile.gettempdir(), "fuzz_minimal_shim.so")
    if _compile_source(_MINIMAL_SHIM_SRC, so_path):
        _shim_cache[key] = so_path
        return so_path
    return None


def build_sancov_shim() -> str | None:
    """Build a sanitizer coverage LD_PRELOAD shim for Clang-instrumented binaries.

    Intercepts __sanitizer_cov_trace_pc_guard and writes edge indices
    to a bitmap file. Provides coverage feedback without AFL or ptrace.

    Returns:
        Path to compiled .so, or None on failure.
    """
    shim_src = os.path.join(os.path.dirname(__file__), "sancov_shim.c")
    if not os.path.exists(shim_src):
        return None

    out_path = os.path.join(tempfile.gettempdir(), "fuzz_sancov_shim.so")
    with open(shim_src) as f:
        src = f.read()
    if _compile_source(src, out_path):
        return out_path
    return None


class BitmapReader:
    """Read the sancov coverage bitmap from a loaded target .so.

    Uses ELF parsing to find the counters section at load time,
    then reads directly from the target's address space after each call.
    """

    def __init__(self, target: str, lib_handle: ctypes.CDLL):
        self.target = target
        self.lib = lib_handle
        self._elf_data: bytes | None = None
        self._counter_offsets: tuple[int, int] | None = None
        self._base_address: int | None = None
        self._bitmap_size: int = 0
        self._snapshot: bytes | None = None
        self._setup()

    def _setup(self):
        # Parse ELF to find sancov counter virtual addresses
        self._counter_offsets = parse_sancov_offsets(self.target)
        if self._counter_offsets is None:
            log.warning("No sancov counters found in %s", self.target)
            return

        # Read ELF data for LOAD segment lookup
        with open(self.target, "rb") as f:
            self._elf_data = f.read()

        # Find base address from /proc/self/maps
        target_name = os.path.basename(self.target)
        try:
            with open(f"/proc/{os.getpid()}/maps") as f:
                for line in f:
                    if target_name in line and "r-xp" in line:
                        self._base_address = int(line.split("-")[0], 16)
                        break
        except Exception:
            pass

        if self._base_address is None:
            # Fallback: try to find via the first LOAD segment
            vaddr, _, _ = find_load_segment(self._elf_data, self._counter_offsets[0]) or (0, 0, 0)
            # Get base from the loaded library's first symbol
            try:
                first_sym = (ctypes.c_char * 1).in_dll(self.lib, "__start___sancov_cntrs")
                self._base_address = ctypes.addressof(first_sym) - self._counter_offsets[0]
            except (ValueError, AttributeError):
                log.warning("Cannot determine base address for %s", self.target)
                return

        # Calculate runtime addresses
        start_elf, stop_elf = self._counter_offsets
        seg = find_load_segment(self._elf_data, start_elf)
        if seg:
            seg_vaddr = seg[0]
            self._runtime_start = self._base_address + (start_elf - seg_vaddr)
            self._runtime_stop = self._base_address + (stop_elf - seg_vaddr)
            self._bitmap_size = self._runtime_stop - self._runtime_start
            log.info(
                "BitmapReader: counters at 0x%x-0x%x (%d bytes)",
                self._runtime_start,
                self._runtime_stop,
                self._bitmap_size,
            )

    @property
    def bitmap_size(self) -> int:
        return self._bitmap_size

    def read_bitmap(self) -> bytes | None:
        """Read the coverage bitmap delta since last reset."""
        if not self._bitmap_size or not self._runtime_start:
            return None
        try:
            current = bytes((ctypes.c_uint8 * self._bitmap_size).from_address(self._runtime_start))
        except (OSError, ValueError):
            return None
        if self._snapshot is None:
            return current
        # Return delta: bytes where current > snapshot (handles size mismatch)
        return bytes(max(0, c - s) for c, s in zip(current, self._snapshot, strict=False))

    def reset_bitmap(self):
        """Reset by snapshotting current state and detecting deltas later."""
        self._snapshot = self.read_bitmap()

    @property
    def valid(self) -> bool:
        return self._bitmap_size > 0


def build_shim(target: str, mode: str = "auto") -> ShimResult:
    """Build a coverage shim for *target* in the given *mode*.

    For "direct" mode: builds a minimal shim that provides
    __sanitizer_cov_8bit_counters_init, and sets up ELF-based bitmap reading.

    For "subprocess" mode: builds a self-contained executable.
    """
    key = _cache_key(target, mode)
    if key in _shim_cache and os.path.exists(_shim_cache[key]):
        info = _inspect_target(target)
        return ShimResult(
            shim_path=_shim_cache[key],
            coverage_type=info["coverage_type"],
            bitmap_size=info.get("bitmap_size", 65536),
            needs_preload=(mode == "direct"),
        )

    info = _inspect_target(target)
    log.info("Target inspection: %s", info)

    if info["coverage_type"] == "none":
        return ShimResult(coverage_type="none")

    if mode == "direct":
        # Build minimal shim — just provides __sanitizer_cov_8bit_counters_init
        shim_path = build_minimal_shim()
        if shim_path:
            # Find sancov counter offsets via ELF parsing
            offsets = parse_sancov_offsets(target)
            bitmap_size = 0
            if offsets:
                bitmap_size = offsets[1] - offsets[0]
            return ShimResult(
                shim_path=shim_path,
                coverage_type=info["coverage_type"],
                bitmap_size=bitmap_size,
                needs_preload=True,
                _elf_offsets=offsets,
            )
        return ShimResult(
            compile_error="Failed to build minimal shim",
            coverage_type=info["coverage_type"],
        )

    # Subprocess mode — no special shim needed, the subprocess loader handles it
    return ShimResult(
        coverage_type=info["coverage_type"],
        bitmap_size=65536,
    )


def load_shim(shim_path: str, mode: str = "direct") -> ctypes.CDLL | None:
    if not shim_path or not os.path.exists(shim_path):
        return None
    if mode == "direct":
        return ctypes.CDLL(shim_path, mode=ctypes.RTLD_GLOBAL)
    return ctypes.CDLL(shim_path)


def read_bitmap(shim_handle: ctypes.CDLL) -> bytes | None:
    if shim_handle is None:
        return None
    try:
        ptr = shim_handle.cov_get_bitmap()
        size = shim_handle.cov_get_size()
        if not ptr or not size:
            return None
        return (ctypes.c_uint8 * size).from_address(ptr).tobytes()
    except (AttributeError, OSError):
        return None


def reset_bitmap(shim_handle: ctypes.CDLL) -> None:
    if shim_handle is None:
        return
    with contextlib.suppress(AttributeError, OSError):
        shim_handle.cov_reset()


def cleanup_shim(shim_path: str | None) -> None:
    if shim_path and os.path.exists(shim_path):
        with contextlib.suppress(OSError):
            os.unlink(shim_path)
