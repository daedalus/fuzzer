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

from fuzzer_tool.core.elf import parse_sancov_offsets

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
    coverage_type: str = "none"  # "inline_8bit", "none"
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
        # Strip ASAN/LSAN from the compiler subprocess — clang/gcc are
        # not built with ASAN and libasan's LeakSanitizer will cause
        # false-positive leak reports that make the compiler exit non-zero.
        env = os.environ.copy()
        env.pop("ASAN_OPTIONS", None)
        env.pop("LSAN_OPTIONS", None)
        # Also remove ASAN from LD_PRELOAD so the compiler isn't slowed down
        ld_preload = env.get("LD_PRELOAD", "")
        if ld_preload:
            parts = [p for p in ld_preload.split(":") if "libasan" not in p]
            env["LD_PRELOAD"] = ":".join(parts) if parts else ""
        result = subprocess.run(
            [compiler, "-shared", "-fPIC", "-O2", "-o", output, src_path],
            capture_output=True,
            timeout=30,
            env=env,
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
    fd, so_path = tempfile.mkstemp(suffix=".so", prefix=f"fuzz_minimal_shim_{os.getpid()}_")
    os.close(fd)
    if _compile_source(_MINIMAL_SHIM_SRC, so_path):
        _shim_cache[key] = so_path
        return so_path
    with contextlib.suppress(OSError):
        os.unlink(so_path)
    return None


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
    try:
        if mode == "direct":
            return ctypes.CDLL(shim_path, mode=ctypes.RTLD_GLOBAL)
        return ctypes.CDLL(shim_path)
    except OSError:
        return None


def cleanup_shim(shim_path: str | None) -> None:
    if shim_path and os.path.exists(shim_path):
        with contextlib.suppress(OSError):
            os.unlink(shim_path)
