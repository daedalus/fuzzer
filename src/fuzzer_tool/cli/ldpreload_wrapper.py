"""LD_PRELOAD wrapper entry point for sanitizer runtimes.

Preloads sanitizer runtimes (ASAN, UBSAN) via LD_PRELOAD at process
start so they initialize with correct shadow mapping, then exec's
the real fuzzer-tool via execvpe.

For ASAN: LD_PRELOAD solves the shadow address computation
(addr>>3 + BASE) matching at process start instead of mid-process
ctypes loading.

For UBSAN: LD_PRELOAD resolves the __ubsan_handle_* symbols that
targets compiled with -fsanitize=undefined leave as unresolved
imports.
"""

import os
import sys


def _find_libasan() -> str | None:
    candidates = [
        "/usr/lib/x86_64-linux-gnu/libasan.so.8",
        "/usr/lib/x86_64-linux-gnu/libasan.so",
        "/usr/lib64/libasan.so.8",
        "/usr/lib64/libasan.so",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    import ctypes.util

    found = ctypes.util.find_library("asan")
    if found:
        return found
    return None


def _find_libubsan() -> str | None:
    """Find the UBSAN standalone runtime shared library via clang -print-resource-dir."""
    import subprocess

    try:
        r = subprocess.run(["clang", "-print-resource-dir"], capture_output=True, timeout=10)
        if r.returncode == 0:
            res_dir = r.stdout.decode().strip()
            so = os.path.join(res_dir, "lib", "linux", "libclang_rt.ubsan_standalone-x86_64.so")
            if os.path.exists(so):
                return so
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    # Fallback: known paths
    for p in [
        "/usr/lib/llvm-19/lib/clang/19/lib/linux/libclang_rt.ubsan_standalone-x86_64.so",
        "/usr/lib/llvm-18/lib/clang/18/lib/linux/libclang_rt.ubsan_standalone-x86_64.so",
        "/usr/lib/llvm-17/lib/clang/17/lib/linux/libclang_rt.ubsan_standalone-x86_64.so",
    ]:
        if os.path.exists(p):
            return p
    return None


def _preload(soname: str, label: str) -> None:
    """Add *soname* to LD_PRELOAD if not already present."""
    current = os.environ.get("LD_PRELOAD", "")
    parts = [p for p in current.split(":") if p] if current else []
    if not any(soname in p for p in parts):
        parts.insert(0, soname)
        os.environ["LD_PRELOAD"] = ":".join(parts)
        print(f"[*] Preloaded {label}: {soname}", file=sys.stdout)


def main() -> None:
    # Preload ASAN runtime (correct shadow mapping at process start)
    libasan = _find_libasan()
    if libasan:
        _preload(libasan, "libasan")
    else:
        print("[!] libasan not found", file=sys.stderr)

    # Preload UBSAN runtime (resolve __ubsan_handle_* symbols)
    libubsan = _find_libubsan()
    if libubsan:
        _preload(libubsan, "libubsan")

    # ASAN options: halt_on_error=0 so ASAN writes report and returns
    asan_opts = os.environ.get("ASAN_OPTIONS", "")
    opt_parts = [p for p in asan_opts.split(":") if p] if asan_opts else []
    seen = {p.split("=")[0] for p in opt_parts}
    for opt in (
        "halt_on_error=0",
        "abort_on_error=0",
        "verify_asan_link_order=0",
        "detect_leaks=0",
    ):
        key = opt.split("=")[0]
        if key not in seen:
            opt_parts.append(opt)
            seen.add(key)
    os.environ["ASAN_OPTIONS"] = ":".join(opt_parts)

    # UBSAN options: halt_on_error=1 so UBSAN aborts on detection
    ubsan_opts = os.environ.get("UBSAN_OPTIONS", "")
    opt_parts = [p for p in ubsan_opts.split(":") if p] if ubsan_opts else []
    seen = {p.split("=")[0] for p in opt_parts}
    for opt in ("halt_on_error=1", "abort_on_error=1", "print_stacktrace=1"):
        key = opt.split("=")[0]
        if key not in seen:
            opt_parts.append(opt)
            seen.add(key)
    os.environ["UBSAN_OPTIONS"] = ":".join(opt_parts)

    # Replace this process with the real fuzzer-tool via execvpe
    cmd = [sys.executable, "-m", "fuzzer_tool"] + sys.argv[1:]
    os.execvpe(sys.executable, cmd, os.environ)


if __name__ == "__main__":
    main()
