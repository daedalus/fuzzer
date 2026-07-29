"""ASAN LD_PRELOAD wrapper entry point.

Sets LD_PRELOAD and ASAN_OPTIONS for proper ASAN initialization
at process start, then exec's the real fuzzer-tool via execvpe.

This fixes Layer 1: the shadow address computation (addr>>3 + BASE)
works correctly when ASAN initializes at process start because the
compiled-in shadow offset matches the runtime shadow mapping.

Usage: fuzzer-tool-asan fuzz <target> [options...]
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


def main() -> None:
    libasan = _find_libasan()
    if libasan is None:
        print("[!] libasan not found — ASAN wrapper ineffective", file=sys.stderr)
    else:
        current = os.environ.get("LD_PRELOAD", "")
        parts = [p for p in current.split(":") if p] if current else []
        if not any("libasan" in p for p in parts):
            parts.insert(0, libasan)
            os.environ["LD_PRELOAD"] = ":".join(parts)

    # Ensure halt_on_error=0 so ASAN doesn't kill the fuzzer process
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

    # Replace this process with the real fuzzer-tool via execvpe
    cmd = [sys.executable, "-m", "fuzzer_tool"] + sys.argv[1:]
    os.execvpe(sys.executable, cmd, os.environ)


if __name__ == "__main__":
    main()
