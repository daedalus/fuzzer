"""LD_PRELOAD wrapper entry point for sanitizer runtimes.

Detects target instrumentation (ASAN, UBSAN) from the fuzz subcommand
arguments and preloads the corresponding runtime via LD_PRELOAD before
exec'ing the real fuzzer-tool via execvpe.

For ASAN: LD_PRELOAD at process start fixes the shadow address
computation (addr>>3 + BASE) so it matches the runtime mapping,
which fails when libasan is loaded mid-process via ctypes.

For UBSAN: LD_PRELOAD resolves the __ubsan_handle_* symbols that
targets compiled with -fsanitize=undefined leave as unresolved imports.
"""

import os
import subprocess
import sys


def _has_undefined_symbol(target: str, name: bytes) -> bool:
    """True if *target* imports *name* as a strong undefined symbol.

    `nm -D` type codes: `U` = undefined (strong — the dynamic linker fails to
    load without a provider); `w` = weak undefined (resolvable-or-NULL, no
    runtime needed, e.g. `__ubsan_handle_cfi_bad_type`); any other type means
    the symbol is *defined* in the binary.
    """
    try:
        r = subprocess.run(["nm", "-D", target], capture_output=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    if r.returncode != 0:
        return False
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) < 2 or name not in parts[-1]:
            continue
        symtype = parts[0] if len(parts) == 2 else parts[1]
        if symtype == b"U":
            return True
    return False


def _detect_asan(target: str) -> bool:
    """Check if *target* has strong undefined __asan_init (ASAN-instrumented).

    Only strong undefined imports count.  Targets that link their own ASAN
    runtime (defined `__asan_init`, the common `-fsanitize=address` build)
    must not trigger a preload — LD_PRELOADing a second libasan on top breaks
    the child's AFL SHM coverage.  The fuzzer falls back to per-child
    LD_PRELOAD in in-process modes when no preload was set at process start.
    """
    return _has_undefined_symbol(target, b"__asan_init")


def _detect_ubsan(target: str) -> bool:
    """Check if *target* has strong undefined __ubsan_handle_* imports.

    Only strong undefined imports count.  Targets that define the handlers
    (link the runtime, the common clang case) or reference them weakly
    (e.g. `__ubsan_handle_cfi_bad_type`, resolvable-or-NULL) must not trigger
    a preload — preloading libasan and the UBSAN standalone together into the
    fuzzer process trips an ASAN init CHECK and hangs startup.
    """
    return _has_undefined_symbol(target, b"__ubsan_handle")


def _resolve_asan() -> str | None:
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


def _resolve_ubsan() -> str | None:
    """Find the UBSAN standalone runtime shared library."""
    try:
        r = subprocess.run(["clang", "-print-resource-dir"], capture_output=True, timeout=10)
        if r.returncode == 0:
            res_dir = r.stdout.decode().strip()
            so = os.path.join(res_dir, "lib", "linux", "libclang_rt.ubsan_standalone-x86_64.so")
            if os.path.exists(so):
                return so
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    for p in [
        "/usr/lib/llvm-19/lib/clang/19/lib/linux/libclang_rt.ubsan_standalone-x86_64.so",
        "/usr/lib/llvm-18/lib/clang/18/lib/linux/libclang_rt.ubsan_standalone-x86_64.so",
        "/usr/lib/llvm-17/lib/clang/17/lib/linux/libclang_rt.ubsan_standalone-x86_64.so",
    ]:
        if os.path.exists(p):
            return p
    return None


def _preload(soname: str, label: str) -> None:
    current = os.environ.get("LD_PRELOAD", "")
    parts = [p for p in current.split(":") if p] if current else []
    if not any(soname in p for p in parts):
        parts.insert(0, soname)
        os.environ["LD_PRELOAD"] = ":".join(parts)
        print(f"[*] Preloaded {label}: {soname}", file=sys.stdout)


def main() -> None:
    # Pick the target path from the subcommand args.
    # Expected: fuzzer-tool fuzz <target> [options...]
    target: str | None = None
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "fuzz" and i + 1 < len(args):
            candidate = args[i + 1]
            if not candidate.startswith("-"):
                target = candidate
                break

    if target and os.path.exists(target):
        if _detect_asan(target):
            libasan = _resolve_asan()
            if libasan:
                _preload(libasan, "libasan")
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

        if _detect_ubsan(target):
            libubsan = _resolve_ubsan()
            if libubsan:
                _preload(libubsan, "libubsan")
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
