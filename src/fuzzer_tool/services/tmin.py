"""Crash minimizer: binary-search for smallest input that still triggers a crash."""

import os
import sys
from pathlib import Path

from fuzzer_tool.adapters.filesystem import hash_data
from fuzzer_tool.core.mutations import minimize_bytes


def tmin(
    target: str,
    crash_file: str,
    timeout: float = 5.0,
    file_mode: bool = False,
    target_args: list[str] | None = None,
    use_coverage: bool = False,
    max_stages: int = 128,
) -> bytes | None:
    """Minimize a crash input to find the smallest reproducer.

    Replays the crash input against the target, then uses delta-debugging
    to find the smallest subset of bytes that still triggers the same crash.

    Args:
        target: Path to the target binary.
        crash_file: Path to the crashing input file.
        timeout: Execution timeout in seconds.
        file_mode: Write input to temp file instead of stdin.
        target_args: Target arguments ({file} placeholder).
        use_coverage: Enable SHM coverage (passed to env).
        max_stages: Maximum reduction stages.

    Returns:
        Minimized bytes, or None if the crash could not be reproduced.
    """
    crash_path = Path(crash_file)
    if not crash_path.is_file():
        print(f"[-] Crash file not found: {crash_file}", file=sys.stderr)
        return None

    data = crash_path.read_bytes()
    if not data:
        print("[-] Crash file is empty", file=sys.stderr)
        return None

    print(f"[*] Crash input: {len(data)} bytes, hash={hash_data(data)}")

    from fuzzer_tool.adapters.process import SIGNAL_CRASH_CODES, run_target_file, run_target_stdin
    from fuzzer_tool.core.sanitizer import SanitizerReport

    tmp_dir = Path("/tmp") / f"tmin_{os.getpid()}"
    if file_mode:
        tmp_dir.mkdir(parents=True, exist_ok=True)

    def _is_crash(data_bytes: bytes) -> bool:
        env = os.environ.copy()
        if use_coverage:
            env["AFL_MAP_SIZE"] = "65536"

        if file_mode:
            returncode, stderr = run_target_file(
                target,
                data_bytes,
                timeout,
                str(tmp_dir),
                target_args or [],
                env=env,
            )
        else:
            returncode, stderr = run_target_stdin(
                target,
                data_bytes,
                timeout,
                env=env,
            )

        if returncode in (-2, -1):
            return False
        report = SanitizerReport.parse(stderr)
        if report and report.is_valid():
            return True
        if returncode in SIGNAL_CRASH_CODES:
            return True
        if returncode < 0:
            return True
        return any(
            sig in stderr
            for sig in [
                "SIGSEGV",
                "SIGABRT",
                "SIGFPE",
                "SIGBUS",
                "Segmentation fault",
                "Aborted",
            ]
        )

    if not _is_crash(data):
        print("[-] Crash not reproduced with original input", file=sys.stderr)
        return None

    print(f"[*] Reproduced. Starting minimization (max {max_stages} stages)...")

    minimized = minimize_bytes(data, _is_crash, max_stages=max_stages)

    print(
        f"[+] Minimized: {len(data)} -> {len(minimized)} bytes "
        f"({100 - len(minimized) / len(data) * 100:.0f}% reduction)"
    )

    if not _is_crash(minimized):
        print("[-] Minimized input no longer crashes! Falling back to original.", file=sys.stderr)
        return data

    return minimized


def main():
    """CLI entry point for fuzzer-tool tmin."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Minimize a crash input to find the smallest reproducer"
    )
    parser.add_argument("target", help="Path to target binary")
    parser.add_argument("crash_file", help="Path to crashing input file")
    parser.add_argument("-t", "--timeout", type=float, default=5, help="Timeout in seconds")
    parser.add_argument(
        "-F", "--file-mode", action="store_true", help="Write input to temp file instead of stdin"
    )
    parser.add_argument(
        "-A",
        "--target-args",
        nargs=argparse.REMAINDER,
        help="Target arguments ({file} placeholder)",
    )
    parser.add_argument("-c", "--coverage", action="store_true", help="Enable SHM coverage")
    parser.add_argument(
        "--max-stages", type=int, default=128, help="Max reduction stages (default: 128)"
    )
    parser.add_argument(
        "-o", "--output", default=None, help="Output file for minimized input (default: stdout)"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.target):
        print(f"[-] Target not found: {args.target}", file=sys.stderr)
        sys.exit(1)

    minimized = tmin(
        target=args.target,
        crash_file=args.crash_file,
        timeout=args.timeout,
        file_mode=args.file_mode,
        target_args=args.target_args,
        use_coverage=args.coverage,
        max_stages=args.max_stages,
    )

    if minimized is None:
        sys.exit(1)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_bytes(minimized)
        print(f"[+] Saved to {args.output}")
    else:
        sys.stdout.buffer.write(minimized)
