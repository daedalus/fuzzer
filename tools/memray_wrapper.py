#!/usr/bin/env python3
"""Run fuzzer-tool fuzz under memray and print the summary report."""

import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time

import memray

sys.argv = ["fuzzer-tool", "fuzz"] + sys.argv[1:]


def _resolve_trace_path() -> str:
    env_path = os.environ.get("MEMRAY_OUTPUT", "").strip()
    if env_path:
        return env_path
    fd, trace_path = tempfile.mkstemp(prefix="memray_wrapper_", suffix=".bin")
    os.close(fd)
    os.unlink(trace_path)
    return trace_path


def _print_summary(trace_path: str) -> None:
    memray_bin = shutil.which("memray")
    if not memray_bin:
        print("[memray] memray CLI not found; skipping summary")
        return
    try:
        proc = subprocess.run(
            [memray_bin, "summary", trace_path],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        print(f"[memray] failed to run summary: {exc}")
        return
    output = (proc.stdout + proc.stderr).strip()
    if output:
        print(f"\n=== memray summary: {trace_path} ===")
        print(
            "Columns: Location | <Total Memory> | Total % | Own Memory | Own % | Allocation Count"
        )
        print("-" * 120)
        print(output)
        print("========================================")
    else:
        print(f"\n[memray] no summary output for {trace_path}")


def main() -> None:
    trace_path = _resolve_trace_path()
    start = time.monotonic()
    try:
        with memray.Tracker(trace_path, native_traces=True, follow_fork=True):
            from fuzzer_tool.cli.commands import main as fuzz_main

            fuzz_main()
    except SystemExit:
        pass
    wall = time.monotonic() - start
    end_rusage = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss_kb = end_rusage.ru_maxrss

    print("\n=== memray wrapper memory profile ===")
    print(f"trace     : {trace_path}")
    print(f"wall time : {wall:.1f}s")
    print(f"peak RSS  : {peak_rss_kb // 1024} MB  (ru_maxrss {peak_rss_kb} KB)")
    print("======================================")
    _print_summary(trace_path)


if __name__ == "__main__":
    main()
