"""Differential fuzzing: run same input through two targets, flag divergence."""

import os
import shutil
import tempfile

from fuzzer_tool.adapters.process import run_target_file, run_target_stdin
from fuzzer_tool.core.sanitizer import SanitizerReport


def diff_run(
    target_a: str,
    target_b: str,
    data: bytes,
    timeout: float = 5.0,
    file_mode: bool = False,
    target_args: list[str] | None = None,
) -> tuple[bool, str]:
    """Run the same input through two targets and compare outputs.

    Returns:
        Tuple of (diverged: bool, description: str).
    """
    env = os.environ.copy()
    tmp_dir = tempfile.mkdtemp(prefix="diff_") if file_mode else None

    try:
        # Run target A
        if file_mode:
            rc_a, stderr_a = run_target_file(
                target_a, data, timeout, tmp_dir, target_args or [], env=env
            )
        else:
            rc_a, stderr_a = run_target_stdin(target_a, data, timeout, env=env)

        # Run target B
        if file_mode:
            rc_b, stderr_b = run_target_file(
                target_b, data, timeout, tmp_dir, target_args or [], env=env
            )
        else:
            rc_b, stderr_b = run_target_stdin(target_b, data, timeout, env=env)
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # Compare results
    diverged = False
    reasons = []

    if rc_a != rc_b:
        diverged = True
        reasons.append(f"returncode: {rc_a} vs {rc_b}")

    report_a = SanitizerReport.parse(stderr_a)
    report_b = SanitizerReport.parse(stderr_b)

    if report_a and report_a.is_valid() and not (report_b and report_b.is_valid()):
        diverged = True
        reasons.append(f"A crashes ({report_a.error_type}), B clean")
    elif report_b and report_b.is_valid() and not (report_a and report_a.is_valid()):
        diverged = True
        reasons.append(f"B crashes ({report_b.error_type}), A clean")
    elif report_a and report_b and report_a.is_valid() and report_b.is_valid():
        if report_a.error_type != report_b.error_type:
            diverged = True
            reasons.append(f"different errors: {report_a.error_type} vs {report_b.error_type}")

    if stderr_a != stderr_b and not diverged and (stderr_a or stderr_b):
        reasons.append("different stderr output")

    description = "; ".join(reasons) if reasons else "identical"
    return diverged, description
