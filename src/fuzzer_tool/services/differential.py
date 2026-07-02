"""Differential fuzzing: run same input through two targets, flag divergence.

Provides two levels of comparison:
1. Per-input: exact match on returncode, sanitizer reports, stderr (diff_run)
2. Statistical: KL divergence on accumulated output distributions across
   many inputs — catches behavioral drift that no single input triggers
"""

import math
import os
import shutil
import tempfile
from collections import Counter

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


class DifferentialTracker:
    """Accumulate output distributions across many inputs for KL-based drift detection.

    Single-input comparison catches exact mismatches. This tracker catches
    statistical drift: e.g., target B produces returncode 1 on 10% of inputs
    while target A produces it 0% — no single input triggers a mismatch, but
    the distributions diverge.

    KL(B || A) is asymmetric: it measures "how much extra information B's
    distribution carries beyond A's". This is the right direction when A is
    the reference implementation and B is the one under test.

    Args:
        drift_threshold: KL divergence above which drift is flagged (default 0.05).
    """

    def __init__(self, drift_threshold: float = 0.05):
        self.drift_threshold = drift_threshold
        self.rc_counts_a: Counter[int] = Counter()
        self.rc_counts_b: Counter[int] = Counter()
        self.sig_counts_a: Counter[str] = Counter()
        self.sig_counts_b: Counter[str] = Counter()
        self.total_inputs: int = 0
        self.last_kl_returncode: float = 0.0
        self.last_kl_signature: float = 0.0
        self.drift_detected: bool = False
        self.drift_description: str = ""

    def record(
        self,
        rc_a: int,
        stderr_a: str,
        rc_b: int,
        stderr_b: str,
    ) -> None:
        """Record outputs from both targets for one input.

        Args:
            rc_a: Return code from target A.
            stderr_a: Stderr from target A.
            rc_b: Return code from target B.
            stderr_b: Stderr from target B.
        """
        self.rc_counts_a[rc_a] += 1
        self.rc_counts_b[rc_b] += 1
        self.total_inputs += 1

        # Extract crash signature for distribution comparison
        sig_a = self._extract_signature(stderr_a, rc_a)
        sig_b = self._extract_signature(stderr_b, rc_b)
        self.sig_counts_a[sig_a] += 1
        self.sig_counts_b[sig_b] += 1

        # Recompute drift periodically (every 10 inputs for efficiency)
        if self.total_inputs % 10 == 0:
            self._check_drift()

    @staticmethod
    def _extract_signature(stderr: str, returncode: int) -> str:
        """Extract a coarse signature from stderr for distribution comparison.

        Uses sanitizer error type if available, otherwise returncode category.
        """
        report = SanitizerReport.parse(stderr)
        if report and report.is_valid():
            return f"asan:{report.error_type}"
        if returncode < 0:
            return f"signal:{abs(returncode)}"
        if returncode != 0:
            return f"exit:{returncode}"
        return "clean"

    def _check_drift(self) -> None:
        """Recompute KL divergence and check for drift."""
        if self.total_inputs < 20:
            return

        self.last_kl_returncode = self._kl_divergence(
            self.rc_counts_b, self.rc_counts_a
        )
        self.last_kl_signature = self._kl_divergence(
            self.sig_counts_b, self.sig_counts_a
        )

        max_kl = max(self.last_kl_returncode, self.last_kl_signature)
        self.drift_detected = max_kl > self.drift_threshold

        if self.drift_detected:
            reasons = []
            if self.last_kl_returncode > self.drift_threshold:
                reasons.append(
                    f"returncode KL={self.last_kl_returncode:.4f} "
                    f"(A: {dict(self.rc_counts_a)}, B: {dict(self.rc_counts_b)})"
                )
            if self.last_kl_signature > self.drift_threshold:
                reasons.append(
                    f"signature KL={self.last_kl_signature:.4f} "
                    f"(A: {dict(self.sig_counts_a)}, B: {dict(self.sig_counts_b)})"
                )
            self.drift_description = "; ".join(reasons)

    @staticmethod
    def _kl_divergence(
        p_counts: Counter, q_counts: Counter, smoothing: float = 0.5
    ) -> float:
        """Compute KL(P || Q) with Laplace smoothing.

        KL(P || Q) = sum_i P(i) * log(P(i) / Q(i))

        Smoothing prevents log(0) and handles categories that appear in P
        but not in Q (the asymmetric blow-up that makes KL useful here —
        it penalizes B for producing outputs A never does).

        Args:
            p_counts: Event counts for distribution P (the "test" distribution).
            q_counts: Event counts for distribution Q (the "reference" distribution).
            smoothing: Laplace smoothing factor (added to all counts).

        Returns:
            KL divergence in [0, +inf). 0 means identical distributions.
        """
        total_p = sum(p_counts.values())
        total_q = sum(q_counts.values())
        if total_p == 0 or total_q == 0:
            return 0.0

        # Collect all categories
        all_keys = set(p_counts) | set(q_counts)

        # Build smoothed distributions
        vocab = len(all_keys)
        p_total = total_p + smoothing * vocab
        q_total = total_q + smoothing * vocab

        kl = 0.0
        for k in all_keys:
            p_prob = (p_counts.get(k, 0) + smoothing) / p_total
            q_prob = (q_counts.get(k, 0) + smoothing) / q_total
            kl += p_prob * math.log(p_prob / q_prob)

        return kl

    def get_report(self) -> dict:
        """Get a summary of the drift analysis.

        Returns:
            Dict with drift status, KL values, and distribution snapshots.
        """
        return {
            "total_inputs": self.total_inputs,
            "drift_detected": self.drift_detected,
            "drift_description": self.drift_description,
            "kl_returncode": self.last_kl_returncode,
            "kl_signature": self.last_kl_signature,
            "returncode_dist_a": dict(self.rc_counts_a),
            "returncode_dist_b": dict(self.rc_counts_b),
            "signature_dist_a": dict(self.sig_counts_a),
            "signature_dist_b": dict(self.sig_counts_b),
        }
