#!/usr/bin/env python3
"""Differential analysis between two fuzzer-bench runs.

Compares crash signatures, edge coverage, and efficiency metrics
from two bench.sh output logs.

Usage:
    python tools/bench_diff.py --baseline /tmp/fuzz_baseline.log --treatment /tmp/fuzz_enhanced.log
    python tools/bench_diff.py --baseline /tmp/b1.log --treatment /tmp/b2.log --verbose

Output:
    Jaccard similarity of crash signatures and edge sets
    Efficiency comparison (crashes per 1000 execs)
    Unique vs shared crash signatures
"""

import argparse
import re
import sys
from pathlib import Path


def extract_metric(log_text: str, pattern: str) -> str | None:
    m = re.search(pattern, log_text, re.MULTILINE)
    return m.group(1) if m else None


def extract_all(pattern: str, log_text: str) -> list[str]:
    return re.findall(pattern, log_text, re.MULTILINE)


def extract_crashes(log_text: str) -> int:
    m = re.search(r"(\d+)\s+crashes?\s+found", log_text)
    return int(m.group(1)) if m else 0


def extract_unique_signatures(log_text: str) -> int:
    m = re.search(r"\((\d+)\s+unique\s+signatures?\)", log_text)
    return int(m.group(1)) if m else 0


def extract_crash_sig_list(log_text: str) -> list[str]:
    """Extract individual crash signature lines."""
    sigs: list[str] = []
    in_sigs = False
    for line in log_text.splitlines():
        if "Crash signatures:" in line:
            in_sigs = True
            continue
        if in_sigs:
            # Lines look like: "    use-after-free in foo (3x)"
            m = re.match(r"\s{4}(\S.+)\((\d+)x\)", line)
            if m:
                sigs.append(m.group(1).strip())
            elif line.strip() == "" or line.startswith("[*]"):
                break
    return sigs


def extract_edges(log_text: str) -> int:
    m = re.search(r"Edges discovered:\s+(\d+)", log_text)
    return int(m.group(1)) if m else 0


def extract_corpus(log_text: str) -> int:
    m = re.search(r"Corpus:\s+(\d+)", log_text)
    return int(m.group(1)) if m else 0


def extract_eps(log_text: str) -> float:
    m = re.search(r"Avg eps:\s+([\d.]+)", log_text)
    return float(m.group(1)) if m else 0.0


def extract_iterations(log_text: str) -> int:
    m = re.search(r"(\d+)\s+execs", log_text)
    return int(m.group(1)) if m else 0


def jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity coefficient."""
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def analyze(baseline_path: str, treatment_path: str, verbose: bool = False) -> int:
    b_text = Path(baseline_path).read_text()
    t_text = Path(treatment_path).read_text()

    # ── Crash counts ──
    b_crashes = extract_crashes(b_text)
    t_crashes = extract_crashes(t_text)
    b_unique = extract_unique_signatures(b_text)
    t_unique = extract_unique_signatures(t_text)

    b_sigs = set(extract_crash_sig_list(b_text))
    t_sigs = set(extract_crash_sig_list(t_text))

    # ── Coverage metrics ──
    b_edges = extract_edges(b_text)
    t_edges = extract_edges(t_text)
    b_corpus = extract_corpus(b_text)
    t_corpus = extract_corpus(t_text)
    b_eps = extract_eps(b_text)
    t_eps = extract_eps(t_text)
    b_iters = extract_iterations(b_text)
    t_iters = extract_iterations(t_text)

    # ── Compute ──
    shared_sigs = b_sigs & t_sigs
    only_baseline = b_sigs - t_sigs
    only_treatment = t_sigs - b_sigs

    crash_jaccard = jaccard(b_sigs, t_sigs)

    # efficiency: unique crashes per 1000 execs
    b_eff = (b_unique / b_iters * 1000) if b_iters else 0.0
    t_eff = (t_unique / t_iters * 1000) if t_iters else 0.0

    # ── Output ──
    print("=" * 60)
    print("  Differential Analysis: Baseline vs Treatment")
    print("=" * 60)
    print(f"  Baseline:  {baseline_path}")
    print(f"  Treatment: {treatment_path}")
    print()

    print(f"{'Metric':<35} {'Baseline':<14} {'Treatment':<14}")
    print("-" * 63)
    print(f"{'Edges discovered':<35} {str(b_edges):<14} {str(t_edges):<14}")
    print(f"{'Corpus entries':<35} {str(b_corpus):<14} {str(t_corpus):<14}")
    print(f"{'Avg eps':<35} {str(b_eps):<14} {str(t_eps):<14}")
    print(f"{'Execs':<35} {str(b_iters):<14} {str(t_iters):<14}")
    print(f"{'Total crashes':<35} {str(b_crashes):<14} {str(t_crashes):<14}")
    print(f"{'Unique crash signatures':<35} {str(b_unique):<14} {str(t_unique):<14}")
    print(f"{'Efficiency (unique/1k execs)':<35} {b_eff:<14.4f} {t_eff:<14.4f}")
    print()

    print(f"  Crash signature Jaccard similarity: {crash_jaccard:.4f}")
    print()

    if shared_sigs:
        print(f"  Shared crash signatures ({len(shared_sigs)}):")
        for sig in sorted(shared_sigs):
            print(f"    ✓ {sig}")
    print()

    if only_baseline:
        print(f"  Baseline-only crashes ({len(only_baseline)}):")
        for sig in sorted(only_baseline):
            print(f"    B {sig}")
    print()

    if only_treatment:
        print(f"  Treatment-only crashes ({len(only_treatment)}):")
        for sig in sorted(only_treatment):
            print(f"    T {sig}")
    print()

    if verbose:
        print("  ── Metrics (raw) ──")
        print(f"  Baseline edges: {b_edges}, corpus: {b_corpus}, eps: {b_eps}")
        print(f"  Treatment edges: {t_edges}, corpus: {t_corpus}, eps: {t_eps}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Differential analysis between two fuzzer bench runs."
    )
    parser.add_argument(
        "--baseline",
        "-b",
        required=True,
        help="Baseline bench log file",
    )
    parser.add_argument(
        "--treatment",
        "-t",
        required=True,
        help="Treatment bench log file",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show raw metrics",
    )

    args = parser.parse_args()
    return analyze(args.baseline, args.treatment, verbose=args.verbose)


if __name__ == "__main__":
    sys.exit(main())
