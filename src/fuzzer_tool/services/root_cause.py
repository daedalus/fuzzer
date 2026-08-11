"""Root-cause byte diff service.

CLI-facing wrapper around ``core.root_cause``: reproduces a crash, picks a
non-crashing baseline (an explicit ``--baseline`` file, or the nearest
corpus seed by Jaccard+Levenshtein similarity), Levenshtein-aligns the two,
and delta-debugs the edit script against the live target to isolate the
minimal set of byte changes that actually cause the crash.
"""

import os
import shutil
import sys
from pathlib import Path

from fuzzer_tool.adapters.filesystem import hash_data
from fuzzer_tool.core.root_cause import (
    apply_edit_subset,
    build_edit_script,
    ddmin_edits,
    edit_indices,
    format_root_cause_report,
)


def _load_corpus(corpus_dir: str) -> list[tuple[str, bytes]]:
    """Read every regular file in *corpus_dir* as a candidate baseline seed."""
    seeds: list[tuple[str, bytes]] = []
    corpus_path = Path(corpus_dir)
    if not corpus_path.is_dir():
        return seeds
    for p in sorted(corpus_path.iterdir()):
        if not p.is_file():
            continue
        try:
            seeds.append((p.name, p.read_bytes()))
        except OSError:
            continue
    return seeds


def root_cause(
    target: str,
    crash_file: str,
    corpus_dir: str | None = None,
    baseline_file: str | None = None,
    timeout: float = 5.0,
    file_mode: bool = False,
    target_args: list[str] | None = None,
    use_coverage: bool = False,
    max_stages: int = 200,
) -> dict | None:
    """Isolate the minimal byte diff, relative to a non-crashing baseline,
    that is responsible for triggering the crash in *crash_file*.

    Either *baseline_file* (an explicit known-good input) or *corpus_dir*
    (searched for the nearest non-crashing seed) must be usable.

    Returns a dict with keys ``baseline_name``, ``baseline``, ``crash``,
    ``script``, ``minimal_indices``, ``minimal_bytes``, ``signature``,
    ``report`` -- or ``None`` if the crash couldn't be reproduced or no
    valid non-crashing baseline was available.
    """
    crash_path = Path(crash_file)
    if not crash_path.is_file():
        print(f"[-] Crash file not found: {crash_file}", file=sys.stderr)
        return None
    crash_data = crash_path.read_bytes()
    if not crash_data:
        print("[-] Crash file is empty", file=sys.stderr)
        return None

    print(f"[*] Crash input: {len(crash_data)} bytes, hash={hash_data(crash_data)}")

    tmp_dir = Path("/tmp") / f"rootcause_{os.getpid()}"
    if file_mode:
        tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        from fuzzer_tool.adapters.process import SIGNAL_CRASH_CODES, run_target_file, run_target_stdin
        from fuzzer_tool.core.sanitizer import SanitizerReport

        def _run_target(data_bytes: bytes) -> tuple[int, str]:
            env = os.environ.copy()
            if use_coverage:
                env["AFL_MAP_SIZE"] = "8192"
            if file_mode:
                rc, stderr, _pid = run_target_file(
                    target, data_bytes, timeout, str(tmp_dir), target_args or [], env=env
                )
                return rc, stderr
            rc, stderr, _pid = run_target_stdin(target, data_bytes, timeout, env=env)
            return rc, stderr

        def _crash_signature(returncode: int, stderr: str) -> str | None:
            if returncode in (-2, -1):
                return None
            report = SanitizerReport.parse(stderr)
            if report and report.is_valid():
                return report.signature
            if returncode in SIGNAL_CRASH_CODES or returncode < 0:
                return f"signal:{abs(returncode)}"
            for sig in ["SIGSEGV", "SIGABRT", "SIGFPE", "SIGBUS", "Segmentation fault", "Aborted"]:
                if sig in stderr:
                    return f"signal:{sig}"
            return None

        def _is_crash(data_bytes: bytes, expected_sig: str | None = None) -> str | None:
            returncode, stderr = _run_target(data_bytes)
            sig = _crash_signature(returncode, stderr)
            if sig is None:
                return None
            if expected_sig is not None and sig != expected_sig:
                return None
            return sig

        original_sig = _is_crash(crash_data)
        if original_sig is None:
            print("[-] Crash not reproduced with original input", file=sys.stderr)
            return None
        print(f"[*] Reproduced. Original signature: {original_sig}")

        # ── Pick the baseline ────────────────────────────────────────
        baseline_name: str | None = None
        baseline: bytes | None = None

        if baseline_file:
            bpath = Path(baseline_file)
            if not bpath.is_file():
                print(f"[-] Baseline file not found: {baseline_file}", file=sys.stderr)
                return None
            candidate = bpath.read_bytes()
            if _is_crash(candidate, expected_sig=original_sig) is not None:
                print(
                    f"[-] Baseline {baseline_file} also reproduces the crash -- "
                    "it isn't non-crashing",
                    file=sys.stderr,
                )
                return None
            baseline_name, baseline = bpath.name, candidate
        else:
            if not corpus_dir:
                print(
                    "[-] Need --baseline or --corpus-dir to find a non-crashing input",
                    file=sys.stderr,
                )
                return None
            corpus = _load_corpus(corpus_dir)
            if not corpus:
                print(f"[-] Corpus is empty or missing: {corpus_dir}", file=sys.stderr)
                return None

            from fuzzer_tool.core.crash_metadata import find_nearest_corpus

            label, sim, _diffs, _summary = find_nearest_corpus(
                crash_data, [b for _, b in corpus], max_check=len(corpus)
            )
            if not label:
                print("[-] Could not find a nearest corpus seed", file=sys.stderr)
                return None
            idx = int(label.rsplit("_", 1)[1])
            candidate_name, candidate = corpus[idx]
            print(f"[*] Nearest corpus seed: {candidate_name} (similarity={sim:.2f})")

            # A stale corpus seed can itself crash against a rebuilt target --
            # confirm it's genuinely non-crashing before trusting it as baseline.
            if _is_crash(candidate, expected_sig=original_sig) is not None:
                print(
                    f"[-] Nearest seed {candidate_name} also reproduces this crash "
                    "signature -- pass --baseline explicitly with a confirmed-good input",
                    file=sys.stderr,
                )
                return None
            baseline_name, baseline = candidate_name, candidate

        if baseline == crash_data:
            print("[-] Baseline and crash input are byte-identical -- nothing to diff", file=sys.stderr)
            return None

        # ── Build and sanity-check the edit script ──────────────────
        script = build_edit_script(baseline, crash_data)
        changes = edit_indices(script)
        if not changes:
            print("[-] No edits found between baseline and crash input", file=sys.stderr)
            return None
        print(
            f"[*] Baseline: {baseline_name} ({len(baseline)} bytes) -> "
            f"crash ({len(crash_data)} bytes), {len(changes)} edit(s) total"
        )

        reconstructed = apply_edit_subset(baseline, script, set(changes))
        if reconstructed != crash_data:
            print(
                "[-] Internal error: edit script did not reconstruct the crash input",
                file=sys.stderr,
            )
            return None
        if _is_crash(reconstructed, expected_sig=original_sig) is None:
            print(
                "[-] Crash did not reproduce deterministically during setup -- "
                "target may be flaky, root-cause diff would be unreliable",
                file=sys.stderr,
            )
            return None

        # ── Delta-debug the edit script ─────────────────────────────
        print(f"[*] Delta-debugging {len(changes)} edit(s) (max {max_stages} stages)...")

        def _interesting(candidate: bytes) -> bool:
            return _is_crash(candidate, expected_sig=original_sig) is not None

        minimal_bytes, minimal_indices = ddmin_edits(
            baseline, script, _interesting, max_stages=max_stages
        )

        report = format_root_cause_report(baseline, crash_data, script, minimal_indices)
        print(f"[+] Isolated {len(minimal_indices)}/{len(changes)} edit(s) as root cause")
        print(report)

        return {
            "baseline_name": baseline_name,
            "baseline": baseline,
            "crash": crash_data,
            "script": script,
            "minimal_indices": minimal_indices,
            "minimal_bytes": minimal_bytes,
            "signature": original_sig,
            "report": report,
        }
    finally:
        if file_mode and tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Root-cause byte diff for a crash")
    parser.add_argument("target", help="Path to target binary")
    parser.add_argument("crash_file", help="Path to crashing input file")
    parser.add_argument("-t", "--timeout", type=float, default=5.0)
    parser.add_argument("-F", "--file-mode", action="store_true")
    parser.add_argument("-A", "--target-args", nargs=argparse.REMAINDER)
    parser.add_argument("-c", "--coverage", action="store_true")
    parser.add_argument("-d", "--corpus-dir", default=None)
    parser.add_argument("-b", "--baseline", default=None)
    parser.add_argument("--max-stages", type=int, default=200)
    parser.add_argument("-O", "--output", default=None)
    args = parser.parse_args()

    result = root_cause(
        target=args.target,
        crash_file=args.crash_file,
        corpus_dir=args.corpus_dir,
        baseline_file=args.baseline,
        timeout=args.timeout,
        file_mode=args.file_mode,
        target_args=args.target_args,
        use_coverage=args.coverage,
        max_stages=args.max_stages,
    )
    if result is None:
        sys.exit(1)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(result["report"])
        print(f"[+] Report saved to {args.output}")


if __name__ == "__main__":
    main()
