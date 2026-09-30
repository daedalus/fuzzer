# Handover: crash-bisect (2026-09-30)

Source: oss-fuzz `infra/bisector.py` (port analysis, item 5).

## Built
- `adapters/git_worktree.py`: `GitWorktree` runs `git bisect` in a detached temp worktree; caller's checkout/index/bisect state untouched.
- `services/crash_bisect.py`: `CrashBisector`; `fuzzer-tool crash-bisect REPO CRASH --good REV [--bad REV] --target PATH --build-cmd CMD [--fixed] [--match class|exact|any]`.
- Endpoints verified first; signature pinned from the crashing endpoint. Unbuildable commits -> `git bisect skip`; skipped-out lists candidates, never guesses.
- `tests/test_crash_bisect.py` (20): shell-script targets (`kill -SEGV $$`), merge-DAG case, pinned-signature adversarial case, control on a no-crash history, CLI end-to-end.

## Not covered
- Not run against a real ASAN library history; only script targets.
- Build cost dominates: one build per probe (~log2 N + 2). No build cache across probes.
- `--match class` treats two distinct bugs with the same sanitizer error type as one.
- No oss-fuzz-style fuzzer-corpus reproducer lookup; the crash file is given.
- Pre-commit full suite not run here (Rule 50); only affected tests.
