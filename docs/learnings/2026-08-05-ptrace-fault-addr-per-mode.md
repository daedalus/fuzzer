# ptrace-fault-addr-per-mode: extending PTRACE_GETSIGINFO capture to the .so modes that don't ptrace

**Date:** 2026-08-05
**Context:** fuzzer-tool (`src/fuzzer_tool/`), extending the landed `8c28481` ptrace capture to the persistent-loader and direct_lite modes (commit `5fdc927`)

## Problem

The two landed commits wired PTRACE_GETSIGINFO fault-address + register capture only into the ptrace `TargetRunner`. The two other `.so` crash paths detect crashes via different mechanisms — the persistent loader's guarded call returns `-sig`, `is_crash` maps waitpid status — so they produced no `si_addr`: NULL-deref vs wild-pointer crashes deduped together as `signal:11` and the sidecar registers stayed zeroed.

## Rejected

- **C-level SA_SIGINFO sigaction in direct_lite** (install a raw siginfo handler via ctypes, parse ucontext/siginfo) — looked plausible because it'd capture in the hot loop; dropped because it's fragile platform-specific struct-offset work for a mode built to avoid overhead, and you'd be parsing registers in the fuzzer's own doomed process.
- **Inline TRACEME + rewrite `_run_c_subprocess`'s `communicate` into a manual waitpid loop** (subprocess loader) — user scoped out: the subprocess loader's tracee is the script *process itself*, so the fuzzer is the tracer and `communicate()` blocks forever on a stopping tracee.
- **direct_lite triage via the existing `_run_target_ptrace`** — dropped: it `exec()`s the target binary, which is useless for `.so` targets.
- **direct_lite triage by re-running through `PersistentLoader`** — heavier than needed (spawns a persistent subprocess for a one-shot triage); the lighter one-shot script won.

## Approach

- **Persistent loader (inline, free).** The fork-per-call grandchild (P2) self-`PTRACE_TRACEME`s after `os.setsid()` before the guarded call (best-effort — an EPERM just runs untraced). Its direct parent (P1) `waitpid`s with a `WUNTRACED` stop loop: on a crash stop capture `si_addr` + rip/rsp/rbp, then `PTRACE_CONT`-with-signal so the guarded call (or default disposition) still runs. Fault state relays P2→P1→P0 by appending `<fa> <rip> <rsp> <rbp>` to the `RC` line.
- **direct_lite (re-run for triage).** Where the target runs inside the fuzzer's own process there's no signal boundary to attach at — so on a crash with no captured address, re-run the bytes once through the subprocess loader script armed with `_PTRACE_TRACEME=1` (`TargetRunner._run_triage_ptrace` loop), capturing the fatal stop with the already-existing `_capture_crash_state` helper. Gated on a lazily-probed `ptrace_available()` and only for `.so`+direct_lite, so the subprocess spawn costs nothing on the hot path.
- **Part C** (standalone-exec clamp: signal-killed targets now exit 128+sig instead of being clamped to 0) and **Part D** (deadline-stop capture gap) were adjacent and folded in.

## Key insight

direct_lite and the persistent loader are structurally different in *where* a signal-carrying process boundary exists. The loader has a real separate child per call, so inline TRACEME + a tracing wait loop are nearly free — genuinely "extend the landed fix." direct_lite loads the target into its own process, so there is never a tracee to attach to; capture must become a *process-on-crash* run, not an in-loop capture. The right mechanism is dictated by the mode's topology, not by the feature being added.

## Verification

- Targeted smoke before tests: persistent loader `CRASHS → fault_addr == 0` (NULL-jump, si_addr 0) + real rsp/rbp; `CRASHA → None` (abort excluded) but regs captured; SAFE → clean.
- direct_lite `_run_triage_ptrace` with a fake fuzzer: `CRASHS → 0`, `CRASHA → None`, SAFE → clean; the on-demand loader tempfile gets written.
- Clamp: standalone-exec segv → loader exit 139, abort → 134, safe → 0; timeout → 137 (not in `SIGNAL_CRASH_CODES`, so no false-crash regression).
- Full suite: 3111 passed (baseline 3099 → +12 new tests in three regression files); existing persistent/subprocess/ptrace-metadata suites green.

## Generalizes to

When extending a capture/diagnostic feature across execution modes, first map each mode's *signal boundary*: separate-child/per-call modes get inline capture at the mode's own wait-loop; same-process modes have no attach point and need a re-run/triage execution instead (extra latency only on the rare crashing input). Reuse the capture helper that already exists — only the loop shape and where it lives differ per boundary.
