# Anti-fuzzing literature survey

Date: 2026-10-02. Scope: published anti-fuzzing techniques, the
countermeasures each paper discusses, and what `fuzzer-tool` ships or should
add in response. Revision 2: the papers were fetched and read (see "Reading
status"); revision 1 worked from abstracts and search snippets.

No code was changed in this revision. Every "Proposal" below is a design
derived from reading the papers and the current code; none has been run.

## Reading status

| Work | What was obtained | Confidence |
|---|---|---|
| AntiFuzz (Güler et al., USENIX Sec '19) | Full text (all sections, tables, refs) | High: numbers below are from the paper |
| Fuzzification (Jung et al., USENIX Sec '19) | Full text | High |
| Escaping the Fuzz (Göransson & Edholm, Chalmers MSc 2016) | Full text (thesis PDF via odr.chalmers.se) | High |
| No-Fuzz (Zhou et al., SecureComm '22) | Abstract only; full text is behind an EAI login / Springer paywall | Low: design details unknown |
| CatchFuzz (Kim & Lee, Comput. & Secur. '24) | Abstract only (Elsevier paywall) | Low |
| obfFuzz (ICSE '25 SVM workshop) | Abstract only | Medium for the claim, none for method |
| SAFTE (Comput. Electr. Eng. '23) | One search, nothing relevant returned | None: revision 1's summary is unverified |
| Practical Anti-Fuzzing w/ Perf. Opt. (IEEE '23) | One search, nothing relevant returned | None: revision 1's summary is unverified |

The AntiFuzz repo README (github.com/RUB-SysSec/antifuzz) was only seen
through an aggregator snippet; the `--enable-*` switch table from revision 1
was not re-verified and is dropped below.

## Corrections to revision 1

- **obfFuzz was mis-described.** Revision 1 called it "counter side:
  selective/dynamic deobfuscation to fuzz obfuscated binaries". Its abstract
  describes an *empirical study* of how obfuscation degrades AFL++ on four
  programs (pdfinfo, exif, tiffinfo, md2roff): -60 % coverage from control-flow
  obfuscation, +70 % time-to-crash from data-flow obfuscation. It supports
  AntiFuzz/Fuzzification's argument that obfuscation hurts fuzzing, but it
  proposes no countermeasure.
- **No-Fuzz numbers.** Revision 1 said "-92.2 % storage cost vs prior". The
  abstract says: under 15 % of the storage cost *per fake block*, and a 95 %
  reduction in total storage cost versus prior work for the same number of
  branch reductions. 92.2 % does not appear in the abstract; treat it as
  unsourced.
- **AntiFuzz configuration.** "Default 10k fake blocks" is imprecise. The
  evaluation used 10,000 fake *functions with constraints* plus 10,000 basic
  blocks for random edge generation, a 750 ms sleep (also applied from the
  crash handler to turn crashes into timeouts), SHA-512 for constant
  comparisons and AES-256-ECB (key derived from a hash of the input) for the
  encrypt/decrypt identity. The fake code adds roughly 25 MB to a binary.
- **Fuzzification does not use fuzzer detection.** Its Table 1 rejects
  "fuzzer identification" as non-generic and trivially bypassed (rename the
  fuzzer). Only AntiFuzz (ptrace/signal checks) and the Chalmers thesis
  (env var, ptrace) use detect-then-act.
- **Detection signals.** Revision 1 listed fork-server fds, `LD_PRELOAD` and
  "tracer present" as signals used by the literature. The three full texts
  name only: `__AFL_SHM_ID` plus `shmat` (Chalmers Listing 6),
  `ptrace(PTRACE_TRACEME)` failing (AntiFuzz §5.2, Chalmers Listing 7),
  `/proc/<pid>/status` signal masks and `strace` output as ways a *fuzzer
  user* can spot an anti-fuzzing binary (Chalmers §5.3.1), and timing
  (No-Fuzz abstract, unverified). Fork-server fds and `LD_PRELOAD` are not
  mentioned in the three texts.
- **Stale TODO.** "Hash-compare constants: no counter beyond cmplog digests"
  is out of date: `Fnv1aEncoder` (commit `6f69c174`) now covers the demo's
  hashed magic. SHA-512 remains uncountered; see AF-7.

## Taxonomy

Anti-fuzzing attacks four fuzzer assumptions (AntiFuzz §3, which surveyed 19
tools and found each relies on at least one):

| Assumption | Attack class |
|---|---|
| (A) Coverage feedback is truthful | Anti-feedback (fake / hidden coverage) |
| (B) Crash = observable signal | Crash masking |
| (C) High exec/s | Anti-fast-execution (delays) |
| (D) Constraints are solvable | Anti-hybrid (hash compares, implicit flows) |

Two activation modes: **always-on / passive** (AntiFuzz coverage and crash
techniques, all of Fuzzification) or **detect-then-act / active** (AntiFuzz's
ptrace check, Chalmers, CatchFuzz, No-Fuzz). Chalmers §4.3 notes the passive
mode also affects normal users (a real crash is masked too); the active mode
needs a reliable detector.

## Papers and what they specify

### AntiFuzz (full text)

| Technique | Mechanism | Reported effect |
|---|---|---|
| Fake coverage, type 1 | Hash of the input selects a few of 10,000 fake functions; each calls the next via a function-pointer table, adding many edges. One bit flip in the input changes the hash, so almost every input "finds new coverage" | Coverage cut 90-95 % on 8 binutils programs vs unprotected (typically <3 % of unprotected coverage in half the runs) |
| Fake coverage, type 2 | Random trees of nested input-dependent conditions, easy and very hard, to look like real parsing | as above |
| Crash masking | Install handlers for crash signals, deliberately raise a SEGV at startup on *every* run (input-independent), handler ignores it; later real crashes end in graceful exit or a timeout. Self-`ptrace` check: if it fails, someone is tracing, so terminate | No fuzzer found a crash in 24 h per config, 100 h with all techniques on |
| Delay | On developer-annotated error paths, sleep (750 ms in eval) or a compute-heavy task | n/a alone |
| Anti-symbolic | SHA-512 comparison instead of constant compare; encrypt-then-decrypt input with AES-256-ECB | KLEE and VUzzer (taint) fail; QSYM/Driller defeated |

Stated limits: a human can remove it; `sleep` on public servers is a DoS
risk (use rate limiting instead); a seed already containing the hashed value
lets concolic engines continue; junk code could be spotted statistically
(they suggest opaque predicates that touch real globals); AES was
recommended to be replaced by a weaker, faster cipher since the goal is to
confuse SMT solvers, not be secure. SPEC CPU2006 overhead: -0.42 % average.

### Fuzzification (full text)

| Technique | Mechanism | Reported effect |
|---|---|---|
| SpeedBump | Profile basic-block frequency on valid inputs, then inject delays into the coldest blocks (start: 30 ms into 3 % of least-executed blocks, tuned to an overhead budget; range 10-300 ms). Delay is CSmith-generated CPU-bound code with data/control dependence on the original program (no `sleep`) | 30 ms delay slows AFL-QEMU >50x; best single technique: -66 % paths (AFL-QEMU), -44 % (Honggfuzz PT), -59 % (QSYM) |
| BranchTrap 1 | Input-sensitive indirect jumps: function epilogues grouped into jump tables of equivalent gadgets, index = XOR of arguments mod table size. Deterministic per input (AFL discards non-deterministic paths) | -23 % / -14 % / -58 % |
| BranchTrap 2 | Inject 10k-100k deterministic branches into one or two cold blocks so the coverage map saturates: 40 % saturation of AFL's 64 KB bitmap cost ~25 % of paths, 80 % cost ~42 % in readelf; 100k branches fill >90 %, +4.6 MB. Argues CollAFL's unique ids do not help because attackers lack source and storage is still fixed-size | as above |
| AntiHybrid | Implicit data-flow copy of strings (bit-by-bit via branches) against taint; comparisons replaced by `CRC_LOOP(v) == CRC_OUT` against symbolic execution | -67 % (QSYM), also hurts plain fuzzers via slowdown |

Combined: -70.3 % paths, -93.0 % unique crashes (real programs),
-67.5 % bugs on LAVA-M, 5 % overhead budget. Self-reported weaknesses:
AntiHybrid code has recognisable patterns (hash/CRC loops); RedQueen's
input-to-state matching bypasses implicit data-flow copies, though it "can
be fooled" by adding the same constant to both comparison operands; delay
calibration is tied to the developer's hardware (a runtime-calibrated loop
would expose a timing syscall or `rdtsc`).

### Escaping the Fuzz (full text)

Evaluated AFL and Honggfuzz on DARPA CGC samples plus MediaInfo, then
masked crashes. Techniques that worked against both, with no crash found:

- **AFL**: wrap `main` so the real program runs in a `fork()` child and the
  parent `waitpid`s, then exits 0 for any signalled child (Listing 5). Cost:
  lower exec/s, and some crashes appeared as AFL *hangs* (the thesis's
  theory: signal delivery slower than AFL's timeout). Signal handlers also
  worked, but a bug in the instrumented build let some crashes through.
- **Honggfuzz**: it observes via ptrace, so handlers cannot hide a signal;
  instead detect it with `ptrace(PTRACE_TRACEME)` and on failure run
  `realMain` with the input swapped for a known-safe one (§5.2), which keeps
  coverage metrics looking legitimate.
- **Active AFL detection**: read `__AFL_SHM_ID`, `shmat` it; the shm is also
  writable, so a target could in principle forge path counts.

The thesis's own suggested defences (§7.1): the fuzzer inspects which
signals the target intercepts, or compares the syscalls executed under the
fuzzer versus a user run, and warns. It expects adoption to be resisted
because of the performance cost.

### Abstract-only works

- **No-Fuzz** (SecureComm '22): automatic anti-fuzzing; <15 % storage cost per
  fake block; for binary-only fuzzing it determines the running environment
  to avoid unnecessary storage; -95 % total storage vs prior for equal
  branch reductions. Mechanism unverified (revision 1's "timing-based env
  detection, random-byte landing space" is not in the abstract).
- **CatchFuzz** (Comput. & Secur. '24): loads the anti-fuzzing logic only
  after detecting the fuzzing environment, then disorders the information
  the fuzzer uses; 99.6 % detection accuracy, 0.5 % false positives,
  -95.39 % unique crashes on nine programs.
- **obfFuzz**: see Corrections.
- NDSS '26 poster "AntiBench" (Ricerca Security) proposes a benchmark that
  decomposes anti-fuzzing into design-level primitives (BranchTrap, delay
  sleep, ...). Seen only as a poster text fragment; a lead for AF-8, not a
  source.

## Shipped defeats (implementation details)

Five countermeasures, one per AntiFuzz technique plus the hash encoders, and
a benchmark target. Tests: `tests/test_coverage_noise.py` (11),
`test_regression_stderr_crash_spoof.py` (14), `test_rq_encodings.py -k fnv`
(5) pass in a clean clone; `test_antifuzz_evade.py` and
`test_regression_antifuzz_demo.py` skip 7 of 9 tests here (they need clang /
an ASAN build).

### 1. Hardened crash oracle: `adapters/process.py`, `services/runner.py`

`stderr_crash_marker(returncode, stderr)` answers crash masking by exit
rewrite (a crash turned into `exit 0` or a timeout). A marker in stderr only
corroborates a *failing* exit:

- Returns `None` when `returncode == 0` or is a sentinel (`-1`, `-2`), so a
  target that prints "Segmentation fault" then exits 0 is treated as **not**
  crashed: it is claiming a crash, not having one.
- Otherwise returns the first of `CRASH_STDERR_MARKERS` (`SIGSEGV`,
  `SIGABRT`, `SIGFPE`, `SIGBUS`, `Segmentation fault`, `Aborted`) in stderr.

`ExecutionRunner.is_interesting` (`runner.py:766`) adds the sanitizer rule:
returncode `-1` or `0` with `ASAN`/`AddressSanitizer` in stderr is
interesting. So a masked crash is recovered when the sanitizer report
reached stderr first. It is **not** recovered when the process is a
non-instrumented build, when the masking wrapper discards the child's
stderr, or when the child is killed before the report prints. Note the same
substring is target-controlled: a target can print "ASAN" and exit 0 to
flood the interesting set (the spoof-regression tests cover the stderr-marker
direction; this one was not audited).

Callers: `services/runner.py` (`_is_crash`), `root_cause.py`, `tmin.py`.

### 2. Coverage-noise probe: `core/coverage_noise.py`

Answers hash-keyed fake coverage (AntiFuzz coverage type 1 / BranchTrap 1).
Pure module, no I/O; the caller runs the target.

- **`classify_noise(base_runs, variant_sets)`**: offline probe. Appends one
  byte (values `1..NOISE_PROBE_VARIANTS`, default 8; `tail_variants`) to a
  seed. If the base seed's repeated runs disagree the verdict is
  `UNMEASURED` (nondeterminism is not the input's fault). If all 8 tail
  variants land on **distinct** edge sets: `SUSPECTED`; else `CLEAN`. Wired
  at `fuzzer.py:_report_coverage_noise`, warn-only. This matches AntiFuzz
  type 1, whose hash covers the whole input; it does **not** match type 2
  (value comparisons on specific bytes, which a tail byte never touches).
- **`AdmissionMonitor`**: runtime sibling. Fires once when admissions reach
  `FLOOD_RATE` (0.5) per execution after `MIN_EXECS` (5000).
- **`fake_novelty_factor(fuzz_count, children_admitted)`**: per-seed weight.
  After `FAKE_NOVELTY_MIN_FUZZ` (256) mutants, a seed whose children flood
  the corpus gets weight `FAKE_NOVELTY_PENALTY` (0.1). Applied in
  `seed_picker.py:_weight_fake_novelty`.

Map saturation (BranchTrap 2): the shim's coverage is an open-addressing
hash table keyed by the full 32-bit edge id, not AFL's 64 KB bitmap, so the
paper's collision mechanism does not apply directly. The analogous failure is
table exhaustion: once full, new edges are dropped and the shim counts them
in a saturating `dropped_edges` header field (`afl_shim.c` offset 24, read via
`ShmCoverage.read_dropped_edges()`). Nothing in the anti-fuzz path consumes
that counter yet; see AF-5.

### 3. `--antifuzz-evade` LD_PRELOAD shim: `adapters/antifuzz_evade.c` + `evade_shim.py`

Answers self-ptrace anti-debug (AntiFuzz §5.2) and delay-on-malformed
(§4.3/5.3) from outside the target, no source access, so binary-only modes
(ptrace coverage, Intel-PT, `--no-shm`) keep working:

- `ptrace(PTRACE_TRACEME, …)` returns 0. Every other request forwards to real
  libc, so the fuzzer's own ptrace coverage is untouched.
- `sleep` / `usleep` / `nanosleep` / `clock_nanosleep` return success
  immediately.
- Opt-out env: `ANTIFUZZ_EVADE_PTRACE=0`, `ANTIFUZZ_EVADE_SLEEP=0`.

`evade_shim.py` is the driver: compiles the `.so` once per process (`clang
-shared -fPIC -O2 … -ldl`, scrubbing `ASAN_OPTIONS`/`LSAN_OPTIONS`/
`LD_PRELOAD` from the compile env) and composes `LD_PRELOAD`. CLI flag
`--antifuzz-evade` (OFF by default, `commands.py:4072`) →
`fuzzer.py:_install_antifuzz_evade`; best-effort, prints a notice and
continues if the shim will not build.

Not covered (all verified against the papers): CPU-bound delays
(Fuzzification SpeedBump uses no `sleep`), statically linked targets
(`LD_PRELOAD` does not apply), crash masking by handler or fork wrapper,
and detection of the shim itself (e.g. `TRACEME` called twice, timing a
`sleep`).

### 4. Benchmark target: `targets/antifuzz_demo.c`

Wraps one real ASAN bug in all four AntiFuzz techniques, each env-gated
(default on):

| Env | Technique | Defeat exercised |
|---|---|---|
| `AF_COVERAGE=0` | hash-keyed fake edges (table of 64, bounded per Hard Rule 54; the paper uses 10k functions) | coverage-noise probe |
| `AF_CRASH=0` | crash masking | hardened oracle + ASAN |
| `AF_SPEED=0` | delay on malformed input (sleep) | `--antifuzz-evade` sleep |
| `AF_PTRACE=0` | self-ptrace anti-debug | `--antifuzz-evade` ptrace |
| `AF_HASHCMP=0` | hashed magic compare → `memcmp` | `Fnv1aEncoder` / cmplog |

`AF_CRASH`/`AF_PTRACE` run once in `main()`; the `.so` skips them (in
`direct_lite` they traced and re-handled the fuzzer's own process).
`--antifuzz-evade` is ignored in in-process mode: `LD_PRELOAD` cannot reach
an already-running host. Bug: input starting with 4-byte magic `crsh`
overflows a stack buffer; the magic is checked via a byte hash unless
`AF_HASHCMP=0`. Built by `tools/build_targets.sh` (ASAN + `afl_shim`, plus a
`fuzz_shm_run` `.so` for `direct_lite`), modelled on `asan_target.c`.

### 5. Hash-compare encoders: `core/rq_encodings.py`

`Crc32Encoder` (the Fuzzification CRC style) and `Fnv1aEncoder`
(`fnv1a_p`/`fnv1a_r`, 4-byte fields compared against a constant) invert the
hash to all preimages, so a cmplog pair of two hash values yields input
patches. `_FNV_MIN_OPERAND` filters operand pairs too small to be hashes.
16 encoders total; none models an additive or XOR constant on both operands
(the case Fuzzification says fools RedQueen).

## Proposals (not implemented)

Each proposal states what it answers, the current state, the design, files,
a test plan and risks. Effort: S ≈ under a day, M ≈ a few days, L ≈ a week+.

### AF-1 Clean-environment differential replay

*Answers:* detect-then-act (AntiFuzz ptrace check, Chalmers env/ptrace
detection, CatchFuzz). The Chalmers thesis proposes exactly this idea
(compare behaviour under the fuzzer vs a user run) but did not build it.

*Today:* every execution runs with fuzzer artefacts present
(`__AFL_SHM_ID`, `AFL_MAP_SIZE`, fork-server fds, optionally tracer). The
instrumented shim is a silent no-op without `__AFL_SHM_ID`
(`afl_shim.c:__afl_map_shm`), so a plain run of the same binary is a "clean"
environment for free.

*Design:* with probability `p` (default 1/500), re-run the input through a
plain `subprocess` runner with no shim env vars, no ptrace and no evade shim,
and compare `(returncode class, stdout digest, stderr digest)` with the
fuzzed run. Divergence on any input, not only crashing ones, shows the
target changes behaviour under fuzzing (Chalmers `fakeMain`, or the
input-swap variant, where outputs for non-"safe" inputs differ). After
`k` divergences print a warning and, optionally, treat clean-mode results as
authoritative for crash status. Also replay every *new corpus admission* in
clean mode: admissions are rare after the start (the `AdmissionMonitor`
already assumes <1 %), so this is cheap and automatically throttled when a
flood is detected.

*Touch points:* `services/runner.py` (second runner), `fuzzer.py` loop,
stats. **Prerequisite:** stdout is not captured in at least the in-process
loader path (`runner.py:704`, `stdout=subprocess.DEVNULL`); other modes were
not audited. Capture a bounded digest first.

*Tests:* extend `targets/antifuzz_demo.c` with `AF_DETECT=1`: when
`__AFL_SHM_ID` is set, print a different line and skip the bug. Expect the
replay to flag it within N runs and to stay silent on `asan_target`.

*Risks:* nondeterministic targets (reuse `classify_noise`'s repeated-run
check to establish a baseline); targets with side effects on replay;
effective only against targets whose behaviour differs on ordinary inputs
(a detector that changes behaviour *only* for crashing inputs is not caught).
Effort: M.

### AF-2 Crash-signal chaining in the evade shim

*Answers:* crash masking by handler or crash→timeout (AntiFuzz §4.2/5.2;
Chalmers §5.3.1). Closes the "oracle does not recover masked crashes" TODO
for dynamically linked targets.

*Today:* handler installs are untouched; the shim only fakes `TRACEME` and
sleeps.

*Design:* interpose `sigaction` and `signal` in `antifuzz_evade.c`. For
`SIGSEGV`, `SIGBUS`, `SIGFPE`, `SIGILL`, `SIGABRT` store the target's
requested action and install a trampoline `evade_crash_tramp(int,
siginfo_t*, void*)` instead. The trampoline writes a fixed-size record
(`signo`, `si_addr`, faulting `rip` from `ucontext`) to a descriptor named in
`ANTIFUZZ_EVADE_SIGFD`, then dispatches to the stored handler (or restores
`SIG_DFL` and re-raises when the target asked for default/ignore). The
runner reads the record after the run and treats it as a crash candidate.
Because the record is written *before* the target's handler runs, a handler
that sleeps then exits (AntiFuzz's crash→timeout) still leaves evidence.

AntiFuzz raises an input-independent SEGV on every run (§4.2), so the
runner must learn a **baseline fingerprint**: run an empty and a known-good
input at startup and record the `(signo, rip-offset)` pairs seen; only a
signal record with a *new* fingerprint counts as a crash. `rip` is made
module-relative (subtract the mapping base) to survive ASLR.

*AF-2b (no shim, cheap warn):* read `/proc/<pid>/status` `SigCgt` of the
first fork-server child and warn when crash-signal bits are set. Chalmers
§5.3.1 notes this is how a human spots handler-based masking. Caveat: ASAN
and other sanitizer runtimes also catch these signals, so only warn on
non-sanitizer builds, or compare with a sanitizer-only baseline.

*Touch points:* `adapters/antifuzz_evade.c`, `evade_shim.py` (pass the fd,
parse records), `services/runner.py`. Tests: `AF_CRASH=1` demo with a
SIGALRM-free real crash; expect one new-fingerprint record per crashing
input and none for clean runs.

*Risks:* ASAN installs its own SEGV handler and, by default, ignores user
`sigaction` for SEGV (`allow_user_segv_handler=0`), so interposition order
with a sanitizer runtime needs testing; statically linked targets are
unaffected; async-signal-safety (only `write` and the stored pointer table
in the trampoline). Effort: M-L.

### AF-3 Child-status hook for fork-wrapper masking

*Answers:* Chalmers Listing 5 (`fork` + `waitpid` + `exit(0)`), which hid
every crash from AFL and was the only technique that worked against it
without any signal handler.

*Today:* `stderr_crash_marker` ignores `exit 0`; the ASAN rule recovers it
only if the child's report reached the parent's stderr.

*Design:* in the same evade `.so`, interpose `waitpid`, `wait4`, `waitid`
(and `wait`). When the returned status has `WIFSIGNALED` with a crash
signal, append a record to `ANTIFUZZ_EVADE_SIGFD` (same format as AF-2)
before returning unchanged. The runner then promotes `exit 0` plus a
child-signal record to a crash. Complements AF-2: handlers catch
in-process faults, this catches faults in children.

*Alternative for binary-only/static targets:* ptrace with
`PTRACE_O_TRACEFORK` (the way Honggfuzz sees signals, per Chalmers §3.3.1).
Needs the `TRACEME` fake from the shim, or a `PTRACE_SEIZE` launch, to avoid
the self-trace check.

*Tests:* demo mode `AF_WRAP=1` with the Listing 5 wrapper; expect the crash
recovered with `exit 0` as the observed status.

*Risks:* the target can write to the same descriptor (same trust model as
stderr, so keep the oracle's rule that records only *upgrade* a run to
crash-candidate for re-verification, ideally via AF-1 clean replay); only
dynamic linking. Effort: S-M.

### AF-4 Reduce the shim's own detection surface

*Answers:* Chalmers §5.3.2 / Listing 6 (read `__AFL_SHM_ID`, `shmat`) and
AntiFuzz-style env checks. Revision 1's TODO.

*Today:* `afl_shim.c` reads `__AFL_SHM_ID` and `AFL_MAP_SIZE`
(`__afl_map_shm`), `__AFL_FORKSRV` (`__afl_start_forkserver`) and keeps them
in the environment; the fork-server uses fds 198/199 (`AFL_FORKSRV_FD`).

*Design:* once attached, copy the values into statics and `unsetenv` the
three variables so later `getenv` calls in `main` see nothing. Two checks
before doing so: (1) the comment in `__afl_map_shm` says `__AFL_SHM_ID` is
"inherited across exec", and the file header mentions a re-call of
`__afl_map_shm` after `fork`; any consumer that re-reads the environment
must read the cached copy; (2) targets that `exec` themselves lose coverage
unless the id is re-exported in a `execve` hook, so gate this behind a flag
(default off, e.g. `--stealth-env`).

*Residual:* a target can still scan `/proc/self/maps` for a SysV shm
mapping, probe fds, or time execution; this only removes the cheapest
(Chalmers) check. Do not claim more.

*Tests:* a target that prints `getenv("__AFL_SHM_ID") != NULL` after
`main` starts must print `0` under `--stealth-env` and still produce
identical edge ids to a run without it. Effort: S.

### AF-5 Fake-coverage handling beyond the tail probe

*Answers:* AntiFuzz coverage type 1 and 2; Fuzzification BranchTrap 1 and 2.

*Today:* tail-byte probe (hash dispatch only), admission-flood monitor,
seed weight penalty; `dropped_edges` counter exists but is unused here.

*Design (three independent pieces):*

1. **Behaviour-neutral edge check** (revision 1's own "flag inputs adding
   many edges with no behaviour change"). For a sample of admitted inputs
   compare the observable output digest of the input and a one-byte
   mutation. If the edge sets differ by at least `T` edges (start `T` =
   32) while `(returncode, stdout, stderr)` digests are equal, count the
   input as *noise-suspect*. Feeds `fake_novelty_factor` instead of only
   the flood-rate heuristic. Needs the stdout digest from AF-1. Also extend
   the probe from "tail byte" to "flip a bit at each of the first K
   positions", which reaches AntiFuzz type 2's per-byte constraints.
2. **Constant edge-block collapsing** (BranchTrap 2, which injects
   10k-100k deterministic branches in a cold block). Maintain, for rare
   edges only, a rolling occurrence signature over the last M executions.
   Edges with an identical signature and a class size of at least 256 are
   treated as one virtual edge for admission: one trigger, no per-edge
   novelty. `ShmCoverage._scan_with_positions()` already gives a stable
   position↔id bijection to key this on.
3. **Table-saturation alarm.** Warn when `read_dropped_edges()` rises
   during a campaign, and when occupancy passes a fraction of
   `AFL_MAP_SIZE` without a matching rise in real coverage. The dropped
   counter is the direct analogue of Fuzzification's bitmap saturation; the
   remedy is already available (`resize`, per the header comments), so
   automating "grow the table when dropped edges appear and a trap block is
   the cause" is the follow-up.

*Tests:* extend the demo's fake-function table from 64 to a configurable
size (AntiFuzz uses 10,000) and add an `AF_TRAP=<n>` gate that executes `n`
deterministic branches from one cold block. Expect piece 2 to collapse the
block and piece 3 to fire for `n` above the table size.

*Risks:* piece 1 false-positives on parsers with many codec-specific paths
and no output difference (this is the existing TODO (c), measure first);
piece 2 can merge real straight-line code, which is harmless for admission
but should not hide it from edge-count stats. Effort: L.

### AF-6 Cold-path delay attribution

*Answers:* Fuzzification SpeedBump (CPU-bound, not `sleep`; >50x slowdown at
30 ms on 3 % of cold blocks) and AntiFuzz's sleep. The `sleep` hook does not
touch SpeedBump.

*Today:* only the `sleep` family is neutralised; per-execution timing is
tracked for timeouts, nothing attributes cost to edges.

*Design:* for each execution record `(wall or CPU time, edge set)`. For edges
that appear in under 5 % of executions (SpeedBump's cold paths), keep
`n_with`, `sum_with` and compare `mean_with` against the global mean. An edge
whose mean-with exceeds the global mean by `Δ` (start 5 ms) over at least 30
samples is a *delay suspect*. Use it to (a) down-weight seeds that reach it,
(b) report it, (c) bias mutation away from the bytes that flipped the branch
into it (position info from `_scan_with_positions`). This cannot remove the
delay, only avoid it; the Chalmers thesis and Fuzzification both agree an
adversary with enough time still finds the bug, and the aim is to stop
wasting most executions on cold error paths.

*Tests:* demo gate `AF_SPEED=cpu` using a busy loop (calibrated like
SpeedBump) on the malformed-input path; expect the error-path edges flagged
and exec/s recovering by the fraction of avoided executions. Measure, don't
assume.

*Risks:* noisy timing on shared hosts (use CPU time, not wall); legitimately
slow paths get avoided too (keep a floor so they are still explored).
Effort: M.

### AF-7 Hash-compare tiers

*Answers:* AntiFuzz SHA-512 constants and AES-ECB identity; Fuzzification CRC
loops and implicit data-flow copies.

| Tier | Case | Counter |
|---|---|---|
| 1 | Invertible or short hash (CRC32, FNV-1a) | Shipped: `Crc32Encoder`, `Fnv1aEncoder` |
| 2 | Non-invertible hash on a small field | Hash-site detection from cmplog (operands ≥ 8 bytes, high-entropy, lhs changes on any single-bit input flip, rhs constant), then bounded brute force over the field (≤ 4 bytes = 2^32 candidates) with a native implementation of the hash. Throughput is unmeasured: benchmark FNV first and extrapolate per hash |
| 3 | Seed already contains the value | AntiFuzz itself notes concolic engines continue from such a seed; prefer valid seed corpora over synthetic ones for hash-guarded formats |
| 4 | SHA-512 on larger fields, AES-ECB identity | No counter; document it. Fuzzification admits a hash-guarded branch can still be hit "rarely" |

Separately, add an additive/XOR-constant encoder (Fuzzification's note that
`(a+c) == (b+c)` defeats RedQueen-style matching; the repo's 16 encoders have
none): infer `c` by flipping an input byte and watching the logged operand
delta, then synthesise patches for `a` and `b`.

Tier 2 needs care: brute forcing is only viable for tiny fields, so cap it
(time budget and field size) and log when it gives up. Effort: M per tier.

### AF-8 Evaluation and the demo target

*Answers:* revision 1's open TODO (ASAN validation) and the benchmark gap
the NDSS '26 AntiBench poster points at (anti-fuzzing evaluated per
primitive, not per tool).

Two parts. (1) Do the tracked `docs/TODO.md` items: build with
`tools/build_targets.sh`, confirm `antifuzz_demo` has `__afl` in the exe and
`fuzz_shm_run` in the `.so`, run `fuzzer-tool fuzz antifuzz_demo` with each
`AF_*` gate on and off (keep `AF_HASHCMP=0`, else no configuration reaches
the bug) and record crashes found and edges with versus without each defeat.
(2) Extend the demo toward the papers' primitives so AF-1…AF-7 each have a
gate: `AF_DETECT`, `AF_WRAP`, `AF_TRAP=<n>`, `AF_SPEED=cpu`, fake-function
count > 64, and an SHA-style non-invertible compare.

Effort: M. Needs clang and an ASAN runtime, absent in the authoring sandbox.

## Open / TODO

- [ ] ASAN-build validation of the shipped defeats (tracked in
  `docs/TODO.md`); 7 of 9 antifuzz tests skip without clang.
- [ ] AF-1 … AF-8 above: none implemented. Suggested order: AF-3 (small, only
  the shim), AF-4 (small, flag-gated), AF-1 (needs a stdout digest), then
  AF-2, AF-6, AF-5, AF-7.
- [ ] SHA-512 / AES-ECB compares: no counter (AF-7 tier 4).
- [ ] Obtain the No-Fuzz, CatchFuzz, SAFTE and IEEE papers (paywalled); until
  then their numbers here are abstract-level and the techniques are
  unverified.
- [ ] Re-verify the AntiFuzz repo README switch list before citing it.

## Sources

- AntiFuzz: https://www.usenix.org/system/files/sec19-guler.pdf (read) ·
  https://github.com/RUB-SysSec/antifuzz (snippet only)
- Fuzzification: https://www.usenix.org/system/files/sec19-jung.pdf (read) ·
  https://github.com/sslab-gatech/fuzzification (named in the paper)
- Escaping the Fuzz (read): https://odr.chalmers.se/handle/20.500.12380/238600
  (earlier link: https://publications.lib.chalmers.se/records/fulltext/238600/238600.pdf)
- No-Fuzz (abstract): https://eudl.eu/doi/10.1007/978-3-031-25538-0_38 ·
  https://link.springer.com/chapter/10.1007/978-3-031-25538-0_38
- CatchFuzz (abstract):
  https://pure.korea.ac.kr/en/publications/catchfuzz-reliable-active-anti-fuzzing-techniques-against-coverag/ ·
  https://www.sciencedirect.com/science/article/abs/pii/S0167404824002062
- obfFuzz (abstract):
  https://conf.researchr.org/details/icse-2025/svm-2025-papers/4/obfFuzz-Empirical-Study-to-Boost-Fuzzing-the-Obfuscated-Software
- SAFTE (not retrieved): https://www.sciencedirect.com/science/article/pii/S0045790623004044
- Practical Anti-Fuzzing (not retrieved): https://ieeexplore.ieee.org/document/10209185/
- AntiBench poster (fragment): https://www.ndss-symposium.org/wp-content/uploads/ndss26-poster-86.pdf
