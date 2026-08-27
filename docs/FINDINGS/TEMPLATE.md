# <TARGET_NAME> Fuzzing Findings

> Replace `<TARGET_NAME>` with the name of the library/tool being fuzzed (e.g. `FFmpeg`, `fgrep`).

---

## Target

[<PROJECT_NAME>](<PROJECT_URL>) <VERSION> — <BRIEF_DESCRIPTION>.

**Library / Component versions** (if applicable):

| Library / Component | Version |
|---------------------|---------|
| <component_1>       | <ver>   |
| <component_2>       | <ver>   |

---

## Fuzz Target(s)

`targets/<TARGET_FILE>.c` — <BRIEF_DESCRIPTION_OF_WHAT_THE_TARGET_DOES>.

| Target | Attack Surface | Input Format |
|--------|---------------|--------------|
| `<target_1>` | `<surface_1>` | `<format_1>` |
| `<target_2>` | `<surface_2>` | `<format_2>` |

**Compilation flags**: <e.g. `-O2 -g -fsanitize=address -mavx2 -lpthread`>

---

## Fuzzing Configuration

- **Engine**: <fuzzer-tool / AFL / libFuzzer / etc.> with <key features: Markov byte generation, coverage-guided mutations, etc.>
- **Coverage**: <AFL SHM bitmap / ASAN / etc.>
- **Inputs generated**: <corpus sources, e.g. PNG corpus, incremental mutations, seed files>
- **Iterations / Duration**: <e.g. 10,000 per target / 10h 43m>

---

## Results

| Metric | Value |
|--------|-------|
| Total crashes discovered | <N> (unique signatures: <M>) |
| Crash input size | <bytes> |
| Crash input type | <format description> |
| Trigger | <function / API call> |
| Crash type | <e.g. `av_assert0(0)`, `SIGFPE`, `heap-buffer-overflow`> |
| Severity | **<LOW / MEDIUM / HIGH / CRITICAL>** |
| Exploitability | <Remotely triggerable / local DoS / etc.> |

---

## Bug: <BUG_TITLE>

**File**: `path/to/source.c:<LINE>`
**Severity**: <LEVEL> — <ONE_LINE_IMPACT_SUMMARY>
**Root cause**: <ONE_SENTENCE_ROOT_CAUSE>
**Discovered by**: <e.g. fuzzer-tool with ASAN via `direct_lite` mode>

### Description

<Detailed explanation of the bug. What code path is reached, what goes wrong, and why it matters. Include relevant code snippets showing the vulnerable logic.>

```c
// Vulnerable code snippet
```

### Trigger Chain

1. **<STEP_1_NAME>** — <description of what happens>
   - e.g. Demuxer identifies the input as a `<format>` stream
   - e.g. `codec_id = <CODEC_ID>`, `codec_type = <TYPE>`
2. **<STEP_2_NAME>** — <description>
3. **<STEP_3_NAME>** — <description of the crash trigger>

### Crash Input

<Hex dump or description of the minimal crashing input.>

```
00000000  <hex_dump>  |<ascii>|
```

- Byte <offset>: `<value>` — <meaning>
- Byte <offset>: `<value>` — <meaning>

### Crash Evidence

<ASAN output, GDB backtrace, or relevant crash logs.>

```
==<PID>==ERROR: AddressSanitizer: <crash_type> on address <addr>
<full ASAN / GDB output>
```

### Crash Metadata

| Field | Value |
|-------|-------|
| Signal | `<e.g. SIGFPE (returncode −8)>` |
| Fault address / RIP | `<address>` |
| RSP | `<stack_pointer>` |
| Execs to find | `<N>` |
| Corpus at find | `<N> entries` |
| Elapsed | `<time>` |
| Parent seed | `<seed_hash>` |
| Target SHA256 | `<hash>` |

### Exploitability Assessment

| Factor | Assessment |
|--------|-----------|
| Crash determinism | <Deterministic / Non-deterministic> |
| Trigger depth | <Shallow / Deep — requires specific setup> |
| Preconditions | <None / specific conditions> |
| Signal type | <SIGFPE, SIGSEGV, assertion, etc.> |
| Memory safety | <OOB read/write, use-after-free, no memory corruption> |
| Reach | <Any application that calls X on untrusted data> |
| Severity | **<LEVEL>** — <one-line severity justification> |

### Suggested Fix

<Description of the fix approach. Include a code diff showing the minimal change.>

```diff
 // Before:
 <vulnerable_code>

 // After:
 <fixed_code>
```

**Why this fix is correct:**

- <reason_1>
- <reason_2>
- <reason_3>

**Alternative approaches** (if applicable):

| Approach | Pros | Cons |
|----------|------|------|
| <approach_1> | <pros> | <cons> |
| <approach_2> | <pros> | <cons> |

### Regression Test

```c
/* Trigger: <brief description of crash trigger> */
static const unsigned char <name>_crash[] = {
    <hex_bytes>
};

/* Expect: <expected behavior after fix> */
```

### Upstream Status

<e.g. Not reported upstream at the time of discovery.>

**To report** (per <link to bug reporting guidelines>):
1. Verify the bug still exists against the **latest development branch** (`git HEAD`), not just the vendored version.
2. Register at <upstream_issue_tracker_url> and submit an issue.
3. Include: the <N>-byte crash input, the GDB backtrace, and the regression test above.
4. Upload the crash sample to <upload_url> (select <project>).

---

## <ADDITIONAL_SECTION_HEADERS>

<Use this section for context-specific information that doesn't fit the standard bug template. Examples from existing findings:>

- `### ffmpeg CLI Reproducibility` — details on whether the CLI triggers the bug and why/why not
- `### Root Cause Analysis` — deeper analysis of contributing factors and design issues
- `### Bug Discovery Method` — technical details on how the fuzzer found the bug
- `### Impact` — broader impact assessment beyond the immediate crash

---

## See Also

- **[<LINK_TEXT>](<PATH>)`** — <one-line description of related finding>
