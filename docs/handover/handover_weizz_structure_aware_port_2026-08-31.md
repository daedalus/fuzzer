# Handover — Weizz structure-aware port (unknown chunk formats)

**Date:** 2026-08-31  
**Sources:**
- `andreafioraldi/weizz-fuzzer` (ISSTA 2020) — local extract under `weizz-src/weizz-fuzzer-master/` (src, include, llvm-tracer only; full QEMU tree omitted).
- Paper: Fioraldi, D'Elia, Coppa — *WEIZZ: Automatic Grey-box Fuzzing for Structured Binary Formats* (https://doi.org/10.1145/3395363.3397372).
- This tree: `daedalus/fuzzer` as of the 2026-08-31 snapshot used for the survey.

**Status: ANALYSIS ONLY — nothing implemented.** This document is the port plan and the ranking against live code. No operators, no tag map, no A/B flag yet. Before writing code, re-grep the items in §3 against `src/fuzzer_tool/`; several neighbouring techniques (cmplog, colorization, TLV/FrameShift, checksum learner) already exist and must not be duplicated.

**Update (implementation status, see `docs/handover/P1_weizz_tags_README.md` for detail):**
P1–P3 and P5 are implemented and tested; P4 (tag-restricted surgical solve,
`_weizz_restricted_find` in `_op_condstmt_solve`) is implemented as of this
update. The only item left from §8's acceptance checklist is the paired
bench against baseline (`tools/bench_paired.py`) — an evaluation run
against a live target, not a code change. The rest of this document is
kept as-written for the original rationale/ranking; treat the "Status"
line above as historical.

**Relation to `docs/port-backlog.md`:** this is a concrete instance of group **A — Structure-aware generation**, closest in spirit to **A1 (Grimoire-style generalization)** but driven by comparison-dependency tags rather than coverage-preserving blanking. It does not replace A1; it is an alternate structure signal that can feed the same mutation registry.

---

## 1. What Weizz is (and is not)

Weizz is an AFL fork specialised for **unknown chunk-based binary formats**. It does **not** take a format specification. Pipeline:

1. **Comparison tracing** — operands, shape, and a temporal counter per comparison site (`cmp_id`).
2. **`get_deps`** — differential / colorization-style execution to build per-byte dependency bitvectors against each comparison operand.
3. **Tag assignment** — every input byte gets a packed tag; contiguous same-`cmp_id` runs become fields; parent/counter nesting approximates chunks.
4. **Higher-order mutations** — field-scoped havoc and AFLSmart-inspired chunk insert/delete/duplicate/swap, guided by the live tag map.
5. **Surgical solve + checksum repair** — input-to-state for individual comparisons; length/CRC candidates inferred from tags and patched after structural edits.

It is **not** a general-purpose fuzzer. The authors restrict the claim to chunk-oriented inputs (media containers and similar). The QEMU tracer and the rest of the AFL C core are scaffolding, not the technique.

### Tag layout (from `include/weizz.h`)

```c
struct tag {
  u16 cmp_id;
  u16 parent;
  u16 counter;    /* temporal / height order */
  u16 depends_on;
  u8  flags;      /* TAG_IS_LEN, TAG_IS_IMPL, … */
} __attribute__((packed));

struct tags_info {
  u32 ntypes;
  u32 max_counter;
  struct tag tags[];  /* one per input byte */
};
```

Key source files in the extract:

| File | Role |
|------|------|
| `src/get_deps.c` | Dependency recovery + colorization |
| `src/tags.c` | Field/chunk inference, `field_mutator`, `higher_order_fuzzing` |
| `src/surgical_fuzz.c` | Per-comparison surgical stage |
| `src/checksums.c` | Checksum location + patch using tags |
| `src/fuzz_one.c` | Stage ordering, getdeps gating (`-L`, once-per-seed) |
| `include/weizz-shared.h` | `cmp_map` / operand log layout |

---

## 2. What this tree already has

Do **not** reimplement these. Port only the missing inference layer and the operators that consume it.

| Capability | Location / notes |
|------------|------------------|
| Comparison tracing | `core/cmplog.py`, shim (`-D__AFL_CMPLOG=1`), libc + `__sanitizer_cov_trace_cmp*` |
| Colorization / differential | `core/colorization.py`, `colorizer.py` |
| Redqueen-style / adaptive solve | adaptive operators, `cond_stmt`, SMT, gradient, magic-byte search |
| Hard-coded format structural mutators | `core/mutations/{png,jpeg,isobmff,nal,mpegts,riff,protobuf,der,elf,zip,sqlite,…}.py` |
| Generic TLV / tree | `tlv_mutate.py`, `tree_mutator.py` |
| Length/offset repair | `structural_constraints.py`, `field_constraints.py`, FrameShift |
| Checksum learning | `checksum_learner.py` (Berlekamp–Massey), `crc32.py` |
| Operator registry + schedulers | 147 operators, Elo / bandits / MOpt / … |

**Gap:** there is no per-seed, comparison-inferred **byte-level tag map** for *unknown* chunk formats, and no mutation operators that treat those tags as fields/chunks. Structure today is either format-specific or generic TLV/tree heuristics.

---

## 3. Port candidates (ranked)

### P1 — Tag map from cmplog + differential deps  (primary)

**What:** After a cmplog (and optional colorization) pass, assign a Weizz-style tag per input byte: same-`cmp_id` runs → fields; parent/counter nesting → approximate chunks; flags for length-like / implicit fields.

**Where it lands:**
- New module, e.g. `core/weizz_tags.py` (or `core/structure_tags.py`), pure Python first.
- Optional blob on seed metadata (run-length or dense array of `(cmp_id, parent, counter, flags)`).
- Collector hooks next to existing cmplog / colorization paths — do not add a second comparison tracer.

**Reuse:** cmplog operand records, colorization executor loop, corpus lineage metadata.

**Effort:** M. Heuristics in `tags.c` / `get_deps.c` are the non-trivial part; the data model is small.

**Gate:** behind a flag (e.g. `--weizz-tags` / `--structure-tags`) and a size limit analogous to Weizz `-L`, so the differential pass is not paid on every seed.

### P2 — Field- and chunk-scoped operators

**What:** Register operators that:
- mutate only inside a tagged field (interesting values, arithmetic, dict tokens);
- treat contiguous same-`cmp_id` / same-parent spans as chunks and do insert / delete / duplicate / swap / crossover of whole chunks.

**Where:** `core/mutations/` + `REGISTRY.register_mutator()`, structural category, so Elo/bandits can schedule them without special cases.

**Depends on:** P1 (tags must exist on the seed).

**Effort:** L–M once P1 exists; mutation bodies are ordinary buffer edits.

### P3 — Tag-guided length/CRC repair

**What:** Feed tag-flagged length and checksum candidates into the existing FrameShift / `structural_constraints` / checksum-learner path after a length-changing tagged mutation.

**Where:** thin glue after P2 mutations; no new repair engine.

**Effort:** L.

### P4 — Surgical solve restricted by tags

**What:** When solving a comparison, restrict or prioritise the bytes that tags say influence that `cmp_id`.

**Where:** existing adaptive / cond_stmt / SMT paths; filter the candidate byte set.

**Effort:** L. Incremental; only worth it after P1 is stable.

### P5 — Derived-tag inheritance

**What:** Children of a getdeps/tagged parent reuse or cheaply re-derive the parent tag map (Weizz `use_derived_tags`).

**Where:** corpus / lineage metadata.

**Effort:** L.

### Do not port

- Full Weizz / AFL C runtime, QEMU tracer, exact AFL stage ordering.
- Hard-coded format mutators Weizz never had — this tree already exceeds that set.
- Anything that duplicates cmplog, colorization, or the checksum learner under a new name.

---

## 4. Suggested implementation order

1. **Data model** — optional `tags` / `structure_map` on seed metadata; document the packed layout (can mirror Weizz or use a Python-friendly run-length form).
2. **Collector (P1)** — tag assignment from cmplog + differential; unit tests on synthetic inputs with known comparison structure.
3. **Cost control** — size limit, once-per-lineage (or energy budget via cost ledger), flag default off.
4. **Operators (P2)** — field then chunk; register under structural; smoke under `--elo` shadow.
5. **Repair glue (P3)** — post-mutation length/CRC using tag flags + existing repair.
6. **Optional (P4, P5)** — only after measured signal from P1–P3.

Every step behind an A/B flag in the `--no-adaptive-havoc` shape; measure with `tools/bench_paired.py`. No coverage claim without a paired run.

---

## 5. Evaluation targets

Weizz was demonstrated on chunky media inputs (e.g. FFmpeg / AVI). Prefer targets where hard-coded format mutators are weak or absent:

- `ffmpeg_read` / container-like seeds (see existing FINDINGS under `docs/FINDINGS/ffmpeg/`).
- Any proprietary or poorly documented chunk format already in the corpus.
- Synthetic chunk format (nested length-prefixed blocks with intentional comparison sites) for ground-truth tag tests — same role as `tools/gen_synthetic_target.py` for coverage mechanics.

Avoid claiming wins on PNG/JPEG/ZIP alone: those already have dedicated mutators and will not isolate the tag signal.

**Metric:** new edges and unique crashes vs baseline with the same exec budget; report whether tags were populated and how often field/chunk operators were selected under Elo.

---

## 6. Constraints and landmines

- **Do not second-guess cmplog.** The shim and operand parsing are load-bearing; tags must consume their output, not re-intercept comparisons.
- **Differential pass cost.** `get_deps`-style work is expensive. Size gate and once-per-lineage are mandatory before enabling on large seeds (Weizz `-L 8k` class limits).
- **Tags are approximate.** Weizz never claims a sound parse. Operators must tolerate wrong or partial tags (same as AFLSmart under a bad model). Prefer “mutate a span that might be a field” over failing closed when tags disagree.
- **Length-changing chunk ops invalidate outer tags.** After insert/delete, either re-tag, inherit with a dirty bit, or drop tags for that seed until the next collector pass. Inherited stale tags will mis-drive P2/P3.
- **Interaction with format mutators.** Field/chunk operators should not fight PNG/ISO-BMFF mutators on the same seed without scheduler awareness; structural category + Elo is the intended arbitration, not hard exclusion.
- **Re-grep before coding.** `docs/port-backlog.md` and neighbouring learnings under `docs/learnings/` have moved quickly; confirm P1 is still a gap.

---

## 7. Local artifacts (this survey)

| Path | Contents |
|------|----------|
| `weizz-src/weizz-fuzzer-master/` | Extracted Weizz sources (src, include, llvm-tracer, README) |
| `weizz-fuzzer.zip` | Full upstream zip (includes QEMU; prefer the extract) |
| `daedalus-fuzzer.zip` / `fuzzer-master/` | Snapshot of this tree used for the comparison |

These paths are workspace artifacts from the 2026-08-31 analysis session; they are not part of the committed tree unless explicitly added later.

---

## 8. Acceptance checklist (when someone implements)

- [x] Tag collector produces stable tags on a synthetic nested-chunk input with known `cmp_id`s.
- [x] At least one field operator and one chunk operator registered and selected under Elo shadow.
- [x] Flag default off; size/lineage gate documented.
- [ ] Paired bench vs baseline on at least one container-like target; numbers recorded under `docs/` or FINDINGS with the flag name.
- [x] No new comparison tracer; cmplog remains the single source of comparison truth.
- [x] Stale-tag behaviour after length-changing mutations defined and tested.

---

## 9. One-line summary

**Port Weizz’s comparison→tag→field/chunk pipeline on top of existing cmplog/colorization; do not port the AFL/QEMU shell.** Primary deliverable is a per-seed tag map plus structural operators gated by flag and cost, measured with `bench_paired.py`.
