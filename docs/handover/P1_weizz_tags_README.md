# P1 — Tag map from cmplog + differential deps

Primary deliverable from
`docs/handover/handover_weizz_structure_aware_port_2026-08-31.md`.

## What landed

| Path | Role |
|------|------|
| `fuzzer_tool/core/weizz_tags.py` | Data model + collector |
| `tests/test_weizz_tags.py` | Unit tests (10/10 green) |

**No new comparison tracer.** Tags consume existing `CmplogCollector.pairs`
(and optional `_pair_pc`) plus optional colorization taint regions.

## API sketch

```python
from fuzzer_tool.core.weizz_tags import (
    TagCollectorConfig,
    build_tag_map_from_cmplog,
    collect_structure_map,
    attach_tags_to_meta,
    load_tags_from_meta,
)

# After an execution that filled cmplog.pairs:
smap = collect_structure_map(seed_bytes, fuzzer._cmplog, colorization_result=cr)

# Or pure function:
smap = build_tag_map_from_cmplog(
    data,
    pairs,
    pair_pcs=pair_pcs,
    taint_regions=[(start, end), ...],  # inclusive ends from colorization
    config=TagCollectorConfig(max_input_len=8192),
)

for start, end, cmp_id in smap.field_spans():
    ...  # field-scoped mutation (P2)

meta = attach_tags_to_meta(seed.meta, smap)
smap2 = load_tags_from_meta(meta)
```

## Integration checklist (into daedalus/fuzzer)

1. Copy `weizz_tags.py` → `src/fuzzer_tool/core/weizz_tags.py`.
2. CLI: add `--weizz-tags` / `--structure-tags` (default **off**) and
   `--weizz-tags-max-len` (default 8192, Weizz `-L` analogue).
3. Collector hook: after cmplog collection on a seed (and optional
   colorization), if flag set and `len(seed) <= max_len` and
   lineage has not been tagged yet (`once_per_lineage`), call
   `collect_structure_map` and `attach_tags_to_meta`.
4. Cost control: skip when `len(data) > max_len`; honour cost ledger /
   once-per-lineage so differential work is not paid every exec.
5. Stale tags: after any length-changing mutator, either
   `smap.mark_dirty(0, len)` + drop, or clear `weizz_tags_*` from meta
   until the next collector pass.
6. Do **not** register field/chunk operators yet — that is **P2**.

## Acceptance (from handover §8)

- [x] Tag collector produces stable tags on synthetic nested / magic+length input
- [ ] Field + chunk operators (P2 — not this change)
- [ ] Flag default off + size/lineage gate (wire in CLI)
- [ ] Paired bench (after P2)
- [x] No new comparison tracer
- [x] Stale-tag behaviour defined (`TagFlags.DIRTY` + meta drop)

## Design notes

- Shorter operands claim bytes first so field boundaries stay tight.
- `cmp_id` prefers PC when cmplog recorded it; otherwise hashes the pair.
- Parent assignment is a single left-to-right stack pass over field runs.
- Colorization taints only soft-tag **still-untagged** bytes (`IS_IMPL`).
- RLE form is what should live on corpus metadata for resume.
