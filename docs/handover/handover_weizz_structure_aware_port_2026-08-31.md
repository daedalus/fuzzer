# Handover — Weizz structure-aware port (unknown chunk formats)

Original: 2026-08-31. Pruned 2026-09-26 to open items; full original:
`git show 1c689e8a^:docs/handover/handover_weizz_structure_aware_port_2026-08-31.md`.
Verified against `4c021daa`.

P1–P5 are live (`core/weizz_tags.py`, `core/mutations/weizz_structural.py`,
`services/operators.py:_op_weizz_*` / `_weizz_restricted_find`,
`services/corpus_manager.py` → `inherit_tags_from_parent`,
`services/fuzzer.py:_maybe_collect_weizz_tags`), gated by `--weizz-tags` /
`--weizz-tags-max-len`.

---

## Open: paired bench vs baseline (pending §E1)

Last unticked item of the acceptance checklist. No coverage claim exists.

- **Do:** add a `weizz-tags` arm (`["--weizz-tags"]`, cmplog on) to `ARMS` in
  `tools/lib/bench_paired.py`; run `tools/benchmark.py paired` against
  baseline on a container-like target (`ffmpeg_read`, or a synthetic nested
  length-prefixed chunk format). Not PNG/JPEG/ZIP alone — dedicated mutators
  mask the tag signal.
- **Metric:** new edges and unique crashes at equal exec budget; also report
  fraction of seeds with a non-dirty tag map and selection share of
  `weizz_*` ops under Elo.
- **Accept:** numbers recorded under `docs/sweeps/` with the flag name.
