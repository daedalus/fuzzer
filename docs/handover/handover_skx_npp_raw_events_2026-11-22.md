# Handover: raw PMU events from the Skylake-X next-page-prefetcher study (2026-11-22)

Source: devsh0, "Reversing the Next-Page Prefetcher in Skylake-X" (2026-11-17).

## Findings used

- SKX has a next-page prefetcher that prefills the dTLB. Trigger: two ordered loads, cache line 52 then 53 of a 4K page. One page ahead only. Forward only; stores and backward walks do not trigger it.
- Counted with `DTLB_LOAD_MISSES.STLB_HIT` (0x2008), `MEM_INST_RETIRED.STLB_MISS_LOADS` (0x11D0), `MEM_INST_RETIRED.STLB_MISS_STORES` (0x12D0).

## Implemented

- `perf_event.PERF_TYPE_RAW`, `resolve_counter`, `parse_raw_name` (`rXXXX`), `cpu_signature`, `counter_supported`, `RAW_PRESETS`.
- `--hw-perf-counters` accepts `rXXXX` and the three `skx_*` presets; presets are rejected by the CLI and skipped by `PerfCounters` off Intel family 6 model 0x55.
- Group read layouts built on demand (`_group_layout`), so more raw counters than `COUNTER_DEFS` entries work.
- Stats: `stlb: Nh Nm` appended to the cache field when the presets are open.

## Not done / caveats

- Never run on real SKX hardware; the sandbox has no PMU. Verify with `--hw-perf-counters skx_dtlb_load_stlb_hit,skx_stlb_miss_loads` on a Skylake-X host.
- Raw configs are not validated against the CPU's event list; a wrong `rXXXX` just counts something else.
- Events are counted per target process with `inherit`, so a high-ratio hit/miss signal is only meaningful for page-sequential workloads (THP off).
- Not wired into scheduling or feedback: microarchitectural counters are noisy outside synthetic loops.
- Next-step candidates: forward scan order in `native_scan.c` only if a profile shows STLB misses there.
