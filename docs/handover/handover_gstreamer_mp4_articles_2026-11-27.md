# Handover: GStreamer MP4 fuzzing articles vs fuzzer-tool (2026-11-27)

Two articles on fuzzing GStreamer's MP4 demuxer (`qtdemux`), mapped against this repo.

- A: Atredis, "Crafting targeted MP4 fuzzers" (2026-11-16) — https://atredis.com/blog/2026/11/16/crafting-targeted-mp4-fuzzers/
- B: GitHub Security Lab, "Uncovering GStreamer secrets" (2024-12-17) — https://github.blog/security/vulnerability-research/uncovering-gstreamer-secrets/

Status: analysis only. No code changed.

## A: Atredis

Target: `isomp4` / `qtdemux` (Good plugins; OSS-Fuzz builds GStreamer without them, so it has no MP4 coverage). Engine: libFuzzer.

| Campaign | Harness | Result |
|---|---|---|
| 1. Stock | `gst-discoverer`, `appsrc://` | 5 CVEs; 3 in subtitle parsers it drifted into (unwanted) |
| 2. Pipeline-constrained | `appsrc ! qtdemux ! fakesink` | CVE-2026-46471 (`qtdemux_parse_uuid`, PlayReady) |
| 3. `stsd`-only | mutated `stsd` spliced into a static video/audio template; `appsrc ! qtdemux ! decodebin ! fakesink` | CVE-2026-46469, -46470 (div-by-zero, audio `stsd`), -46472 (global OOB read); fixed 1.28.2 |

Campaign 3 custom mutator:
- Byte 0 picks the track type (`vide`/`soun`).
- Restores the `stsd` fourcc on every mutation.
- Fixes the `stsd` size field 50% of the time.
- Clamps a zero or oversized entry count.
- Ancestor sizes (`moov`→`stbl`) are rewritten at fixed offsets as `orig + diff`.

Limitations the author lists: only `stsd` is mutated, templates are static, only video/audio tracks, demux only.

## B: GitHub Security Lab

Reports 29 CVEs across MP4, Matroska, WAV, AVI, Vorbis, Opus and subtitle parsers. The post gives no fixed versions. Believed fixed in 1.24.10 — unverified, check the GHSL advisories.

MP4 bugs, all in `qtdemux`:

| CVE | Site |
|---|---|
| CVE-2024-47537 | OOB write, `qtdemux.c` |
| CVE-2024-47543 | OOB read, `qtdemux_parse_container` |
| CVE-2024-47544 | NULL deref, `qtdemux_parse_sbgp` |
| CVE-2024-47545 | int underflow, `FOURCC_strf` |
| CVE-2024-47596 | OOB read, `FOURCC_SMI_` |
| CVE-2024-47597 | OOB read, `qtdemux_parse_samples` |
| CVE-2024-47598 | OOB read, `qtdemux_merge_sample_table` |
| CVE-2024-47606 | memcpy overlap → OOB write, `qtdemux_parse_theora_extension` |

Method: a corpus generator, not a mutator. It wrote about 4M small files, each built like this:
1. Random tree shape (child count per node drawn level by level).
2. Containers labelled from `CONTAINER_LIST` (`moov`, `trak`, `udta`, `clip`, `vttc`, …), leaves from `FOURCC_LIST` (`crgn`, `kmat`, `elst`, `load`, …).
3. Payload = fourcc + `min_size` NULs + 4–16 B of `0x41` filler, so byte mutations keep sizes valid.
4. Sizes computed bottom-up.

Generator source: https://github.com/github/securitylab/tree/main/Fuzzing/GStreamer

Not in the post: harness, engine, coverage measurement.

## Mapping to this repo

| Article technique | fuzzer-tool | Gap |
|---|---|---|
| Constrain input to `qtdemux` (A2) | `targets/gstreamer_read.c`: typefind → demuxer, pushed directly | none |
| `stsd`-only mutation + ancestor size fixup (A3) | `mp4_stsd_mutate` + `mp4_path.splice`: walks the real tree, also fixes `stco`/`co64` | none |
| 50% size repair (A3) | 50/50 repair draws in `mp4_stsd`, `mp4_atom` | none |
| Static templates (A3 limitation) | `mp4_atom_transplant` grafts donor atoms between corpus files | none |
| Single atom (A3 limitation) | `mp4_atom_mutate`: ~25 atom types (`stts`, `stsc`, `stsz`, `trun`, `elst`, …) | none |
| Video/audio only (A3 limitation) | `mp4_stsd.Kind.OTHER` is opaque: `tx3g`, `wvtt`, `stpp`, `c608` fields untouched | **yes** |
| `decodebin` after demux (A3) | harness stops at the demuxer | **yes**: codec bugs (Vorbis/Opus/JPEG in B) unreachable |
| Random box trees (B) | `core/prufer.py` + `tlv_reshape` exist, but `parse_tlv` is tag-first with a header-exclusive length; ISOBMFF is size-first, header-inclusive, so it never applies. `isobmff.py` has swap/delete/duplicate, no reparent/nest | **yes** |
| Obscure fourccs (B) | `dictionaries/isomp4.dict`: 718 `qtdemux` fourccs; structural mutators don't use them as labels | **partial** |
| Known-bug seeds | `tools/gen_ffmpeg_gst_seeds.py` has no `qtdemux` `stsd` seeds (its div-by-zero seed is AVI `vprp`) | **yes** |

## TODO (ordered)

1. **Calibration (do first).**
   - Build GStreamer 1.28.1 (A's 3 bugs) and 1.24.9 (B's 8 `qtdemux` bugs): `tools/build_targets.sh --gstreamer-version=<v>`.
   - Fuzz with the MP4 operators and count rediscoveries out of 11.
   - Record the result in `docs/AB_LEDGER.md` (Hard Rule 59), with an A/A control.
   - This decides whether items 2–4 are worth building.
2. **ISOBMFF box-tree reshape.**
   - Add a size-first, header-inclusive layout to `parse_tlv` / `resize_tlv_value` (`core/structural_constraints.py`).
   - That arms `tlv_reshape` for MP4, giving reparent and nest edits through Prüfer trees with no new operator.
   - Seeds: `tools/gen_isobmff_tree_seeds.py` (`prufer` shapes, labels from `isomp4.dict`), output under `~/fuzzing/`.
3. **Field-aware text/subtitle sample entries** in `mp4_stsd` (`tx3g`, `wvtt`, `stpp`, `c608`).
4. **`qtdemux` CVE seeds** in `gen_ffmpeg_gst_seeds.py` (`gs_` prefix), built from the 1.28.2 / 1.24.10 fix diffs.
5. **Decode path.** Optional `decodebin` after the demuxer in `gstreamer_read.c`, so codec parsers are reachable.

Each item needs the TDD, falsification and adversarial tests, and `docs/DEEP_DIVE.md` updates the repo rules require.
