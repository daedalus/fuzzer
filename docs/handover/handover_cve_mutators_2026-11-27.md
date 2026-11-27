# Handover: CVE-derived mutators for the FFmpeg and GStreamer targets (2026-11-27)

Follow-up to `hevc_ps_mutate` (same day). Task: "add mutators based on CVEs in both
ffmpeg and gstreamer". Three operators shipped, one per trigger shape no existing
op could express. Status: code + tests + docs done; A/B owed (`docs/AB_PENDING.md`
§Operators).

## 1. CVE selection

Candidates from live CVE feeds (NVD/OpenCVE/GStreamer Security Center/VulnCheck/
AISLE), filtered by two rules: (a) the fuzzer's targets actually reach the buggy
code (`ffmpeg_canary_read.c` runs demux→decode on one in-memory file;
`gstreamer_read.c` typefinds → demuxer → parser from the calling thread), and
(b) no existing operator already expresses the trigger (else it is a calibration
gap, not a mutator gap).

| CVE | Target | Site | Trigger shape | Shipped op |
|---|---|---|---|---|
| CVE-2026-125239 | FFmpeg ≤ 9.0.2 | `ff_dxva2_hevc_fill_picture_parameters()` (`dxva2_hevc.c`) | many non-uniformly spaced tiles in the HEVC PPS write 16-bit values past fixed DXVA arrays | `hevc_tile_mutate` |
| CVE-2026-39043 (SA-2026-0022) | GStreamer < 1.28.2 | matroskademux bz2 track | decompressed-buffer size calc, missing parentheses | `mkv_codec_cfg_mutate` (CompAlgo → 1) |
| CVE-2026-17072 (SA-2026-0073) | GStreamer < 1.28.6 | matroskademux FLAC stream header parser | metadata block whose length lands the block exactly at the CodecPrivate boundary (next header reads 4 past), or loop entry with <4 bytes left | `mkv_codec_cfg_mutate` (FLAC mode) |
| GStreamer-SA-2026-0089 | GStreamer < 1.28.6 | matroskademux Xiph codec data parser | integer overflow + OOB read on A_VORBIS count/lacing bytes | `mkv_codec_cfg_mutate` (Xiph mode) |
| CVE-2026-64830 | FFmpeg 2.1–8.1.2 | vobsub demuxer (`libavformat/mpeg.c`), fix dbd495f066a8 | more distinct `id:` tags than the fixed-size `vobsub->q[]` array; writes past it via `ff_subtitles_queue_insert` | `vobsub_idx_mutate` |

Rejected as already covered or out of target reach:

- **CVE-2026-73472** (avidemux superindex/subindex integer overflow, SA-2026-0072,
  fixed 1.28.6): `avi_field_mutate`'s INDX table already pokes `wLongsPerEntry`
  (0/1/0x4000/0x4001), `bIndexType`, `nEntriesInUse`, `base_offset` in `indx`/`ix##`
  chunks — field-level trigger reachable; calibration gap only.
- **GHSL qtdemux set** (CVE-2024-47537..47606, 2024) and **Atredis set**
  (CVE-2026-46469..46472, fixed 1.28.2): `mp4_stsd_mutate`/`mp4_atom_mutate`/
  `mp4_atom_transplant`/`mp4_box_grow` cover the atom structures
  (`handover_gstreamer_mp4_articles_2026-11-27.md` §Mapping; its TODO item 1 =
  calibration). Open piece there: `uuid`/PlayReady graft (CVE-2026-46471) — same
  transplant pattern would work; left as TODO (c).
- **CVE-2026-125262** (LCEVC enhancement data in H.264/HEVC, `lcevcdec.c`):
  trigger is SEI payloads with conflicting LCEVC global configs; needs an
  SEI-body mutator (TODO (c)).
- **CVE-2026-135706** (modplug_read_header, `max_size` 0) and **CVE-2026-135722**
  (wavenc peak writer): muxer-side or allocation-shape triggers, not a
  structure-aware mutator gap.
- GStreamer SA-2026-0094..0101 (H.265 bitstream parser uninit, RTP payloaders,
  DVB descriptors, ASF muxer, VMnc decoder): decode/RTP paths behind external
  libs or muxers — not in either target's demux/parsing layer.

## 2. Operators

### `hevc_tile_mutate` (`core/mutations/hevc_tiles.py`, FFmpeg)

`hevc_ps_mutate` keeps the PPS tiles section verbatim (its `_pps` returns at
`tiles_enabled_flag`); this op parses the whole grid and rewrites one thing:

- **count**: `num_tile_columns_minus1`/`num_tile_rows_minus1` to 0/1/2, spec
  bounds (18/19/21), 255, 2^32-2. An explicit grid is forced uniform on a count
  edit (its width list is for the *old* count; 2^32-2 entries cannot pack).
- **spacing**: uniform↔explicit flip; the rebuilt lists draw width edges
  (0 = 1 luma sample = max tiles per picture, 1, 65, 255, 2^32-2).
- **width**: one `column_width_minus1_i`/`row_height_minus1_i` edge (sums that
  cannot match `pic_width_in_luma_samples`).
- **lf**: `loop_filter_across_tiles_enabled_flag` flip.

Grids ≥ 64 columns/rows decline (spec bound is 20; the width list is that many
codewords). PPS prefix/tail parsing shared with `hevc_ps` (`_pps_prefix`/
`_pps_tail` split there in this change — one source of truth). Decoder-reach
test: FFmpeg logs `num_tile_columns_minus1 out of range` on the 2^32-2 edit.
Note: libx265 4.1 (Debian) dropped tile encoding (`Unknown option: tile-columns`),
so the ffmpeg-integration fixture splices a hand-built grid PPS into a real
libx265 stream.

### `mkv_codec_cfg_mutate` (`core/mutations/mkv_codec_cfg.py`, GStreamer + FFmpeg)

Uses `webm.py`'s EBML parse/serialize; the edits are in-place leaf rewrites or a
ContentEncodings subtree spliced into the TrackEntry (parent size vints recomputed
bottom-up by `serialize_webm`):

- **flac** (CodecID `A_FLAC`/`A/FLAC`, CodecPrivate starts `fLaC`): rewrite one
  metadata block length to end-exact at the CodecPrivate boundary, leave a 1–3
  byte tail, or overshoot by 16.
- **xiph** (CodecID `A_VORBIS`/`A/VORBIS`): count byte → 0/1/3/255; one lacing
  length byte → 0/1/0x7F/255.
- **comp**: rewrite `ContentCompAlgo` (0x4254, 1-byte payload) to 1 (bz2, the
  CVE-2026-39043 trigger)/2/3/0, or inject a minimal ContentEncodings subtree
  (ContentEncoding→ContentEncodingType 0→ContentCompression→ContentCompAlgo)
  into any track that has none.

FFmpeg's Matroska demuxer parses the same leaves, so both targets see the edits.

### `vobsub_idx_mutate` (`core/mutations/vobsub_idx.py`, FFmpeg)

The `.idx` text grammar is kept (`id: xx, index: n`, `timestamp: h:m:s:ms,
filepos: %09x`): inject 1..8 fresh `id:` lines per call (compounding: 40
distinct ids > the 32-slot `vobsub->q[]`), re-tag one id, filepos edges
(0/1/0xFFFF/0x7FFFFFFF/2^32-1), timestamp edges (0, max, out-of-range).

## 3. Wiring (Hard Rule 12)

Registry format band + sniffers with the CVE ids in comments
(`operator_registry.py`), handlers in `services/operators.py`, no mypy
exemption (all three modules pass strict; `hevc_ps` was *removed* from the
exemption list in this change — the previous commit had grown it to 483 > the
482 ceiling and `test_exemption_list_does_not_grow` failed). Tests
`tests/test_{hevc_tiles,mkv_codec_cfg,vobsub_idx}_mutator.py`; battery entries
in the no-op sweep. Docs: DEEP_DIVE §format ops, CHANGELOG, TODO
**CVE-derived mutators** item, AB_PENDING §Operators.

## 4. Also fixed in this pass

- mypy ratchet 483 → 482 (see above); `hevc_ps._edge` now draws edge *indices*
  instead of `rng.choice` so it type-checks strictly; the four scripted-RNG call
  sites in `tests/test_hevc_ps_mutator.py` follow.
- No-op sweep reachability: `inet_forms` (needed a dotted-quad battery entry;
  it sat outside the sweep on the 2% trickle) and `rasc_chunk_mutate` (needed an
  entry with `RASC` at bytes 8:12 — its gate offset — after the battery grew
  and the trickle stopped covering it).

## 5. Open

1. **A/B** (`AB_PENDING.md`): each op masked out of the tree vs with, A/A first;
   FFmpeg target for hevc_tile/vobsub_idx, both for mkv_codec_cfg.
2. **Seeds**: no VobSub `.idx`, no FLAC/Vorbis-in-MKV seeds in either corpus —
   ops fire via the 2% trickle until `gen_ffmpeg_gst_seeds.py` grows them
   (vobsub `.idx` is trivial text; FLAC/Vorbis mkv via `ffmpeg -c:a flac/libvorbis`).
   A tile-PPS HEVC seed needs the hand-built splice (libx265 4.1 emits no tiles).
3. **Rediscovery calibration** (handover_gstreamer_mp4_articles item 1): build
   GStreamer 1.24.9/1.28.1 and count rediscoveries out of the GHSL/Atredis
   sets — decides whether the qtdemux `uuid`/PlayReady graft op (CVE-2026-46471)
   is worth building.
4. **More CVE families** (TODO item (c)): LCEVC SEI (CVE-2026-125262),
   modplug (CVE-2026-135706, needs a header-field story, not `max_size`), the
   H.265 bitstream parser uninit (SA-2026-0094) once the target links real
   decoders.
