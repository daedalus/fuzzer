# Handover: codec mutator gaps vs Wikipedia "List of codecs" (2026-11-19)

Source: `en.wikipedia.org/wiki/List_of_codecs` (oldid 1383540516, edited 2026-11-03). Survey only, nothing implemented.

TODO: `docs/TODO.md` §Codec mutator gaps.

## Scope and method

- Compared every codec on the page against the structure-aware ops in `core/operator_registry.py` (`*_chunk_mutate`, `g711_mutate`, `nal_chunk_mutate`). **[verified]** by grep at `cb29842`.
- Consumer: `targets/ffmpeg_read.c` (FFmpeg 9.0.1, `avcodec_send_packet` path). Only codecs FFmpeg decodes count; encoder-only, analog, neural, military/radio vocoders are out.
- FFmpeg decoder presence is **[unverified]** — FFmpeg sources are not vendored in this container. Confirm each name in `libavcodec/allcodecs.c` before building.
- Page sections not retrieved (fetch truncated): lossy video §3.4.2–3.4.12 (screen-capture, game, intra-only, CD-ROM), point cloud, mesh. Second pass needed; likely adds Bink/Smacker video, Cinepak, ProRes, DNxHD, Indeo.

## Already covered

| Op | Codecs on the page it reaches |
|---|---|
| `mp3_chunk_mutate` | MP1, MP2, MP3 (Layer I/II/III frame headers) |
| `adts_chunk_mutate` | AAC-LC/HE-AAC in ADTS |
| `flac_chunk_mutate` | FLAC (metadata blocks) |
| `ogg_chunk_mutate` | Vorbis, Opus, Theora, Speex, FLAC-in-Ogg (page framing only) |
| `nal_chunk_mutate` | H.264, HEVC (start codes + NAL header) |
| `av1_rtp_chunk_mutate`, `avif_chunk_mutate` | AV1 (RTP aggregation, AVIF items) |
| `webm_chunk_mutate` | VP8/VP9/Opus/Vorbis via EBML framing |
| `riff_chunk_mutate`, `asf_chunk_mutate`, `isobmff_chunk_mutate`, `flv_chunk_mutate`, `mpegts_chunk_mutate` | containers carrying all of the above, plus WMA, ALAC, MS-ADPCM (container level only) |
| `g711_mutate` | G.711 A-law/µ-law |
| `shorten_chunk_mutate`, `magicyuv_chunk_mutate`, `cfhd_chunk_mutate`, `rasc_chunk_mutate` | Shorten, MagicYUV, CineForm, RASC |
| `jpeg2000_chunk_mutate`, `jpeg_chunk_mutate`, `png_chunk_mutate` | Motion JPEG 2000, MJPEG, CorePNG frames |
| `dvbsub_chunk_mutate`, `pgs_chunk_mutate` | DVB / PGS subtitles |

Common blind spot: container and framing ops stop at the packet boundary. The codec payload inside Ogg/WebM/RIFF/ASF (Vorbis setup header, Opus TOC, VP8/VP9 frame header, AC-3 BSI) only gets flat byte mutation.

## Gaps

Rank = FFmpeg history of decoder bugs × how much structure a flat mutator breaks × whether seeds are cheap. Cost S/M/L.

### Tier 1 — build

| ID | Op | Codecs | Structure to target | Cost |
|---|---|---|---|---|
| C1 | `vorbis_hdr_mutate` | Vorbis | 3 setup headers: codebook dims/lookup type, floor/residue/mapping counts. Inner payload of `ogg_chunk_mutate` pages. | M |
| C2 | `opus_toc_mutate` | Opus (SILK/CELT/hybrid) | TOC byte (config, stereo, code 0–3), frame-count byte, VBR length coding (RFC 6716 §3.2). `OpusHead` channel mapping family. | S |
| C3 | `vpx_frame_mutate` | VP8, VP9 | VP8: 3-byte frame tag (keyframe, partition size), start code `9d 01 2a`, dimensions. VP9: uncompressed header (profile, show_existing, refresh flags, tile cols). IVF framing for raw seeds. | M |
| C4 | `ac3_frame_mutate` | AC-3, E-AC-3 | Sync `0B 77`, fscod/frmsizecod, bsid (≤10 vs 16 = E-AC-3), acmod/lfeon, CRC1/CRC2 fix-up. | S |
| C5 | `dca_frame_mutate` | DTS core, DTS-HD MA (XLL) | Sync `7FFE8001` (+14-bit variants), FSIZE/NBLKS, substream header `64582025`, XLL chunk. | M |
| C6 | `ape_frame_mutate` | Monkey's Audio | `MAC ` descriptor + header (compression level, blocks/frame, seek table length). Seek table vs file size is the classic OOB. | S |
| C7 | `wavpack_blk_mutate` | WavPack | `wvpk` block header (ckSize, block_samples, flags), metadata sub-blocks (id, odd-size bit, large bit). TLV-like; reuse `tlv_mutate` helpers. | S |

### Tier 2 — build after Tier 1 A/B

| ID | Op | Codecs | Note | Cost |
|---|---|---|---|---|
| C8 | `amr_frame_mutate` | AMR-NB, AMR-WB | `#!AMR\n`/`#!AMR-WB\n` magic, per-frame TOC (FT, Q). FT drives frame length. | S |
| C9 | `tta_frame_mutate` | TTA | `TTA1` header, seek table + CRC32 fix-up. | S |
| C10 | `alac_cfg_mutate` | ALAC | 24/36-byte magic cookie (frameLength, kb, mb, pb) inside `isobmff` `alac` atom. | S |
| C11 | `mlp_frame_mutate` | MLP, TrueHD | Major sync `F8726FBA`/`BB`, substream directory, restart headers. | M |
| C12 | `ffv1_hdr_mutate` | FFV1 | Range-coded config record (version, coder type, slice count) + slice footers with CRC. | M |
| C13 | `huffyuv_mutate` | Huffyuv, Ut Video, Lagarith | Huffman tables in extradata; Ut Video plane slice offsets. Shares a code-length table mutator with `huffman_tree_mutate`. | M |
| C14 | `atrac_frame_mutate` | ATRAC1/3/3+/9 | Sound-unit header, BFU counts; ATRAC9 config in RIFF/`at9`. | M |
| C15 | `gsm_frame_mutate` | GSM 06.10, MS GSM | 33-byte frame, magic nibble `0xD`; MS GSM 65-byte pairs. | S |
| C16 | `adpcm_hdr_mutate` | MS/IMA/Xbox/VAG/GCADPCM/ADX/HCA | Block-header predictor/step index out of range; ADX/HCA header fields. One op, per-variant table. | M |

### Tier 3 — record, do not build yet

- **Speech codecs with FFmpeg decoders** (G.723.1, G.728, G.729, iLBC, QCELP, EVRC, TrueSpeech, Nellymoser, Siren/G.722.1, SBC/aptX): fixed-size frames, little structure beyond a mode byte. Flat mutation already reaches most states. Revisit only if per-codec coverage plateaus.
- **Musepack SV7/SV8, TAK, OptimFROG, RealAudio Cook, WMA Pro/Lossless, TwinVQ, QDM2, Bink/Smacker audio, XMA**: real decoders, niche seeds. Cheapest path is a generic "length-prefixed packet header" op, not one op each.
- **VVC, EVC, AVS2/3, VC-1, MPEG-2/4 Part 2, H.261/H.263, Dirac/VC-2**: start-code video. C17 below covers these cheaply.
- **Lossless screen/game video** (ZMBV, TSCC, FPS1, MSRLE, QTRLE, DXA, Lagarith, SheerVideo, ScreenPressor …): zlib/RLE inner payloads. Covered partly by `zlib_chunk_mutate`/`recompress`.
- **Out of scope**: uncompressed (LPCM, v210, YUV), analog RF, neural codecs (Lyra, EnCodec, Satin), military/radio vocoders (MELP, AMBE, IMBE, TETRA), Bluetooth-only codecs without FFmpeg decode (LDAC, LHDC, LC3plus), research drafts (Thor, xvc, OAC).

### C17 — generalize `nal_chunk_mutate` (cheap win)

`nal.py` handles H.264/HEVC headers only **[verified]** (`core/mutations/nal.py:1-12`). Start-code scanning is shared by MPEG-1/2 video (`00 00 01 B3/B5/00`), MPEG-4 Part 2 (`00 00 01 B0/B6`), VC-1 (`00 00 01 0F/0E/0D`), VVC (2-byte NAL header, `nuh_unit_type` 5 bits), AVS (`00 00 01 B0/B3`). Add a per-standard header table; one op, sniffer picks the table. Cost S–M.

## Sketch (C2, representative)

```python
# core/mutations/opus.py
_TOC_CODE_MASK = 0x03          # RFC 6716 §3.1
_MAX_FRAMES = 48               # RFC 6716 §3.2.5

def mutate(data: bytes, max_len: int, rng: RandPool) -> bytes:
    """Mutate one Opus packet's TOC / frame-count / length coding."""
    if not data:
        return data
    op = rng.randint(0, 3)
    buf = bytearray(data)

    # TOC config/stereo/code: flips mode SILK<->CELT<->hybrid
    if op == 0:
        buf[0] ^= 1 << rng.randint(0, 7)
        return bytes(buf)

    # code 3 frame count M: 0, 1, 48, 49, 63 (limit edges)
    ...
```

Source Opus packets come from `ogg_chunk_mutate`'s page parser (`parse_ogg_pages`, `core/mutations/ogg.py:56`) and `webm` SimpleBlocks; the op mutates the packet and re-serializes with CRC fix-up. Same pattern for C1, C3.

## Wiring (each op, Hard Rule 12)

1. `core/mutations/<fmt>.py` — parser + `mutate(data, max_len, rng)`; `RandPool` only (Rule 16).
2. `core/operator_registry.py` — name in the `format` band of `_CATEGORIES`; sniffer in `_AVAILABLE` (e.g. `_sniff_ac3`: `0B 77` within the first 4 KiB, or WAV fmt tag `0x2000`).
3. `services/operators.py` — `_op_<name>` handler on `OperatorEngine`.
4. Seeds: extend the format seed generator (`handover_format_seed_generator_2026-09-21.md`) — without seeds the op fires only via the trickle (same open item as `g711_mutate`, `docs/TODO.md`).
5. Dictionary: `dictionaries/<fmt>.dict` for magics/sync words.
6. Tests: `tests/test_regression_operator_registry.py` placement + gating; per-op falsification + adversarial test (Rule 23); scripted RNG (Rule 39).
7. A/B on FFmpeg target → `docs/AB_LEDGER.md` (Rule 59). `docs/DEEP_DIVE.md` §Format mutators; `docs/architecture.svg` only if a new subsystem.

## Open questions

- Payload-in-container ops (C1–C3) need packet extraction from Ogg/WebM/ISO-BMFF. Reuse each container's parser, or add one shared "packet view" layer under `core/mutations/`? The second avoids N×M coupling (Rule 35).
- Does `ffmpeg_read.c` probe raw `.ac3`/`.dts`/`.ape`/`.wv`/`.amr` without a container? Expected yes via `av_probe_input_format` **[unverified]**.
- Finish the page: re-fetch §3.4.2–3.4.12 before closing this survey.
