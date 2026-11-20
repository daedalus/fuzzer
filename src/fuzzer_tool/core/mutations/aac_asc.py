"""AAC AudioSpecificConfig mutator (MP4 ``esds`` DecoderSpecificInfo).

The ASC picks FFmpeg's AAC decoder setup before any frame is read
(ISO/IEC 14496-3 §1.6.2.1):

  audioObjectType 5 [+6 if 31] | samplingFrequencyIndex 4 [+24 if 15]
  | channelConfiguration 4 | (SBR/PS: ext rate index, core object type)
  | GASpecificConfig: frameLengthFlag, dependsOnCoreCoder [+14], extensionFlag
  | rest (PCE, sync extensions) kept verbatim

One field is rewritten (object type edges cross into SBR/PS/ER/escape,
rate index 13-15, channel 0/7-15); fields keep their widths, so the ASC
length is stable, but descriptor lengths are still fixed if it changes.
"""

from __future__ import annotations

from fuzzer_tool.core.mutations.bitrec import Field, FieldParser, Tail, pack
from fuzzer_tool.core.mutations.mp4_path import (
    BOX_HDR,
    BoxHit,
    EsdsLayout,
    find_boxes,
    parse_esds,
    splice,
    write_descr_len,
)
from fuzzer_tool.core.rand_pool import RandPool

AOT_ESCAPE = 31
AOT_ESCAPE_BASE = 32
SFI_EXPLICIT = 15
SBR_AOTS = frozenset({5, 29})
GA_AOTS = frozenset({1, 2, 3, 4, 6, 7, 17, 19, 20, 21, 22, 23})

AOT_EDGES = (0, 1, 2, 4, 5, 22, 23, 29, 30, 31)
FREQ_EDGES = (0, 1, 7350, 96000, 0xFFFFFF)
_EDGES = {
    "audio_object_type": AOT_EDGES,
    "ext_audio_object_type": AOT_EDGES,
    "sampling_frequency": FREQ_EDGES,
    "extension_sampling_frequency": FREQ_EDGES,
}


def _aot(p: FieldParser, name: str, ext_name: str) -> int:
    aot = p.u(name, 5)
    return AOT_ESCAPE_BASE + p.u(ext_name, 6) if aot == AOT_ESCAPE else aot


def _rate(p: FieldParser, name: str, freq_name: str) -> None:
    if p.u(name, 4) == SFI_EXPLICIT:
        p.u(freq_name, 24)


def _asc(p: FieldParser) -> None:
    aot = _aot(p, "audio_object_type", "aot_ext")
    _rate(p, "sampling_frequency_index", "sampling_frequency")
    p.u("channel_configuration", 4)
    if aot in SBR_AOTS:
        _rate(p, "extension_sampling_frequency_index", "extension_sampling_frequency")
        aot = _aot(p, "ext_audio_object_type", "ext_aot_ext")

    if aot not in GA_AOTS:
        return
    p.u("frame_length_flag", 1)
    if p.u("depends_on_core_coder", 1):
        p.u("core_coder_delay", 14)
    p.u("extension_flag", 1)


def parse_asc(asc: bytes) -> tuple[list[Field], str] | None:
    """(fields, rest bits incl. padding) of an ASC, or None if truncated."""
    p = FieldParser(asc)
    try:
        _asc(p)
    except ValueError:
        return None  # ran out of bits: not an ASC we can edit
    return p.fields, p.reader.rest(Tail.NONE)


def _edge(field: Field, rng: RandPool) -> int:
    if field.width == 1:
        return field.value ^ 1
    edges = _EDGES.get(field.name)
    return rng.choice(edges) if edges else rng.randint(0, (1 << field.width) - 1)


def edit_asc(asc: bytes, rng: RandPool) -> bytes | None:
    """Rewrite one ASC field. Draw order: field index, value."""
    parsed = parse_asc(asc)
    if parsed is None:
        return None
    fields, rest = parsed
    field = fields[rng.randint(0, len(fields) - 1)]
    field.value = _edge(field, rng)
    return pack(fields, rest, Tail.NONE)


def find_asc(data: bytes) -> list[tuple[BoxHit, EsdsLayout]]:
    """esds boxes that carry a DecoderSpecificInfo."""
    found = []
    for hit in find_boxes(data, frozenset({b"esds"})):
        layout = parse_esds(data[hit.start + BOX_HDR : hit.start + hit.size])
        if layout is not None and layout.dsi is not None:
            found.append((hit, layout))
    return found


def mutate_aac_asc(data: bytes, max_len: int = 65536, *, rng: RandPool) -> bytes | None:
    """Rewrite one ASC field in an MP4 esds; None if declined.

    Draw order: esds index, field index, value.
    """
    found = find_asc(data)
    if not found:
        return None
    hit, layout = found[rng.randint(0, len(found) - 1)]
    dsi = layout.dsi
    payload = data[hit.start + BOX_HDR : hit.start + hit.size]

    asc = edit_asc(payload[dsi.payload_off : dsi.end], rng)
    if asc is None:
        return None
    new = bytearray(payload[: dsi.payload_off] + asc + payload[dsi.end :])

    # Enclosing descriptor lengths sit before the ASC: offsets are stable.
    delta = len(asc) - dsi.length
    for d in (layout.es, layout.dc, dsi) if delta else ():
        if not write_descr_len(new, d.len_off, d.len_width, d.length + delta):
            return None

    out = splice(data, hit, bytes(new))
    return None if out is None else out[:max_len]
