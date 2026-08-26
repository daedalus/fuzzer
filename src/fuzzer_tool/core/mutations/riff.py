"""Structure-aware RIFF container mutator (AVI, WAV).

RIFF is a flat, top-level chunk grammar:

  "RIFF" + size(4 LE) + form_type(4, e.g. "WAVE", "AVI ") + chunks...

Each chunk is:

  fourcc(4) + size(4 LE) + data(size bytes, padded to even length)

AVI nests further via "LIST" chunks (a chunk whose data begins with a
4-byte list_type, e.g. "hdrl"/"strl"/"movi", followed by more chunks),
but this mutator operates at the *top-level* chunk-framing layer only —
the layer libavformat's `riffdec.c`/`avidec.c`/`wavdec.c` walk first and
where a corrupted or overflowing chunk `size` field causes the classic
desync/OOB-read bugs. It does not recurse into LIST bodies structurally;
a LIST chunk's body (including its list_type and any nested chunks) is
carried as one opaque blob, same as any other chunk's data. This keeps
the round-trip exact and the mutation surface honest, at the cost of not
independently corrupting nested strl/strh sub-chunks — the corpus's own
copies of those still travel through unmodified inside the LIST body,
and top-level `duplicate_chunk`/`mutate_chunk_size` on a LIST chunk still
exercises the recursive-descent entry into it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

RIFF_HEADER_LEN = 12  # "RIFF" + size(4) + form_type(4)
KNOWN_FOURCCS = [
    b"fmt ", b"data", b"fact", b"LIST", b"JUNK", b"idx1",
    b"hdrl", b"strl", b"strh", b"strf", b"strd", b"strn",
    b"movi", b"avih", b"vprp", b"INFO", b"PAD ",
]


@dataclass
class RiffChunk:
    fourcc: bytes
    declared_size: int  # size field as stored/mutated; may not equal len(data)
    data: bytes  # actual payload bytes present

    def to_bytes(self) -> bytes:
        out = self.fourcc + self.declared_size.to_bytes(4, "little") + self.data
        if len(self.data) % 2 == 1:
            out += b"\x00"
        return out


def parse_riff_chunks(data: bytes) -> tuple[bytes, list[RiffChunk]] | None:
    """Parse top-level RIFF chunk framing. Returns (form_type, chunks) or
    None. Declines RIFF/WEBP so it doesn't overlap webp_chunk_mutate."""
    if len(data) < RIFF_HEADER_LEN or data[:4] != b"RIFF":
        return None
    form_type = data[8:12]
    if form_type == b"WEBP":
        return None

    pos = RIFF_HEADER_LEN
    n = len(data)
    chunks: list[RiffChunk] = []
    while pos + 8 <= n:
        fourcc = data[pos : pos + 4]
        size = int.from_bytes(data[pos + 4 : pos + 8], "little")
        body_start = pos + 8
        body_end = min(body_start + size, n)
        body = data[body_start:body_end]
        chunks.append(RiffChunk(fourcc, size, body))
        consumed = body_end - body_start
        pos = body_end + (consumed % 2)
        if consumed <= 0 and size != 0:
            break  # malformed/truncated; stop rather than loop

    return (form_type, chunks) if chunks else None


def serialize_riff(form_type: bytes, chunks: list[RiffChunk]) -> bytes:
    body = form_type + b"".join(c.to_bytes() for c in chunks)
    return b"RIFF" + len(body).to_bytes(4, "little") + body


class RiffMutator:
    """Structure-aware RIFF top-level chunk mutator."""

    def mutate(self, data: bytes, max_len: int = 65536, rng=None) -> bytes:
        rng = rng or random
        parsed = parse_riff_chunks(data)
        if not parsed:
            return self._generate_random_riff(max_len=max_len, rng=rng)
        form_type, chunks = parsed

        op = rng.randint(0, 7)
        if op == 7:
            # Outer RIFF size field, independent of any chunk edit — the
            # field libavformat trusts to bound the whole read.
            out = bytearray(serialize_riff(form_type, chunks))
            bogus = rng.choice([0, 0xFFFFFFFF, len(out) // 2, rng.randint(0, 0xFFFFFFFF)])
            out[4:8] = bogus.to_bytes(4, "little")
            return bytes(out)[:max_len]

        mutators = [
            self._mutate_chunk_size,
            self._mutate_fourcc,
            self._mutate_list_type,
            self._duplicate_chunk,
            self._delete_chunk,
            self._reorder_chunks,
            self._truncate_chunk_data,
        ]
        mutators[op](chunks, rng)
        return serialize_riff(form_type, chunks)[:max_len]

    def _mutate_chunk_size(self, chunks: list[RiffChunk], rng) -> None:
        """Corrupt a declared chunk size independent of its real data length
        — the classic RIFF-parser desync/OOB-read trigger."""
        target = rng.choice(chunks)
        target.declared_size = rng.choice(
            [0, 0xFFFFFFFF, len(target.data) + 1, max(0, len(target.data) - 1), rng.randint(0, 0xFFFFFFFF)]
        )

    def _mutate_fourcc(self, chunks: list[RiffChunk], rng) -> None:
        """Relabel a chunk's ID — tests type-dispatch confusion (e.g. a
        "data" chunk masquerading as "fmt ")."""
        target = rng.choice(chunks)
        target.fourcc = rng.choice(KNOWN_FOURCCS + [bytes(rng.randint(0, 255) for _ in range(4))])

    def _mutate_list_type(self, chunks: list[RiffChunk], rng) -> None:
        """If a chunk is LIST-typed, corrupt the embedded list_type (first 4
        bytes of its data) without touching the nested chunks after it."""
        candidates = [c for c in chunks if c.fourcc == b"LIST" and len(c.data) >= 4]
        if not candidates:
            self._mutate_fourcc(chunks, rng)
            return
        target = rng.choice(candidates)
        new_type = rng.choice([b"hdrl", b"strl", b"movi", bytes(rng.randint(0, 255) for _ in range(4))])
        target.data = new_type + target.data[4:]

    def _duplicate_chunk(self, chunks: list[RiffChunk], rng) -> None:
        idx = rng.randint(0, len(chunks) - 1)
        orig = chunks[idx]
        chunks.insert(idx + 1, RiffChunk(orig.fourcc, orig.declared_size, orig.data))

    def _delete_chunk(self, chunks: list[RiffChunk], rng) -> None:
        if len(chunks) > 1:
            chunks.pop(rng.randint(0, len(chunks) - 1))

    def _reorder_chunks(self, chunks: list[RiffChunk], rng) -> None:
        if len(chunks) >= 2:
            i, j = rng.sample(range(len(chunks)), 2)
            chunks[i], chunks[j] = chunks[j], chunks[i]

    def _truncate_chunk_data(self, chunks: list[RiffChunk], rng) -> None:
        """Shrink a chunk's real payload while leaving declared_size as-is
        — inverse of _mutate_chunk_size's grow case."""
        target = rng.choice(chunks)
        if target.data:
            cut = rng.randint(0, len(target.data) - 1)
            target.data = target.data[:cut]

    def _generate_random_riff(self, max_len: int = 65536, rng=None) -> bytes:
        rng = rng or random
        # Minimal PCM WAVE: "fmt " (16-byte PCM format struct) + "data".
        fmt_data = (
            (1).to_bytes(2, "little")  # wFormatTag = PCM
            + (2).to_bytes(2, "little")  # nChannels
            + (44100).to_bytes(4, "little")  # nSamplesPerSec
            + (176400).to_bytes(4, "little")  # nAvgBytesPerSec
            + (4).to_bytes(2, "little")  # nBlockAlign
            + (16).to_bytes(2, "little")  # wBitsPerSample
        )
        pcm = bytes(rng.randint(0, 255) for _ in range(128))
        chunks = [
            RiffChunk(b"fmt ", len(fmt_data), fmt_data),
            RiffChunk(b"data", len(pcm), pcm),
        ]
        return serialize_riff(b"WAVE", chunks)[:max_len]
