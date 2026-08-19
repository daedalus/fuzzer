#!/usr/bin/env python3
"""Generate a JPEG corpus for fuzzing libjpeg-turbo.

Creates JPEGs with structurally varied COM/APPn segments, including large
arbitrary payloads in segments libjpeg does not parse for metadata.
Those bytes are the intended "coverage-dead" candidates for the item 4
real-corpus sensitivity sweep.

Usage:
    python tools/corpus_jpeg.py [--out DIR] [--count N]
"""

from __future__ import annotations

import argparse
import io
import os
import random
import struct
import sys
from pathlib import Path

SEED = 42


def _pillow_jpeg(width: int, height: int, color: str = "L") -> bytes:
    from PIL import Image

    img = Image.new(color, (width, height), color=128)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _find_marker(jpeg: bytes, marker: bytes) -> int:
    """Find the first occurrence of a 2-byte marker in jpeg data."""
    for i in range(len(jpeg) - 1):
        if jpeg[i : i + 2] == marker:
            return i
    return -1


def _inject_segment_after_soi(jpeg: bytes, marker: int, payload: bytes) -> bytes:
    """Inject a marker segment immediately after SOI."""
    if len(payload) > 65533:
        raise ValueError("segment payload must fit in 16-bit length")
    segment = bytes([marker >> 8, marker & 0xFF]) + struct.pack(">H", len(payload) + 2) + payload
    # SOI is always at offset 0 in a valid JPEG
    return jpeg[:2] + segment + jpeg[2:]


def _inject_com_segment(jpeg: bytes, payload: bytes) -> bytes:
    """Inject a COM (comment) segment immediately after SOI."""
    return _inject_segment_after_soi(jpeg, 0xFFFE, payload)


def _append_segment_before_eoi(jpeg: bytes, marker: int, payload: bytes) -> bytes:
    """Append a marker segment before the EOI marker."""
    if len(payload) > 65533:
        raise ValueError("segment payload must fit in 16-bit length")
    segment = bytes([marker >> 8, marker & 0xFF]) + struct.pack(">H", len(payload) + 2) + payload
    eoi = b"\xff\xd9"
    pos = _find_marker(jpeg, eoi)
    if pos == -1:
        raise ValueError("no EOI marker found")
    return jpeg[:pos] + segment + jpeg[pos:]


def _make_seed_variants() -> list[tuple[str, bytes]]:
    rng = random.Random(SEED)
    seeds: list[tuple[str, bytes]] = []

    # Baseline JPEGs at sizes large enough for multi-region profiling.
    # profile_buffer uses window=4096 and requires >=512-byte regions,
    # so we need seeds >4096 bytes to get more than one region.
    for w, h, color in [
        (64, 64, "L"),
        (128, 128, "L"),
        (128, 128, "RGB"),
        (256, 256, "L"),
        (256, 256, "RGB"),
        (512, 512, "L"),
        (1024, 1024, "L"),
    ]:
        name = f"baseline_{color.lower()}_{w}x{h}.jpg"
        seeds.append((name, _pillow_jpeg(w, h, color)))

    # COM segments of increasing size, injected after SOI.
    # COM is the simplest skippable segment; libjpeg just discards it.
    for size in [64, 256, 1024, 4096, 16384, 65533]:
        payload = bytes(rng.randint(0, 255) for _ in range(size))
        jpeg = _pillow_jpeg(128, 128, "L")
        jpeg = _inject_com_segment(jpeg, payload)
        seeds.append((f"com_after_soi_{size}.jpg", jpeg))

    # APP2-APP15 segments with large payloads after SOI.
    # Unknown APPn segments are skipped by length alone.
    for marker in range(0xFFE2, 0xFFEF):
        payload = bytes(rng.randint(0, 255) for _ in range(4096))
        jpeg = _pillow_jpeg(128, 128, "L")
        jpeg = _inject_segment_after_soi(jpeg, marker, payload)
        name = f"app{marker & 0xF}_after_soi_4096.jpg"
        seeds.append((name, jpeg))

    # APP2 segment appended before EOI.
    for size in [4096, 16384, 65533]:
        payload = bytes(rng.randint(0, 255) for _ in range(size))
        jpeg = _pillow_jpeg(128, 128, "L")
        jpeg = _append_segment_before_eoi(jpeg, 0xFFE2, payload)
        seeds.append((f"app2_before_eoi_{size}.jpg", jpeg))

    # Repo source embedded as COM payload (mirrors the png corpus technique).
    try:
        src = (
            Path(__file__).resolve().parent.parent
            / "src"
            / "fuzzer_tool"
            / "core"
            / "live_bit_mask.py"
        )
        src_bytes = src.read_bytes()
    except Exception:
        src_bytes = b"#!/usr/bin/env python3\n# live_bit_mask.py\n" * 512

    # Split source into two COM segments so the payload spans multiple regions.
    chunk_size = min(len(src_bytes), 65533 - 2)
    chunk = src_bytes[:chunk_size]
    jpeg = _pillow_jpeg(128, 128, "L")
    jpeg = _inject_com_segment(jpeg, chunk)
    seeds.append(("src_embedded_com.jpg", jpeg))

    return seeds


def _validate_jpeg(data: bytes) -> bool:
    """Quick structural validation: SOI, EOI, and at least one scan."""
    if len(data) < 4:
        return False
    if data[:2] != b"\xff\xd8":
        return False
    if data[-2:] != b"\xff\xd9":
        return False
    # Must contain an SOS marker
    return b"\xff\xda" in data


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate JPEG corpus for fuzzing")
    parser.add_argument("--out", default="corpus_jpeg", help="Corpus directory")
    parser.add_argument("--count", type=int, default=0, help="Max seeds (0=all)")
    args = parser.parse_args()

    seeds_dir = os.path.join(args.out, "seeds")
    os.makedirs(seeds_dir, exist_ok=True)

    raw_seeds = _make_seed_variants()
    seeds = [(name, data) for name, data in raw_seeds if _validate_jpeg(data)]

    if args.count > 0:
        seeds = seeds[: args.count]

    for name, data in seeds:
        path = os.path.join(seeds_dir, name)
        with open(path, "wb") as f:
            f.write(data)

    print(f"[*] Generated {len(seeds)}/{len(raw_seeds)} valid seed JPEGs in {seeds_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
