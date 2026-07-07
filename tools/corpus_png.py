#!/usr/bin/env python3
"""Generate a diverse PNG corpus for fuzzing libpng.

Creates minimal PNGs with varied properties (dimensions, color types,
bit depths, compression levels, filter types, ancillary chunks) plus
zlib-aware seeds for decompression testing and downloads real-world
PNGs from public sources when network is available.

Usage:
    python tools/corpus_png.py [--out DIR] [--count N] [--download]

Output is a directory of .png files suitable as fuzzer seed corpus.
"""

import argparse
import os
import random
import struct
import sys
import urllib.request
import zlib


def make_chunk(chunk_type: bytes, data: bytes) -> bytes:
    chunk = chunk_type + data
    crc = struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)
    return struct.pack(">I", len(data)) + chunk + crc


def make_png(
    width: int,
    height: int,
    color_type: int,
    bit_depth: int,
    pixel_func=None,
    compression: int = 9,
    filter_type: int = 0,
    interlace: int = 0,
    extra_chunks: list[tuple[bytes, bytes]] | None = None,
) -> bytes:
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = make_chunk(
        b"IHDR",
        struct.pack(">IIBBBBB", width, height, bit_depth, color_type, 0, 0, interlace),
    )

    if pixel_func is None:
        channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color_type]

        def pixel_func(w, y, c=channels):
            return bytes([0x80] * (w * c))

    raw = b""
    for y in range(height):
        raw += bytes([filter_type & 0x07])
        raw += pixel_func(width, y)

    compressed = zlib.compress(raw, compression)
    idat = make_chunk(b"IDAT", compressed)
    iend = make_chunk(b"IEND", b"")

    parts = [sig, ihdr]
    if extra_chunks:
        for ct, cd in extra_chunks:
            parts.append(make_chunk(ct, cd))
    parts.extend([idat, iend])
    return b"".join(parts)


def make_variants() -> list[tuple[str, bytes]]:
    """Generate a set of structurally diverse PNGs."""
    seeds = []

    # Color type × bit depth matrix
    for ct_name, ct, bd in [
        ("gray", 0, 8), ("gray", 0, 16), ("gray", 0, 1),
        ("rgb", 2, 8), ("rgb", 2, 16),
        ("palette", 3, 1), ("palette", 3, 2), ("palette", 3, 4), ("palette", 3, 8),
        ("gray_alpha", 4, 8),
        ("rgba", 6, 8), ("rgba", 6, 16),
    ]:
        for w, h in [(1, 1), (8, 8), (64, 64)]:
            name = f"{ct_name}_{bd}bit_{w}x{h}.png"
            seeds.append((name, make_png(w, h, ct, bd)))

    # Different compression levels
    for level in [0, 1, 5, 9]:
        name = f"compress_level{level}_32x32.png"
        seeds.append((name, make_png(32, 32, 2, 8, compression=level)))

    # Different filter types
    for ft in range(5):
        name = f"filter_{ft}_32x32.png"
        seeds.append((name, make_png(32, 32, 2, 8, filter_type=ft)))

    # Stress sizes
    for w, h in [(256, 256), (512, 512), (1024, 1)]:
        name = f"stress_{w}x{h}_rgb.png"
        seeds.append((name, make_png(w, h, 2, 8)))

    # Noise patterns
    rng = random.Random(42)
    for w, h in [(64, 64), (128, 128)]:
        def noise(w, y, _rng=rng):
            return bytes(_rng.randint(0, 255) for _ in range(w * 3))
        seeds.append((f"noise_{w}x{h}.png", make_png(w, h, 2, 8, pixel_func=noise)))

    # Gradient
    def gradient(w, y):
        buf = bytearray()
        for x in range(w):
            buf += bytes([min(255, y), min(255, x * 4), max(0, 255 - x * 4)])
        return bytes(buf)
    seeds.append(("gradient_256x256.png", make_png(256, 256, 2, 8, pixel_func=gradient)))

    # Stripes
    def stripes(w, y):
        v = 255 if (y // 4) % 2 == 0 else 0
        return bytes([v, 0, 0]) * w
    seeds.append(("stripes_128x128.png", make_png(128, 128, 2, 8, pixel_func=stripes)))

    return seeds


def make_zlib_variants() -> list[tuple[str, bytes]]:
    """Generate zlib-aware seeds for decompression path coverage."""
    seeds = []
    rng = random.Random(42)

    # Raw zlib streams at various compression levels and sizes
    for level in range(0, 10):
        for size in [64, 256, 1024, 4096]:
            raw = bytes(rng.randint(0, 255) for _ in range(size))
            compressed = zlib.compress(raw, level)
            seeds.append((f"zlib_L{level}_s{size}.bin", compressed))

    # PNGs with complex (random) pixel data
    for w, h in [(1, 1), (8, 8), (32, 32), (64, 64), (128, 128)]:
        for ct, bd in [(0, 8), (2, 8), (3, 8), (4, 8), (6, 8)]:
            channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[ct]
            raw = b""
            for y in range(h):
                raw += bytes([0])  # filter None
                raw += bytes(rng.randint(0, 255) for _ in range(w * channels))
            # Bind raw data into a closure for the pixel_func
            pixel_data = raw
            def _make_pf(pd):
                def pf(w, y):
                    row_size = w * {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[ct]
                    start = y * (row_size + 1) + 1  # skip filter byte
                    return pd[start:start + row_size]
                return pf
            seeds.append((f"complex_ct{ct}_bd{bd}_{w}x{h}.png",
                          make_png(w, h, ct, bd, pixel_func=_make_pf(pixel_data))))

    # PNGs with multiple IDAT chunks (split compressed data)
    for w, h in [(16, 16), (32, 32), (64, 64)]:
        raw = bytes(rng.randint(0, 255) for _ in range(w * h * 3))
        compressed = zlib.compress(raw, 6)
        sig = b"\x89PNG\r\n\x1a\n"
        ihdr = make_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        mid = len(compressed) // 2
        idat1 = make_chunk(b"IDAT", compressed[:mid])
        idat2 = make_chunk(b"IDAT", compressed[mid:])
        iend = make_chunk(b"IEND", b"")
        seeds.append((f"split_idat_{w}x{h}.png", sig + ihdr + idat1 + idat2 + iend))

    # PNGs with ancillary chunks
    for chunk_name, chunk_data in [
        (b"tEXt", b"Key=Value"),
        (b"gAMA", struct.pack(">I", 45455)),
        (b"pHYs", struct.pack(">IIb", 3780, 3780, 1)),
        (b"tIME", struct.pack(">HBBBBB", 2024, 1, 1, 0, 0, 0)),
    ]:
        seeds.append((f"chunk_{chunk_name.decode()}.png",
                      make_png(8, 8, 2, 8,
                               pixel_func=lambda w, y: bytes([0x80] * (w * 3)),
                               extra_chunks=[(chunk_name, chunk_data)])))

    # Minimal valid PNGs for all color_type × bit_depth combos
    for ct, bd in [
        (0, 1), (0, 2), (0, 4), (0, 8), (0, 16),
        (2, 8), (2, 16),
        (3, 1), (3, 2), (3, 4), (3, 8),
        (4, 8),
        (6, 8), (6, 16),
    ]:
        channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[ct]
        row_bytes = max(1, (channels * bd + 7) // 8)
        raw = bytes([0]) + bytes([0x80] * row_bytes)
        seeds.append((f"minimal_ct{ct}_bd{bd}.png", make_png(1, 1, ct, bd,
                      pixel_func=lambda w, y, _r=raw: _r[1:1 + w * {0:1,2:3,3:1,4:2,6:4}[ct]])))

    # Interlaced PNGs
    for w, h in [(16, 16), (32, 32)]:
        seeds.append((f"interlace_{w}x{h}.png",
                      make_png(w, h, 2, 8, interlace=1,
                               pixel_func=lambda w, y: bytes(rng.randint(0, 255) for _ in range(w * 3)))))

    # Zlib-wrapped raw data (for decompression testing)
    for data_name, raw in [
        ("zeros", b"\x00" * 1024),
        ("repeating", b"\x41\x42\x43\x44" * 256),
        ("random", bytes(rng.randint(0, 255) for _ in range(1024))),
        ("incremental", bytes(range(256)) * 4),
    ]:
        for level in [0, 6, 9]:
            compressed = zlib.compress(raw, level)
            seeds.append((f"zlib_{data_name}_L{level}.bin", compressed))

    # PNGs with palette data
    for palette_size in [2, 4, 16, 256]:
        w, h = 16, 16
        palette = bytes(rng.randint(0, 255) for _ in range(palette_size * 3))
        raw = b""
        for y in range(h):
            raw += bytes([0])
            raw += bytes(rng.randint(0, palette_size - 1) for _ in range(w))
        sig = b"\x89PNG\r\n\x1a\n"
        ihdr = make_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 3, 0, 0, 0))
        plte = make_chunk(b"PLTE", palette)
        compressed = zlib.compress(raw, 6)
        idat = make_chunk(b"IDAT", compressed)
        iend = make_chunk(b"IEND", b"")
        seeds.append((f"palette_{palette_size}.png",
                      sig + ihdr + plte + idat + iend))

    return seeds


DOWNLOAD_URLS = [
    ("google_logo.png", "https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png"),
    ("w3c_logo.png", "https://www.w3.org/TR/png/images/logo-h-rgb-32.png"),
    ("wikipedia_transparency.png", "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"),
]


def download_pngs(out_dir: str) -> int:
    """Download real-world PNGs. Returns count of successful downloads."""
    count = 0
    for name, url in DOWNLOAD_URLS:
        path = os.path.join(out_dir, name)
        try:
            urllib.request.urlretrieve(url, path)
            size = os.path.getsize(path)
            print(f"  downloaded {name} ({size} bytes)")
            count += 1
        except Exception as e:
            print(f"  skipped {name}: {e}", file=sys.stderr)
    return count


def main():
    parser = argparse.ArgumentParser(description="Generate PNG corpus for fuzzing")
    parser.add_argument("--out", default="corpus_png", help="Output directory (default: corpus_png)")
    parser.add_argument("--count", type=int, default=0, help="Max seeds (0=all)")
    parser.add_argument("--download", action="store_true", help="Also download real-world PNGs")
    parser.add_argument("--zlib-only", action="store_true",
                        help="Only generate zlib-aware seeds (skip basic variants)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"[*] Generating PNG corpus in {args.out}/")
    seeds = []
    if not args.zlib_only:
        seeds.extend(make_variants())
    seeds.extend(make_zlib_variants())

    if args.count > 0:
        seeds = seeds[: args.count]

    for name, data in seeds:
        path = os.path.join(args.out, name)
        with open(path, "wb") as f:
            f.write(data)

    print(f"[*] Generated {len(seeds)} seed PNGs")

    if args.download:
        print("[*] Downloading real-world PNGs...")
        dl_count = download_pngs(args.out)
        print(f"[*] Downloaded {dl_count} PNGs")

    total = len(os.listdir(args.out))
    print(f"[*] Corpus ready: {total} files in {args.out}/")


if __name__ == "__main__":
    main()
