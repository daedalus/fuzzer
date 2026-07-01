#!/usr/bin/env python3
"""Generate a diverse PNG corpus for fuzzing libpng.

Creates minimal PNGs with varied properties (dimensions, color types,
bit depths, compression levels, filter types) plus downloads real-world
PNGs from public sources when network is available.

Usage:
    python tools/corpus_png.py [--out DIR] [--count N] [--download]

Output is a directory of .png files suitable as fuzzer seed corpus.
"""

import argparse
import os
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
) -> bytes:
    sig = b"\x89PNG\r\n\x1a\n"
    # IHDR: compression_method=0 (deflate), filter_method=0 (adaptive), interlace
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
    return sig + ihdr + idat + iend


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
    import random
    random.seed(42)
    for w, h in [(64, 64), (128, 128)]:
        def noise(w, y, _rng=random):
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
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"[*] Generating PNG corpus in {args.out}/")
    seeds = make_variants()
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
