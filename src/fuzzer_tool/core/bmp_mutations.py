"""Structure-aware BMP mutations.

Parses BMP file/dib headers, color tables, and pixel data.
Applies targeted corruption to header fields, color tables,
compression modes, and pixel data layout. Falls back to random
BMP generation when input is not valid BMP.
"""

from __future__ import annotations

import random
import struct
from dataclasses import dataclass

# BMP signatures
BMP_SIGNATURE = b"BM"

# Compression modes
BI_RGB = 0
BI_RLE8 = 1
BI_RLE4 = 2
BI_BITFIELDS = 3
BI_JPEG = 4
BI_PNG = 5

# Common DIB header sizes
DIB_INFOHEADER = 40
DIB_V4HEADER = 108
DIB_V5HEADER = 124


@dataclass
class BmpInfo:
    """Parsed BMP header information."""

    file_size: int
    pixel_offset: int
    dib_size: int
    width: int
    height: int
    planes: int
    bit_count: int
    compression: int
    image_size: int
    x_ppm: int
    y_ppm: int
    colors_used: int
    colors_important: int
    # Extended fields for V4/V5
    red_mask: int = 0
    green_mask: int = 0
    blue_mask: int = 0
    alpha_mask: int = 0
    # Raw data
    header: bytearray = None
    color_table: bytes = b""
    pixel_data: bytes = b""

    def __post_init__(self):
        if self.header is None:
            self.header = bytearray()


def parse_bmp(data: bytes) -> BmpInfo | None:
    """Parse BMP file header and DIB header.

    Returns None if data doesn't start with 'BM' or is too small.
    """
    if len(data) < 54 or data[:2] != BMP_SIGNATURE:
        return None

    file_size = struct.unpack("<I", data[2:6])[0]
    pixel_offset = struct.unpack("<I", data[10:14])[0]
    dib_size = struct.unpack("<I", data[14:18])[0]

    if dib_size < DIB_INFOHEADER or pixel_offset > len(data):
        return None

    # BITMAPINFOHEADER (common case)
    width = struct.unpack("<i", data[18:22])[0]
    height = struct.unpack("<i", data[22:26])[0]
    planes = struct.unpack("<H", data[26:28])[0]
    bit_count = struct.unpack("<H", data[28:30])[0]
    compression = struct.unpack("<I", data[30:34])[0]
    image_size = struct.unpack("<I", data[34:38])[0]
    x_ppm = struct.unpack("<i", data[38:42])[0]
    y_ppm = struct.unpack("<i", data[42:46])[0]
    colors_used = struct.unpack("<I", data[46:50])[0]
    colors_important = struct.unpack("<I", data[50:54])[0]

    info = BmpInfo(
        file_size=file_size,
        pixel_offset=pixel_offset,
        dib_size=dib_size,
        width=width,
        height=height,
        planes=planes,
        bit_count=bit_count,
        compression=compression,
        image_size=image_size,
        x_ppm=x_ppm,
        y_ppm=y_ppm,
        colors_used=colors_used,
        colors_important=colors_important,
        header=bytearray(data[:pixel_offset]),
    )

    # Extended headers (V4/V5)
    if dib_size >= DIB_V4HEADER and len(data) >= 58:
        info.red_mask = struct.unpack("<I", data[54:58])[0] if len(data) >= 58 else 0
        info.green_mask = struct.unpack("<I", data[58:62])[0] if len(data) >= 62 else 0
        info.blue_mask = struct.unpack("<I", data[62:66])[0] if len(data) >= 66 else 0
        info.alpha_mask = struct.unpack("<I", data[66:70])[0] if len(data) >= 70 else 0

    # Color table
    if bit_count <= 8:
        table_size = (colors_used if colors_used else (1 << bit_count)) * 4
        table_start = 14 + dib_size
        info.color_table = data[table_start : table_start + table_size]

    # Pixel data
    if pixel_offset < len(data):
        info.pixel_data = data[pixel_offset:]

    return info


def serialize_bmp(info: BmpInfo) -> bytes:
    """Serialize BmpInfo back to BMP bytes."""
    return bytes(info.header) + info.pixel_data


def _corrupt_field(data: bytearray, offset: int, size: int, signed: bool = False) -> None:
    """Apply random corruption to a field in the header."""
    method = random.randint(0, 4)
    if size == 2:
        val = struct.unpack("<H", data[offset : offset + 2])[0]
        if method == 0:
            val ^= 1 << random.randint(0, 15)
        elif method == 1:
            val = random.choice([0, 1, 0x7FFF, 0xFFFF])
        elif method == 2:
            val = max(0, val + random.choice([-2, -1, 1, 2, 16, 256]))
        elif method == 3:
            val = random.randint(0, 0xFFFF)
        else:
            val = random.randint(0, 16)
        struct.pack_into("<H", data, offset, val)
    elif size == 4:
        val = struct.unpack("<I", data[offset : offset + 4])[0]
        if signed:
            val = struct.unpack("<i", data[offset : offset + 4])[0]
            if method == 0:
                val ^= 1 << random.randint(0, 31)
            elif method == 1:
                val = random.choice([0, 1, -1, 0x7FFFFFFF, -0x80000000])
            elif method == 2:
                val = max(
                    -0x80000000, min(0x7FFFFFFF, val + random.choice([-2, -1, 1, 2, 256, 65536]))
                )
            elif method == 3:
                val = random.randint(-0x80000000, 0x7FFFFFFF)
            else:
                val = random.randint(0, 256)
            val = max(-0x80000000, min(0x7FFFFFFF, val))
            struct.pack_into("<i", data, offset, val)
        else:
            if method == 0:
                val ^= 1 << random.randint(0, 31)
            elif method == 1:
                val = random.choice([0, 1, 0x7FFFFFFF, 0xFFFFFFFF])
            elif method == 2:
                val = max(0, min(0xFFFFFFFF, val + random.choice([-2, -1, 1, 2, 256, 65536])))
            elif method == 3:
                val = random.randint(0, 0xFFFFFFFF)
            else:
                val = random.randint(0, 256)
            val = max(0, min(0xFFFFFFFF, val))
            struct.pack_into("<I", data, offset, val)


class BmpMutator:
    """Structure-aware BMP mutator.

    When ``use_wfc=True``, pixel data generation uses 2D Wave Function
    Collapse for locally-coherent pixel textures instead of random bytes.

    Dispatches one of 16 mutation operations per call, targeting
    specific BMP structures for maximum code-path diversity.
    """

    use_wfc: bool = False  # set to True by Fuzzer when --wfc is active

    def mutate(self, data: bytes, max_len: int = 4096) -> bytes:
        """Apply one structure-aware BMP mutation."""
        info = parse_bmp(data)
        if info is None:
            return self._generate_random_bmp(max_len)

        op = random.randint(0, 15)
        mutators = [
            self._mutate_dimensions,
            self._mutate_bit_count,
            self._mutate_compression,
            self._mutate_resolution,
            self._mutate_color_table,
            self._mutate_pixel_data,
            self._corrupt_file_size,
            self._corrupt_pixel_offset,
            self._swap_color_channels,
            self._mutate_bitfields,
            self._duplicate_bmp_header,
            self._truncate_bmp,
            self._inject_junk_before_pixels,
            self._mutate_planes,
            self._mutate_colors_used,
            self._generate_random_bmp,
        ]
        result = mutators[op](info, max_len)
        if isinstance(result, BmpInfo):
            return serialize_bmp(result)[:max_len]
        return result[:max_len]

    def _mutate_dimensions(self, info: BmpInfo, max_len: int) -> BmpInfo:
        """Corrupt width or height in DIB header."""
        if info.dib_size >= DIB_INFOHEADER and len(info.header) >= 26:
            field = random.choice(["width", "height"])
            if field == "width":
                _corrupt_field(info.header, 18, 4, signed=True)
            else:
                _corrupt_field(info.header, 22, 4, signed=True)
        return info

    def _mutate_bit_count(self, info: BmpInfo, max_len: int) -> BmpInfo:
        """Corrupt bits-per-pixel field."""
        if info.dib_size >= DIB_INFOHEADER and len(info.header) >= 30:
            info.header[28] = random.choice([1, 2, 4, 8, 16, 24, 32, 0, 255])
            info.header[29] = 0
        return info

    def _mutate_compression(self, info: BmpInfo, max_len: int) -> BmpInfo:
        """Corrupt compression mode."""
        if info.dib_size >= DIB_INFOHEADER and len(info.header) >= 34:
            info.header[30] = random.choice([0, 1, 2, 3, 4, 5, 0xFF])
            for i in range(31, 34):
                info.header[i] = 0
        return info

    def _mutate_resolution(self, info: BmpInfo, max_len: int) -> BmpInfo:
        """Corrupt pixels-per-meter resolution."""
        if info.dib_size >= DIB_INFOHEADER and len(info.header) >= 46:
            field = random.choice(["x_ppm", "y_ppm"])
            offset = 38 if field == "x_ppm" else 42
            _corrupt_field(info.header, offset, 4, signed=True)
        return info

    def _mutate_color_table(self, info: BmpInfo, max_len: int) -> BmpInfo:
        """Corrupt a color table entry."""
        if info.bit_count <= 8 and len(info.color_table) >= 4:
            idx = random.randint(0, len(info.color_table) - 4)
            table = bytearray(info.color_table)
            table[idx : idx + 4] = bytes(random.randint(0, 255) for _ in range(4))
            info.color_table = bytes(table)
        return info

    def _mutate_pixel_data(self, info: BmpInfo, max_len: int) -> BmpInfo:
        """Flip random bytes in pixel data."""
        if not info.pixel_data:
            return info
        if self.use_wfc and abs(info.width) >= 2 and abs(info.height) >= 2:
            return self._wfc_pixels(info, max_len)
        pixels = bytearray(info.pixel_data)
        for _ in range(random.randint(1, min(8, len(pixels)))):
            idx = random.randint(0, len(pixels) - 1)
            pixels[idx] ^= 1 << random.randint(0, 7)
        info.pixel_data = bytes(pixels)
        return info

    def _wfc_pixels(self, info: BmpInfo, max_len: int) -> BmpInfo:
        """Generate pixel data using 1D Wave Function Collapse.

        Treats each row as a sequence of pixel-sample tiles, uses WFC
        to generate row content with locally-coherent color transitions.
        Falls back to random bytes if WFC fails.
        """
        if not info.pixel_data:
            return info
        w = abs(info.width)
        h = abs(info.height)
        bpp = max(1, info.bit_count // 8)
        stride = ((w * bpp + 3) // 4) * 4
        if w < 2 or h < 2 or len(info.pixel_data) < stride * 2:
            return info

        from fuzzer_tool.core.wfc import AdjacencyTable, Tile, WaveGrid

        # Build per-pixel tiles from the first row (used as tile alphabet)
        pixels = info.pixel_data
        unique_tiles: dict[bytes, int] = {}
        for x in range(w):
            start = x * bpp
            sample = pixels[start : start + bpp]
            unique_tiles[sample] = unique_tiles.get(sample, 0) + 1

        tile_list = [Tile(name=t, weight=c) for t, c in unique_tiles.items()]

        # Build adjacency from existing pixel data
        adj = AdjacencyTable()
        for x in range(w - 1):
            a = pixels[x * bpp : (x + 1) * bpp]
            b = pixels[(x + 1) * bpp : (x + 2) * bpp]
            if a in unique_tiles and b in unique_tiles:
                adj.add_forward(a, b)

        # Generate each row via WFC
        new_pixels = bytearray()
        for row_y in range(h):
            wave = WaveGrid(tile_list, adj, width=w, height=1)
            row_result = wave.run(
                seed=random.randint(0, 2**31),
                max_restarts=2,
                ac3_budget=2000,
            )
            row_data = bytearray()
            if row_result and row_result[0]:
                for tile_name in row_result[0]:
                    if tile_name is not None:
                        row_data.extend(tile_name)
                    else:
                        row_data.extend(pixels[row_y * stride : row_y * stride + bpp])
            else:
                # Fallback: copy original row
                row_data.extend(pixels[row_y * stride : row_y * stride + bpp])
            # Pad to stride
            while len(row_data) < stride:
                row_data.append(0)
            new_pixels.extend(row_data[:stride])

        info.pixel_data = bytes(new_pixels)
        return info

    def _corrupt_file_size(self, info: BmpInfo, max_len: int) -> BmpInfo:
        """Corrupt the file size field in the file header."""
        if len(info.header) >= 6:
            _corrupt_field(info.header, 2, 4)
        return info

    def _corrupt_pixel_offset(self, info: BmpInfo, max_len: int) -> BmpInfo:
        """Corrupt the pixel data offset field."""
        if len(info.header) >= 14:
            _corrupt_field(info.header, 10, 4)
        return info

    def _swap_color_channels(self, info: BmpInfo, max_len: int) -> BmpInfo:
        """Swap R and B channels in pixel data (BGR → RGB corruption)."""
        if info.bit_count == 24 and len(info.pixel_data) >= 3:
            pixels = bytearray(info.pixel_data)
            for i in range(0, len(pixels) - 2, 3):
                pixels[i], pixels[i + 2] = pixels[i + 2], pixels[i]
            info.pixel_data = bytes(pixels)
        return info

    def _mutate_bitfields(self, info: BmpInfo, max_len: int) -> BmpInfo:
        """Corrupt BI_BITFIELDS color masks."""
        if info.dib_size >= DIB_V4HEADER and len(info.header) >= 70:
            mask_offsets = [54, 58, 62, 66]
            idx = random.choice(mask_offsets)
            _corrupt_field(info.header, idx, 4)
        elif info.compression == BI_BITFIELDS and len(info.header) >= 58:
            # Inject bitfields mask after BITMAPINFOHEADER
            mask = struct.pack(
                "<III",
                random.choice([0xFF0000, 0x00FF00, 0x0000FF]),
                random.choice([0xFF00, 0xFF0000, 0xFF]),
                random.choice([0xFF, 0xFF00, 0xFF0000]),
            )
            header = bytearray(info.header)
            header[30:34] = struct.pack("<I", BI_BITFIELDS)
            header.extend(mask)
            info.header = header
        return info

    def _duplicate_bmp_header(self, info: BmpInfo, max_len: int) -> BmpInfo:
        """Insert a duplicate DIB header (tests parser re-entrancy)."""
        if info.dib_size >= DIB_INFOHEADER and len(info.header) >= 14 + info.dib_size:
            dib = info.header[14 : 14 + info.dib_size]
            header = bytearray(info.header)
            header[14:14] = dib
            info.header = header
        return info

    def _truncate_bmp(self, info: BmpInfo, max_len: int) -> bytes:
        """Remove trailing pixel data to produce a truncated BMP."""
        if info.dib_size >= DIB_INFOHEADER and len(info.header) >= 14 + info.dib_size:
            # Keep only the headers
            return bytes(info.header)[:max_len]
        return serialize_bmp(info)[:max_len]

    def _inject_junk_before_pixels(self, info: BmpInfo, max_len: int) -> BmpInfo:
        """Inject random bytes between headers and pixel data."""
        junk = bytes(random.randint(0, 255) for _ in range(random.randint(4, 64)))
        header = bytearray(info.header)
        insert_pos = min(info.pixel_offset, len(header))
        header[insert_pos:insert_pos] = junk
        # Update pixel offset
        if len(header) >= 14:
            struct.pack_into("<I", header, 10, info.pixel_offset + len(junk))
        info.header = header
        return info

    def _mutate_planes(self, info: BmpInfo, max_len: int) -> BmpInfo:
        """Corrupt the planes field (must be 1 in valid BMP)."""
        if info.dib_size >= DIB_INFOHEADER and len(info.header) >= 28:
            info.header[26] = random.choice([0, 2, 3, 0xFF])
            info.header[27] = 0
        return info

    def _mutate_colors_used(self, info: BmpInfo, max_len: int) -> BmpInfo:
        """Corrupt colors_used and colors_important fields."""
        if info.dib_size >= DIB_INFOHEADER and len(info.header) >= 54:
            field = random.choice(["used", "important"])
            offset = 46 if field == "used" else 50
            _corrupt_field(info.header, offset, 4)
        return info

    def _generate_random_bmp(self, info_or_max=None, max_len: int = 4096) -> bytes:
        """Generate a minimal random BMP from scratch.

        Called from dispatch as _generate_random_bmp(info, max_len) or
        standalone as _generate_random_bmp(max_len=N).
        """
        if isinstance(info_or_max, BmpInfo):
            max_len = max_len
        elif isinstance(info_or_max, int):
            max_len = info_or_max

        width = random.randint(1, 64)
        height = random.randint(1, 64)
        bit_count = random.choice([1, 4, 8, 24, 32])

        # Calculate row stride (padded to 4-byte boundary)
        bits_per_row = width * bit_count
        stride = ((bits_per_row + 31) // 32) * 4

        # Color table size
        if bit_count <= 8:
            num_colors = 1 << bit_count
            color_table_size = num_colors * 4
        else:
            num_colors = 0
            color_table_size = 0

        # Pixel data
        pixel_size = stride * abs(height)
        pixel_offset = 14 + DIB_INFOHEADER + color_table_size
        file_size = pixel_offset + pixel_size

        # Build header
        header = bytearray(14 + DIB_INFOHEADER)
        # File header
        header[0:2] = BMP_SIGNATURE
        struct.pack_into("<I", header, 2, file_size)
        struct.pack_into("<I", header, 10, pixel_offset)
        # DIB header (BITMAPINFOHEADER)
        struct.pack_into("<I", header, 14, DIB_INFOHEADER)
        struct.pack_into("<i", header, 18, width)
        struct.pack_into("<i", header, 22, height)
        struct.pack_into("<H", header, 26, 1)  # planes
        struct.pack_into("<H", header, 28, bit_count)
        struct.pack_into("<I", header, 30, BI_RGB)
        struct.pack_into("<I", header, 34, pixel_size)
        struct.pack_into("<i", header, 38, 2835)  # 72 DPI
        struct.pack_into("<i", header, 42, 2835)

        # Color table (grayscale: each entry is [i, i, i, 0])
        color_table = bytearray(num_colors * 4)
        if bit_count <= 8:
            for i in range(num_colors):
                color_table[i * 4 : i * 4 + 4] = bytes([i, i, i, 0])

        # Random pixel data
        pixels = random.randbytes(pixel_size)

        return bytes(header) + bytes(color_table) + pixels
