"""Regression test: verify rng parameter is actually threaded through format mutators.

This catches the docstring bug pattern where self._rng = rng or random was
accidentally pasted inside a docstring instead of as executable code.
"""

import random

import pytest


class MockRng:
    """Minimal mock that tracks which methods are called."""

    def __init__(self):
        self.call_count = 0
        self.last_method = None

    def randint(self, a, b):
        self.call_count += 1
        self.last_method = "randint"
        return random.randint(a, b)

    def choice(self, seq):
        self.call_count += 1
        self.last_method = "choice"
        return random.choice(seq)

    def shuffle(self, seq):
        self.call_count += 1
        self.last_method = "shuffle"
        random.shuffle(seq)

    def sample(self, population, k):
        self.call_count += 1
        self.last_method = "sample"
        return random.sample(population, k)

    def randbytes(self, n):
        self.call_count += 1
        self.last_method = "randbytes"
        return random.randbytes(n)

    def random(self):
        self.call_count += 1
        self.last_method = "random"
        return random.random()


BMP_HEADER = (
    b"BM"
    + b"\x00\x00\x00\x00"  # file size placeholder
    + b"\x00\x00\x00\x00"  # reserved
    + b"\x36\x00\x00\x00"  # pixel data offset
    + b"\x28\x00\x00\x00"  # DIB header size (40)
    + b"\x04\x00\x00\x00"  # width (4)
    + b"\x04\x00\x00\x00"  # height (4)
    + b"\x01\x00"  # planes
    + b"\x18\x00"  # bits per pixel (24)
    + b"\x00\x00\x00\x00"  # compression
    + b"\x00\x00\x00\x00"  # image size
    + b"\x00\x00\x00\x00"  # x ppm
    + b"\x00\x00\x00\x00"  # y ppm
    + b"\x00\x00\x00\x00"  # colors used
    + b"\x00\x00\x00\x00"  # important colors
    + b"\x00" * 48  # pixel data (4x4 @ 24bpp)
)

GZIP_HEADER = (
    b"\x1f\x8b"  # magic
    + b"\x08"  # method (deflate)
    + b"\x00"  # flags
    + b"\x00\x00\x00\x00"  # mtime
    + b"\x00"  # xfl
    + b"\xff"  # os
)

ZLIB_HEADER = b"\x78\x9c" + b"\x00" * 10

JPEG_HEADER = (
    b"\xff\xd8"  # SOI
    + b"\xff\xe0"  # APP0 marker
    + b"\x00\x10"  # length
    + b"JFIF\x00"  # identifier
    + b"\x01\x01"  # version
    + b"\x00"  # units
    + b"\x00\x01\x00\x01"  # density
    + b"\x00\x00"  # thumbnail
    + b"\xff\xd9"  # EOI
)


@pytest.mark.parametrize(
    "module_name,cls_name,header,method_name",
    [
        ("fuzzer_tool.core.bmp_mutations", "BmpMutator", BMP_HEADER, "mutate"),
        ("fuzzer_tool.core.gzip_mutations", "GzipMutator", GZIP_HEADER, "mutate"),
        ("fuzzer_tool.core.jpeg_mutations", "JpegMutator", JPEG_HEADER, "mutate"),
        ("fuzzer_tool.core.zlib_mutations", "ZlibMutator", ZLIB_HEADER, "mutate"),
        ("fuzzer_tool.core.png_mutations", "PngChunkMutator", b"\x89PNG\r\n\x1a\n", "mutate"),
    ],
)
def test_rng_parameter_is_actually_used(module_name, cls_name, header, method_name):
    """Pass a MockRng and verify it gets invoked (not bare random)."""
    import importlib

    mod = importlib.import_module(module_name)
    cls = getattr(mod, cls_name)
    mutator = cls()

    mock_rng = MockRng()
    mutate_fn = getattr(mutator, method_name)

    # Call mutate multiple times — at least one should use the mock
    for _ in range(20):
        try:
            mutate_fn(header, max_len=4096, rng=mock_rng)
        except Exception:
            pass  # Some inputs may fail to parse — that's OK

    assert mock_rng.call_count > 0, (
        f"{cls_name}.{method_name}(rng=mock_rng) never invoked mock_rng — "
        f"the rng parameter is being ignored (likely dead text in docstring)"
    )


@pytest.mark.parametrize(
    "module_name,cls_name,header,method_name",
    [
        ("fuzzer_tool.core.bmp_mutations", "BmpMutator", BMP_HEADER, "_generate_random_bmp"),
        ("fuzzer_tool.core.gzip_mutations", "GzipMutator", GZIP_HEADER, "_generate_random_gzip"),
        ("fuzzer_tool.core.jpeg_mutations", "JpegMutator", JPEG_HEADER, "_generate_random_jpeg"),
        ("fuzzer_tool.core.zlib_mutations", "ZlibMutator", ZLIB_HEADER, "_generate_random_zlib"),
    ],
)
def test_rng_parameter_in_generate_random(module_name, cls_name, header, method_name):
    """Pass a MockRng to _generate_random_* and verify it gets invoked."""
    import importlib

    mod = importlib.import_module(module_name)
    cls = getattr(mod, cls_name)
    mutator = cls()

    mock_rng = MockRng()
    gen_fn = getattr(mutator, method_name)

    try:
        gen_fn(max_len=4096, rng=mock_rng)
    except Exception:
        pass

    assert mock_rng.call_count > 0, (
        f"{cls_name}.{method_name}(rng=mock_rng) never invoked mock_rng — "
        f"the rng parameter is being ignored"
    )
