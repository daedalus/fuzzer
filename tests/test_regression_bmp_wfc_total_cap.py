"""BMP WFC total-tile cap: 512x64 took 7.4 s per mutation under the 200k cap."""

import struct

from fuzzer_tool.core.mutations import bmp
from fuzzer_tool.core.mutations.bmp import BmpMutator, parse_bmp
from fuzzer_tool.core.wfc import WaveGrid

_BPP = 3
_PALETTE = (b"\x00\x00\x00", b"\xff\x00\x00", b"\x00\xff\x00")
_HDR = 54


def _bmp24(w: int, h: int) -> bytes:
    """24-bpp BMP, 3 distinct tiles, row width a multiple of 4 bytes."""
    stride = ((w * _BPP + 3) // 4) * 4
    row = b"".join(_PALETTE[i % len(_PALETTE)] for i in range(w))
    row += b"\0" * (stride - len(row))
    px = row * h
    file_hdr = b"BM" + struct.pack("<IHHI", _HDR + len(px), 0, 0, _HDR)
    dib = struct.pack("<IiiHHIIiiII", 40, w, h, 1, 24, 0, len(px), 2835, 2835, 0, 0)
    return file_hdr + dib + px


def _run(w: int, h: int, monkeypatch) -> int:
    built = []
    real = WaveGrid.__init__

    def counting(self, *a, **k):
        built.append(1)
        real(self, *a, **k)

    monkeypatch.setattr(WaveGrid, "__init__", counting)
    m = BmpMutator(seed=1)
    m.use_wfc = True
    m._wfc_pixels(parse_bmp(_bmp24(w, h)), 1 << 20)
    return len(built)


def test_cap_is_small():
    assert bmp.BMP_WFC_MAX_TOTAL <= 2_000


def test_512x64_skips_wfc(monkeypatch):
    """Falsification: the reported slow shape never builds a grid."""
    assert _run(512, 64, monkeypatch) == 0


def test_small_image_still_uses_wfc(monkeypatch):
    """Adversarial: the cap must not disable WFC for small images."""
    assert _run(32, 32, monkeypatch) == 1


def test_cap_boundary(monkeypatch):
    """Exactly at the cap runs WFC; one row more falls back."""
    h = bmp.BMP_WFC_MAX_TOTAL // 32
    assert _run(32, h, monkeypatch) == 1
    assert _run(32, h + 1, monkeypatch) == 0
