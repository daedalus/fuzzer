"""BMP WFC rebuilt an identical WaveGrid (adjacency matrix included) per row.

The tile alphabet and adjacency are fixed for the whole image, and run(seed)
reseeds the grid's RNG, so one grid cleared between rows produces the same
pixels. Rebuilding cost ~30% of a pixel mutation (233k compatible() calls
for a 20-row image).
"""

from fuzzer_tool.core.mutations import bmp
from fuzzer_tool.core.mutations.bmp import BmpMutator, parse_bmp
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.wfc import WaveGrid


def _bmp(seed: int) -> bytes:
    m, r = BmpMutator(seed=seed), RandPool(seed)
    while True:
        b = m._generate_random_bmp(4096, rng=r)
        info = parse_bmp(b)
        if info and abs(info.width) >= 4 and abs(info.height) >= 4 and info.pixel_data:
            return b


def _pixels(data: bytes, seed: int) -> bytes:
    m = BmpMutator(seed=seed)
    m.use_wfc = True
    return m._wfc_pixels(parse_bmp(data), 4096).pixel_data


def test_regression_bmp_wfc_shared_grid(monkeypatch):
    """One WaveGrid per image, not per row."""
    data = _bmp(1)
    built = []
    real = WaveGrid.__init__

    def counting(self, *a, **k):
        built.append(1)
        real(self, *a, **k)

    monkeypatch.setattr(WaveGrid, "__init__", counting)
    _pixels(data, 5)
    assert len(built) == 1


def _old_pixels(data: bytes, seed: int) -> bytes:
    """The pre-change _wfc_pixels: a fresh WaveGrid per row, verbatim."""
    from fuzzer_tool.core.wfc import Tile

    m = BmpMutator(seed=seed)
    info = parse_bmp(data)
    w, h = abs(info.width), abs(info.height)
    bpp = max(1, info.bit_count // 8)
    stride = ((w * bpp + 3) // 4) * 4
    if w < 2 or h < 2 or len(info.pixel_data) < stride * 2:
        return info.pixel_data
    sample = bpp
    per_row = stride // sample
    if per_row < 2:
        return info.pixel_data
    pixels = info.pixel_data
    unique = bmp._count_tiles(pixels, per_row, sample)
    if (
        per_row > bmp.BMP_WFC_MAX_CELLS
        or len(unique) > bmp.BMP_WFC_MAX_TILES
        or per_row * h > bmp.BMP_WFC_MAX_TOTAL
    ):
        return m._flip_pixels(info).pixel_data
    tiles = [Tile(name=t, weight=c) for t, c in unique.items()]
    adj = bmp._first_row_adjacency(pixels, unique, per_row, sample)
    out = bytearray()
    for row_y in range(h):
        wave = WaveGrid(tiles, adj, width=per_row, height=1)
        row = wave.run(seed=m._rng.randint(0, 2**31), max_restarts=2, ac3_budget=2000)
        out.extend(bmp._assemble_row(row, pixels, row_y, stride, sample))
    return bytes(out)


def test_shared_grid_same_pixels():
    """Byte-identical to a fresh grid per row (the old construction)."""
    for seed in range(6):
        data = _bmp(seed)
        assert _pixels(data, seed) == _old_pixels(data, seed)


def test_clear_restores_full_superposition():
    """Adversarial: a collapsed grid comes back fully open after clear()."""
    data = parse_bmp(_bmp(2))
    assert data is not None
    from fuzzer_tool.core.wfc import AdjacencyTable, Tile

    adj = AdjacencyTable()
    tiles = [Tile(name=b"a", weight=1), Tile(name=b"b", weight=1)]
    g = WaveGrid(tiles, adj, width=4)
    g.run(seed=1)
    g.clear()
    assert g.superpositions.all() and not g.contradiction
