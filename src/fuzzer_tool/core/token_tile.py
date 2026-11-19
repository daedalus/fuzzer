"""Word-break tiling: cut the input into dictionary tokens, mutate whole tokens.

Byte-level dict ops insert or overwrite a token at a random offset and
usually land mid-token. This op first finds the tiling of the input into
dictionary tokens plus literal gap bytes that covers the most bytes, then
edits one tile as a unit::

    input   GET /a HTTP           tokens GET, /a, HTTP
    tiles   [GET] [/a] [HTTP]     gaps " ", " "
    REPLACE [HTTP] /a HTTP        DUPLICATE GETGET /a HTTP
    DELETE   /a HTTP              SWAP      HTTP /a GET

Matches come from :class:`~fuzzer_tool.core.aho_corasick.TokenScanner`
(overlapping occurrences; ``find`` / integer windows below 512 tokens, the
automaton above). Max coverage is weighted interval scheduling over those
matches, sorted by end; equal coverage keeps the longer token (fewer tiles).
Scalar below TILE_NP_MIN_MATCHES matches, numpy-assisted above.

Bounds: the first MAX_TILE_SCAN bytes are tiled (the tail is kept as is),
MAX_TILE_TOKENS tokens per call (a window from a random start, wrapping),
MAX_TILE_MATCHES matches (lowest offsets kept), SCANNER_CACHE scanners.
"""

from __future__ import annotations

import enum
import functools
from bisect import bisect_right

import numpy as np

from fuzzer_tool.core.aho_corasick import TokenScanner

#: Input prefix tiled; the rest is copied through.
MAX_TILE_SCAN = 4096
#: Tokens scanned per call: bounds scan and DP cost.
MAX_TILE_TOKENS = 128
#: Matches fed to the DP (zero-filled input x run tokens: one per offset per token).
MAX_TILE_MATCHES = 16384
#: Token windows whose scanner is kept (the dictionary repeats across calls).
SCANNER_CACHE = 32
#: Matches from which the numpy DP beats the scalar one (min-of-11: scalar
#: 1.9x faster at 10 matches, numpy 1.07x at 233, 1.56x at 32k).
TILE_NP_MIN_MATCHES = 256


class Edit(enum.Enum):
    """Whole-token edit applied to one tile."""

    REPLACE = enum.auto()
    DUPLICATE = enum.auto()
    DELETE = enum.auto()
    SWAP = enum.auto()


_EDITS = tuple(Edit)


@functools.lru_cache(maxsize=SCANNER_CACHE)
def _prepare(window: tuple) -> tuple[tuple[bytes, ...], TokenScanner]:
    """Distinct non-empty byte tokens of *window* and their scanner."""
    toks = tuple(dict.fromkeys(t.encode() if isinstance(t, str) else bytes(t) for t in window))
    toks = tuple(t for t in toks if t)
    return toks, TokenScanner(toks)


def _window(tokens, rng) -> tuple:
    """At most MAX_TILE_TOKENS tokens from a random start, wrapping."""
    n = len(tokens)
    if n <= MAX_TILE_TOKENS:
        return tuple(tokens)

    first = rng.randint(0, n - 1)
    return tuple(tokens[(first + i) % n] for i in range(MAX_TILE_TOKENS))


def _tile_py(found: dict[bytes, list[int]]) -> list[tuple[int, bytes]]:
    """Weighted interval scheduling (weight = length), scalar: few matches.

    ``best[k]``: max bytes covered up to ``ends[k]``. Matches sorted by
    (end, start): at equal coverage the strict ``>`` keeps the longer token.
    """
    spans = sorted((s + len(t), s, t) for t, offs in found.items() for s in offs)
    ends, best, pick = [0], [0], [None]
    for e, s, t in spans:
        cand = best[bisect_right(ends, s) - 1] + e - s
        if e != ends[-1]:
            ends.append(e)
            best.append(best[-1])  # carry: gap byte(s) up to this end
            pick.append(None)
        if cand > best[-1]:
            best[-1] = cand
            pick[-1] = (s, t)

    tiles = []
    k = len(ends) - 1
    while k > 0:
        p = pick[k]
        if p is None:
            k -= 1
            continue
        tiles.append(p)
        k = bisect_right(ends, p[0]) - 1
    tiles.reverse()
    return tiles


def _tile_np(found: dict[bytes, list[int]]) -> list[tuple[int, bytes]]:
    """``_tile_py`` with the sort and predecessor search in numpy: many matches.

    Only this path can pass MAX_TILE_MATCHES (> TILE_NP_MIN_MATCHES); it is capped here.

    A match's predecessor slot depends only on its start, so it is one
    ``searchsorted``; only the max-recurrence stays a loop.
    """
    toks = list(found)
    counts = [len(found[t]) for t in toks]
    starts = np.concatenate([np.array(found[t], np.int64) for t in toks])
    which = np.repeat(np.arange(len(toks)), counts)
    ends = starts + np.repeat([len(t) for t in toks], counts)
    if len(starts) > MAX_TILE_MATCHES:  # keep the lowest starts: a prefix
        # Stable sort: per-token runs are already ascending (beats argpartition 2.6x).
        keep = np.argsort(starts, kind="stable")[:MAX_TILE_MATCHES]
        starts, ends, which = starts[keep], ends[keep], which[keep]

    order = np.lexsort((starts, ends))
    starts, ends, which = starts[order], ends[order], which[order]

    uniq = np.concatenate(([0], np.unique(ends)))
    prev = (np.searchsorted(uniq, starts, "right") - 1).tolist()
    slot = np.searchsorted(uniq, ends).tolist()
    lens = (ends - starts).tolist()
    best = [0] * len(uniq)
    pick = [-1] * len(uniq)
    last = 0
    for j, k in enumerate(slot):
        if k != last:
            best[k] = best[last]
            last = k
        cand = best[prev[j]] + lens[j]
        if cand > best[k]:
            best[k] = cand
            pick[k] = j

    st, wl = starts.tolist(), which.tolist()
    tiles = []
    k = len(uniq) - 1
    while k > 0:
        j = pick[k]
        if j < 0:
            k -= 1
            continue
        tiles.append((st[j], toks[wl[j]]))
        k = prev[j]
    tiles.reverse()
    return tiles


def _tile(data: bytes, scanner: TokenScanner) -> list[tuple[int, bytes]]:
    found = scanner.scan(data[:MAX_TILE_SCAN])
    if not found:
        return []

    n = sum(map(len, found.values()))
    return _tile_np(found) if n >= TILE_NP_MIN_MATCHES else _tile_py(found)


def tile(data: bytes, tokens) -> list[tuple[int, bytes]]:
    """Max-coverage tiling of ``data[:MAX_TILE_SCAN]``: ascending (offset, token)."""
    return _tile(data, _prepare(tuple(tokens))[1])


# ── edits: (data, tiles, i, toks, rng) -> bytes | None ───────────────────


def _replace(data, tiles, i, toks, rng):
    at, tok = tiles[i]
    new = toks[rng.randint(0, len(toks) - 1)]
    if new == tok:
        return None
    return data[:at] + new + data[at + len(tok) :]


def _duplicate(data, tiles, i, _toks, _rng):
    at, tok = tiles[i]
    end = at + len(tok)
    return data[:end] + tok + data[end:]


def _delete(data, tiles, i, _toks, _rng):
    at, tok = tiles[i]
    return data[:at] + data[at + len(tok) :]


def _swap(data, tiles, i, _toks, rng):
    j = rng.randint(0, len(tiles) - 1)
    if tiles[i][1] == tiles[j][1]:
        return None

    (lo, a), (hi, b) = sorted((tiles[i], tiles[j]))
    return data[:lo] + b + data[lo + len(a) : hi] + a + data[hi + len(b) :]


_APPLY = {
    Edit.REPLACE: _replace,
    Edit.DUPLICATE: _duplicate,
    Edit.DELETE: _delete,
    Edit.SWAP: _swap,
}


def token_tile(data: bytes, tokens, start: int, rng, max_len: int) -> bytes | None:
    """Tile *data* into *tokens*, then replace / duplicate / delete / swap one tile.

    Args:
        data: Input bytes.
        tokens: Dictionary / cmplog literals (str or bytes).
        start: Planned mutation site; the tile ending after it is edited.
        rng: ``RandPool`` (or ScriptedRng): window start (only past
            MAX_TILE_TOKENS), edit, then the REPLACE token / SWAP partner.
        max_len: Output ceiling.

    Returns:
        New bytes, or None: no tile, a no-op edit, or too long.
    """
    if not data or not tokens:
        return None

    toks, scanner = _prepare(_window(tokens, rng))
    if not toks:
        return None

    tiles = _tile(data, scanner)
    if not tiles:
        return None

    i = bisect_right([at + len(t) for at, t in tiles], start)
    i = 0 if i == len(tiles) else i
    out = _APPLY[_EDITS[rng.randint(0, len(_EDITS) - 1)]](data, tiles, i, toks, rng)
    if out is None or len(out) > max_len:
        return None
    return out
