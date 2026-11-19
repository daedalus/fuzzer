"""Byte and object mutators, crash keys and crash files.

All randomness comes from a caller-owned ``RandPool`` (Hard Rule 16), so a
campaign seed reproduces every case exactly.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fuzzer_tool.adapters.filesystem import hash_data_crypto

if TYPE_CHECKING:
    from collections.abc import Callable

    from fuzzer_tool.core.rand_pool import RandPool

# Bounded memory (Hard Rule 54): no mutant grows past this.
MAX_MUTANT_LEN = 1 << 20
MAX_SPAN = 64
MAX_STACK = 8
MAX_OBJ_DEPTH = 6
PACKAGE_MARKER = "fuzzer_tool"

# Boundary integers: sign, width and size-field edges.
INTERESTING_INTS = (
    0,
    1,
    -1,
    0x7F,
    0x80,
    0xFF,
    0x7FFF,
    0xFFFF,
    0x7FFFFFFF,
    0xFFFFFFFF,
    -(1 << 31),
    (1 << 63) - 1,
    1 << 64,
    10**5000,
)
INTERESTING_OBJS: tuple[Any, ...] = (None, "", "zz", b"", [], {}, 1.5, float("nan"), True)


# ── bytes ────────────────────────────────────────────────────────────────────


def _span(rng: RandPool, n: int) -> tuple[int, int]:
    """Random [a, b) inside n bytes, at most MAX_SPAN long."""
    a = rng.randrange(n)
    return a, min(n, a + rng.randint(1, MAX_SPAN))


def _flip(rng: RandPool, buf: bytearray) -> None:
    """Overwrite one byte."""
    buf[rng.randrange(len(buf))] = rng.randrange(256)


def _int_word(rng: RandPool, buf: bytearray) -> None:
    """Overwrite a 1/2/4/8-byte little-endian word with a boundary value."""
    width = rng.choice((1, 2, 4, 8))
    pos = rng.randrange(len(buf))
    val = rng.choice(INTERESTING_INTS) & ((1 << (8 * width)) - 1)
    buf[pos : pos + width] = val.to_bytes(width, "little")


def _delete(rng: RandPool, buf: bytearray) -> None:
    """Delete a span."""
    a, b = _span(rng, len(buf))
    del buf[a:b]


def _dup(rng: RandPool, buf: bytearray) -> None:
    """Repeat a span a few times."""
    a, b = _span(rng, len(buf))
    buf[b:b] = buf[a:b] * rng.randint(1, 8)


def _truncate(rng: RandPool, buf: bytearray) -> None:
    """Cut at a random position (headers without bodies)."""
    del buf[rng.randrange(len(buf)) :]


def _insert(rng: RandPool, buf: bytearray) -> None:
    """Insert random bytes."""
    pos = rng.randrange(len(buf))
    buf[pos:pos] = rng.randbytes(rng.randint(1, MAX_SPAN))


_BYTE_OPS: tuple[Callable[[RandPool, bytearray], None], ...] = (
    _flip,
    _flip,
    _int_word,
    _int_word,
    _delete,
    _dup,
    _truncate,
    _insert,
)


def mutate_bytes(rng: RandPool, data: bytes) -> bytes:
    """Stack 1..MAX_STACK byte mutations onto *data*."""
    buf = bytearray(data)
    for _ in range(rng.randint(1, MAX_STACK)):
        # Empty buffers only grow; every other op needs a byte to aim at.
        if not buf:
            buf += rng.randbytes(rng.randint(1, MAX_SPAN))
            continue

        rng.choice(_BYTE_OPS)(rng, buf)

    return bytes(buf[:MAX_MUTANT_LEN])


# ── objects (persisted-state shapes) ─────────────────────────────────────────


def _leaf(rng: RandPool, value: Any) -> Any:
    """Replace a scalar with a boundary or wrong-typed value."""
    if isinstance(value, int) and not isinstance(value, bool) and rng.random() < 0.5:
        return rng.choice(INTERESTING_INTS)
    return rng.choice(INTERESTING_OBJS + INTERESTING_INTS)


def _mutate_dict(rng: RandPool, value: dict, depth: int) -> dict:
    """Mutate one entry: recurse, drop it, or add a junk key."""
    out = dict(value)
    roll = rng.random()
    if out and roll < 0.6:
        key = rng.choice(list(out))
        out[key] = mutate_obj(rng, out[key], depth + 1)
    elif out and roll < 0.8:
        del out[rng.choice(list(out))]
    else:
        out[rng.choice(("zz", "-1", "", "ff" * 3))] = _leaf(rng, 0)
    return out


def _mutate_list(rng: RandPool, value: list, depth: int) -> list:
    """Mutate one item, or append a wrong-typed one."""
    out = list(value)
    if out and rng.random() < 0.7:
        i = rng.randrange(len(out))
        out[i] = mutate_obj(rng, out[i], depth + 1)
        return out

    out.append(_leaf(rng, 0))
    return out


def mutate_obj(rng: RandPool, value: Any, depth: int = 0) -> Any:
    """Structurally mutate a pickle-able state object (dict/list/scalars)."""
    if depth > MAX_OBJ_DEPTH:
        return _leaf(rng, value)
    if isinstance(value, dict):
        return _mutate_dict(rng, value, depth)
    if isinstance(value, list | tuple | set):
        return _mutate_list(rng, list(value), depth)
    return _leaf(rng, value)


# ── crash bookkeeping ────────────────────────────────────────────────────────


def crash_key(exc: BaseException) -> str:
    """Dedup key: exception type + innermost fuzzer_tool frame."""
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if PACKAGE_MARKER in f.filename]
    where = f"{Path(frames[-1].filename).name}:{frames[-1].lineno}" if frames else "?"
    return f"{type(exc).__name__}@{where}"


def save_crash(crash_dir: Path, target: str, ext: str, data: bytes) -> Path:
    """Write ``<target>__<hash>.<ext>``; the name is what replay dispatches on."""
    crash_dir.mkdir(parents=True, exist_ok=True)
    path = crash_dir / f"{target}__{hash_data_crypto(data)[:12]}.{ext}"
    path.write_bytes(data)
    return path
