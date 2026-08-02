"""Configurable CRC-32 computation.

This module is the **single point of dispatch** for all CRC-32 operations
in the fuzzer.  Format mutators (``png.py``, ``zip.py``, ``gzip.py``),
the SMT solver, and the seed picker all import from here instead of
calling ``zlib.crc32`` directly.

When the fuzzer has recovered a non-standard checksum polynomial via
``ChecksumLearner``, the recovered model is activated and used
automatically.  Otherwise the standard ``zlib.crc32`` (polynomial
``0xEDB88320``, reflected I/O, init/final XOR ``0xFFFFFFFF``) is used,
so the hot path is unchanged.

A *model* is ``(poly, width, init, final_xor, reflect_in, reflect_out)``
— the full configuration of the checksum function.  The active model is
set via :func:`set_active_model` (full control) or :func:`set_active_poly`
(GCD-domain shorthand: non-reflected, ``init=0``, ``final_xor=0``, which
is the convention recovered by GCD-of-syndromes).  The fuzzer calls these
when recovery succeeds or state is restored.
"""

from __future__ import annotations

import threading
import zlib

# Standard CRC-32 (reversed) polynomial — matches zlib.crc32.
_STANDARD_POLY = 0xEDB88320

# Exact configuration of zlib.crc32: reflected I/O, init/final XOR
# 0xFFFFFFFF, no output reversal.  Models equal to this shortcut to the
# hardware-accelerated implementation.
_STANDARD_MODEL = (_STANDARD_POLY, 32, 0xFFFFFFFF, 0xFFFFFFFF, True, False)

# Module-level active model: (poly, width, init, final_xor, reflect_in,
# reflect_out).  poly=None means "no recovered polynomial — use zlib.crc32".
# Protected by a lock because recovery runs in the main fuzz loop while
# mutations may read it from worker threads.
_active_model: tuple[int | None, int, int, int, bool, bool] = (None, 32, 0, 0, False, False)
_poly_lock = threading.Lock()


def set_active_model(
    poly: int | None,
    width: int = 32,
    init: int = 0,
    final_xor: int = 0,
    reflect_in: bool = False,
    reflect_out: bool = False,
) -> None:
    """Set the active checksum model (full configuration).

    Call this from the fuzzer when recovery succeeds or state is
    restored from ``state.json``.
    """
    with _poly_lock:
        global _active_model
        _active_model = (poly, width, init, final_xor, reflect_in, reflect_out)


def set_active_poly(poly: int | None) -> None:
    """Set the active checksum polynomial (GCD-domain shorthand).

    Equivalent to ``set_active_model(poly, 32, 0, 0, False, False)`` —
    the non-reflected convention recovered by GCD-of-syndromes.
    """
    set_active_model(poly, width=32, init=0, final_xor=0, reflect_in=False, reflect_out=False)


def get_active_model() -> tuple[int | None, int, int, int, bool, bool]:
    """Return the active ``(poly, width, init, final_xor, reflect_in, reflect_out)``."""
    with _poly_lock:
        return _active_model


def get_active_poly() -> int | None:
    """Return the active polynomial, or ``None`` for the standard zlib CRC-32."""
    with _poly_lock:
        return _active_model[0]


def crc32(data: bytes, poly: int | None = None) -> int:
    """Compute a CRC-32 over *data* using the active checksum model.

    Args:
        data: Input bytes.
        poly: Optional explicit polynomial in the GCD domain (non-reflected,
            ``init=0``, ``final_xor=0``).  When ``None`` the module's
            active model is used.

    Returns:
        CRC-32 as an unsigned 32-bit integer.
    """
    model = (poly, 32, 0, 0, False, False) if poly is not None else get_active_model()
    epoly, width, init, final_xor, reflect_in, reflect_out = model
    if epoly is None or model == _STANDARD_MODEL:
        return zlib.crc32(data) & 0xFFFFFFFF

    # Non-standard model: software LFSR implementation.
    from fuzzer_tool.core.berlekamp_massey import compute_checksum

    return compute_checksum(
        data,
        poly=epoly,
        width=width,
        init=init,
        final_xor=final_xor,
        reflect_in=reflect_in,
        reflect_out=reflect_out,
    )


def crc32_ieee(data: bytes) -> int:
    """Standard IEEE 802.3 CRC-32 (same as ``zlib.crc32``).

    Kept as an explicit named alias for callers that want to be clear
    about using the standard polynomial regardless of any learned one.
    """
    return zlib.crc32(data) & 0xFFFFFFFF
