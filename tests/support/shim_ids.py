"""Python mirror of the C shim's location mapping (afl_shim.c, "Guard numbering").

Tests that compile a harness calling ``__afl_map_edge(<literal>)`` used to
assert the literal back as the edge id. The shim now mixes hand-written ids
into the same hashed location space as trace-pc-guard ids -- fed in raw,
small sequential constants aliased (``prev >> 1`` drops bit 0 of the
previous id, and XORs of neighbouring constants collide) -- so expected ids
have to be computed. Keep this byte-for-byte in step with the C.
"""

from __future__ import annotations

_M64 = (1 << 64) - 1
#: __afl_loc_mask before any guard_init widens it (manual-only builds).
MANUAL_ONLY_MASK = 0xFFFF
_MANUAL_SALT = 0x6A09E667F3BCC909


def guard_mix(x: int) -> int:
    """splitmix64 finalizer, truncated to 32 bits (``__afl_guard_mix``)."""
    x = (x + 0x9E3779B97F4A7C15) & _M64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _M64
    return (x ^ (x >> 31)) & 0xFFFFFFFF


def manual_loc(cur_loc: int, mask: int = MANUAL_ONLY_MASK) -> int:
    """The location ``__afl_map_edge(cur_loc)`` actually records."""
    v = guard_mix((cur_loc ^ _MANUAL_SALT) & _M64) & mask
    return v or 1


def edge_chain(manual_ids: list[int], mask: int = MANUAL_ONLY_MASK) -> list[int]:
    """Context-free edge ids for consecutive manual calls after a reset."""
    out, prev = [], 0
    for cur in manual_ids:
        loc = manual_loc(cur, mask)
        out.append(((prev ^ loc) & 0xFFFFFFFF) or 1)
        prev = loc >> 1
    return out
