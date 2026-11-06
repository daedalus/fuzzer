"""TL-UL bus instruction mutator for hardware fuzzing harnesses.

Input = a stream of Hardware Fuzzing instructions (Trippel et al., "Fuzzing
Hardware Like Software", 2021). Encoding matches the hw-fuzzing reference
defaults: constant opcodes, variable frames, little-endian.

    wait   01                     1 byte   advance the clock one period
    read   02 AAAAAAAA            5 bytes  TL-UL Get
    write  03 AAAAAAAA DDDDDDDD   9 bytes  TL-UL PutFullData

Byte mutators break the frame boundaries; these edits keep the stream
decodable and work on whole bus transactions:

    mode          edit                                      why
    ------------  ----------------------------------------  ---------------------
    insert_instr  new wait/read/write at a boundary         grow the sequence
    delete_instr  drop one instruction                      skip a config step
    dup_instr     repeat one instruction                    FIFO pushes, polling
    swap_instr    swap two neighbours                       config/trigger order
    addr_swap     address := seen register or edge          other CSR, OOB window
    addr_step     address += +-4 / 1..3 / +-0x40 / 0x1000   neighbour, misaligned
    data_edge     write data := 0, ~0, 1<<31, 0x55.., ...   CSR field boundaries
    data_flip     flip one write-data bit                   single CSR field bit
    wait_run      insert N waits                            let the core finish
    rw_flip       read <-> write                            read-only/write-only

Returns None to decline when the input is not a stream, the mode has no
target instruction, the edit is a no-op or the output is over budget.
"""

import struct
from array import array
from enum import IntEnum


class TlulOp(IntEnum):
    """Constant opcode encoding (0 and 4..255 are invalid)."""

    WAIT = 1
    READ = 2
    WRITE = 3


OPS = (TlulOp.WAIT, TlulOp.READ, TlulOp.WRITE)

# Frame length per opcode byte; 0 marks an invalid opcode.
_FRAME_SIZES = {TlulOp.WAIT: 1, TlulOp.READ: 5, TlulOp.WRITE: 9}
_FRAME_LEN = bytes(_FRAME_SIZES.get(b, 0) for b in range(256))

_ADDR = struct.Struct("<I")
_READ = struct.Struct("<BI")
_WRITE = struct.Struct("<BII")
_WAIT = bytes((TlulOp.WAIT,))
_U32_MASK = 0xFFFFFFFF
_U32_BITS = 32

# OpenTitan IP register windows start at 0 and are a few KiB wide.
ADDR_EDGES = (0x0, 0x4, 0x10, 0x100, 0xFFC, 0x1000, 0x7FFFFFFC, 0xFFFFFFFC, 0xFFFFFFFF)
ADDR_DELTAS = (4, -4, 1, 2, 3, 0x40, -0x40, 0x1000)
DATA_EDGES = (
    0x0,
    0x1,
    0x7FFFFFFF,
    0x80000000,
    0xFFFFFFFF,
    0x55555555,
    0xAAAAAAAA,
    0x0000FFFF,
    0xFFFF0000,
)

# Paper: AES 12-16 cycles/block, KMAC 72, HMAC 80 (2 waits = 1 clock edge pair).
WAIT_RUNS = (2, 16, 80, 160, 512)

# Sniff at most this many bytes; mutate re-parses the whole input.
_SNIFF_WINDOW = 256


def frames(data: bytes) -> array | None:
    """Instruction start offsets, or None unless *data* is exactly a stream.

    e.g. ``03 <a> <d> 01 02 <a>`` -> ``[0, 9, 10]``.
    """
    n = len(data)
    if not n:
        return None

    offs = array("l")
    off = 0
    while off < n:
        size = _FRAME_LEN[data[off]]
        if not size:
            return None
        offs.append(off)
        off += size

    return offs if off == n else None


def sniff_tlul(data: bytes) -> bool:
    """Prefix decodes as instructions and holds at least one bus op."""
    n = len(data)
    window = min(n, _SNIFF_WINDOW)
    bus = False
    off = 0
    while off < window:
        op = data[off]
        size = _FRAME_LEN[op]
        if not size:
            return False
        bus = bus or op != TlulOp.WAIT
        off += size

    return bus and off <= n


# ── Frame helpers ──


def _end(data: bytes, offs: array, i: int) -> int:
    return offs[i + 1] if i + 1 < len(offs) else len(data)


def _cut(data: bytes, offs: array, i: int) -> bytes:
    return data[offs[i] : _end(data, offs, i)]


def _replace(data: bytes, offs: array, i: int, new: bytes) -> bytes:
    return data[: offs[i]] + new + data[_end(data, offs, i) :]


def _bus(data: bytes, offs: array) -> list[int]:
    return [i for i, off in enumerate(offs) if data[off] != TlulOp.WAIT]


def _writes(data: bytes, offs: array) -> list[int]:
    return [i for i, off in enumerate(offs) if data[off] == TlulOp.WRITE]


def _addr(data: bytes, off: int) -> int:
    return _ADDR.unpack_from(data, off + 1)[0]


def _addr_pool(data: bytes, offs: array) -> tuple[int, ...]:
    """Addresses the seed already uses (order kept, deduped), then edges."""
    known = dict.fromkeys(_addr(data, offs[i]) for i in _bus(data, offs))
    return (*known, *ADDR_EDGES)


def _instr(op: int, addr: int, value: int) -> bytes:
    if op == TlulOp.WAIT:
        return _WAIT
    if op == TlulOp.READ:
        return _READ.pack(op, addr)
    return _WRITE.pack(op, addr, value)


def _set_addr(data: bytes, off: int, addr: int) -> bytes:
    return data[: off + 1] + _ADDR.pack(addr & _U32_MASK) + data[off + 5 :]


def _set_data(data: bytes, off: int, value: int) -> bytes:
    return data[: off + 5] + _ADDR.pack(value) + data[off + 9 :]


# ── Modes ──


def insert_instr(data: bytes, offs: array, rng) -> bytes:
    at = rng.choice((*offs, len(data)))
    op = rng.choice(OPS)
    if op == TlulOp.WAIT:
        return data[:at] + _WAIT + data[at:]

    addr = rng.choice(_addr_pool(data, offs))
    value = rng.choice(DATA_EDGES) if op == TlulOp.WRITE else 0
    return data[:at] + _instr(op, addr, value) + data[at:]


def delete_instr(data: bytes, offs: array, rng) -> bytes | None:
    if len(offs) < 2:
        return None
    return _replace(data, offs, rng.choice(range(len(offs))), b"")


def dup_instr(data: bytes, offs: array, rng) -> bytes:
    i = rng.choice(range(len(offs)))
    return _replace(data, offs, i, _cut(data, offs, i) * 2)


def swap_instr(data: bytes, offs: array, rng) -> bytes | None:
    if len(offs) < 2:
        return None

    i = rng.choice(range(len(offs) - 1))
    pair = _cut(data, offs, i + 1) + _cut(data, offs, i)
    return data[: offs[i]] + pair + data[_end(data, offs, i + 1) :]


def addr_swap(data: bytes, offs: array, rng) -> bytes | None:
    bus = _bus(data, offs)
    if not bus:
        return None

    off = offs[rng.choice(bus)]
    return _set_addr(data, off, rng.choice(_addr_pool(data, offs)))


def addr_step(data: bytes, offs: array, rng) -> bytes | None:
    bus = _bus(data, offs)
    if not bus:
        return None

    off = offs[rng.choice(bus)]
    return _set_addr(data, off, _addr(data, off) + rng.choice(ADDR_DELTAS))


def data_edge(data: bytes, offs: array, rng) -> bytes | None:
    writes = _writes(data, offs)
    if not writes:
        return None
    return _set_data(data, offs[rng.choice(writes)], rng.choice(DATA_EDGES))


def data_flip(data: bytes, offs: array, rng) -> bytes | None:
    writes = _writes(data, offs)
    if not writes:
        return None

    off = offs[rng.choice(writes)]
    value = _ADDR.unpack_from(data, off + 5)[0]
    return _set_data(data, off, value ^ (1 << rng.randint(0, _U32_BITS - 1)))


def wait_run(data: bytes, offs: array, rng) -> bytes:
    at = rng.choice((*offs, len(data)))
    return data[:at] + _WAIT * rng.choice(WAIT_RUNS) + data[at:]


def rw_flip(data: bytes, offs: array, rng) -> bytes | None:
    bus = _bus(data, offs)
    if not bus:
        return None

    i = rng.choice(bus)
    off = offs[i]
    addr = _addr(data, off)

    # write -> read drops the data word; read -> write gains an edge value.
    if data[off] == TlulOp.WRITE:
        return _replace(data, offs, i, _instr(TlulOp.READ, addr, 0))
    return _replace(data, offs, i, _instr(TlulOp.WRITE, addr, rng.choice(DATA_EDGES)))


MODES = (
    insert_instr,
    delete_instr,
    dup_instr,
    swap_instr,
    addr_swap,
    addr_step,
    data_edge,
    data_flip,
    wait_run,
    rw_flip,
)


def tlul_mutate(data: bytes, rng, max_len: int) -> bytes | None:
    """Apply one instruction-level edit; None when it cannot apply."""
    offs = frames(data)
    if offs is None:
        return None

    out = rng.choice(MODES)(data, offs, rng)
    if out is None or out == data or len(out) > max_len:
        return None
    return out
