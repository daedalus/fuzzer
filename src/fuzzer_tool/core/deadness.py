"""Sound static dead-function proofs (soksolv ``deadzone/deadzone.go``).

soksolv proves a region dead by searching a *relaxed* problem to
exhaustion: if even the relaxation cannot reach it, the real problem
cannot. Here the relaxation is an over-approximate call graph of the
binary, built from raw bytes without trusting the disassembler:

    edges   every byte offset decoded as call/jmp/jcc rel32 or jmp/jcc rel8;
            a target anywhere inside function G adds an edge to G
    roots   e_entry, main, .dynsym exports, code outside any function
            symbol, and every function whose address may be formed:
              RIP-relative disp32 (the byte before is a ModRM with mod=00,
              rm=101; immediate tails 0/1/2/4), any relocation (symbol +
              addend, or addend), and in ET_EXEC any 4/8-byte value in a
              loaded section

Indirect calls and libc callbacks can only reach address-formed functions,
which are roots, so a function the relaxation never reaches is DEAD: a
proof relative to two assumptions, (1) the symbol table's function ranges
are accurate (overlapping ranges -> UNKNOWN) and (2) code addresses are
never built by arithmetic beyond one such reference. Anything the analysis
cannot model (not ELF64 x86-64, malformed sections, SHT_REL) is UNKNOWN,
never DEAD. ALIVE means "not proven dead", not "reachable".

Byte scans run in bounded chunks (``_CHUNK``) so memory stays flat on
ffmpeg-size text.
"""

import struct
from collections import OrderedDict
from enum import Enum

import numpy as np

from fuzzer_tool.core.elf import _elf64_le, _shdrs_in_bounds, _sym_sections, func_table


class Verdict(Enum):
    """Static verdict for a function (or a block inside one)."""

    DEAD = "dead"  # proven unreachable
    ALIVE = "alive"  # not proven dead
    UNKNOWN = "unknown"  # analysis could not run; never cached


# Verdicts kept in a VerdictCache (LRU).
MAX_VERDICTS = 1 << 16

# ELF constants (spec).
_EM_X86_64 = 62
_ET_EXEC = 2
_SHT_PROGBITS = 1
_SHT_RELA = 4
_SHT_NOBITS = 8
_SHT_REL = 9
_SHF_ALLOC = 0x2
_SHF_EXECINSTR = 0x4
_STT_FUNC = 2
_STT_GNU_IFUNC = 10
_RELA_SIZE = 24
_SYM_SIZE = 24

# x86-64 encodings (spec).
_CALL_REL32 = 0xE8
_JMP_REL32 = 0xE9
_JMP_REL8 = 0xEB
_TWO_BYTE = 0x0F
_JCC_REL32 = range(0x80, 0x90)  # after 0x0F
_JCC_REL8 = range(0x70, 0x80)
_LOOP_REL8 = range(0xE0, 0xE4)  # loopne/loope/loop/jrcxz
_MODRM_RIP_MASK = 0xC7
_MODRM_RIP = 0x05
_RIP_TAILS = (0, 1, 2, 4)  # immediate bytes after a RIP-relative disp32

# Bytes per scan chunk, and the overlap a window may read past it.
_CHUNK = 1 << 22
_OVERLAP = 8

_REL8_OPS = np.zeros(256, dtype=bool)
_REL8_OPS[[_JMP_REL8, *_JCC_REL8, *_LOOP_REL8]] = True


def _u32(b: np.ndarray, k: int) -> np.ndarray:
    """Little-endian u32 at offsets 0..k-1 of *b* (len(b) >= k + 3)."""
    out = b[:k].astype(np.uint32)
    for s in (1, 2, 3):
        out |= b[s : s + k].astype(np.uint32) << np.uint32(8 * s)
    return out


def _i32(b: np.ndarray, k: int) -> np.ndarray:
    return _u32(b, k).view(np.int32).astype(np.int64)


def _sections(elf: bytes) -> list[tuple[int, int, int, int, int, int, int]] | None:
    """(type, flags, addr, off, size, link, entsize) per section; None if malformed."""
    e_shoff = struct.unpack_from("<Q", elf, 40)[0]
    e_shentsize, e_shnum = struct.unpack_from("<HH", elf, 58)
    if e_shnum == 0 or not _shdrs_in_bounds(len(elf), e_shoff, e_shnum, e_shentsize):
        return None
    out = []
    for i in range(e_shnum):
        sh = e_shoff + i * e_shentsize
        sh_type, sh_flags, addr, off, size, link = struct.unpack_from("<IQQQQI", elf, sh + 4)
        entsize = struct.unpack_from("<Q", elf, sh + 56)[0]
        has_bytes = sh_type != _SHT_NOBITS and size
        if has_bytes and (off > len(elf) or size > len(elf) - off):
            return None
        out.append((sh_type, sh_flags, addr, off, size, link, entsize))
    return out


class DeadProver:
    """Function-level reachability over the relaxed call graph of one ELF."""

    def __init__(self, elf: bytes):
        self._ok = False
        self._exec: list[tuple[int, int]] = []
        self._starts = np.zeros(0, dtype=np.uint64)
        self._ends = np.zeros(0, dtype=np.uint64)
        self._names: list[str] = []
        self._src = np.zeros(0, dtype=np.int64)
        self._dst = np.zeros(0, dtype=np.int64)
        self._reached = np.zeros(0, dtype=bool)
        self._analyse(elf)

    # ── queries ──────────────────────────────────────────────────────

    def verdict(self, addr: int) -> Verdict:
        """Verdict for the function holding *addr*."""
        if not self._ok:
            return Verdict.UNKNOWN
        i = self._func_index(addr)
        if i < 0:
            in_code = any(a <= addr < a + n for a, n in self._exec)
            return Verdict.ALIVE if in_code else Verdict.UNKNOWN
        return Verdict.ALIVE if self._reached[i] else Verdict.DEAD

    def _func_index(self, addr: int) -> int:
        idx = int(np.searchsorted(self._starts, np.uint64(addr), side="right")) - 1
        if idx < 0 or addr >= int(self._ends[idx]):
            return -1
        return idx

    def _map(self, addrs: np.ndarray) -> np.ndarray:
        """Function index per address; -1 outside every function."""
        a = addrs.astype(np.uint64, copy=False)
        idx = np.searchsorted(self._starts, a, side="right").astype(np.int64) - 1
        ok = idx >= 0
        ok[ok] &= a[ok] < self._ends[idx[ok]]
        return np.where(ok, idx, -1)

    def _direct_reach_from_main(self) -> np.ndarray:
        """The naive check: direct edges from main only (tests show it unsound)."""
        main = [i for i, n in enumerate(self._names) if n == "main"]
        return self._reach(np.array(main, dtype=np.int64))

    # ── analysis ─────────────────────────────────────────────────────

    def _analyse(self, elf: bytes) -> None:
        secs = self._modelled(elf)
        if secs is None:
            return

        roots: list[np.ndarray] = [self._named_roots(elf)]
        src, dst = [], []
        for s in secs:
            if not s[1] & _SHF_EXECINSTR or s[0] != _SHT_PROGBITS:
                continue
            for a, b, k in self._chunks(elf, s[2], s[3], s[4]):
                e_src, e_dst, r = self._scan_code(a, b, k)
                src.append(e_src)
                dst.append(e_dst)
                roots.append(r)
        roots.append(self._data_refs(elf, secs))
        roots.append(self._reloc_refs(elf, secs))

        self._src = np.concatenate(src) if src else self._src
        self._dst = np.concatenate(dst) if dst else self._dst
        self._reached = self._reach(np.unique(np.concatenate(roots)))
        self._ok = True

    def _modelled(self, elf: bytes) -> list | None:
        """Sections when the analysis can model this ELF; None means UNKNOWN."""
        if not _elf64_le(elf) or struct.unpack_from("<H", elf, 18)[0] != _EM_X86_64:
            return None
        secs = _sections(elf)
        ft = func_table(elf)
        if secs is None or ft is None or any(s[0] == _SHT_REL for s in secs):
            return None
        if not self._load_funcs(ft):
            return None
        self._exec = [(s[2], s[4]) for s in secs if s[1] & _SHF_EXECINSTR and s[4]]
        return secs if self._exec else None

    def _load_funcs(self, ft) -> bool:
        """Unique starts (aliases merged); False when ranges overlap."""
        starts = np.asarray(ft.starts, dtype=np.uint64)
        ends = np.asarray(ft.ends, dtype=np.uint64)
        uniq, first = np.unique(starts, return_index=True)
        top = np.zeros(uniq.size, dtype=np.uint64)
        np.maximum.at(top, np.searchsorted(uniq, starts), ends)
        if uniq.size > 1 and (top[:-1] > uniq[1:]).any():
            return False

        # A gap (size-0 CRT symbols, padding) joins the function before it:
        # merging ranges only adds edges, and keeps CRT bytes out of the
        # root region, where stray decodes would root their targets.
        top[:-1] = uniq[1:]
        self._starts, self._ends = uniq, top
        self._names = [ft.names[i] for i in first.tolist()]
        return True

    @staticmethod
    def _chunks(elf: bytes, addr: int, off: int, size: int):
        """(addr, bytes) windows of a section, overlapping by _OVERLAP."""
        whole = np.frombuffer(elf, dtype=np.uint8, count=size, offset=off)
        pad = np.zeros(_OVERLAP, dtype=np.uint8)
        for lo in range(0, size, _CHUNK):
            b = whole[lo : lo + _CHUNK + _OVERLAP]
            yield addr + lo, np.concatenate([b, pad]), min(_CHUNK, size - lo)

    def _scan_code(self, a: int, b: np.ndarray, k: int) -> tuple[np.ndarray, ...]:
        """Branch edges and address-formed roots in the first *k* bytes of *b*."""
        pos = np.arange(k, dtype=np.int64)
        op1 = b[1 : k + 1]
        rel32 = (b[:k] == _CALL_REL32) | (b[:k] == _JMP_REL32)
        jcc32 = (b[:k] == _TWO_BYTE) & (op1 >= _JCC_REL32.start) & (op1 < _JCC_REL32.stop)
        rel8 = _REL8_OPS[b[:k]]
        d1 = _i32(b[1:], k)
        d2 = _i32(b[2:], k)
        d8 = b[1 : k + 1].view(np.int8).astype(np.int64)

        src_pos = np.concatenate([pos[rel32], pos[jcc32], pos[rel8]])
        tgt = np.concatenate(
            [
                a + pos[rel32] + 5 + d1[rel32],
                a + pos[jcc32] + 6 + d2[jcc32],
                a + pos[rel8] + 2 + d8[rel8],
            ]
        )
        fs = self._map(a + src_pos)
        ft = self._map(tgt)
        hit = ft >= 0

        # Branches from code outside any function: their targets are roots.
        roots = [ft[hit & (fs < 0)]]
        edge = hit & (fs >= 0)

        # RIP-relative references: disp32 right after a mod=00 rm=101 ModRM.
        rip = (b[:k] & _MODRM_RIP_MASK) == _MODRM_RIP
        disp = d1[rip]
        base = a + pos[rip] + 5
        for tail in _RIP_TAILS:
            r = self._map(base + tail + disp)
            roots.append(r[r >= 0])
        return fs[edge], ft[edge], np.concatenate(roots)

    def _named_roots(self, elf: bytes) -> np.ndarray:
        """e_entry, main, .dynsym exports (IFUNC resolvers included)."""
        e_entry = struct.unpack_from("<Q", elf, 24)[0]
        roots = [self._map(np.array([e_entry], dtype=np.uint64))]
        roots.append(
            np.array([i for i, n in enumerate(self._names) if n == "main"], dtype=np.int64)
        )
        roots.append(self._dynsym_roots(elf))
        return np.concatenate([r[r >= 0] for r in roots])

    def _dynsym_roots(self, elf: bytes) -> np.ndarray:
        syms = _sym_sections(elf)
        if syms is None or syms[2] is None:
            return np.zeros(0, dtype=np.int64)
        off, size = struct.unpack_from("<QQ", elf, syms[2] + 24)
        values, info, shndx = _sym_fields(elf, off, size)
        typ = info & 0xF
        keep = ((typ == _STT_FUNC) | (typ == _STT_GNU_IFUNC)) & (shndx != 0) & (values > 0)
        return self._map(values[keep])

    def _data_refs(self, elf: bytes, secs) -> np.ndarray:
        """ET_EXEC: any 4/8-byte value in a loaded section naming a function."""
        if struct.unpack_from("<H", elf, 16)[0] != _ET_EXEC:
            return np.zeros(0, dtype=np.int64)
        out = []
        for s in secs:
            if not s[1] & _SHF_ALLOC or s[0] == _SHT_NOBITS or not s[4]:
                continue
            for _a, b, k in self._chunks(elf, *s[2:5]):
                lo = _u32(b, k).astype(np.uint64)
                hi = _u32(b[4:], k).astype(np.uint64)
                for v in (lo, lo | (hi << np.uint64(32))):
                    r = self._map(v)
                    out.append(r[r >= 0])
        return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)

    def _reloc_refs(self, elf: bytes, secs) -> np.ndarray:
        """Every RELA addend, and symbol value + addend, naming a function."""
        out = []
        for s in secs:
            if s[0] != _SHT_RELA or not s[4]:
                continue
            n = s[4] // _RELA_SIZE
            rela = np.frombuffer(elf, dtype="<u8", count=n * 3, offset=s[3]).reshape(n, 3)
            addend = rela[:, 2]
            symi = (rela[:, 1] >> np.uint64(32)).astype(np.int64)
            cands = [addend]
            if 0 <= s[5] < len(secs) and secs[s[5]][4]:
                link = secs[s[5]]
                values, _, _ = _sym_fields(elf, link[3], link[4])
                ok = symi < values.size
                cands.append(values[symi[ok]] + addend[ok])
            for v in cands:
                r = self._map(v)
                out.append(r[r >= 0])
        return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)

    def _reach(self, roots: np.ndarray) -> np.ndarray:
        """Functions reachable from *roots* (level-synchronous BFS on sorted edges).

        Self-contained: horizon's CSR helpers import icfg, which imports this
        module's caller (analyzer_distance).
        """
        n = self._starts.size
        seen = np.zeros(n, dtype=bool)
        if not n or not roots.size:
            return seen
        order = np.argsort(self._src, kind="stable")
        src, dst = self._src[order], self._dst[order]
        seen[roots] = True
        frontier = np.unique(roots)
        while frontier.size:
            lo = np.searchsorted(src, frontier, side="left")
            hi = np.searchsorted(src, frontier, side="right")
            take = np.repeat(lo, hi - lo) + _ramp(hi - lo)
            kids = np.unique(dst[take])
            kids = kids[~seen[kids]]
            seen[kids] = True
            frontier = kids
        return seen


def _ramp(counts: np.ndarray) -> np.ndarray:
    """0..c-1 for each count c, concatenated."""
    total = int(counts.sum())
    starts = np.cumsum(counts) - counts
    return np.arange(total, dtype=np.int64) - np.repeat(starts, counts)


def _sym_fields(elf: bytes, off: int, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(st_value, st_info, st_shndx) arrays of one symbol table."""
    n = size // _SYM_SIZE
    dt = np.dtype(
        [("name", "<u4"), ("info", "u1"), ("other", "u1"), ("shndx", "<u2"),
         ("value", "<u8"), ("size", "<u8")]
    )  # fmt: skip
    syms = np.frombuffer(elf, dtype=dt, count=n, offset=off)
    return syms["value"].astype(np.uint64), syms["info"], syms["shndx"]


class VerdictCache:
    """Bounded LRU of address -> DEAD/ALIVE for one binary build.

    UNKNOWN is never stored: an unknown must be re-derived, not remembered.
    State is tied to *identity* (build id, or size+mtime) so a rebuilt
    target never inherits proofs about the old one.
    """

    def __init__(self, identity: bytes, cap: int = MAX_VERDICTS):
        self.identity = identity
        self.cap = cap
        self._v: OrderedDict[int, Verdict] = OrderedDict()

    def get(self, addr: int) -> Verdict | None:
        v = self._v.get(addr)
        if v is not None:
            self._v.move_to_end(addr)
        return v

    def put(self, addr: int, verdict: Verdict) -> None:
        """
        Raises:
            ValueError: *verdict* is UNKNOWN.
        """
        if verdict is Verdict.UNKNOWN:
            raise ValueError("UNKNOWN verdicts are never cached")
        self._v[addr] = verdict
        self._v.move_to_end(addr)
        while len(self._v) > self.cap:
            self._v.popitem(last=False)

    def state_dict(self) -> dict:
        return {"identity": self.identity, "verdicts": {a: v.value for a, v in self._v.items()}}

    def load_state_dict(self, state: dict) -> None:
        """Restore; ignored for another build, UNKNOWN entries dropped."""
        if state.get("identity") != self.identity:
            return
        for addr, value in state.get("verdicts", {}).items():
            if value in (Verdict.DEAD.value, Verdict.ALIVE.value):
                self.put(int(addr), Verdict(value))
