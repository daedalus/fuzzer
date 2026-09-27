"""Pure-Python DWARF line-table parser: resolve ``file.c:line`` → addresses.

Implements just enough of DWARF 4 and DWARF 5 to walk the line-number
program of each compilation unit:

  - .debug_info CU DIEs (via .debug_abbrev) for DW_AT_stmt_list,
    DW_AT_comp_dir, DW_AT_name, DW_AT_addr_size.
  - .debug_line line programs: standard/special/extended opcodes,
    include-directory and file tables (v4 inline strings, v5
    DW_LNCT/DW_FORM-encoded entries), address-size handling.

Compressed DWARF sections (SHF_COMPRESSED, zlib) are decompressed.
Only 32-bit DWARF (standard clang output) is supported.

The result is a mapping (basename → line → sorted addresses) used by
``TargetDistance`` to turn AFLGo-style ``file.c:123`` targets into
concrete addresses. A source line can legitimately map to several
addresses (inlined code); all are returned and treated as targets.
"""

import logging
import struct
import zlib
from pathlib import Path
from typing import NamedTuple

log = logging.getLogger(__name__)

# ── DWARF constants ────────────────────────────────────────────────────

_DW_TAG_compile_unit = 0x11
_DW_TAG_skeleton_unit = 0x4A

_DW_AT_name = 0x03
_DW_AT_stmt_list = 0x10
_DW_AT_comp_dir = 0x1B
_DW_AT_addr_size = 0x57

_DW_FORM_data1 = 0x0B
_DW_FORM_data2 = 0x05
_DW_FORM_data4 = 0x06
_DW_FORM_data8 = 0x07
_DW_FORM_string = 0x08
_DW_FORM_udata = 0x0F
_DW_FORM_strp = 0x0E
_DW_FORM_sec_offset = 0x17
_DW_FORM_line_strp = 0x1F
_DW_FORM_strx = 0x1A
_DW_FORM_strx1 = 0x25
_DW_FORM_strx2 = 0x26
_DW_FORM_strx3 = 0x27
_DW_FORM_strx4 = 0x28
_DW_FORM_addrx = 0x1B
_DW_FORM_addrx1 = 0x29
_DW_FORM_sdata = 0x0D
_DW_FORM_flag = 0x0C
_DW_FORM_flag_present = 0x19
_DW_FORM_exprloc = 0x18
_DW_FORM_ref1 = 0x11
_DW_FORM_ref2 = 0x12
_DW_FORM_ref4 = 0x13
_DW_FORM_ref8 = 0x14
_DW_FORM_ref_udata = 0x15
_DW_FORM_ref_addr = 0x10
_DW_FORM_ref_sig8 = 0x20
_DW_FORM_block = 0x09
_DW_FORM_block1 = 0x0A
_DW_FORM_block2 = 0x03
_DW_FORM_block4 = 0x04
_DW_FORM_data16 = 0x1E
_DW_FORM_addr = 0x01
_DW_FORM_implicit_const = 0x21

_DW_LNS_copy = 1
_DW_LNS_advance_pc = 2
_DW_LNS_advance_line = 3
_DW_LNS_set_file = 4
_DW_LNS_set_column = 5
_DW_LNS_negate_stmt = 6
_DW_LNS_set_basic_block = 7
_DW_LNS_const_add_pc = 8
_DW_LNS_fixed_advance_pc = 9
_DW_LNS_set_prologue_end = 10
_DW_LNS_set_epilogue_begin = 11
_DW_LNS_set_isa = 12

_DW_LNE_end_sequence = 1
_DW_LNE_set_address = 2
_DW_LNE_define_file = 3
_DW_LNE_set_discriminator = 4

_DW_LNCT_path = 1
_DW_LNCT_directory_index = 2

_SHF_COMPRESSED = 0x800


# ── LEB128 readers ─────────────────────────────────────────────────────


def _uleb(data: bytes, off: int) -> tuple[int, int]:
    """Read an unsigned LEB128 at *off*, return (value, new_offset)."""
    result = 0
    shift = 0
    while True:
        if off >= len(data):
            raise ValueError("ULEB128 truncated")
        b = data[off]
        off += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, off
        shift += 7
        if shift > 63:
            raise ValueError("ULEB128 too long")


def _sleb(data: bytes, off: int) -> tuple[int, int]:
    """Read a signed LEB128 at *off*, return (value, new_offset)."""
    result = 0
    shift = 0
    while True:
        if off >= len(data):
            raise ValueError("SLEB128 truncated")
        b = data[off]
        off += 1
        result |= (b & 0x7F) << shift
        shift += 7
        if not (b & 0x80):
            if b & 0x40:  # sign extend
                result -= 1 << shift
            return result, off
        if shift > 63:
            raise ValueError("SLEB128 too long")


def _cstr(data: bytes, off: int) -> tuple[bytes, int]:
    """Read a null-terminated string at *off*."""
    end = data.find(b"\x00", off)
    if end < 0:
        raise ValueError("unterminated string")
    return data[off:end], end + 1


# ── ELF section extraction ─────────────────────────────────────────────


def _elf_sections(elf_data: bytes) -> dict[bytes, tuple[int, int, int]]:
    """Map section name → (file_offset, size, flags), decompressing
    SHF_COMPRESSED sections. Returns {} for non-ELF or malformed input."""
    if len(elf_data) < 64 or elf_data[:4] != b"\x7fELF" or elf_data[4] != 2:
        return {}
    try:
        e_shoff = struct.unpack_from("<Q", elf_data, 40)[0]
        e_shentsize = struct.unpack_from("<H", elf_data, 58)[0]
        e_shnum = struct.unpack_from("<H", elf_data, 60)[0]
        e_shstrndx = struct.unpack_from("<H", elf_data, 62)[0]
        if e_shnum == 0 or e_shstrndx >= e_shnum:
            return {}
        shstr = e_shoff + e_shstrndx * e_shentsize
        shstr_off = struct.unpack_from("<Q", elf_data, shstr + 24)[0]
        sections = {}
        for i in range(e_shnum):
            sh = e_shoff + i * e_shentsize
            name_off = struct.unpack_from("<I", elf_data, sh)[0]
            name = elf_data[shstr_off + name_off :].split(b"\x00", 1)[0]
            flags = struct.unpack_from("<Q", elf_data, sh + 8)[0]
            offset = struct.unpack_from("<Q", elf_data, sh + 24)[0]
            size = struct.unpack_from("<Q", elf_data, sh + 32)[0]
            data = elf_data[offset : offset + size]
            if flags & _SHF_COMPRESSED and len(data) >= 24:
                ch_type, ch_size = struct.unpack_from("<IQ", data, 0)
                if ch_type == 1:  # ELFCOMPRESS_ZLIB
                    data = zlib.decompress(data[24 : 24 + (size - 24)])[:ch_size]
            sections[name] = data
        return sections
    except Exception:
        log.debug("ELF section parse failed", exc_info=True)
        return {}


# ── CU DIE parsing (for stmt_list / comp_dir / addr_size) ─────────────


def _str_at(sections: dict[bytes, bytes], sec: bytes, strp: int) -> str:
    """NUL-terminated string at *strp* in section *sec*; "" when out of range."""
    dbg_str = sections.get(sec, b"")
    if strp < len(dbg_str):
        return dbg_str[strp:].split(b"\x00", 1)[0].decode(errors="replace")
    return ""


# Attribute readers: (sections, data, off, addr_size, str_base) → (value, new_off).
# Truncated input raises (IndexError / struct.error / ValueError) like the
# underlying reads; callers catch.


def _rd_string(sections, data, off, addr_size, str_base):
    s, off = _cstr(data, off)
    return s.decode(errors="replace"), off


def _rd_strp(sections, data, off, addr_size, str_base):
    strp = struct.unpack_from("<I", data, off)[0]
    return _str_at(sections, b".debug_str", strp), off + 4


def _rd_line_strp(sections, data, off, addr_size, str_base):
    strp = struct.unpack_from("<I", data, off)[0]
    return _str_at(sections, b".debug_line_str", strp), off + 4


# DW_FORM_strx* index readers: (data, off) → (index, new_off)
_STRX_INDEX = {
    _DW_FORM_strx: _uleb,
    _DW_FORM_strx1: lambda data, off: (data[off], off + 1),
    _DW_FORM_strx2: lambda data, off: (struct.unpack_from("<H", data, off)[0], off + 2),
    _DW_FORM_strx3: lambda data, off: (int.from_bytes(data[off : off + 3], "little"), off + 3),
    _DW_FORM_strx4: lambda data, off: (struct.unpack_from("<I", data, off)[0], off + 4),
}


def _rd_strx(sections, data, off, addr_size, str_base, form=_DW_FORM_strx):
    """Indexed string: index into .debug_str_offsets → .debug_str."""
    idx, off = _STRX_INDEX[form](data, off)
    str_offsets = sections.get(b".debug_str_offsets", b"")
    entry = str_base + idx * 4
    if entry + 4 > len(str_offsets):
        return "", off
    strp = struct.unpack_from("<I", str_offsets, entry)[0]
    return _str_at(sections, b".debug_str", strp), off


def _rd_uleb(sections, data, off, addr_size, str_base):
    return _uleb(data, off)


def _rd_sleb(sections, data, off, addr_size, str_base):
    return _sleb(data, off)


def _rd_flag_present(sections, data, off, addr_size, str_base):
    return True, off


def _rd_flag(sections, data, off, addr_size, str_base):
    return bool(data[off]), off + 1


def _rd_u8(sections, data, off, addr_size, str_base):
    return data[off], off + 1


def _rd_u16(sections, data, off, addr_size, str_base):
    return struct.unpack_from("<H", data, off)[0], off + 2


def _rd_u32(sections, data, off, addr_size, str_base):
    return struct.unpack_from("<I", data, off)[0], off + 4


def _rd_u64(sections, data, off, addr_size, str_base):
    return struct.unpack_from("<Q", data, off)[0], off + 8


def _rd_uleb_block(sections, data, off, addr_size, str_base):
    length, off = _uleb(data, off)
    return None, off + length


def _rd_block1(sections, data, off, addr_size, str_base):
    return None, off + 1 + data[off]


def _rd_block2(sections, data, off, addr_size, str_base):
    return None, off + 2 + struct.unpack_from("<H", data, off)[0]


def _rd_block4(sections, data, off, addr_size, str_base):
    return None, off + 4 + struct.unpack_from("<I", data, off)[0]


def _rd_data16(sections, data, off, addr_size, str_base):
    return data[off : off + 16], off + 16


def _rd_addr(sections, data, off, addr_size, str_base):
    return int.from_bytes(data[off : off + addr_size], "little"), off + addr_size


def _rd_implicit(sections, data, off, addr_size, str_base):
    # value lives in the abbrev table; consumes no DIE bytes
    return None, off


def _strx_reader(form: int):
    """Bind *form* into _rd_strx so the table keeps one reader signature."""
    return lambda sections, data, off, addr_size, str_base: _rd_strx(
        sections, data, off, addr_size, str_base, form
    )


# DW_FORM → reader
_FORM_READERS = {
    _DW_FORM_string: _rd_string,
    _DW_FORM_strp: _rd_strp,
    _DW_FORM_line_strp: _rd_line_strp,
    **{form: _strx_reader(form) for form in _STRX_INDEX},
    _DW_FORM_udata: _rd_uleb,
    _DW_FORM_addrx: _rd_uleb,
    _DW_FORM_ref_udata: _rd_uleb,
    _DW_FORM_sdata: _rd_sleb,
    _DW_FORM_flag_present: _rd_flag_present,
    _DW_FORM_flag: _rd_flag,
    _DW_FORM_data1: _rd_u8,
    _DW_FORM_addrx1: _rd_u8,
    _DW_FORM_ref1: _rd_u8,
    _DW_FORM_data2: _rd_u16,
    _DW_FORM_ref2: _rd_u16,
    _DW_FORM_data4: _rd_u32,
    _DW_FORM_ref4: _rd_u32,
    _DW_FORM_ref_addr: _rd_u32,
    _DW_FORM_sec_offset: _rd_u32,
    _DW_FORM_data8: _rd_u64,
    _DW_FORM_ref8: _rd_u64,
    _DW_FORM_ref_sig8: _rd_u64,
    _DW_FORM_exprloc: _rd_uleb_block,
    _DW_FORM_block: _rd_uleb_block,
    _DW_FORM_block1: _rd_block1,
    _DW_FORM_block2: _rd_block2,
    _DW_FORM_block4: _rd_block4,
    _DW_FORM_data16: _rd_data16,
    _DW_FORM_addr: _rd_addr,
    _DW_FORM_implicit_const: _rd_implicit,
}


def _read_attr(
    sections: dict[bytes, bytes],
    form: int,
    data: bytes,
    off: int,
    addr_size: int = 8,
    str_base: int = 8,
):
    """Read one attribute value. Returns (value, new_offset).

    *str_base* is the offset into ``.debug_str_offsets`` used by the
    DW_FORM_strx* indexed-string forms (per DWARF5, default 8).
    """
    reader = _FORM_READERS.get(form)
    if reader is None:
        raise ValueError(f"unsupported form 0x{form:x}")
    return reader(sections, data, off, addr_size, str_base)


def _find_abbrev(abbrev: bytes, off: int, code: int):
    """Scan the abbrev table at *off* for *code* → (tag, [(name, form)]).

    Returns (None, []) when the table ends before *code* is found.
    """
    while True:
        acode, off = _uleb(abbrev, off)
        if acode == 0:
            return None, []
        atag, off = _uleb(abbrev, off)
        off += 1  # has-children flag
        aspecs = []
        while True:
            name, off = _uleb(abbrev, off)
            form, off = _uleb(abbrev, off)
            if name == 0 and form == 0:
                break
            aspecs.append((name, form))
            if form == _DW_FORM_implicit_const:
                # value embedded in the abbrev table, not the DIE
                _v, off = _sleb(abbrev, off)
        if acode == code:
            return atag, aspecs


def _cu_attrs(sections, info: bytes, die_off: int, specs, addr_size: int):
    """Read the CU DIE's attributes → (stmt_list, comp_dir, name)."""
    stmt_list = None
    comp_dir = None
    name = None
    str_base = 8  # DWARF5 default when DW_AT_str_offsets_base is absent
    for attr_name, form in specs:
        value, die_off = _read_attr(
            sections, form, info, die_off, addr_size=addr_size, str_base=str_base
        )
        if attr_name == _DW_AT_stmt_list:
            stmt_list = value
        elif attr_name == _DW_AT_comp_dir:
            comp_dir = value if isinstance(value, str) else None
        elif attr_name == _DW_AT_name:
            name = value if isinstance(value, str) else None
        elif attr_name == 0x74 and isinstance(value, int):  # DW_AT_str_offsets_base
            str_base = value
    return stmt_list, comp_dir, name


def _parse_cu_die(sections: dict[bytes, bytes], info: bytes, cu_off: int):
    """Parse the compile-unit DIE at *cu_off*, returning
    (stmt_list, comp_dir, name, addr_size) or None."""
    try:
        unit_length = struct.unpack_from("<I", info, cu_off)[0]
        if unit_length == 0xFFFFFFFF:
            return None  # DWARF64 unsupported
        version = struct.unpack_from("<H", info, cu_off + 4)[0]
        if version >= 5:
            # unit_length(4) version(2) unit_type(1) addr_size(1)
            # abbrev_offset(4) → DIE at +12
            abbrev_off = struct.unpack_from("<I", info, cu_off + 8)[0]
            addr_size = info[cu_off + 7]
            die_off = cu_off + 12
        else:
            abbrev_off = struct.unpack_from("<I", info, cu_off + 6)[0]
            addr_size = info[cu_off + 10]
            die_off = cu_off + 11
        abbrev = sections.get(b".debug_abbrev", b"")
        if abbrev_off >= len(abbrev):
            return None
        # The DIE in .debug_info begins with its abbrev code (ULEB); the
        # abbrev table is a hash-ish list in arbitrary order, so scan it
        # for that code (gcc emits helper DIEs like formal_parameter
        # before the compile_unit entry; clang usually puts it first).
        code, die_off = _uleb(info, die_off)
        if code == 0:
            return None
        tag, specs = _find_abbrev(abbrev, abbrev_off, code)
        if tag not in (_DW_TAG_compile_unit, _DW_TAG_skeleton_unit):
            return None
        stmt_list, comp_dir, name = _cu_attrs(sections, info, die_off, specs, addr_size)
        return stmt_list, comp_dir, name, addr_size
    except Exception:
        log.debug("CU DIE parse failed", exc_info=True)
        return None


# ── Line program parsing ───────────────────────────────────────────────


def _read_v5_table(data: bytes, off: int, sections: dict[bytes, bytes]):
    """Read one v5 include-directory or file-name table.

    LLVM (clang) emits these as [format_count][(ct, form) pairs][count]
    [entries] — the entry count follows the format list (verified against
    MCDwarf.cpp's emitV5FileDirTables and readelf). Returns
    (entries, new_offset) where entries is a list of dicts keyed by
    content type.
    """
    format_count, off = _uleb(data, off)
    formats = []
    for _ in range(format_count):
        ct, off = _uleb(data, off)
        form, off = _uleb(data, off)
        formats.append((ct, form))
    count, off = _uleb(data, off)
    entries = []
    for _ in range(count):
        fields, off = _read_v5_entry(data, off, formats, sections)
        entries.append(fields)
    return entries, off


def _read_v5_entry(data: bytes, off: int, formats, sections: dict[bytes, bytes]):
    """Read one v5 file/dir entry per *formats*. Returns (fields, off)
    where fields is a dict keyed by content type."""
    fields = {}
    for ct, form in formats:
        value, off = _read_attr(sections, form, data, off)
        fields[ct] = value
    return fields, off


class _LineHdr(NamedTuple):
    """Fixed .debug_line header fields after header_length."""

    min_inst_length: int
    max_ops: int
    default_is_stmt: int
    line_base: int
    line_range: int
    opcode_base: int
    std_opcode_lengths: list[int]


def _line_header(data: bytes, p: int, version: int) -> tuple[_LineHdr, int]:
    """Read min_inst_length … standard_opcode_lengths → (hdr, new_p)."""
    min_inst_length = data[p]
    p += 1
    if version >= 4:
        max_ops = data[p]
        p += 1
    else:
        max_ops = 1
    default_is_stmt = data[p]
    p += 1
    line_base = struct.unpack_from("<b", data, p)[0]
    p += 1
    line_range = data[p]
    p += 1
    opcode_base = data[p]
    p += 1
    std_opcode_lengths = list(data[p : p + opcode_base - 1])
    p += opcode_base - 1
    hdr = _LineHdr(
        min_inst_length,
        max_ops,
        default_is_stmt,
        line_base,
        line_range,
        opcode_base,
        std_opcode_lengths,
    )
    return hdr, p


def _line_file_tables(data, p, header_end, version, comp_dir, sections):
    """Directory and file tables → (dirs, files[(name, dir index)]).

    v5: directory index 0 is the compilation directory. v4: index 0
    means "no directory" (name is relative to the CU's comp dir).
    """
    dirs: list[str] = [""]
    files: list[tuple[str, int]] = []  # (display name, dir index)
    if version >= 5:
        dirs[0] = comp_dir or ""
        dir_entries, p = _read_v5_table(data, p, sections)
        for fields in dir_entries:
            dirs.append(str(fields.get(_DW_LNCT_path, "")))
        file_entries, p = _read_v5_table(data, p, sections)
        for fields in file_entries:
            files.append(
                (
                    str(fields.get(_DW_LNCT_path, "")),
                    int(fields.get(_DW_LNCT_directory_index, 0)),
                )
            )
        return dirs, files

    while p < header_end and data[p] != 0:
        name, p = _cstr(data, p)
        dirs.append(name.decode(errors="replace"))
    p += 1  # trailing null
    while p < header_end and data[p] != 0:
        name, p = _cstr(data, p)
        dir_idx, p = _uleb(data, p)
        _mtime, p = _uleb(data, p)
        _size, p = _uleb(data, p)
        files.append((name.decode(errors="replace"), dir_idx))
    return dirs, files


def _line_display(version: int, files: list, dirs: list[str]):
    """File-number → "dir/name" resolver.

    v5 file numbers are 0-based (entry 0 is the root file); v2-4 are
    1-based. *files* is shared, so DW_LNE_define_file additions are seen.
    """
    first = 0 if version >= 5 else 1

    def _display(file_idx: int) -> str:
        if not first <= file_idx < len(files) + first:
            return ""
        fname, dir_idx = files[file_idx - first]
        if dir_idx < len(dirs) and dirs[dir_idx]:
            return f"{dirs[dir_idx]}/{fname}"
        return fname

    return _display


class _LineProgram:
    """DWARF line-number state machine over one program's opcodes."""

    def __init__(self, data, end, hdr: _LineHdr, version, addr_size, files, display):
        self.data = data
        self.end = end
        self.hdr = hdr
        self.version = version
        self.addr_size = addr_size
        self.files = files
        self.display = display
        self._reset()

    def _reset(self) -> None:
        """Initial register state (start of program / after end_sequence)."""
        self.address = 0
        self.file_idx = 1
        self.line_no = 1
        self.is_stmt = bool(self.hdr.default_is_stmt)

    def _row(self) -> tuple[str, int, int]:
        return (self.display(self.file_idx), self.line_no, self.address)

    def run(self, p: int) -> list[tuple[str, int, int]]:
        """Execute opcodes from *p* to end → rows (display, line, address)."""
        data, end, hdr = self.data, self.end, self.hdr
        rows: list[tuple[str, int, int]] = []
        while p < end:
            opcode = data[p]
            p += 1
            if opcode == 0:  # extended
                p = self._extended(p)
                if p is None:
                    break
                continue
            if opcode >= hdr.opcode_base:  # special opcode
                adjusted = opcode - hdr.opcode_base
                self.address += (adjusted // hdr.line_range) * hdr.min_inst_length * hdr.max_ops
                self.line_no += hdr.line_base + (adjusted % hdr.line_range)
                rows.append(self._row())
                continue
            if opcode == _DW_LNS_copy:
                rows.append(self._row())
                continue
            p = self._standard(opcode, p)
        return rows

    def _extended(self, p: int) -> int | None:
        """Execute one extended opcode; None when its length overruns the unit."""
        data, end = self.data, self.end
        ext_len, p = _uleb(data, p)
        ext_end = p + ext_len
        if ext_end > end:
            return None
        if p >= end:
            return ext_end
        sub = data[p]
        p += 1
        if sub == _DW_LNE_end_sequence:
            # The end_sequence row marks the address one past the
            # last instruction of the sequence; per DWARF it does
            # not belong to any source line. Emitting it with the
            # stale line_no attributed the end-of-function address
            # to that function's last line, so a file:line target
            # could resolve to an address past the function body.
            self._reset()
        elif sub == _DW_LNE_set_address:
            if p + self.addr_size <= ext_end:
                self.address = int.from_bytes(data[p : p + self.addr_size], "little")
        elif sub == _DW_LNE_define_file and self.version < 5:
            fname, p2 = _cstr(data, p)
            dir_idx, p2 = _uleb(data, p2)
            _mtime, p2 = _uleb(data, p2)
            _size, p2 = _uleb(data, p2)
            self.files.append((fname.decode(errors="replace"), dir_idx))
        # set_discriminator and unknown sub-opcodes: skip
        return ext_end

    def _standard(self, opcode: int, p: int) -> int:
        """Execute one standard opcode (other than DW_LNS_copy) → new p."""
        data, hdr = self.data, self.hdr
        if opcode == _DW_LNS_advance_pc:
            adv, p = _uleb(data, p)
            self.address += adv * hdr.min_inst_length * hdr.max_ops
        elif opcode == _DW_LNS_advance_line:
            adv, p = _sleb(data, p)
            self.line_no += adv
        elif opcode == _DW_LNS_set_file:
            self.file_idx, p = _uleb(data, p)
        elif opcode == _DW_LNS_set_column:
            _col, p = _uleb(data, p)
        elif opcode == _DW_LNS_negate_stmt:
            self.is_stmt = not self.is_stmt
        elif opcode == _DW_LNS_set_basic_block:
            pass
        elif opcode == _DW_LNS_const_add_pc:
            self.address += (
                ((255 - hdr.opcode_base) // hdr.line_range) * hdr.min_inst_length * hdr.max_ops
            )
        elif opcode == _DW_LNS_fixed_advance_pc:
            if p + 2 <= self.end:
                self.address += struct.unpack_from("<H", data, p)[0]
                p += 2
        else:  # prologue_end, epilogue_begin, set_isa, unknown
            n = opcode - 1
            if 0 <= n < len(hdr.std_opcode_lengths):
                for _ in range(hdr.std_opcode_lengths[n]):
                    _v, p = _uleb(data, p)
        return p


class DwarfLineResolver:
    """Resolve ``file.c:line`` → addresses from a binary's DWARF info.

    Args:
        target: Path to the ELF binary (executable or shared library).
    """

    def __init__(self, target: str):
        self.target = target
        # basename → {line → sorted addresses}
        self._by_basename: dict[str, dict[int, list[int]]] = {}
        self._loaded = False

    def load(self) -> bool:
        """Parse DWARF sections. Returns True if any line info was found."""
        try:
            elf_data = Path(self.target).read_bytes()
        except OSError as e:
            log.warning("Cannot read target ELF: %s", e)
            return False
        sections = _elf_sections(elf_data)
        info = sections.get(b".debug_info")
        line = sections.get(b".debug_line")
        if not info or not line:
            log.debug("No DWARF info/line sections in %s", self.target)
            return False

        try:
            cu_off = 0
            while cu_off < len(info):
                parsed = _parse_cu_die(sections, info, cu_off)
                if parsed is None:
                    break
                stmt_list, comp_dir, _cu_name, addr_size = parsed
                if stmt_list is not None:
                    self._parse_line_program(line, stmt_list, comp_dir, addr_size, sections)
                # advance to next CU
                unit_length = struct.unpack_from("<I", info, cu_off)[0]
                if unit_length == 0xFFFFFFFF or unit_length == 0:
                    break
                cu_off += 4 + unit_length
        except Exception:
            log.debug("DWARF parse failed for %s", self.target, exc_info=True)

        self._loaded = bool(self._by_basename)
        if self._loaded:
            log.info(
                "DwarfLineResolver: %d files, %d lines in %s",
                len(self._by_basename),
                sum(len(v) for v in self._by_basename.values()),
                self.target,
            )
        return self._loaded

    def _parse_line_program(
        self,
        line: bytes,
        stmt_list: int,
        comp_dir: str | None,
        addr_size: int,
        sections: dict[bytes, bytes],
    ):
        """Parse one .debug_line program; populate ``self._by_basename``."""
        if stmt_list >= len(line):
            return
        data = line[stmt_list:]
        off = 0
        unit_length = struct.unpack_from("<I", data, 0)[0]
        if unit_length == 0xFFFFFFFF or unit_length == 0:
            return
        end = off + 4 + unit_length
        version = struct.unpack_from("<H", data, 4)[0]
        if version < 2 or version > 5:
            return
        if version >= 5:
            # unit_length(4) version(2) addr_size(1) seg_sel(1)
            # header_length(4) → header fields at +12
            addr_size = data[6]
            header_length = struct.unpack_from("<I", data, 8)[0]
            p = 12
        else:
            header_length = struct.unpack_from("<I", data, 6)[0]
            p = 10
        header_end = p + header_length
        if header_end > end:
            return

        hdr, p = _line_header(data, p, version)
        dirs, files = _line_file_tables(data, p, header_end, version, comp_dir, sections)
        display = _line_display(version, files, dirs)
        prog = _LineProgram(data, end, hdr, version, addr_size, files, display)
        self._index_line_rows(prog.run(header_end))

    def _index_line_rows(self, rows: list[tuple[str, int, int]]) -> None:
        """Index rows by basename → line → addresses (dedup, first address
        of each contiguous run wins via sorted dedup below)."""
        seen: set[tuple[str, int, int]] = set()
        for display, ln, addr in rows:
            if addr <= 0 or not display:
                continue
            base = display.rsplit("/", 1)[-1]
            key = (base, ln, addr)
            if key in seen:
                continue
            seen.add(key)
            self._by_basename.setdefault(base, {}).setdefault(ln, []).append(addr)

    def resolve(self, file_spec: str, line: int) -> list[int]:
        """Return sorted addresses for ``file_spec:line`` ([] if unknown)."""
        base = file_spec.rsplit("/", 1)[-1]
        per_line = self._by_basename.get(base)
        if not per_line:
            return []
        addrs = sorted(set(per_line.get(line, [])))
        # A source line usually spans several rows (address advances while
        # the line stays the same); the meaningful entry point is the
        # lowest address.
        return addrs
