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
    if form == _DW_FORM_string:
        s, off = _cstr(data, off)
        return s.decode(errors="replace"), off
    if form == _DW_FORM_strp:
        strp, off = struct.unpack_from("<I", data, off), off + 4
        dbg_str = sections.get(b".debug_str", b"")
        if strp[0] < len(dbg_str):
            s = dbg_str[strp[0] :].split(b"\x00", 1)[0]
            return s.decode(errors="replace"), off
        return "", off
    if form == _DW_FORM_line_strp:
        strp, off = struct.unpack_from("<I", data, off), off + 4
        dbg_str = sections.get(b".debug_line_str", b"")
        if strp[0] < len(dbg_str):
            s = dbg_str[strp[0] :].split(b"\x00", 1)[0]
            return s.decode(errors="replace"), off
        return "", off
    if form in (_DW_FORM_strx, _DW_FORM_strx1, _DW_FORM_strx2, _DW_FORM_strx3, _DW_FORM_strx4):
        # indexed string: index into .debug_str_offsets → .debug_str
        if form == _DW_FORM_strx:
            idx, off = _uleb(data, off)
        elif form == _DW_FORM_strx1:
            idx, off = data[off], off + 1
        elif form == _DW_FORM_strx2:
            idx, off = struct.unpack_from("<H", data, off)[0], off + 2
        elif form == _DW_FORM_strx3:
            idx, off = int.from_bytes(data[off : off + 3], "little"), off + 3
        else:
            idx, off = struct.unpack_from("<I", data, off)[0], off + 4
        str_offsets = sections.get(b".debug_str_offsets", b"")
        entry = str_base + idx * 4
        if entry + 4 <= len(str_offsets):
            strp = struct.unpack_from("<I", str_offsets, entry)[0]
            dbg_str = sections.get(b".debug_str", b"")
            if strp < len(dbg_str):
                s = dbg_str[strp:].split(b"\x00", 1)[0]
                return s.decode(errors="replace"), off
        return "", off
    if form == _DW_FORM_udata:
        v, off = _uleb(data, off)
        return v, off
    if form == _DW_FORM_addrx:
        v, off = _uleb(data, off)
        return v, off
    if form == _DW_FORM_sdata:
        v, off = _sleb(data, off)
        return v, off
    if form == _DW_FORM_flag_present:
        return True, off
    if form == _DW_FORM_flag:
        return bool(data[off]), off + 1
    if form in (_DW_FORM_data1, _DW_FORM_addrx1):
        return data[off], off + 1
    if form in (_DW_FORM_data2, _DW_FORM_ref2):
        return struct.unpack_from("<H", data, off)[0], off + 2
    if form in (_DW_FORM_data4, _DW_FORM_ref4, _DW_FORM_ref_addr):
        return struct.unpack_from("<I", data, off)[0], off + 4
    if form in (_DW_FORM_data8, _DW_FORM_ref8, _DW_FORM_ref_sig8):
        return struct.unpack_from("<Q", data, off)[0], off + 8
    if form in (_DW_FORM_ref1,):
        return data[off], off + 1
    if form == _DW_FORM_ref_udata:
        v, off = _uleb(data, off)
        return v, off
    if form in (_DW_FORM_sec_offset,):
        return struct.unpack_from("<I", data, off)[0], off + 4
    if form == _DW_FORM_exprloc:
        length, off = _uleb(data, off)
        return None, off + length
    if form in (_DW_FORM_block,):
        length, off = _uleb(data, off)
        return None, off + length
    if form in (_DW_FORM_block1,):
        return None, off + 1 + data[off]
    if form in (_DW_FORM_block2,):
        return None, off + 2 + struct.unpack_from("<H", data, off)[0]
    if form in (_DW_FORM_block4,):
        return None, off + 4 + struct.unpack_from("<I", data, off)[0]
    if form == _DW_FORM_data16:
        return data[off : off + 16], off + 16
    if form == _DW_FORM_addr:
        return int.from_bytes(data[off : off + addr_size], "little"), off + addr_size
    if form == _DW_FORM_implicit_const:
        # value lives in the abbrev table; consumes no DIE bytes
        return None, off
    raise ValueError(f"unsupported form 0x{form:x}")


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
        tag = None
        specs = []
        off = abbrev_off
        while True:
            acode, off = _uleb(abbrev, off)
            if acode == 0:
                break
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
                tag, specs = atag, aspecs
                break
        if tag not in (_DW_TAG_compile_unit, _DW_TAG_skeleton_unit):
            return None
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

        # File tables.
        # v5: directory index 0 is the compilation directory. v4: index 0
        # means "no directory" (name is relative to the CU's comp dir).
        dirs: list[str] = [""]
        if version >= 5:
            dirs[0] = comp_dir or ""
            dir_entries, p = _read_v5_table(data, p, sections)
            for fields in dir_entries:
                dirs.append(str(fields.get(_DW_LNCT_path, "")))
            file_entries, p = _read_v5_table(data, p, sections)
            files: list[tuple[str, int]] = []
            for fields in file_entries:
                files.append(
                    (
                        str(fields.get(_DW_LNCT_path, "")),
                        int(fields.get(_DW_LNCT_directory_index, 0)),
                    )
                )
        else:
            files: list[tuple[str, int]] = []  # (display name, dir index)
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

        # v5 file numbers are 0-based (entry 0 is the root file); v2-4
        # are 1-based.
        if version >= 5:

            def _display(file_idx: int) -> str:
                if 0 <= file_idx < len(files):
                    fname, dir_idx = files[file_idx]
                    if dir_idx < len(dirs) and dirs[dir_idx]:
                        return f"{dirs[dir_idx]}/{fname}"
                    return fname
                return ""
        else:

            def _display(file_idx: int) -> str:
                if 1 <= file_idx <= len(files):
                    fname, dir_idx = files[file_idx - 1]
                    if dir_idx < len(dirs) and dirs[dir_idx]:
                        return f"{dirs[dir_idx]}/{fname}"
                    return fname
                return ""

        # Line program.
        address = 0
        file_idx = 1
        line_no = 1
        is_stmt = bool(default_is_stmt)
        p = header_end
        rows: list[tuple[str, int, int]] = []  # (display, line, address)
        while p < end:
            opcode = data[p]
            p += 1
            if opcode == 0:  # extended
                ext_len, p = _uleb(data, p)
                ext_end = p + ext_len
                if ext_end > end:
                    break
                if p < end:
                    sub = data[p]
                    p += 1
                    if sub == _DW_LNE_end_sequence:
                        rows.append((_display(file_idx), line_no, address))
                        address = 0
                        file_idx = 1
                        line_no = 1
                        is_stmt = bool(default_is_stmt)
                    elif sub == _DW_LNE_set_address:
                        if p + addr_size <= ext_end:
                            address = int.from_bytes(data[p : p + addr_size], "little")
                    elif sub == _DW_LNE_define_file and version < 5:
                        fname, p2 = _cstr(data, p)
                        dir_idx, p2 = _uleb(data, p2)
                        _mtime, p2 = _uleb(data, p2)
                        _size, p2 = _uleb(data, p2)
                        files.append((fname.decode(errors="replace"), dir_idx))
                    # set_discriminator and unknown sub-opcodes: skip
                p = ext_end
                continue
            if opcode >= opcode_base:  # special opcode
                adjusted = opcode - opcode_base
                address += (adjusted // line_range) * min_inst_length * max_ops
                line_no += line_base + (adjusted % line_range)
                rows.append((_display(file_idx), line_no, address))
                continue
            # standard opcode
            if opcode == _DW_LNS_copy:
                rows.append((_display(file_idx), line_no, address))
            elif opcode == _DW_LNS_advance_pc:
                adv, p = _uleb(data, p)
                address += adv * min_inst_length * max_ops
            elif opcode == _DW_LNS_advance_line:
                adv, p = _sleb(data, p)
                line_no += adv
            elif opcode == _DW_LNS_set_file:
                file_idx, p = _uleb(data, p)
            elif opcode == _DW_LNS_set_column:
                _col, p = _uleb(data, p)
            elif opcode == _DW_LNS_negate_stmt:
                is_stmt = not is_stmt
            elif opcode == _DW_LNS_set_basic_block:
                pass
            elif opcode == _DW_LNS_const_add_pc:
                address += ((255 - opcode_base) // line_range) * min_inst_length * max_ops
            elif opcode == _DW_LNS_fixed_advance_pc:
                if p + 2 <= end:
                    address += struct.unpack_from("<H", data, p)[0]
                    p += 2
            else:  # prologue_end, epilogue_begin, set_isa, unknown
                n = opcode - 1
                if 0 <= n < len(std_opcode_lengths):
                    for _ in range(std_opcode_lengths[n]):
                        _v, p = _uleb(data, p)

        # Index rows by basename → line → addresses (dedup, first address
        # of each contiguous run wins via sorted dedup below).
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
