"""Edge coverage via ptrace breakpoints on closed-source binaries."""

import ctypes
import logging
import os
import struct

log = logging.getLogger(__name__)

try:
    from capstone import CS_ARCH_X86, CS_MODE_64, Cs
    from capstone.x86_const import X86_GRP_CALL, X86_GRP_INT, X86_GRP_JUMP, X86_GRP_RET

    HAS_CAPSTONE = True
except ImportError:
    HAS_CAPSTONE = False

PTRACE_TRACEME = 0
PTRACE_PEEKDATA = 2
PTRACE_POKEDATA = 5
PTRACE_CONT = 7
PTRACE_SINGLESTEP = 9
PTRACE_GETREGS = 12
PTRACE_SETREGS = 13
PTRACE_SETOPTIONS = 0x4200
PTRACE_O_TRACESYSGOOD = 1
INT3 = 0xCC


class PtraceCoverage:
    """Edge coverage via ptrace breakpoints on closed-source binaries.

    Strategy: disassemble the first bytes of each function (from ELF symtab/dynsym),
    place int3 at each basic block entry, record (prev, curr) edges.
    With --deep-coverage, uses capstone to discover all basic blocks.
    """

    def __init__(
        self,
        target_path: str,
        map_size: int = 65536,
        deep_coverage: bool = False,
        max_bps: int = 50000,
    ):
        self.target_path = target_path
        self.map_size = map_size
        self.bb_addrs: list[int] = []
        self.original_bytes: dict[int, int] = {}
        self.edge_map: bytearray = bytearray(map_size)
        self.prev_location = 0
        self.total_edges = 0
        self.cumulative_edges = 0
        self.total_bp_hits = 0
        self._base_address: int | None = None
        self._map_snapshot = bytes(self.edge_map)
        self.deep_coverage = deep_coverage and HAS_CAPSTONE
        self.max_bps = max_bps
        self._discovered_bbs: set[int] = set()
        self._func_ranges: list[tuple[int, int]] = []
        self._elf_data: bytes = b""
        self._load_segments: list[tuple[int, int, int, int]] = []
        self._is_pie: bool = True  # assume PIE until proven otherwise
        self._is_x86_64: bool = False  # cached platform check
        self._stack_initialized: bool = False  # True after first valid RSP seen

        if self.deep_coverage:
            self._disassembler = Cs(CS_ARCH_X86, CS_MODE_64)
            self._disassembler.detail = True
            self._parse_elf_segments()

        self._collect_basic_blocks()

        # Cache platform check once (avoids import + call in hot SIGTRAP loop)
        import platform as _platform

        self._is_x86_64 = _platform.machine() == "x86_64"

    def _collect_basic_blocks(self):
        try:
            with open(self.target_path, "rb") as f:
                data = f.read()
        except Exception:
            log.debug("Failed to read ELF from %s", self.target_path, exc_info=True)
            return

        if data[:4] != b"\x7fELF":
            return

        is_64 = data[4] == 2
        is_le = data[5] == 1
        if not (is_64 and is_le):
            return

        e_type = struct.unpack_from("<H", data, 16)[0]
        self._is_pie = e_type == 3  # ET_DYN = PIE, ET_EXEC = non-PIE
        self._elf_entry = struct.unpack_from("<Q", data, 24)[0]  # e_entry

        e_shoff = struct.unpack_from("<Q", data, 40)[0]
        e_shnum = struct.unpack_from("<H", data, 60)[0]
        e_shentsize = struct.unpack_from("<H", data, 58)[0]
        e_shstrndx = struct.unpack_from("<H", data, 62)[0]

        if e_shnum == 0 or e_shstrndx >= e_shnum:
            return

        shstr_off = e_shoff + e_shstrndx * e_shentsize
        if shstr_off + e_shentsize > len(data):
            return
        shstr_offset = struct.unpack_from("<Q", data, shstr_off + 24)[0]

        symtab_sec = None
        strtab_sec = None
        dynsym_sec = None
        dynstr_sec = None
        text_start = 0
        text_end = 0
        for i in range(e_shnum):
            sh = e_shoff + i * e_shentsize
            if sh + e_shentsize > len(data):
                return
            sh_type = struct.unpack_from("<I", data, sh + 4)[0]
            sh_name_idx = struct.unpack_from("<I", data, sh)[0]
            name = data[shstr_offset + sh_name_idx : shstr_offset + sh_name_idx + 32].split(
                b"\x00"
            )[0]
            if sh_type == 2:
                symtab_sec = sh
            elif sh_type == 11:
                dynsym_sec = sh
            elif sh_type == 3:
                if name == b".strtab" and strtab_sec is None:
                    strtab_sec = sh
                elif name == b".dynstr" and dynstr_sec is None:
                    dynstr_sec = sh
            elif name == b".text":
                text_start = struct.unpack_from("<Q", data, sh + 24)[0]
                text_size = struct.unpack_from("<Q", data, sh + 32)[0]
                text_end = text_start + text_size

        self._parse_symbol_table(data, symtab_sec, strtab_sec, text_start, text_end)
        if not self.bb_addrs:
            self._parse_symbol_table(data, dynsym_sec, dynstr_sec, text_start, text_end)

        if self.deep_coverage and HAS_CAPSTONE:
            for func_va, func_size in self._func_ranges:
                self._collect_function_bbs(func_va, func_size)

        # Exclude _start (entry point) — stack not set up yet, re-executing
        # instructions there causes SIGSEGV from push to RSP=0.
        # Must run after all collection (symbol table + capstone discovery).
        self.bb_addrs = [a for a in set(self.bb_addrs) if a != self._elf_entry]
        self.bb_addrs.sort()

    def _parse_symbol_table(
        self,
        data: bytes,
        sym_sec: int | None,
        str_sec: int | None,
        text_start: int = 0,
        text_end: int = 0,
    ):
        if sym_sec is None or str_sec is None:
            return

        sym_offset = struct.unpack_from("<Q", data, sym_sec + 24)[0]
        sym_size = struct.unpack_from("<Q", data, sym_sec + 32)[0]
        sym_entsize = struct.unpack_from("<Q", data, sym_sec + 56)[0]
        if sym_entsize == 0:
            log.debug("sym_entsize == 0 in section, skipping malformed symbol table")
            return
        sym_count = sym_size // sym_entsize if sym_entsize else 0

        # Valid x86-64 function entry opcodes (first byte of instruction)
        valid_opcodes = {
            0xF3,  # endbr64 prefix
            0x55,  # push %rbp
            0x48,  # rex.W prefix (mov, sub, lea, etc.)
            0x41,  # rex.B prefix (push, mov, etc.)
            0x53,  # push %rbx
            0x56,  # push %rsi
            0x57,  # push %rdi
            0x83,  # sub $imm, r/m
            0x81,  # sub $imm32, r/m
            0x31,  # xor r/m, r
            0x33,  # xor r, r/m
            0x89,  # mov r/m, r
            0xE8,  # call rel32
            0xFF,  # call/jmp indir
            0xB8,  # mov eax, imm32
            0xC3,  # ret (shouldn't be entry, but safe)
        }

        for i in range(min(sym_count, 10000)):
            sym = sym_offset + i * sym_entsize
            if sym + sym_entsize > len(data):
                break
            st_info = data[sym + 4]
            st_value = struct.unpack_from("<Q", data, sym + 8)[0]
            st_size = struct.unpack_from("<Q", data, sym + 16)[0]
            st_type = st_info & 0xF
            if (
                st_type == 2
                and st_value > 0
                and st_size > 0
                and st_value < len(data)
                and data[st_value] in valid_opcodes
                and (text_start == 0 or text_start <= st_value < text_end)
            ):
                self.bb_addrs.append(st_value)
                self._func_ranges.append((st_value, st_value + st_size))

        self.bb_addrs.sort()
        self._func_ranges.sort()

    def _parse_elf_segments(self):
        try:
            with open(self.target_path, "rb") as f:
                self._elf_data = f.read()
        except Exception:
            log.debug("Failed to read ELF segments from %s", self.target_path, exc_info=True)
            return

        data = self._elf_data
        if len(data) < 64 or data[:4] != b"\x7fELF":
            return
        if data[4] != 2 or data[5] != 1:
            return

        e_phoff = struct.unpack_from("<Q", data, 32)[0]
        e_phentsize = struct.unpack_from("<H", data, 54)[0]
        e_phnum = struct.unpack_from("<H", data, 56)[0]

        for i in range(min(e_phnum, 100)):
            ph = e_phoff + i * e_phentsize
            if ph + 56 > len(data):
                break
            p_type = struct.unpack_from("<I", data, ph)[0]
            if p_type == 1:
                p_offset = struct.unpack_from("<Q", data, ph + 8)[0]
                p_vaddr = struct.unpack_from("<Q", data, ph + 16)[0]
                p_filesz = struct.unpack_from("<Q", data, ph + 32)[0]
                p_memsz = struct.unpack_from("<Q", data, ph + 40)[0]
                self._load_segments.append((p_vaddr, p_offset, p_filesz, p_memsz))

    def _read_func_bytes(self, func_va: int, max_len: int = 512) -> bytes | None:
        for vaddr, offset, filesz, _memsz in self._load_segments:
            if vaddr <= func_va < vaddr + filesz:
                file_offset = offset + (func_va - vaddr)
                end = min(file_offset + max_len, offset + filesz)
                return self._elf_data[file_offset:end]
        return None

    def _collect_function_bbs(self, func_va: int, func_size: int):
        if not self.deep_coverage or not HAS_CAPSTONE:
            return

        func_bytes = self._read_func_bytes(func_va, max_len=min(func_size, 2048))
        if not func_bytes:
            return

        try:
            for insn in self._disassembler.disasm(func_bytes, func_va):
                is_jump = X86_GRP_JUMP in insn.groups
                is_call = X86_GRP_CALL in insn.groups
                is_ret = X86_GRP_RET in insn.groups
                is_int = X86_GRP_INT in insn.groups

                if is_jump:
                    if insn.op_str.startswith("0x"):
                        target = int(insn.op_str, 16)
                        if (
                            func_va <= target < func_va + func_size
                            and target not in self._discovered_bbs
                        ):
                            self.bb_addrs.append(target)
                            self._discovered_bbs.add(target)
                    next_addr = insn.address + insn.size
                    if next_addr not in self._discovered_bbs:
                        self.bb_addrs.append(next_addr)
                        self._discovered_bbs.add(next_addr)
                elif is_call or is_ret or is_int:
                    next_addr = insn.address + insn.size
                    if next_addr not in self._discovered_bbs:
                        self.bb_addrs.append(next_addr)
                        self._discovered_bbs.add(next_addr)
        except Exception:
            log.debug("Failed to disassemble function at %#x", func_va, exc_info=True)

    def discover_new_bbs(self, pid: int, bp_addr: int, max_discover: int = 32):
        if not self.deep_coverage or len(self.original_bytes) >= self.max_bps:
            return 0

        rel_addr = bp_addr - self._base_address if self._base_address is not None else bp_addr

        func_start = None
        func_size = 0
        for start, end in self._func_ranges:
            if start <= rel_addr < end:
                func_start = start
                func_size = end - start
                break
        if func_start is None:
            return 0

        scan_start = rel_addr
        func_bytes = self._read_func_bytes(
            scan_start,
            max_len=min(func_size - (scan_start - func_start), 512),
        )
        if not func_bytes:
            return 0

        count = 0
        try:
            for insn in self._disassembler.disasm(func_bytes, scan_start):
                if count >= max_discover or len(self.original_bytes) >= self.max_bps:
                    break

                is_jump = X86_GRP_JUMP in insn.groups
                is_call = X86_GRP_CALL in insn.groups
                is_ret = X86_GRP_RET in insn.groups
                is_int = X86_GRP_INT in insn.groups

                new_targets = []
                if is_jump and insn.op_str.startswith("0x"):
                    target = int(insn.op_str, 16)
                    if func_start <= target < func_start + func_size:
                        new_targets.append(target)
                    next_addr = insn.address + insn.size
                    if func_start <= next_addr < func_start + func_size:
                        new_targets.append(next_addr)
                elif is_call or is_ret or is_int:
                    next_addr = insn.address + insn.size
                    if func_start <= next_addr < func_start + func_size:
                        new_targets.append(next_addr)

                for target in new_targets:
                    if target in self._discovered_bbs:
                        continue
                    self._discovered_bbs.add(target)
                    abs_target = self._resolve_addr(target)
                    try:
                        val = self._read_memory(pid, abs_target)
                        orig = val & 0xFF
                        if orig != INT3:
                            self.original_bytes[abs_target] = orig
                            self._write_memory(pid, abs_target, (val & ~0xFF) | INT3)
                            self.bb_addrs.append(target)
                            count += 1
                    except Exception:
                        log.debug("Failed to install bp at %#x", target, exc_info=True)
        except Exception:
            log.debug("Failed to discover BBs from %#x", rel_addr, exc_info=True)

        return count

    def resolve_base(self, pid: int):
        try:
            with open(f"/proc/{pid}/maps") as f:
                for line in f:
                    if self.target_path in line or line.split()[-1].endswith(
                        "/" + os.path.basename(self.target_path)
                    ):
                        parts = line.split()
                        addr_range = parts[0].split("-")
                        self._base_address = int(addr_range[0], 16)
                        return
            with open(f"/proc/{pid}/maps") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2 and parts[1] == "r-xp":
                        addr_range = parts[0].split("-")
                        self._base_address = int(addr_range[0], 16)
                        return
        except Exception:
            log.debug("Failed to resolve base address from /proc/%d/maps", pid, exc_info=True)

    def _resolve_addr(self, rel_addr: int) -> int:
        if self._is_pie and self._base_address is not None:
            return self._base_address + rel_addr
        return rel_addr

    def _ptrace(self, request, pid, addr=None, data=None):
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.ptrace.argtypes = [
            ctypes.c_long,
            ctypes.c_long,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        libc.ptrace.restype = ctypes.c_long
        ctypes.set_errno(0)
        result = libc.ptrace(
            request,
            pid,
            ctypes.c_void_p(addr) if addr else None,
            ctypes.c_void_p(data) if data else None,
        )
        return result

    def _read_memory(self, pid: int, addr: int) -> int:
        val = self._ptrace(PTRACE_PEEKDATA, pid, addr)
        return val & 0xFFFFFFFFFFFFFFFF

    def _write_memory(self, pid: int, addr: int, data_8: int):
        self._ptrace(PTRACE_POKEDATA, pid, addr, data_8)

    def install_breakpoints(self, pid: int):
        self.resolve_base(pid)
        if self._is_pie and self._base_address is None:
            log.warning("Could not resolve PIE base address, breakpoints may be incorrect")
        self.original_bytes.clear()
        for rel_addr in self.bb_addrs:
            addr = self._resolve_addr(rel_addr)
            try:
                val = self._read_memory(pid, addr)
                self.original_bytes[addr] = val & 0xFF
                new_val = (val & ~0xFF) | INT3
                self._write_memory(pid, addr, new_val)
            except Exception:
                log.debug("Failed to install bp at %#x", addr, exc_info=True)

    def remove_breakpoints(self, pid: int):
        for addr, orig in self.original_bytes.items():
            try:
                val = self._read_memory(pid, addr)
                new_val = (val & ~0xFF) | orig
                self._write_memory(pid, addr, new_val)
            except Exception:
                log.debug("Failed to restore bp at %#x", addr, exc_info=True)
        self.original_bytes.clear()

    def reset_edge_map(self):
        self.prev_location = 0
        self.total_edges = 0
        self._map_snapshot = bytes(self.edge_map)

    def record_edge(self, addr: int) -> bool:
        # Use relative addresses for edge hashing (consistent with AFL).
        # Absolute addresses cause spurious collisions under PIE/ASLR.
        rel = addr - self._base_address if self._base_address else addr
        bucket = (rel ^ self.prev_location) % self.map_size
        self.prev_location = rel % self.map_size
        self.total_bp_hits += 1
        if self.edge_map[bucket] == 0:
            self.edge_map[bucket] = 1
            self.total_edges += 1
            self.cumulative_edges += 1
            return True
        return False

    def is_new_coverage(self) -> bool:
        return bytes(self.edge_map) != self._map_snapshot
