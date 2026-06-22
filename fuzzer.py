#!/usr/bin/env python3
"""Binary fuzzer with ASAN/MSAN detection, dictionary and Markov mutations."""

import argparse
import collections
import ctypes
import ctypes.util
import hashlib
import math
import os
import random
import re
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path

try:
    from capstone import Cs, CS_ARCH_X86, CS_MODE_64, CS_OPT_DETAIL
    from capstone.x86_const import X86_GRP_JUMP, X86_GRP_CALL, X86_GRP_RET, X86_GRP_INT
    HAS_CAPSTONE = True
except ImportError:
    HAS_CAPSTONE = False


INTERESTING_8 = [0, 1, 0x7F, 0x80, 0xFF]
INTERESTING_16 = [0x7FFF, 0x8000, 0xFFFF, 0, 1]
INTERESTING_32 = [0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0, 1]

MUTATIONS = [
    "bit_flip",
    "byte_flip",
    "interesting_8",
    "interesting_16",
    "interesting_32",
    "random_bytes",
    "block_insert",
    "block_delete",
    "block_duplicate",
    "havoc",
]

DICT_MUTATIONS = [
    "dict_insert",
    "dict_replace",
]


class MarkovChain:
    """Byte-level Markov chain for fuzz input generation."""

    def __init__(self, order=1, smoothing=1e-6):
        self.order = order
        self.smoothing = smoothing
        self.transitions = collections.defaultdict(lambda: collections.Counter())
        self._contexts_seen = 0

    def train(self, data: bytes):
        for i in range(len(data)):
            ctx = bytes(data[max(0, i - self.order):i]) if self.order else b""
            self.transitions[ctx][data[i]] += 1
            self._contexts_seen += 1

    def train_corpus(self, corpus: list[bytes]):
        for data in corpus:
            self.train(data)

    def generate(self, length: int) -> bytes:
        result = bytearray()
        ctx = b"\x00" * self.order
        for _ in range(length):
            counts = self.transitions.get(ctx)
            if counts is None or not counts:
                result.append(random.randint(0, 255))
                ctx = bytes(result[max(0, len(result) - self.order):])
                continue
            total = sum(counts.values()) + self.smoothing * 256
            r = random.random() * total
            cumulative = 0.0
            for byte_val, count in counts.items():
                cumulative += count + self.smoothing
                if r <= cumulative:
                    result.append(byte_val)
                    break
            else:
                result.append(random.randint(0, 255))
            ctx = bytes(result[max(0, len(result) - self.order):])
        return bytes(result)

    def sample_byte(self, ctx: bytes) -> int:
        counts = self.transitions.get(ctx)
        if counts is None or not counts:
            return random.randint(0, 255)
        total = sum(counts.values()) + self.smoothing * 256
        r = random.random() * total
        cumulative = 0.0
        for byte_val, count in counts.items():
            cumulative += count + self.smoothing
            if r <= cumulative:
                return byte_val
        return random.randint(0, 255)

    def is_trained(self) -> bool:
        return self._contexts_seen > 0


class MonteCarloScheduler:
    """Thompson sampling bandit for mutation ops + CEM byte distribution."""

    ELITE_MAX = 200

    def __init__(self, elite_frac=0.1, refit_interval=1000):
        self.arm_alpha: dict[str, float] = {}
        self.arm_beta: dict[str, float] = {}
        self.elite_frac = elite_frac
        self.refit_interval = refit_interval
        self.execs_since_refit = 0
        self.elite_set: list[tuple[int, bytes]] = []
        self.byte_freq: dict[int, dict[int, int]] = {}
        self.cem_fitted = False

    def init_arm(self, name: str):
        if name not in self.arm_alpha:
            self.arm_alpha[name] = 1.0
            self.arm_beta[name] = 1.0

    def select_op(self, ops: list[str]) -> str:
        best_op = ops[0]
        best_val = -1.0
        for op in ops:
            a = self.arm_alpha.get(op, 1.0)
            b = self.arm_beta.get(op, 1.0)
            val = random.betavariate(a, b)
            if val > best_val:
                best_val = val
                best_op = op
        return best_op

    def record(self, name: str, success: bool):
        if success:
            self.arm_alpha[name] = self.arm_alpha.get(name, 1.0) + 1
        else:
            self.arm_beta[name] = self.arm_beta.get(name, 1.0) + 1

    def add_elite(self, data: bytes, score: int):
        self.elite_set.append((score, data))
        if len(self.elite_set) > self.ELITE_MAX:
            self.elite_set.sort(key=lambda x: x[0])
            self.elite_set.pop(0)

    def maybe_refit(self):
        self.execs_since_refit += 1
        if self.execs_since_refit < self.refit_interval:
            return
        self.execs_since_refit = 0
        if not self.elite_set:
            return
        n_elite = max(1, int(len(self.elite_set) * self.elite_frac))
        sorted_elite = sorted(self.elite_set, key=lambda x: x[0], reverse=True)
        elite = [d for _, d in sorted_elite[:n_elite]]
        self.byte_freq = {}
        for pos in range(max(len(d) for d in elite)):
            freq: dict[int, int] = {}
            for data in elite:
                if pos < len(data):
                    b = data[pos]
                    freq[b] = freq.get(b, 0) + 1
            self.byte_freq[pos] = freq
        self.cem_fitted = True

    def cem_byte(self, pos: int) -> int:
        freq = self.byte_freq.get(pos)
        if not freq:
            return random.randint(0, 255)
        total = sum(freq.values()) + 256
        r = random.random() * total
        cumulative = 0
        for byte_val, count in freq.items():
            cumulative += count + 1
            if r <= cumulative:
                return byte_val
        return random.randint(0, 255)

    def cem_sample(self, length: int) -> bytes:
        return bytes(self.cem_byte(i) for i in range(length))

    def bandit_stats(self) -> dict[str, tuple[float, float]]:
        result = {}
        for name in sorted(self.arm_alpha):
            a = self.arm_alpha[name]
            b = self.arm_beta[name]
            result[name] = (a - 1, b - 1)
        return result


# Sanitizer error patterns
SANITIZER_PATTERNS = [
    (r"AddressSanitizer:\s*(heap-buffer-overflow|stack-buffer-overflow|heap-use-after-free"
     r"|global-buffer-overflow|stack-buffer-underflow|heap-buffer-overflow-|"
     r"dynamic-stack-buffer-overflow|stack-use-after-return|stack-use-after-scope"
     r"|allocation-size-too-big|double-free|invalid-malloc-size"
     r"|attempting-free-on-non-deallocated-memory|"
     r"negative-size-param|heap-use-after-scope", "ASAN"),
    (r"MemorySanitizer:\s*(use-of-uninitialized-value)", "MSAN"),
    (r"ThreadSanitizer:\s*(data-race|heap-use-after-race|lock-order-inversion", "TSAN"),
    (r"LeakSanitizer:\s*(leak)", "LSAN"),
    (r"UndefinedBehaviorSanitizer:\s*(undefined|shift-exponent|signed-integer-overflow"
     r"|null-pointer-use|integer-divide-by-zero)", "UBSAN"),
]

SANITIZER_ERROR_RE = re.compile(
    r"(AddressSanitizer|MemorySanitizer|ThreadSanitizer|LeakSanitizer|UndefinedBehaviorSanitizer)"
    r":\s*(\S+(?:\s+\S+)?)",
    re.IGNORECASE,
)
SANITIZER_STACK_FRAME_RE = re.compile(
    r"#\d+\s+0x[0-9a-f]+\s+in\s+(\S+)\s+.*"
)
SANITIZER_FAULT_ADDR_RE = re.compile(
    r"(?:Address|Memory)Sanitizer.*(?:on|at) address\s+(0x[0-9a-f]+)",
    re.IGNORECASE,
)


class SanitizerReport:
    """Parse sanitizer output from a crashed process."""

    __slots__ = ("sanitizer", "error_type", "fault_addr", "frames", "raw", "signature")

    def __init__(self, sanitizer: str, error_type: str, fault_addr: str,
                 frames: list[str], raw: str):
        self.sanitizer = sanitizer
        self.error_type = error_type
        self.fault_addr = fault_addr
        self.frames = frames
        self.raw = raw
        self.signature = self._build_signature()

    def _build_signature(self) -> str:
        key = f"{self.sanitizer}:{self.error_type}"
        for f in self.frames[:6]:
            key += f"@{f}"
        return key

    @classmethod
    def parse(cls, stderr: str) -> "SanitizerReport | None":
        m = SANITIZER_ERROR_RE.search(stderr)
        if not m:
            return None
        sanitizer = m.group(1)
        error_type = m.group(2).strip()

        fault_addr = ""
        addr_m = SANITIZER_FAULT_ADDR_RE.search(stderr)
        if addr_m:
            fault_addr = addr_m.group(1)

        frames = SANITIZER_STACK_FRAME_RE.findall(stderr)
        return cls(sanitizer, error_type, fault_addr, frames, stderr)

    def is_valid(self) -> bool:
        return bool(self.sanitizer and self.error_type)


def parse_dict_line(line: str) -> bytes | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.split("=", 1)
    token = parts[-1] if len(parts) == 2 else line
    return token.encode("raw_unicode_escape").decode("unicode_escape").encode("latin-1")


def load_dictionary(path: str) -> list[bytes]:
    d = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            tok = parse_dict_line(line)
            if tok is not None:
                d.append(tok)
    return d


# ptrace constants
PTRACE_TRACEME = 0
PTRACE_PEEKDATA = 2
PTRACE_POKEDATA = 5
PTRACE_CONT = 7
PTRACE_SINGLESTEP = 9
PTRACE_GETREGS = 12
PTRACE_SETREGS = 13
PTRACE_SETOPTIONS = 0x4200
PTRACE_O_TRACESYSGOOD = 1
PTRACE_EVENT_FORK = 1
PTRACE_EVENT_VFORK = 2
PTRACE_EVENT_CLONE = 3
PTRACE_EVENT_EXEC = 4
INT3 = 0xCC
INT3_BYTE = bytes([INT3])


class PtraceCoverage:
    """Edge coverage via ptrace breakpoints on closed-source binaries.

    Strategy: disassemble the first bytes of each function (from ELF symtab/dynsym),
    place int3 at each basic block entry, record (prev, curr) edges.
    With --deep-coverage, uses capstone to discover all basic blocks.
    """

    def __init__(self, target_path: str, map_size: int = 65536,
                 deep_coverage: bool = False, max_bps: int = 50000):
        self.target_path = target_path
        self.map_size = map_size
        self.bb_addrs: list[int] = []  # relative (file) addresses
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
        self._func_ranges: list[tuple[int, int]] = []  # (start, end) sorted
        self._elf_data: bytes = b""
        self._load_segments: list[tuple[int, int, int, int]] = []  # vaddr, offset, filesz, memsz

        if self.deep_coverage:
            self._disassembler = Cs(CS_ARCH_X86, CS_MODE_64)
            self._disassembler.detail = True
            self._parse_elf_segments()

        self._collect_basic_blocks()

    def _collect_basic_blocks(self):
        """Parse ELF to find function entry points as basic block targets.
        With deep_coverage, also discovers internal basic blocks via capstone."""
        try:
            with open(self.target_path, "rb") as f:
                data = f.read()
        except Exception:
            return

        if data[:4] != b"\x7fELF":
            return

        is_64 = data[4] == 2
        is_le = data[5] == 1
        if not (is_64 and is_le):
            return

        e_shoff = struct.unpack_from("<Q", data, 40)[0]
        e_shnum = struct.unpack_from("<H", data, 60)[0]
        e_shentsize = struct.unpack_from("<H", data, 58)[0]
        e_shstrndx = struct.unpack_from("<H", data, 62)[0]

        if e_shnum == 0 or e_shstrndx >= e_shnum:
            return

        shstr_off = e_shoff + e_shstrndx * e_shentsize
        shstr_offset = struct.unpack_from("<Q", data, shstr_off + 24)[0]

        symtab_sec = None
        strtab_sec = None
        dynsym_sec = None
        dynstr_sec = None
        for i in range(e_shnum):
            sh = e_shoff + i * e_shentsize
            sh_type = struct.unpack_from("<I", data, sh + 4)[0]
            sh_name_idx = struct.unpack_from("<I", data, sh)[0]
            name = data[shstr_offset + sh_name_idx:shstr_offset + sh_name_idx + 32].split(b"\x00")[0]
            if sh_type == 2:  # SHT_SYMTAB
                symtab_sec = sh
            elif sh_type == 11:  # SHT_DYNSYM
                dynsym_sec = sh
            elif sh_type == 3:
                if name == b".strtab" and strtab_sec is None:
                    strtab_sec = sh
                elif name == b".dynstr" and dynstr_sec is None:
                    dynstr_sec = sh

        self._parse_symbol_table(data, symtab_sec, strtab_sec)
        if not self.bb_addrs:
            self._parse_symbol_table(data, dynsym_sec, dynstr_sec)

        if self.deep_coverage and HAS_CAPSTONE:
            func_entries = list(self.bb_addrs)
            for func_va, func_size in self._func_ranges:
                self._collect_function_bbs(func_va, func_size)
            self.bb_addrs = sorted(set(self.bb_addrs))

    def _parse_symbol_table(self, data: bytes, sym_sec: int | None, str_sec: int | None):
        if sym_sec is None or str_sec is None:
            return

        sym_offset = struct.unpack_from("<Q", data, sym_sec + 24)[0]
        sym_size = struct.unpack_from("<Q", data, sym_sec + 32)[0]
        sym_entsize = struct.unpack_from("<Q", data, sym_sec + 56)[0]
        if sym_entsize == 0:
            sym_entsize = 24
        sym_count = sym_size // sym_entsize if sym_entsize else 0
        strtab_offset = struct.unpack_from("<Q", data, str_sec + 24)[0]

        for i in range(min(sym_count, 10000)):
            sym = sym_offset + i * sym_entsize
            st_info = data[sym + 4]
            st_value = struct.unpack_from("<Q", data, sym + 8)[0]
            st_size = struct.unpack_from("<Q", data, sym + 16)[0]
            st_type = st_info & 0xf
            if st_type == 2 and st_value > 0 and st_size > 0:  # STT_FUNC
                self.bb_addrs.append(st_value)
                self._func_ranges.append((st_value, st_value + st_size))
        self.bb_addrs.sort()
        self._func_ranges.sort()

    def _parse_elf_segments(self):
        """Parse ELF program headers to find PT_LOAD segments."""
        try:
            with open(self.target_path, "rb") as f:
                self._elf_data = f.read()
        except Exception:
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
            if p_type == 1:  # PT_LOAD
                p_offset = struct.unpack_from("<Q", data, ph + 8)[0]
                p_vaddr = struct.unpack_from("<Q", data, ph + 16)[0]
                p_filesz = struct.unpack_from("<Q", data, ph + 32)[0]
                p_memsz = struct.unpack_from("<Q", data, ph + 40)[0]
                self._load_segments.append((p_vaddr, p_offset, p_filesz, p_memsz))

    def _read_func_bytes(self, func_va: int, max_len: int = 512) -> bytes | None:
        """Read bytes from ELF file for a function at virtual address func_va."""
        for vaddr, offset, filesz, memsz in self._load_segments:
            if vaddr <= func_va < vaddr + filesz:
                file_offset = offset + (func_va - vaddr)
                end = min(file_offset + max_len, offset + filesz)
                return self._elf_data[file_offset:end]
        return None

    def _collect_function_bbs(self, func_va: int, func_size: int):
        """Disassemble a function and discover basic block entries."""
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
                        if func_va <= target < func_va + func_size:
                            if target not in self._discovered_bbs:
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
            pass

    def discover_new_bbs(self, pid: int, bp_addr: int, max_discover: int = 32):
        """After hitting a breakpoint, disassemble forward and install new BPs."""
        if not self.deep_coverage or len(self.original_bytes) >= self.max_bps:
            return 0

        # Convert absolute bp_addr back to relative for ELF lookup
        if self._base_address is not None:
            rel_addr = bp_addr - self._base_address
        else:
            rel_addr = bp_addr

        # Find which function this belongs to
        func_start = None
        func_size = 0
        for start, end in self._func_ranges:
            if start <= rel_addr < end:
                func_start = start
                func_size = end - start
                break
        if func_start is None:
            return 0

        # Read bytes from ELF for this function, starting from bp
        scan_start = rel_addr
        func_bytes = self._read_func_bytes(scan_start, max_len=min(func_size - (scan_start - func_start), 512))
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
                        pass
        except Exception:
            pass

        return count

    def resolve_base(self, pid: int):
        """Read /proc/pid/maps to find the base address of the main executable."""
        try:
            with open(f"/proc/{pid}/maps") as f:
                for line in f:
                    if self.target_path in line or line.split()[-1].endswith("/" + os.path.basename(self.target_path)):
                        parts = line.split()
                        addr_range = parts[0].split("-")
                        self._base_address = int(addr_range[0], 16)
                        return
            # fallback: first r-xp mapping
            with open(f"/proc/{pid}/maps") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2 and parts[1] == "r-xp":
                        addr_range = parts[0].split("-")
                        self._base_address = int(addr_range[0], 16)
                        return
        except Exception:
            pass

    def _resolve_addr(self, rel_addr: int) -> int:
        if self._base_address is not None:
            return self._base_address + rel_addr
        return rel_addr

    def _ptrace(self, request, pid, addr=None, data=None):
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.ptrace.argtypes = [ctypes.c_long, ctypes.c_long,
                                ctypes.c_void_p, ctypes.c_void_p]
        libc.ptrace.restype = ctypes.c_long
        ctypes.set_errno(0)
        result = libc.ptrace(request, pid,
                            ctypes.c_void_p(addr) if addr else None,
                            ctypes.c_void_p(data) if data else None)
        return result

    def _read_memory(self, pid: int, addr: int) -> int:
        val = self._ptrace(PTRACE_PEEKDATA, pid, addr)
        return val & 0xFFFFFFFFFFFFFFFF

    def _write_memory(self, pid: int, addr: int, data_8: int):
        self._ptrace(PTRACE_POKEDATA, pid, addr, data_8)

    def install_breakpoints(self, pid: int):
        self.resolve_base(pid)
        self.original_bytes.clear()
        for rel_addr in self.bb_addrs:
            addr = self._resolve_addr(rel_addr)
            try:
                val = self._read_memory(pid, addr)
                self.original_bytes[addr] = val & 0xFF
                new_val = (val & ~0xFF) | INT3
                self._write_memory(pid, addr, new_val)
            except Exception:
                pass

    def remove_breakpoints(self, pid: int):
        for addr, orig in self.original_bytes.items():
            try:
                val = self._read_memory(pid, addr)
                new_val = (val & ~0xFF) | orig
                self._write_memory(pid, addr, new_val)
            except Exception:
                pass
        self.original_bytes.clear()

    def reset_edge_map(self):
        self.prev_location = 0
        self.total_edges = 0
        self._map_snapshot = bytes(self.edge_map)

    def record_edge(self, addr: int) -> bool:
        bucket = (addr ^ self.prev_location) % self.map_size
        self.prev_location = addr % self.map_size
        self.total_bp_hits += 1
        if self.edge_map[bucket] == 0:
            self.edge_map[bucket] = 1
            self.total_edges += 1
            self.cumulative_edges += 1
            return True
        return False

    def is_new_coverage(self) -> bool:
        return bytes(self.edge_map) != self._map_snapshot


class Fuzzer:
    def __init__(self, target, corpus_dir, crashes_dir, max_len=4096,
                 timeout=5, mutations_per_input=8, use_coverage=False,
                 deep_coverage=False, max_bps=50000,
                 dictionary=None, file_mode=False, target_args=None,
                 markov_order=1, markov_generate=False,
                 mc_bandit=False, mc_cem=False,
                 mc_elite_frac=0.1, mc_refit_interval=1000):
        self.target = target
        self.corpus_dir = Path(corpus_dir)
        self.crashes_dir = Path(crashes_dir)
        self.max_len = max_len
        self.timeout = timeout
        self.mutations_per_input = mutations_per_input
        self.use_coverage = use_coverage
        self.dictionary = dictionary or []
        self.file_mode = file_mode
        self.target_args = target_args or []
        self._tmp_dir = Path("/tmp") / f"fuzzer_{os.getpid()}"
        if self.file_mode:
            self._tmp_dir.mkdir(parents=True, exist_ok=True)

        self.ptrace_cov: PtraceCoverage | None = None
        if self.use_coverage:
            cov = PtraceCoverage(target, deep_coverage=deep_coverage, max_bps=max_bps)
            if cov.bb_addrs:
                self.ptrace_cov = cov
                mode = "deep (capstone)" if cov.deep_coverage else "function-entry"
                print(f"[*] Coverage: {len(cov.bb_addrs)} breakpoints ({mode}), "
                      f"map={cov.map_size}")
            else:
                print("[!] Coverage: no symbols found in ELF, "
                      "coverage disabled (use -g to compile with symbols)")
                print("[!] For closed-source binaries, use AFL++ QEMU mode: "
                      "afl-qemu-trace ./target")

        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        self.crashes_dir.mkdir(parents=True, exist_ok=True)

        self.corpus: list[bytes] = []
        self.seen_hashes: set[str] = set()
        self.crash_hashes: set[str] = set()
        self.crash_sigs: dict[str, int] = {}  # signature -> count
        self.exec_count = 0
        self.crash_count = 0
        self.start_time = time.time()
        self.last_report: SanitizerReport | None = None

        self.markov = MarkovChain(order=markov_order)
        self.markov_generate = markov_generate
        self.markov_trained = False

        self._load_corpus()
        if self.corpus:
            self.markov.train_corpus(self.corpus)
            self.markov_trained = self.markov.is_trained()

        self.mc_bandit = mc_bandit
        self.mc_cem = mc_cem
        self.mc = MonteCarloScheduler(
            elite_frac=mc_elite_frac,
            refit_interval=mc_refit_interval,
        ) if (mc_bandit or mc_cem) else None
        self._last_ops_used: list[str] = []

        if self.mc and self.mc_bandit:
            for op in MUTATIONS:
                self.mc.init_arm(op)
            for op in DICT_MUTATIONS:
                self.mc.init_arm(op)

    def _hash(self, data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()[:16]

    def _load_corpus(self):
        for f in self.corpus_dir.iterdir():
            if f.is_file():
                data = f.read_bytes()
                h = self._hash(data)
                if h not in self.seen_hashes:
                    self.seen_hashes.add(h)
                    self.corpus.append(data)
        if not self.corpus:
            self.corpus.append(b"AAAAAAAA")

    def _run_target(self, data: bytes) -> tuple[int, str]:
        """Execute target, return (returncode, stderr)."""
        if self.ptrace_cov:
            return self._run_target_ptrace(data)
        try:
            env = os.environ.copy()
            if self.use_coverage:
                env["AFL_MAP_SIZE"] = "65536"

            if self.file_mode:
                tmp_file = self._tmp_dir / f"fuzz_{os.getpid()}"
                tmp_file.write_bytes(data)
                cmd = [self.target] + [
                    a.replace("{file}", str(tmp_file)) for a in self.target_args
                ]
                try:
                    proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        env=env,
                        preexec_fn=os.setsid,
                    )
                    _, stderr = proc.communicate(timeout=self.timeout)
                    return proc.returncode, stderr.decode(errors="replace")
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    proc.wait()
                    return -1, "timeout"
                finally:
                    try:
                        tmp_file.unlink()
                    except OSError:
                        pass
            else:
                proc = subprocess.Popen(
                    [self.target],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    env=env,
                    preexec_fn=os.setsid,
                )
                try:
                    _, stderr = proc.communicate(input=data, timeout=self.timeout)
                    return proc.returncode, stderr.decode(errors="replace")
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    proc.wait()
                    return -1, "timeout"
        except Exception as e:
            return -2, str(e)

    def _run_target_ptrace(self, data: bytes) -> tuple[int, str]:
        """Run target under ptrace for edge coverage."""
        cov = self.ptrace_cov
        cov.reset_edge_map()
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.ptrace.argtypes = [ctypes.c_long, ctypes.c_long,
                                ctypes.c_void_p, ctypes.c_void_p]
        libc.ptrace.restype = ctypes.c_long

        stdin_r, stdin_w = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.setsid()
            os.dup2(stdin_r, 0)
            os.close(stdin_r)
            os.close(stdin_w)
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, 1)
            os.dup2(devnull, 2)
            os.close(devnull)
            libc.ptrace(PTRACE_TRACEME, 0, None, None)
            signal.signal(signal.SIGTRAP, signal.SIG_IGN)
            os.execv(self.target, [self.target])
            os._exit(127)

        os.close(stdin_r)
        os.write(stdin_w, data)
        os.close(stdin_w)

        try:
            _, status = os.waitpid(pid, 0)
            if not (os.WIFSTOPPED(status) and os.WSTOPSIG(status) == signal.SIGTRAP):
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
                return -2, "exec failed"

            cov.install_breakpoints(pid)
            libc.ptrace(PTRACE_CONT, pid, None, None)

            deadline = time.time() + self.timeout

            last_action = None
            while time.time() < deadline:
                _, status = os.waitpid(pid, os.WNOHANG | os.WUNTRACED)
                if status == 0:
                    time.sleep(0.0005)
                    continue

                if os.WIFEXITED(status) or os.WIFSIGNALED(status):
                    break

                if os.WIFSTOPPED(status):
                    sig = os.WSTOPSIG(status)
                    if sig == signal.SIGTRAP:
                        regs_buf = (ctypes.c_char * (27 * 8))()
                        libc.ptrace(PTRACE_GETREGS, pid, None, regs_buf)
                        rip = struct.unpack_from("<Q", bytes(regs_buf), 128)[0]
                        bp_addr = rip - 1

                        if bp_addr in cov.original_bytes:
                            orig = cov.original_bytes[bp_addr]
                            cov.record_edge(bp_addr)
                            val = cov._read_memory(pid, bp_addr)
                            cov._write_memory(pid, bp_addr, (val & ~0xFF) | orig)
                            del cov.original_bytes[bp_addr]
                            # Dynamic BB discovery for deep coverage
                            cov.discover_new_bbs(pid, bp_addr)
                            # fix RIP: INT3 sets RIP past the byte, move back
                            regs_buf2 = (ctypes.c_char * (27 * 8))()
                            libc.ptrace(PTRACE_GETREGS, pid, None, regs_buf2)
                            regs = bytearray(regs_buf2)
                            struct.pack_into("<Q", regs, 128, bp_addr)
                            libc.ptrace(PTRACE_SETREGS, pid, None, bytes(regs))
                        libc.ptrace(PTRACE_CONT, pid, None, None)
                        last_action = "cont"
                    else:
                        break
                else:
                    break

            if last_action == "cont":
                _, status = os.waitpid(pid, 0)
            else:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)

            returncode = (os.WEXITSTATUS(status) if os.WIFEXITED(status)
                          else -abs(os.WTERMSIG(status)))
            return returncode, ""

        except ChildProcessError:
            return 0, ""
        except Exception as e:
            try:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
            except Exception:
                pass
            return -2, str(e)

    def _getregs(self, pid: int) -> int:
        """Get instruction pointer (RIP on x86-64)."""
        buf = (ctypes.c_char * (27 * 8))()
        ctypes.CDLL(None).ptrace(PTRACE_GETREGS, pid, None, buf)
        return struct.unpack_from("<Q", bytes(buf), 16)[0]  # RIP offset 16

    SIGNAL_CRASH_CODES = {134, 135, 136, 139}  # SIGABRT, SIGBUS, SIGFPE, SIGSEGV

    def _is_interesting(self, returncode: int, stderr: str) -> bool:
        if returncode in self.SIGNAL_CRASH_CODES:
            return True
        if returncode < 0 and returncode != -1:
            return True
        if returncode in (-1, 0) and "ASAN" in stderr:
            return True
        if "Segmentation fault" in stderr:
            return True
        if "Aborted" in stderr:
            return True
        return False

    def _is_crash(self, returncode: int, stderr: str) -> bool:
        self.last_report = None
        if returncode == -2:
            return False

        report = SanitizerReport.parse(stderr)
        if report and report.is_valid():
            self.last_report = report
            return True

        if returncode in self.SIGNAL_CRASH_CODES:
            return True
        if returncode < 0:
            return True
        if any(sig in stderr for sig in ["SIGSEGV", "SIGABRT", "SIGFPE", "SIGBUS",
                                         "Segmentation fault", "Aborted"]):
            return True
        return False

    def mutate(self, data: bytes) -> bytes:
        buf = bytearray(data)
        if not buf:
            buf = bytearray(b"\x00" * random.randint(1, 32))

        ops = list(MUTATIONS)
        if self.dictionary:
            ops.extend(DICT_MUTATIONS)
        if self.markov_trained:
            ops.append("markov_bytes")
        if self.mc and self.mc_cem and self.mc.cem_fitted:
            ops.append("cem_bytes")
            self.mc.init_arm("cem_bytes")
        if self.mc and self.mc_bandit:
            for op in ops:
                self.mc.init_arm(op)

        self._last_ops_used = []

        for _ in range(self.mutations_per_input):
            op = self.mc.select_op(ops) if self.mc and self.mc_bandit else random.choice(ops)
            self._last_ops_used.append(op)

            if op == "bit_flip" and buf:
                byte_idx = random.randint(0, len(buf) - 1)
                bit_idx = random.randint(0, 7)
                buf[byte_idx] ^= (1 << bit_idx)

            elif op == "byte_flip" and buf:
                byte_idx = random.randint(0, len(buf) - 1)
                buf[byte_idx] ^= 0xFF

            elif op == "interesting_8" and buf:
                idx = random.randint(0, len(buf) - 1)
                buf[idx] = random.choice(INTERESTING_8)

            elif op == "interesting_16" and len(buf) >= 2:
                idx = random.randint(0, len(buf) - 2)
                val = random.choice(INTERESTING_16)
                struct.pack_into("<H", buf, idx, val)

            elif op == "interesting_32" and len(buf) >= 4:
                idx = random.randint(0, len(buf) - 4)
                val = random.choice(INTERESTING_32)
                struct.pack_into("<I", buf, idx, val)

            elif op == "random_bytes" and buf:
                idx = random.randint(0, len(buf) - 1)
                buf[idx] = random.randint(0, 255)

            elif op == "block_insert" and len(buf) < self.max_len:
                idx = random.randint(0, len(buf))
                size = random.randint(1, min(32, self.max_len - len(buf)))
                buf[idx:idx] = bytes(random.randint(0, 255) for _ in range(size))

            elif op == "block_delete" and len(buf) > 1:
                idx = random.randint(0, len(buf) - 1)
                max_size = min(32, len(buf) - idx, len(buf) - 1)
                if max_size >= 1:
                    size = random.randint(1, max_size)
                    del buf[idx:idx + size]

            elif op == "block_duplicate" and len(buf) < self.max_len:
                idx = random.randint(0, len(buf) - 1)
                size = random.randint(1, min(16, len(buf) - idx))
                block = buf[idx:idx + size]
                ins = random.randint(0, len(buf))
                buf[ins:ins] = block

            elif op == "dict_insert" and self.dictionary:
                token = random.choice(self.dictionary)
                if len(buf) + len(token) <= self.max_len:
                    idx = random.randint(0, len(buf))
                    buf[idx:idx] = token

            elif op == "dict_replace" and self.dictionary and buf:
                token = random.choice(self.dictionary)
                idx = random.randint(0, len(buf) - 1)
                end = min(idx + len(token), len(buf))
                buf[idx:end] = token[:end - idx]

            elif op == "markov_bytes" and buf:
                idx = random.randint(0, len(buf) - 1)
                ctx = bytes(buf[max(0, idx - self.markov.order):idx]) if self.markov.order else b""
                buf[idx] = self.markov.sample_byte(ctx)

            elif op == "cem_bytes" and self.mc and self.mc.cem_fitted:
                if buf:
                    idx = random.randint(0, len(buf) - 1)
                    buf[idx] = self.mc.cem_byte(idx)
                else:
                    length = random.randint(1, min(32, self.max_len))
                    buf = bytearray(self.mc.cem_sample(length))

            elif op == "havoc":
                return bytes(self._havoc_mutate(buf))

        return bytes(buf)

    def _havoc_mutate(self, buf: bytearray) -> bytearray:
        for _ in range(random.randint(2, 8)):
            self._apply_single_mutation(buf)
        return buf

    def _apply_single_mutation(self, buf: bytearray):
        if not buf:
            buf.extend(random.randint(0, 255) for _ in range(random.randint(1, 16)))
            return
        op = random.randint(0, 4)
        if op == 0:
            idx = random.randint(0, len(buf) - 1)
            buf[idx] ^= (1 << random.randint(0, 7))
        elif op == 1:
            idx = random.randint(0, len(buf) - 1)
            buf[idx] = random.randint(0, 255)
        elif op == 2 and len(buf) > 1:
            i, j = random.sample(range(len(buf)), 2)
            buf[i], buf[j] = buf[j], buf[i]
        elif op == 3 and len(buf) < self.max_len:
            idx = random.randint(0, len(buf))
            buf.insert(idx, random.randint(0, 255))
        elif op == 4 and len(buf) > 1:
            idx = random.randint(0, len(buf) - 1)
            size = random.randint(1, min(len(buf) - 1, len(buf) - idx))
            del buf[idx:idx + size]

    def save_crash(self, data: bytes, returncode: int, stderr: str):
        h = self._hash(data)
        if h in self.crash_hashes:
            return

        report = SanitizerReport.parse(stderr)
        sig = report.signature if report and report.is_valid() else f"signal:{abs(returncode)}"
        self.crash_hashes.add(h)
        self.crash_sigs[sig] = self.crash_sigs.get(sig, 0) + 1

        ts = int(time.time())
        crash_file = self.crashes_dir / f"crash_{ts}_{h}"
        crash_file.write_bytes(data)

        meta = crash_file.with_suffix(".txt")
        lines = [f"returncode: {returncode}"]
        if report and report.is_valid():
            lines.extend([
                f"sanitizer: {report.sanitizer}",
                f"error: {report.error_type}",
                f"fault_addr: {report.fault_addr}",
                f"signature: {sig}",
                f"seen: {self.crash_sigs[sig]}x",
                "",
                "=== stack trace ===",
            ])
            for i, frame in enumerate(report.frames[:12]):
                lines.append(f"  #{i} {frame}")
            lines.extend(["", "=== raw stderr ===", report.raw])
        else:
            lines.extend(["", "=== stderr ===", stderr])
        meta.write_text("\n".join(lines))

    def save_to_corpus(self, data: bytes):
        h = self._hash(data)
        if h in self.seen_hashes:
            return
        self.seen_hashes.add(h)
        self.corpus.append(data)
        self.markov.train(data)
        self.markov_trained = self.markov.is_trained()
        corpus_file = self.corpus_dir / f"id_{h}"
        corpus_file.write_bytes(data)

    def _pick_seed(self) -> bytes:
        if self.markov_generate and self.markov_trained and random.random() < 0.15:
            length = random.randint(1, self.max_len)
            return self.markov.generate(length)
        if self.corpus:
            return random.choice(self.corpus)
        return b"AAAAAAAA"

    def fuzz_one(self, data: bytes) -> bool:
        mutated = self.mutate(data)
        returncode, stderr = self._run_target(mutated)
        self.exec_count += 1

        is_crash = self._is_crash(returncode, stderr)
        is_interesting = self._is_interesting(returncode, stderr)
        has_new_coverage = (self.ptrace_cov and self.ptrace_cov.is_new_coverage())
        success = is_crash or is_interesting or has_new_coverage

        if self.mc and self.mc_bandit:
            seen = set()
            for op in self._last_ops_used:
                if op not in seen:
                    self.mc.record(op, success)
                    seen.add(op)

        if is_crash:
            self.crash_count += 1
            self.save_crash(mutated, returncode, stderr)
            if self.mc and self.mc_cem:
                self.mc.add_elite(mutated, 3)
                self.mc.maybe_refit()
            return True

        if is_interesting or has_new_coverage:
            self.save_to_corpus(mutated)
            if self.mc and self.mc_cem:
                self.mc.add_elite(mutated, 2)
                self.mc.maybe_refit()
            return True

        return False

    def print_stats(self):
        elapsed = time.time() - self.start_time
        eps = self.exec_count / elapsed if elapsed > 0 else 0
        dict_str = f" | dict: {len(self.dictionary)}" if self.dictionary else ""
        markov_str = " | markov: trained" if self.markov_trained else ""
        if self.markov_generate:
            markov_str += "+gen"
        cov_str = ""
        if self.ptrace_cov:
            cov_str = f" | edges: {self.ptrace_cov.cumulative_edges} hits: {self.ptrace_cov.total_bp_hits}"
            if self.ptrace_cov.deep_coverage:
                cov_str += f" bps:{len(self.ptrace_cov.original_bytes)}"
        mc_str = ""
        if self.mc:
            parts = []
            if self.mc_bandit:
                parts.append("bandit")
            if self.mc_cem:
                parts.append(f"cem:{len(self.mc.elite_set)}")
            if parts:
                mc_str = " | mc: " + "+".join(parts)
        sig_str = f" | sigs: {len(self.crash_sigs)}" if self.crash_sigs else ""
        print(f"\r[*] execs: {self.exec_count} | corpus: {len(self.corpus)} | "
              f"crashes: {self.crash_count}{sig_str} | eps: {eps:.0f} | "
              f"time: {elapsed:.0f}s{dict_str}{markov_str}{cov_str}{mc_str}", end="", flush=True)

    def run(self, iterations=0):
        print(f"[*] Target: {self.target}")
        print(f"[*] Corpus: {self.corpus_dir} ({len(self.corpus)} seeds)")
        print(f"[*] Crashes: {self.crashes_dir}")
        print(f"[*] Max input length: {self.max_len}")
        print(f"[*] Timeout: {self.timeout}s")
        if self.dictionary:
            print(f"[*] Dictionary: {len(self.dictionary)} tokens")
        if self.markov_trained:
            print(f"[*] Markov chain: order={self.markov.order}, "
                  f"transitions={len(self.markov.transitions)}")
        if self.markov_generate:
            print(f"[*] Markov generation: enabled (15% of seeds)")
        if self.mc:
            if self.mc_bandit:
                print(f"[*] MC bandit: Thompson sampling over {len(self.mc.arm_alpha)} arms")
            if self.mc_cem:
                print(f"[*] MC CEM: elite_frac={self.mc.elite_frac}, "
                      f"refit_interval={self.mc.refit_interval}")
        print(f"[*] Starting fuzzing...\n")

        i = 0
        try:
            while True:
                if iterations and i >= iterations:
                    break
                seed = self._pick_seed()
                self.fuzz_one(seed)
                i += 1
                if i % 100 == 0:
                    self.print_stats()
        except KeyboardInterrupt:
            pass

        self.print_stats()
        print(f"\n\n[*] Fuzzing stopped. {self.crash_count} crashes found "
              f"({len(self.crash_sigs)} unique signatures).")
        if self.crash_sigs:
            print("[*] Crash signatures:")
            for sig, count in sorted(self.crash_sigs.items(), key=lambda x: -x[1]):
                print(f"    {sig} ({count}x)")
            print(f"\n[*] Crash files in: {self.crashes_dir}")


def main():
    parser = argparse.ArgumentParser(description="Coverage-guided binary fuzzer")
    parser.add_argument("target", help="Path to target binary")
    parser.add_argument("-d", "--corpus", default=None,
                        help="Corpus directory (default: ~/fuzzing/<target>/corpus)")
    parser.add_argument("-o", "--crashes", default=None,
                        help="Crashes directory (default: ~/fuzzing/<target>/crashes)")
    parser.add_argument("-m", "--max-len", type=int, default=4096, help="Max input length")
    parser.add_argument("-t", "--timeout", type=float, default=5, help="Timeout in seconds")
    parser.add_argument("-n", "--iterations", type=int, default=0, help="Number of iterations (0=infinite)")
    parser.add_argument("-M", "--mutations", type=int, default=8, help="Mutations per input")
    parser.add_argument("-c", "--coverage", action="store_true", help="Enable coverage-guided mode")
    parser.add_argument("--deep-coverage", action="store_true",
                        help="Enable capstone-based basic block discovery (requires -c)")
    parser.add_argument("--max-bps", type=int, default=50000,
                        help="Max breakpoints for deep coverage (default: 50000)")
    parser.add_argument("-D", "--dict", help="Dictionary file (one token per line, NAME=value or raw bytes)")
    parser.add_argument("-F", "--file-mode", action="store_true",
                        help="Write input to temp file instead of stdin")
    parser.add_argument("-A", "--target-args", nargs=argparse.REMAINDER,
                        help="Target arguments (use {file} as placeholder for temp file)")
    parser.add_argument("--markov", action="store_true",
                        help="Enable Markov chain mutation (trained on corpus)")
    parser.add_argument("--markov-gen", action="store_true",
                        help="Enable Markov chain seed generation (15%% of seeds)")
    parser.add_argument("--markov-order", type=int, default=1,
                        help="Markov chain order (default: 1)")
    parser.add_argument("--mc-bandit", action="store_true",
                        help="Enable Thompson sampling for mutation operator selection")
    parser.add_argument("--mc-cem", action="store_true",
                        help="Enable cross-entropy method for byte distribution learning")
    parser.add_argument("--mc-elite-frac", type=float, default=0.1,
                        help="Fraction of elite set to fit CEM (default: 0.1)")
    parser.add_argument("--mc-refit-int", type=int, default=1000,
                        help="Refit CEM every N executions (default: 1000)")
    args = parser.parse_args()

    if not os.path.isfile(args.target):
        print(f"[-] Target not found: {args.target}")
        sys.exit(1)

    if not os.access(args.target, os.X_OK):
        print(f"[-] Target not executable: {args.target}")
        sys.exit(1)

    target_name = os.path.basename(os.path.abspath(args.target))
    fuzz_dir = Path.home() / "fuzzing" / target_name
    corpus_dir = args.corpus or str(fuzz_dir / "corpus")
    crashes_dir = args.crashes or str(fuzz_dir / "crashes")

    dictionary = []
    if args.dict:
        if not os.path.isfile(args.dict):
            print(f"[-] Dictionary not found: {args.dict}")
            sys.exit(1)
        dictionary = load_dictionary(args.dict)
        print(f"[*] Loaded {len(dictionary)} tokens from {args.dict}")

    use_markov = args.markov or args.markov_gen

    fuzzer = Fuzzer(
        target=args.target,
        corpus_dir=corpus_dir,
        crashes_dir=crashes_dir,
        max_len=args.max_len,
        timeout=args.timeout,
        mutations_per_input=args.mutations,
        use_coverage=args.coverage,
        deep_coverage=args.deep_coverage,
        max_bps=args.max_bps,
        dictionary=dictionary,
        file_mode=args.file_mode,
        target_args=args.target_args,
        markov_order=args.markov_order if use_markov else 0,
        markov_generate=args.markov_gen,
        mc_bandit=args.mc_bandit,
        mc_cem=args.mc_cem,
        mc_elite_frac=args.mc_elite_frac,
        mc_refit_interval=args.mc_refit_int,
    )
    fuzzer.run(iterations=args.iterations)


if __name__ == "__main__":
    main()
