"""Fuzzer orchestration: coordinates mutations, execution, and coverage."""

import atexit
import contextlib
import ctypes
import json
import logging
import math
import os
import random
import resource
import signal
import struct
import threading
import time
from pathlib import Path

from fuzzer_tool.adapters.filesystem import load_corpus, save_crash, save_to_corpus
from fuzzer_tool.adapters.process import (
    SIGNAL_CRASH_CODES,
    _child_pids,
    run_target_file,
    run_target_stdin,
)
from fuzzer_tool.adapters.shm import ShmCoverage
from fuzzer_tool.core.bloom import BloomFilter
from fuzzer_tool.core.markov import MarkovChain
from fuzzer_tool.core.montecarlo import MonteCarloScheduler
from fuzzer_tool.core.mutations import (
    DICT_MUTATIONS,
    INTERESTING_8,
    INTERESTING_16,
    INTERESTING_32,
    MUTATIONS,
    splice,
)
from fuzzer_tool.core.sanitizer import SanitizerReport

log = logging.getLogger(__name__)

_shutdown = False


def _kill_children(sig=None, frame=None):
    global _shutdown
    _shutdown = True
    for pid in list(_child_pids):
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.killpg(os.getpgid(pid), signal.SIGKILL)
    _child_pids.clear()


atexit.register(_kill_children)
signal.signal(signal.SIGTERM, _kill_children)
signal.signal(signal.SIGINT, _kill_children)


try:
    from capstone import CS_ARCH_X86, CS_MODE_64, Cs
    from capstone.x86_const import X86_GRP_CALL, X86_GRP_INT, X86_GRP_JUMP, X86_GRP_RET

    HAS_CAPSTONE = True
except ImportError:
    HAS_CAPSTONE = False


def _write_and_close(fd: int, data: bytes) -> None:
    """Write *data* to *fd* then close it — designed to run in a thread."""
    try:
        os.write(fd, data)
    finally:
        try:
            os.close(fd)
        except OSError:
            log.debug("Failed to close fd %d (already closed?)", fd)


def _cleanup_tmp_dir(path: Path) -> None:
    """Remove temp directory on exit."""
    import shutil

    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        log.debug("Failed to clean up %s", path, exc_info=True)


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

        if self.deep_coverage:
            self._disassembler = Cs(CS_ARCH_X86, CS_MODE_64)
            self._disassembler.detail = True
            self._parse_elf_segments()

        self._collect_basic_blocks()

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

        self._parse_symbol_table(data, symtab_sec, strtab_sec)
        if not self.bb_addrs:
            self._parse_symbol_table(data, dynsym_sec, dynstr_sec)

        if self.deep_coverage and HAS_CAPSTONE:
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
            log.debug("sym_entsize == 0 in section, skipping malformed symbol table")
            return
        sym_count = sym_size // sym_entsize if sym_entsize else 0

        for i in range(min(sym_count, 10000)):
            sym = sym_offset + i * sym_entsize
            st_info = data[sym + 4]
            st_value = struct.unpack_from("<Q", data, sym + 8)[0]
            st_size = struct.unpack_from("<Q", data, sym + 16)[0]
            st_type = st_info & 0xF
            if st_type == 2 and st_value > 0 and st_size > 0:
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


class Fuzzer:
    def __init__(
        self,
        target,
        corpus_dir,
        crashes_dir,
        max_len=4096,
        timeout=5,
        mutations_per_input=8,
        use_coverage=False,
        deep_coverage=False,
        max_bps=50000,
        dictionary=None,
        file_mode=False,
        target_args=None,
        markov_order=1,
        markov_generate=False,
        mc_bandit=False,
        mc_cem=False,
        mc_elite_frac=0.1,
        mc_refit_interval=1000,
        stats_file=None,
        stats_interval=1000,
        coverage_report=None,
        coverage_log=None,
        grammar=None,
        persistent=False,
        seed=42,
    ):
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
        self.coverage_report = Path(coverage_report) if coverage_report else None
        self.coverage_log = Path(coverage_log) if coverage_log else None
        self.grammar = grammar
        self.persistent = persistent
        self.seed = seed
        random.seed(seed)
        self._tmp_dir = Path("/tmp") / f"fuzzer_{os.getpid()}"
        if self.file_mode:
            self._tmp_dir.mkdir(parents=True, exist_ok=True)
            atexit.register(_cleanup_tmp_dir, self._tmp_dir)

        self.ptrace_cov: PtraceCoverage | None = None
        self.shm_cov: ShmCoverage | None = None
        if self.use_coverage:
            try:
                self.shm_cov = ShmCoverage()
                print(f"[*] Coverage: AFL SHM bitmap, id={self.shm_cov.env_id}")
            except OSError:
                cov = PtraceCoverage(target, deep_coverage=deep_coverage, max_bps=max_bps)
                if cov.bb_addrs:
                    self.ptrace_cov = cov
                    mode = "deep (capstone)" if cov.deep_coverage else "function-entry"
                    print(
                        f"[*] Coverage: {len(cov.bb_addrs)} breakpoints ({mode}), map={cov.map_size}"
                    )
                else:
                    print(
                        "[!] Coverage: no symbols found in ELF, "
                        "coverage disabled (use -g to compile with symbols)"
                    )
                    print(
                        "[!] For closed-source binaries, use AFL++ QEMU mode: afl-qemu-trace ./target"
                    )

        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        self.crashes_dir.mkdir(parents=True, exist_ok=True)

        self.corpus: list[bytes] = []
        self.seen_hashes: set[str] = set()
        self.bloom = BloomFilter(capacity=100_000)
        self.crash_hashes: set[str] = set()
        self.crash_sigs: dict[str, int] = {}
        self.exec_count = 0
        self.crash_count = 0
        self.timeout_count = 0
        self.start_time = time.time()
        self.last_report: SanitizerReport | None = None
        self.op_counts: dict[str, int] = {}
        self.op_success: dict[str, int] = {}
        self._peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self.stats_file = Path(stats_file) if stats_file else None
        self.stats_interval = stats_interval

        self.markov = MarkovChain(order=markov_order)
        self.markov_generate = markov_generate
        self.markov_trained = False

        self._load_corpus()
        self._init_seed_metadata()
        if self.corpus:
            self.markov.train_corpus(self.corpus)
            self.markov_trained = self.markov.is_trained()

        self.mc_bandit = mc_bandit
        self.mc_cem = mc_cem
        self.mc = (
            MonteCarloScheduler(
                elite_frac=mc_elite_frac,
                refit_interval=mc_refit_interval,
            )
            if (mc_bandit or mc_cem)
            else None
        )
        self._last_ops_used: list[str] = []

        if self.mc and self.mc_bandit:
            for op in MUTATIONS:
                self.mc.init_arm(op)
            for op in DICT_MUTATIONS:
                self.mc.init_arm(op)
            self.mc.init_arm("markov_bytes")
            self.mc.init_arm("cem_bytes")
            if self.grammar:
                self.mc.init_arm("grammar_mutate")

        self._persistent_runner = None
        if self.persistent:
            from fuzzer_tool.adapters.persistent import PersistentRunner

            self._persistent_runner = PersistentRunner(target=self.target, timeout=self.timeout)
            if self._persistent_runner.start():
                print("[*] Persistent mode: target started")
            else:
                print("[!] Persistent mode: failed to start target, falling back to fork")
                self._persistent_runner = None

    def _load_corpus(self):
        self.corpus, self.seen_hashes = load_corpus(self.corpus_dir, self.bloom)

    def _init_seed_metadata(self):
        now = time.time()
        self.seed_meta: dict[bytes, dict] = {}
        for seed in self.corpus:
            self.seed_meta[seed] = {
                "fuzz_count": 0,
                "coverage_edges": 0,
                "added_at": now,
            }

    def _run_target(self, data: bytes) -> tuple[int, str]:
        if self._persistent_runner:
            return self._persistent_runner.run_one(data)

        if self.ptrace_cov:
            return self._run_target_ptrace(data)

        if self.shm_cov:
            self.shm_cov.reset_edge_map()

        env = os.environ.copy()
        if self.use_coverage:
            env["AFL_MAP_SIZE"] = "65536"
        if self.shm_cov:
            env["__AFL_SHM_ID"] = self.shm_cov.env_id

        if self.file_mode:
            return run_target_file(
                self.target,
                data,
                self.timeout,
                str(self._tmp_dir),
                self.target_args,
                env=env,
            )
        return run_target_stdin(self.target, data, self.timeout, env=env)

    def _run_target_ptrace(self, data: bytes) -> tuple[int, str]:
        cov = self.ptrace_cov
        cov.reset_edge_map()
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.ptrace.argtypes = [
            ctypes.c_long,
            ctypes.c_long,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        libc.ptrace.restype = ctypes.c_long

        stdin_r, stdin_w = os.pipe()
        writer = None
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
        # Write data in a thread to avoid deadlock when data > PIPE_BUF (~64KB).
        # The child may be stopped at exec's SIGTRAP before reading stdin, so a
        # blocking write would stall the parent before it can call waitpid.
        writer = threading.Thread(target=_write_and_close, args=(stdin_w, data))
        writer.start()

        try:
            _, status = os.waitpid(pid, 0)
            if os.WIFSTOPPED(status) and os.WSTOPSIG(status) == signal.SIGTRAP:
                pass  # normal: child stopped at exec, install breakpoints
            elif os.WIFSTOPPED(status):
                sig = os.WSTOPSIG(status)
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
                return -sig, ""  # child crashed before we could instrument it
            elif os.WIFSIGNALED(status):
                return -os.WTERMSIG(status), ""
            elif os.WIFEXITED(status):
                return os.WEXITSTATUS(status), ""
            else:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
                return -2, "exec failed"

            cov.install_breakpoints(pid)
            libc.ptrace(PTRACE_CONT, pid, None, None)

            deadline = time.time() + self.timeout

            last_action = None
            last_sig = 0
            while time.time() < deadline:
                _, status = os.waitpid(pid, os.WNOHANG | os.WUNTRACED)
                if status == 0:
                    time.sleep(0.0005)
                    continue

                if os.WIFEXITED(status) or os.WIFSIGNALED(status):
                    break

                if os.WIFSTOPPED(status):
                    sig = os.WSTOPSIG(status)
                    last_sig = sig
                    if sig == signal.SIGTRAP:
                        regs_buf = (ctypes.c_char * (27 * 8))()
                        libc.ptrace(PTRACE_GETREGS, pid, None, regs_buf)
                        # RIP is at offset 128 in user_regs_struct (x86-64 Linux)
                        rip = struct.unpack_from("<Q", bytes(regs_buf), 128)[0]
                        bp_addr = rip - 1

                        if bp_addr in cov.original_bytes:
                            orig = cov.original_bytes[bp_addr]
                            cov.record_edge(bp_addr)
                            val = cov._read_memory(pid, bp_addr)
                            cov._write_memory(pid, bp_addr, (val & ~0xFF) | orig)
                            del cov.original_bytes[bp_addr]
                            cov.discover_new_bbs(pid, bp_addr)
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

            if last_action == "cont" and last_sig == signal.SIGTRAP:
                _, status = os.waitpid(pid, os.WNOHANG | os.WUNTRACED)
                if os.WIFSTOPPED(status):
                    libc.ptrace(PTRACE_CONT, pid, None, None)
                    _, status = os.waitpid(pid, 0)
            else:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)

            if os.WIFSIGNALED(status):
                returncode = -os.WTERMSIG(status)
            elif os.WIFEXITED(status):
                returncode = os.WEXITSTATUS(status)
            elif os.WIFSTOPPED(status):
                returncode = -os.WSTOPSIG(status)
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
            else:
                returncode = 0
            return returncode, ""

        except ChildProcessError:
            return 0, ""
        except Exception as e:
            try:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
            except Exception:
                log.debug("Failed to kill orphan pid %d", pid, exc_info=True)
            return -2, str(e)
        finally:
            if writer is not None:
                writer.join(timeout=self.timeout)

    def _is_interesting(self, returncode: int, stderr: str) -> bool:
        if returncode in SIGNAL_CRASH_CODES:
            return True
        if returncode < 0 and returncode != -1:
            return True
        if returncode in (-1, 0) and "ASAN" in stderr:
            return True
        if "Segmentation fault" in stderr:
            return True
        return "Aborted" in stderr

    def _is_crash(self, returncode: int, stderr: str) -> bool:
        self.last_report = None
        if returncode in (-2, -1):
            return False

        report = SanitizerReport.parse(stderr)
        if report and report.is_valid():
            self.last_report = report
            return True

        if returncode in SIGNAL_CRASH_CODES:
            return True
        if returncode < 0:
            return True
        return any(
            sig in stderr
            for sig in [
                "SIGSEGV",
                "SIGABRT",
                "SIGFPE",
                "SIGBUS",
                "Segmentation fault",
                "Aborted",
            ]
        )

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
        if self.grammar:
            ops.append("grammar_mutate")

        self._last_ops_used = []

        for _ in range(self.mutations_per_input):
            op = self.mc.select_op(ops) if self.mc and self.mc_bandit else random.choice(ops)
            self._last_ops_used.append(op)

            if op == "bit_flip" and buf:
                byte_idx = random.randint(0, len(buf) - 1)
                bit_idx = random.randint(0, 7)
                buf[byte_idx] ^= 1 << bit_idx

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
                    del buf[idx : idx + size]

            elif op == "block_duplicate" and len(buf) < self.max_len:
                idx = random.randint(0, len(buf) - 1)
                size = random.randint(1, min(16, len(buf) - idx))
                block = buf[idx : idx + size]
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
                buf[idx:end] = token[: end - idx]

            elif op == "markov_bytes" and buf:
                idx = random.randint(0, len(buf) - 1)
                ctx = (
                    bytes(buf[max(0, idx - self.markov.order) : idx]) if self.markov.order else b""
                )
                buf[idx] = self.markov.sample_byte(ctx)

            elif op == "cem_bytes" and self.mc and self.mc.cem_fitted:
                if buf:
                    idx = random.randint(0, len(buf) - 1)
                    buf[idx] = self.mc.cem_byte(idx)
                else:
                    length = random.randint(1, min(32, self.max_len))
                    buf = bytearray(self.mc.cem_sample(length))

            elif op == "splice" and len(self.corpus) >= 2:
                a = random.choice(self.corpus)
                b = random.choice(self.corpus)
                if a is not data and b is not data:
                    buf = bytearray(splice(a, b)[: self.max_len])
                else:
                    others = [c for c in self.corpus if c is not data]
                    if others:
                        other = random.choice(others)
                        buf = bytearray(splice(bytes(buf), other)[: self.max_len])

            elif op == "grammar_mutate" and self.grammar:
                mutated = self.grammar.mutate(bytes(buf), max_len=self.max_len)
                buf = bytearray(mutated[: self.max_len])

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
            buf[idx] ^= 1 << random.randint(0, 7)
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
            del buf[idx : idx + size]

    def save_crash(self, data: bytes, returncode: int, stderr: str):
        save_crash(
            data,
            returncode,
            stderr,
            self.crashes_dir,
            self.crash_hashes,
            self.crash_sigs,
        )

    def save_to_corpus(self, data: bytes):
        if save_to_corpus(data, self.corpus_dir, self.seen_hashes, self.bloom):
            self.corpus.append(data)
            self.seed_meta[data] = {
                "fuzz_count": 0,
                "coverage_edges": 0,
                "added_at": time.time(),
            }
            self.markov.train(data)
            self.markov_trained = self.markov.is_trained()

    def _pick_seed(self) -> bytes:
        if self.markov_generate and self.markov_trained and random.random() < 0.15:
            length = random.randint(1, self.max_len)
            return self.markov.generate(length)
        if self.corpus and self.seed_meta:
            return self._weighted_pick_seed()
        if self.corpus:
            return random.choice(self.corpus)
        return b"AAAAAAAA"

    def _weighted_pick_seed(self) -> bytes:
        now = time.time()
        weights = []
        for seed in self.corpus:
            meta = self.seed_meta.get(seed)
            if meta is None:
                weights.append(1.0)
                continue
            fuzz_count = max(meta["fuzz_count"], 1)
            coverage = meta["coverage_edges"]
            age = now - meta["added_at"]
            w = (1.0 / math.sqrt(fuzz_count)) * (1.0 + coverage * 0.5) / (1.0 + age * 0.01)
            weights.append(max(w, 1e-6))
        return random.choices(self.corpus, weights=weights, k=1)[0]

    def fuzz_one(self, data: bytes) -> bool:
        meta = self.seed_meta.get(data)
        if meta is not None:
            meta["fuzz_count"] += 1

        mutated = self.mutate(data)
        returncode, stderr = self._run_target(mutated)
        self.exec_count += 1
        if self.mc:
            self.mc.execs_since_refit += 1

        if self.exec_count % 100 == 0:
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if rss > self._peak_rss:
                self._peak_rss = rss

        for op in set(self._last_ops_used):
            self.op_counts[op] = self.op_counts.get(op, 0) + 1

        is_timeout = returncode == -1 and stderr == "timeout"
        if is_timeout:
            self.timeout_count += 1

        is_crash = self._is_crash(returncode, stderr)
        is_interesting = self._is_interesting(returncode, stderr)
        has_new_coverage = (self.ptrace_cov and self.ptrace_cov.is_new_coverage()) or (
            self.shm_cov and self.shm_cov.is_new_coverage()
        )
        success = is_crash or is_interesting or has_new_coverage

        if success:
            for op in set(self._last_ops_used):
                self.op_success[op] = self.op_success.get(op, 0) + 1

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
            if meta is not None and has_new_coverage:
                meta["coverage_edges"] += 1
            self.save_to_corpus(mutated)
            if self.mc and self.mc_cem:
                self.mc.add_elite(mutated, 2)
                self.mc.maybe_refit()
            return True

        return False

    def _dump_stats(self):
        if not self.stats_file:
            return
        elapsed = time.time() - self.start_time
        eps = self.exec_count / elapsed if elapsed > 0 else 0
        stats = {
            "timestamp": time.time(),
            "exec_count": self.exec_count,
            "crash_count": self.crash_count,
            "timeout_count": self.timeout_count,
            "corpus_size": len(self.corpus),
            "unique_crash_sigs": len(self.crash_sigs),
            "eps": round(eps, 1),
            "elapsed_sec": round(elapsed, 1),
            "peak_rss_kb": self._peak_rss,
            "op_counts": dict(self.op_counts),
            "op_success": dict(self.op_success),
        }
        if self.mc and self.mc_bandit:
            stats["bandit_stats"] = {
                k: {"successes": v[0], "failures": v[1]} for k, v in self.mc.bandit_stats().items()
            }
        if self.mc and self.mc_cem:
            stats["cem_elite_size"] = len(self.mc.elite_set)
            stats["cem_fitted"] = self.mc.cem_fitted
        try:
            self.stats_file.parent.mkdir(parents=True, exist_ok=True)
            self.stats_file.write_text(json.dumps(stats, indent=2))
        except OSError:
            log.debug("Failed to write stats to %s", self.stats_file, exc_info=True)

    def _dump_coverage_report(self):
        if not self.coverage_report:
            return
        edge_map = None
        if self.shm_cov:
            edge_map = self.shm_cov.edge_map
        elif self.ptrace_cov:
            edge_map = self.ptrace_cov.edge_map
        if edge_map is None:
            print("[!] No coverage data available for report")
            return

        hit_edges = []
        cumulative = 0
        for i, val in enumerate(edge_map):
            if val:
                hit_edges.append(i)
                cumulative += 1

        report = {
            "map_size": len(edge_map),
            "cumulative_edges": cumulative,
            "hit_edges": hit_edges,
            "coverage_pct": round(cumulative / len(edge_map) * 100, 4),
            "exec_count": self.exec_count,
            "corpus_size": len(self.corpus),
        }
        self.coverage_report.parent.mkdir(parents=True, exist_ok=True)
        self.coverage_report.write_text(json.dumps(report, indent=2))
        print(
            f"\n[*] Coverage report: {self.coverage_report} "
            f"({cumulative}/{len(edge_map)} edges, {report['coverage_pct']}%)"
        )

    def _append_coverage_log(self):
        if not self.coverage_log:
            return
        cumulative = 0
        if self.shm_cov:
            cumulative = self.shm_cov.cumulative_edges
        elif self.ptrace_cov:
            cumulative = self.ptrace_cov.cumulative_edges
        elapsed = time.time() - self.start_time
        line = (
            f"{elapsed:.1f},{self.exec_count},{cumulative},{len(self.corpus)},{self.crash_count}\n"
        )
        self.coverage_log.parent.mkdir(parents=True, exist_ok=True)
        with open(self.coverage_log, "a") as f:
            f.write(line)

    def print_stats(self):
        elapsed = time.time() - self.start_time
        eps = self.exec_count / elapsed if elapsed > 0 else 0
        dict_str = f" | dict: {len(self.dictionary)}" if self.dictionary else ""
        markov_str = " | markov: trained" if self.markov_trained else ""
        if self.markov_generate:
            markov_str += "+gen"
        cov_str = ""
        if self.shm_cov:
            cov_str = f" | shm-edges: {self.shm_cov.cumulative_edges}"
        elif self.ptrace_cov:
            cov_str = (
                f" | edges: {self.ptrace_cov.cumulative_edges}"
                f" hits: {self.ptrace_cov.total_bp_hits}"
            )
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
        sig_str = f"({len(self.crash_sigs)}sigs)" if self.crash_sigs else ""
        timeout_str = f" | timeouts: {self.timeout_count}" if self.timeout_count else ""
        rss_kb = self._peak_rss
        rss_str = f" | rss: {rss_kb // 1024}MB" if rss_kb >= 1024 else f" | rss: {rss_kb}KB"
        ops_str = ""
        if self.op_counts:
            rates = []
            for op, count in sorted(self.op_counts.items(), key=lambda x: -x[1])[:3]:
                succ = self.op_success.get(op, 0)
                pct = succ / count * 100 if count else 0
                rates.append(f"{op}:{pct:.0f}%")
            ops_str = " | ops: " + " ".join(rates)
        print(
            f"\r[*] execs: {self.exec_count} | corpus: {len(self.corpus)} | "
            f"crashes: {self.crash_count}{sig_str}{timeout_str} | eps: {eps:.0f} | "
            f"time: {elapsed:.0f}s{rss_str}{ops_str}{dict_str}{markov_str}{cov_str}{mc_str}",
            end="",
            flush=True,
        )

    def run(self, iterations=0):
        print(f"[*] Target: {self.target}")
        print(f"[*] Corpus: {self.corpus_dir} ({len(self.corpus)} seeds)")
        print(f"[*] Crashes: {self.crashes_dir}")
        print(f"[*] Max input length: {self.max_len}")
        print(f"[*] Timeout: {self.timeout}s")
        print(f"[*] Seed: {self.seed}")
        if self.grammar:
            print(f"[*] Grammar: {len(self.grammar.rules)} rules")
        if self.persistent:
            print("[*] Persistent mode: enabled")
        if self.dictionary:
            print(f"[*] Dictionary: {len(self.dictionary)} tokens")
        if self.markov_trained:
            print(
                f"[*] Markov chain: order={self.markov.order}, "
                f"transitions={len(self.markov.transitions)}"
            )
        if self.markov_generate:
            print("[*] Markov generation: enabled (15% of seeds)")
        if self.mc:
            if self.mc_bandit:
                print(f"[*] MC bandit: Thompson sampling over {len(self.mc.arm_alpha)} arms")
            if self.mc_cem:
                print(
                    f"[*] MC CEM: elite_frac={self.mc.elite_frac}, "
                    f"refit_interval={self.mc.refit_interval}"
                )
        if self.stats_file:
            print(f"[*] Stats: {self.stats_file} every {self.stats_interval} iterations")
        print("[*] Starting fuzzing...\n")

        i = 0
        try:
            while not _shutdown:
                if iterations and i >= iterations:
                    break
                seed = self._pick_seed()
                self.fuzz_one(seed)
                i += 1
                if i % 100 == 0:
                    self.print_stats()
                    self._append_coverage_log()
                if self.stats_file and i % self.stats_interval == 0:
                    self._dump_stats()
        except (KeyboardInterrupt, SystemExit, OSError):
            pass

        self._dump_stats()
        self._dump_coverage_report()
        self.print_stats()
        print(
            f"\n\n[*] Fuzzing stopped. {self.crash_count} crashes found "
            f"({len(self.crash_sigs)} unique signatures)."
        )
        if self.crash_sigs:
            print("[*] Crash signatures:")
            for sig, count in sorted(self.crash_sigs.items(), key=lambda x: -x[1]):
                print(f"    {sig} ({count}x)")
            print(f"\n[*] Crash files in: {self.crashes_dir}")
        if self.mc and self.mc_bandit:
            print("\n[*] Bandit convergence:")
            for name, (succ, fail) in sorted(
                self.mc.bandit_stats().items(),
                key=lambda x: -(x[1][0] / max(x[1][0] + x[1][1], 1)),
            ):
                total = succ + fail
                pct = succ / total * 100 if total else 0
                print(f"    {name:20s}: {succ:.0f}/{fail:.0f} ({pct:.0f}% success)")
