"""Persistent subprocess loader for coverage-guided fuzzing.

Keeps one Python subprocess alive across iterations, communicating
via stdin/stdout pipes. Eliminates Python startup + ctypes.CDLL load
overhead on every iteration.

Protocol:
  Init:   "INIT <target> <func>\n"  ->  "READY\n"
  Run:    "RUN <len>\n<data>"       ->  "RC <rc> <bmp_len>\n<bmp>"
  Quit:   "QUIT\n"
"""

import contextlib
import logging
import os
import subprocess
import sys
import tempfile

log = logging.getLogger(__name__)

_PERSISTENT_LOADER = r"""
import ctypes, os, struct, sys

lib = None
fn = None
bitmap_start = 0
bitmap_size = 0

def parse_elf_sancov(path):
    # Parse ELF to find __start/__stop___sancov_cntrs virtual addresses.
    try:
        with open(path, 'rb') as f:
            elf = f.read()
        if len(elf) < 64 or elf[:4] != b'\x7fELF' or elf[4] != 2 or elf[5] != 1:
            return None, None
        e_shoff = struct.unpack_from('<Q', elf, 40)[0]
        e_shnum = struct.unpack_from('<H', elf, 60)[0]
        e_shentsize = struct.unpack_from('<H', elf, 58)[0]
        e_shstrndx = struct.unpack_from('<H', elf, 62)[0]
        if e_shnum == 0 or e_shstrndx >= e_shnum:
            return None, None
        shstr_off = e_shoff + e_shstrndx * e_shentsize
        shstr_offset = struct.unpack_from('<Q', elf, shstr_off + 24)[0]
        symtab_sec = strtab_sec = None
        for i in range(e_shnum):
            sh = e_shoff + i * e_shentsize
            sh_type = struct.unpack_from('<I', elf, sh + 4)[0]
            sh_name_idx = struct.unpack_from('<I', elf, sh)[0]
            name = elf[shstr_offset + sh_name_idx:shstr_offset + sh_name_idx + 32].split(b'\x00')[0]
            if sh_type == 2 and name == b'.symtab':
                symtab_sec = sh
            elif sh_type == 3 and name == b'.strtab':
                strtab_sec = sh
        if symtab_sec is None or strtab_sec is None:
            return None, None
        sym_offset = struct.unpack_from('<Q', elf, symtab_sec + 24)[0]
        sym_size = struct.unpack_from('<Q', elf, symtab_sec + 32)[0]
        sym_entsize = struct.unpack_from('<Q', elf, symtab_sec + 56)[0]
        if sym_entsize == 0:
            return None, None
        strtab_offset = struct.unpack_from('<Q', elf, strtab_sec + 24)[0]
        start_addr = stop_addr = None
        for i in range(sym_size // sym_entsize):
            sym = sym_offset + i * sym_entsize
            st_value = struct.unpack_from('<Q', elf, sym + 8)[0]
            st_name_idx = struct.unpack_from('<I', elf, sym)[0]
            name = elf[strtab_offset + st_name_idx:strtab_offset + st_name_idx + 64].split(b'\x00')[0].decode(errors='replace')
            if name == '__start___sancov_cntrs' and st_value > 0:
                start_addr = st_value
            elif name == '__stop___sancov_cntrs' and st_value > 0:
                stop_addr = st_value
        if start_addr is not None and stop_addr is not None:
            return start_addr, stop_addr
    except Exception:
        pass
    return None, None

def find_base_addr(target_path):
    # Find the runtime base address from /proc/self/maps.
    basename = os.path.basename(target_path)
    try:
        with open(f'/proc/{os.getpid()}/maps') as f:
            for line in f:
                if basename in line and 'r-xp' in line:
                    return int(line.split('-')[0], 16)
    except Exception:
        pass
    return None

# Read init
header = sys.stdin.buffer.readline().decode()
parts = header.strip().split()
target_path = parts[1]
func_name = parts[2]

# Load target
shim_path = os.environ.get("_COV_SHM_PATH")
if shim_path and os.path.exists(shim_path):
    ctypes.CDLL(shim_path, mode=ctypes.RTLD_GLOBAL)

lib = ctypes.CDLL(target_path)
fn_ptr = getattr(lib, func_name)
fn_ptr.restype = ctypes.c_int
fn_ptr.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]

# Find sancov counters via ELF parsing + /proc/self/maps
elf_start, elf_stop = parse_elf_sancov(target_path)
if elf_start is not None and elf_stop is not None:
    base = find_base_addr(target_path)
    if base is not None:
        # The ELF vaddr is relative to the first LOAD segment (vaddr=0)
        bitmap_start = base + elf_start
        bitmap_size = elf_stop - elf_start

sys.stdout.buffer.write(b"READY\n")
sys.stdout.buffer.flush()

while True:
    line = sys.stdin.buffer.readline()
    if not line:
        break
    cmd = line.decode().strip()
    if cmd == "QUIT":
        break
    if cmd.startswith("RUN "):
        data_len = int(cmd.split()[1])
        data = sys.stdin.buffer.read(data_len)
        buf = (ctypes.c_uint8 * len(data))(*data)
        try:
            rc = fn_ptr(buf, len(data))
        except Exception:
            rc = -11

        bmp = b""
        if bitmap_start and bitmap_size > 0:
            try:
                bmp = bytes((ctypes.c_uint8 * bitmap_size).from_address(bitmap_start))
            except Exception:
                pass

        resp = f"RC {rc} {len(bmp)}\n".encode()
        sys.stdout.buffer.write(resp)
        if bmp:
            sys.stdout.buffer.write(bmp)
        sys.stdout.buffer.flush()
"""


class PersistentLoader:
    """Persistent subprocess — one process, many calls.

    The subprocess loads the target .so once and calls
    LLVMFuzzerTestOneInput repeatedly via stdin/stdout pipes.
    """

    def __init__(self, target: str, function_name: str = "LLVMFuzzerTestOneInput",
                 timeout: float = 5.0):
        self.target = target
        self.function_name = function_name
        self.timeout = timeout
        self._proc: subprocess.Popen | None = None
        self._loader_path: str | None = None
        self._ready = False
        self._last_bitmap: bytes | None = None

    def start(self) -> bool:
        if self._proc and self._proc.poll() is None:
            return True

        fd, self._loader_path = tempfile.mkstemp(suffix=".py", prefix="fuzz_persist_")
        os.write(fd, _PERSISTENT_LOADER.encode())
        os.close(fd)

        env = os.environ.copy()
        # Pass shim path for coverage
        from fuzzer_tool.adapters.shim_factory import build_minimal_shim
        shim = build_minimal_shim()
        if shim:
            env["_COV_SHM_PATH"] = shim

        self._proc = subprocess.Popen(
            [sys.executable, self._loader_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
        )

        init = f"INIT {self.target} {self.function_name}\n"
        self._proc.stdin.write(init.encode())
        self._proc.stdin.flush()

        resp = self._proc.stdout.readline()
        if resp.strip() == b"READY":
            self._ready = True
            log.info("Persistent loader started: %s", self.target)
            return True

        log.warning("Persistent loader failed to start")
        return False

    def run_one(self, data: bytes) -> tuple[int, bytes | None]:
        if not self._ready or not self._proc:
            return -2, None

        cmd = f"RUN {len(data)}\n"
        self._proc.stdin.write(cmd.encode())
        self._proc.stdin.write(data)
        self._proc.stdin.flush()

        header = self._proc.stdout.readline()
        if not header:
            return -2, None

        parts = header.decode().strip().split()
        if len(parts) < 3 or parts[0] != "RC":
            return -2, None

        rc = int(parts[1])
        bmp_len = int(parts[2])

        bitmap = None
        if bmp_len > 0:
            bitmap = self._proc.stdout.read(bmp_len)

        self._last_bitmap = bitmap
        return rc, bitmap

    def stop(self):
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.stdin.write(b"QUIT\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=2)
            except Exception:
                with contextlib.suppress(Exception):
                    self._proc.kill()
                    self._proc.wait()
        self._proc = None
        self._ready = False
        if self._loader_path and os.path.exists(self._loader_path):
            with open(self._loader_path, "w") as f:
                f.write("")
            os.unlink(self._loader_path)
            self._loader_path = None

    def __del__(self):
        self.stop()
