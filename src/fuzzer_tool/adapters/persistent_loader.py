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
import threading

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
    # Find the runtime base address and ELF vaddr of the r-xp LOAD segment.
    # Both are needed to compute the ELF load bias.
    basename = os.path.basename(target_path)
    try:
        with open(f'/proc/{os.getpid()}/maps') as f:
            for line in f:
                if basename in line and 'r-xp' in line:
                    return int(line.split('-')[0], 16)
    except Exception:
        pass
    return None

def find_rxp_vaddr(elf_data):
    # Find the ELF virtual address of the first r-xp LOAD segment.
    if len(elf_data) < 64 or elf_data[:4] != b'\\x7fELF' or elf_data[4] != 2 or elf_data[5] != 1:
        return None
    e_phoff = struct.unpack_from('<Q', elf_data, 32)[0]
    e_phentsize = struct.unpack_from('<H', elf_data, 54)[0]
    e_phnum = struct.unpack_from('<H', elf_data, 56)[0]
    for i in range(e_phnum):
        off = e_phoff + i * e_phentsize
        p_type = struct.unpack_from('<I', elf_data, off)[0]
        if p_type == 1:  # PT_LOAD
            p_vaddr = struct.unpack_from('<Q', elf_data, off + 16)[0]
            p_flags = struct.unpack_from('<I', elf_data, off + 4)[0]
            if p_flags & 0x5:  # PF_R | PF_X = r-x
                return p_vaddr
    return None

# Read init
header = sys.stdin.buffer.readline().decode()
parts = header.strip().split()
target_path = parts[1]
func_name = parts[2]

# Detect target type
is_executable = (os.path.isfile(target_path) and os.access(target_path, os.X_OK)
                 and not target_path.endswith(('.so', '.dylib', '.dll')))

if is_executable:
    # Standalone executable — run as subprocess, read bitmap from file
    import subprocess as _subprocess
    bitmap_out = os.environ.get('_COV_BITMAP_OUT', '')

    def run_executable(data):
        env = os.environ.copy()
        if bitmap_out:
            env['_COV_BITMAP_OUT'] = bitmap_out
        proc = _subprocess.Popen(
            [target_path],
            stdin=_subprocess.PIPE,
            stdout=_subprocess.DEVNULL,
            stderr=_subprocess.DEVNULL,
            env=env,
        )
        try:
            proc.communicate(input=data, timeout=float(os.environ.get('_TIMEOUT', '5')))
        except _subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        bmp = b''
        if bitmap_out and os.path.exists(bitmap_out):
            try:
                with open(bitmap_out, 'rb') as f:
                    bmp = f.read()
            except Exception:
                pass
        return proc.returncode, bmp

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
            rc, bmp = run_executable(data)
            resp = f"RC {rc} {len(bmp)}\n".encode()
            sys.stdout.buffer.write(resp)
            if bmp:
                sys.stdout.buffer.write(bmp)
            sys.stdout.buffer.flush()

else:
    # Shared library — load via ctypes
    shim_path = os.environ.get("_COV_SHM_PATH")
    if shim_path and os.path.exists(shim_path):
        ctypes.CDLL(shim_path, mode=ctypes.RTLD_GLOBAL)

    lib = ctypes.CDLL(target_path)
    fn_ptr = getattr(lib, func_name)
    fn_ptr.restype = ctypes.c_int
    fn_ptr.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]

    elf_start, elf_stop = parse_elf_sancov(target_path)
    if elf_start is not None and elf_stop is not None:
        base = find_base_addr(target_path)
        if base is not None:
            # Read ELF to find r-xp segment vaddr for bias computation
            with open(target_path, 'rb') as f:
                elf_data = f.read()
            rxp_vaddr = find_rxp_vaddr(elf_data)
            if rxp_vaddr is not None:
                # ELF load bias = runtime_rxp_start - elf_rxp_vaddr
                # This single offset translates ANY ELF vaddr to runtime
                bias = base - rxp_vaddr
                bitmap_start = bias + elf_start
                bitmap_size = elf_stop - elf_start
            else:
                # Fallback: assume base is already the bias (PIE with 0-base)
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

    Uses a C loader (compiled at startup) for maximum speed.
    Falls back to Python loader if C compilation fails.
    """

    _c_loader_path: str | None = None

    def __init__(
        self, target: str, function_name: str = "LLVMFuzzerTestOneInput", timeout: float = 5.0
    ):
        self.target = target
        self.function_name = function_name
        self.timeout = timeout
        self._proc: subprocess.Popen | None = None
        self._loader_path: str | None = None
        self._ready = False
        self._last_bitmap: bytes | None = None
        self._bitmap_out: str | None = None
        self._restarting: bool = False

    @classmethod
    def _ensure_c_loader(cls) -> str | None:
        """Compile the C loader if not already done. Returns path to binary."""
        if cls._c_loader_path and os.path.exists(cls._c_loader_path):
            return cls._c_loader_path

        loader_c = os.path.join(os.path.dirname(__file__), "fuzz_loader.c")
        if not os.path.exists(loader_c):
            return None

        # Compile C loader — include PID to avoid races under --jobs N
        fd, out_path = tempfile.mkstemp(suffix="_loader", prefix=f"fuzz_loader_{os.getpid()}_")
        os.close(fd)
        try:
            result = subprocess.run(
                ["gcc", "-O2", "-o", out_path, loader_c],
                capture_output=True,
                timeout=10,
            )
            if result.returncode == 0 and os.path.exists(out_path):
                cls._c_loader_path = out_path
                return out_path
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Try clang as fallback
        try:
            result = subprocess.run(
                ["clang", "-O2", "-o", out_path, loader_c],
                capture_output=True,
                timeout=10,
            )
            if result.returncode == 0 and os.path.exists(out_path):
                cls._c_loader_path = out_path
                return out_path
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        with contextlib.suppress(OSError):
            os.unlink(out_path)
        return None

    def start(self) -> bool:
        if self._proc and self._proc.poll() is None:
            return True

        # Try C loader first, fall back to Python
        c_loader = self._ensure_c_loader()
        if c_loader:
            self._use_c_loader(c_loader)
        else:
            self._use_python_loader()

        if self._proc and self._ready:
            return True

        log.warning("Persistent loader failed to start")
        return False

    def _use_c_loader(self, c_loader_path: str):
        """Start the C loader subprocess."""
        fd, self._bitmap_out = tempfile.mkstemp(suffix=".cov", prefix=f"fuzz_cov_{os.getpid()}_")
        os.close(fd)
        env = os.environ.copy()
        env["_COV_BITMAP_OUT"] = self._bitmap_out
        env["_TIMEOUT"] = str(self.timeout)

        self._proc = subprocess.Popen(
            [c_loader_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
        )

        init = f"INIT {self.target} {self.function_name} {self._bitmap_out} {int(self.timeout)}\n"
        self._proc.stdin.write(init.encode())
        self._proc.stdin.flush()

        resp = self._proc.stdout.readline()
        if resp.strip() == b"READY":
            self._ready = True
            log.info("Persistent C loader started: %s", self.target)
            return

        log.warning("Persistent C loader failed to start")

    def _use_python_loader(self):
        """Start the Python loader subprocess (fallback)."""
        fd, self._loader_path = tempfile.mkstemp(suffix=".py", prefix="fuzz_persist_")
        os.write(fd, _PERSISTENT_LOADER.encode())
        os.close(fd)

        env = os.environ.copy()
        from fuzzer_tool.adapters.shim_factory import build_minimal_shim

        shim = build_minimal_shim()
        if shim:
            env["_COV_SHM_PATH"] = shim
        fd, self._bitmap_out = tempfile.mkstemp(suffix=".cov", prefix=f"fuzz_cov_{os.getpid()}_")
        os.close(fd)
        env["_COV_BITMAP_OUT"] = self._bitmap_out
        env["_TIMEOUT"] = str(self.timeout)

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
            log.info("Persistent Python loader started: %s", self.target)

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
            # Use a thread to read bitmap with timeout, preventing indefinite block
            # if the subprocess stalls mid-response
            result = [None]

            def _read():
                result[0] = self._proc.stdout.read(bmp_len)

            t = threading.Thread(target=_read, daemon=True)
            t.start()
            t.join(timeout=self.timeout)
            if t.is_alive():
                log.warning("Bitmap read timed out after %.1fs, restarting loader", self.timeout)
                with contextlib.suppress(Exception):
                    self._proc.kill()
                    self._proc.wait()
                self._ready = False
                # Auto-restart (once) to avoid unbounded recursion
                if not self._restarting:
                    self._restarting = True
                    try:
                        if self.start():
                            return self.run_one(data)
                    finally:
                        self._restarting = False
                return -1, None
            bitmap = result[0]

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
