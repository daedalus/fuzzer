"""In-process target execution via ctypes or direct Python call.

Three modes:
  - direct (--inprocess-direct): ctypes.CDLL call, zero overhead.
    Target MUST handle errors internally (setjmp/longjmp or noexcept).
    A SIGSEGV in the target kills the fuzzer process.
  - subprocess (--inprocess): subprocess loader, process isolation.
    When coverage is enabled, uses a persistent subprocess that stays
    alive across iterations to eliminate Python startup overhead.
  - python: direct in-process call.
"""

import contextlib
import ctypes
import importlib
import logging
import os
import signal
import subprocess
import sys
import tempfile
from collections.abc import Callable

from fuzzer_tool.adapters.shim_factory import (
    BitmapReader,
    ShimResult,
    build_shim,
    cleanup_shim,
    load_shim,
)

log = logging.getLogger(__name__)

_LOADER_SCRIPT = """\
import ctypes
import ctypes.util
import os
import subprocess
import sys

target = sys.argv[1]
func_name = sys.argv[2]
data = sys.stdin.buffer.read()

# Standalone executable — run directly
if os.path.isfile(target) and os.access(target, os.X_OK) \
        and not target.endswith(('.so', '.dylib', '.dll')):
    proc = subprocess.Popen(
        [target],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        _, stderr = proc.communicate(input=data, timeout=int(os.environ.get('_TIMEOUT', '5')))
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    sys.exit(max(0, min(proc.returncode, 125)))

# Shared library — load via ctypes
shim_path = os.environ.get("_COV_SHM_PATH")
if shim_path and os.path.exists(shim_path):
    ctypes.CDLL(shim_path, mode=ctypes.RTLD_GLOBAL)

lib = ctypes.CDLL(target)
fn = getattr(lib, func_name)
fn.restype = ctypes.c_int
fn.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]

buf = (ctypes.c_uint8 * len(data))(*data)
rc = fn(buf, len(data))

# Read coverage bitmap from shim
if shim_path and os.path.exists(shim_path):
    try:
        shim = ctypes.CDLL(shim_path)
        bmp_ptr = shim.cov_get_bitmap()
        bmp_size = shim.cov_get_size()
        if bmp_ptr and bmp_size:
            bitmap = (ctypes.c_uint8 * bmp_size).from_address(bmp_ptr)
            out_path = os.environ.get("_COV_BITMAP_OUT")
            if out_path:
                with open(out_path, "wb") as f:
                    f.write(bytes(bitmap))
    except OSError:
        pass

# Also try reading from SHM (AFL shim targets)
shm_id_str = os.environ.get("__AFL_SHM_ID")
if shm_id_str:
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
        libc.shmat.restype = ctypes.c_void_p
        ptr = libc.shmat(int(shm_id_str), None, 0)
        if ptr and ptr != -1:
            map_size = int(os.environ.get("AFL_MAP_SIZE", "65536"))
            bitmap = (ctypes.c_uint8 * map_size).from_address(ptr)
            out_path = os.environ.get("_COV_BITMAP_OUT")
            if out_path:
                with open(out_path, "wb") as f:
                    f.write(bytes(bitmap))
    except Exception:
        pass

sys.exit(max(0, min(rc, 125)))
"""


class InProcessRunner:
    """Call target function with minimal overhead.

    direct=True: ctypes.CDLL call in-process (fastest, but target must
    not SIGSEGV — use setjmp/longjmp or ASAN-instrumented builds).
    direct=False: subprocess loader with process isolation.
    When coverage is enabled in subprocess mode, uses a persistent
    subprocess that stays alive to eliminate Python startup overhead.
    """

    def __init__(
        self,
        target: str,
        function_name: str = "LLVMFuzzerTestOneInput",
        timeout: float = 5.0,
        shm_size: int = 65536,
        direct: bool = False,
        direct_lite: bool = False,
        coverage_env_id: str | None = None,
        cov: bool = False,
        debug: bool = False,
    ):
        self.target = target
        self.function_name = function_name
        self.timeout = timeout
        self.shm_size = shm_size
        self.direct = direct
        self.direct_lite = direct_lite
        self.coverage_env_id = coverage_env_id
        self.debug = debug

        self._func: Callable[[bytes], int] | None = None
        self._lib: ctypes.CDLL | None = None
        self._func_ptr = None  # cached function pointer
        self._is_c = False
        self._loader_path: str | None = None
        self._bitmap_out: str | None = None

        # Shim state
        self._shim: ShimResult | None = None
        self._shim_handle: ctypes.CDLL | None = None
        self._bitmap_reader: BitmapReader | None = None

        # Persistent loader state
        self._persistent = None

        # Forkserver state
        self._forkserver = None

        self._start()

    def _start(self):
        target_lower = self.target.lower()
        if target_lower.endswith((".so", ".dylib", ".dll")) or (
            "." not in self.target and self.function_name
        ):
            self._start_c()
        else:
            self._start_python()

    def _start_c(self):
        mode = "direct" if self.direct else "subprocess"
        cov = bool(self.coverage_env_id)

        # Build coverage shim via factory
        if cov:
            self._shim = build_shim(self.target, mode=mode)
            if self._shim.compile_error:
                log.warning("Shim build failed: %s", self._shim.compile_error)

        if self.direct or self.direct_lite:
            # Direct mode: load shim with RTLD_GLOBAL, then load target
            shim_loaded = False
            if self._shim and self._shim.shim_path and self._shim.needs_preload:
                self._shim_handle = load_shim(self._shim.shim_path, mode="direct")
                shim_loaded = True
            # Set __AFL_SHM_ID and AFL_MAP_SIZE BEFORE loading library so
            # the instrumented code can attach to SHM during initialization
            # with the correct size. Without AFL_MAP_SIZE, the compiled-in
            # shim defaults to 65536 which mismatches the auto-sized SHM
            # and causes OOB writes that hang or corrupt memory.
            if self.coverage_env_id:
                os.environ["__AFL_SHM_ID"] = self.coverage_env_id
                os.environ["AFL_MAP_SIZE"] = str(self.shm_size)
            if cov and not self._bitmap_out:
                fd, self._bitmap_out = tempfile.mkstemp(suffix=".cov", prefix="fuzz_cov_")
                os.close(fd)
            try:
                self._lib = ctypes.CDLL(self.target)
                # The constructor __afl_auto_init may not fire during
                # ctypes.CDLL loading (getenv() may not see os.environ
                # updates at constructor time). Check if __afl_area was
                # attached; if not, call __afl_map_shm() manually.
                try:
                    afl_area = ctypes.c_void_p.in_dll(self._lib, '__afl_area')
                    if self.debug:
                        print(
                            f"  [debug] __afl_area={afl_area.value}, "
                            f"__AFL_SHM_ID={os.environ.get('__AFL_SHM_ID')}, "
                            f"AFL_MAP_SIZE={os.environ.get('AFL_MAP_SIZE')}",
                            flush=True,
                        )
                    if not afl_area.value and self.coverage_env_id:
                        getattr(self._lib, '__afl_map_shm')()
                        if self.debug:
                            afl_area2 = ctypes.c_void_p.in_dll(self._lib, '__afl_area')
                            print(
                                f"  [debug] After manual __afl_map_shm: __afl_area={afl_area2.value}",
                                flush=True,
                            )
                except (OSError, AttributeError):
                    pass
                fn_ptr = getattr(self._lib, self.function_name)
                fn_ptr.restype = ctypes.c_int
                fn_ptr.argtypes = [
                    ctypes.POINTER(ctypes.c_uint8),
                    ctypes.c_size_t,
                ]
                self._func_ptr = fn_ptr  # cache the resolved symbol
                if cov and self._lib:
                    self._bitmap_reader = BitmapReader(self.target, self._lib)
                    if not self._bitmap_reader.valid:
                        log.warning("BitmapReader: no sancov counters in target")
            except OSError as e:
                if shim_loaded:
                    log.warning("Direct mode failed (%s), falling back to subprocess", e)
                    self.direct = False
                    self._lib = None
                else:
                    raise

        if not self.direct and not self.direct_lite:
            # Set __AFL_SHM_ID and AFL_MAP_SIZE in process env so
            # subprocess loaders and forkserver inherit the correct values.
            # Without AFL_MAP_SIZE, the forkserver defaults to 65536 which
            # mismatches the auto-sized SHM and causes OOB writes.
            if self.coverage_env_id:
                os.environ["__AFL_SHM_ID"] = self.coverage_env_id
                os.environ["AFL_MAP_SIZE"] = str(self.shm_size)

            # Try persistent subprocess first (faster: one process, many calls)
            if cov:
                from fuzzer_tool.adapters.persistent_loader import PersistentLoader

                self._persistent = PersistentLoader(
                    target=self.target,
                    function_name=self.function_name,
                    timeout=self.timeout,
                )
                if not self._persistent.start():
                    log.warning("Persistent loader failed, falling back to per-call")
                    self._persistent = None

            # Try forkserver (C binary) for standalone executables only.
            # .so targets use PersistentLoader (crash isolation via fork-per-call).
            is_so = self.target.lower().endswith((".so", ".dylib", ".dll"))
            if not self._persistent and not is_so:
                from fuzzer_tool.adapters.forkserver import ForkserverRunner

                self._forkserver = ForkserverRunner(
                    target=self.target,
                    function_name=self.function_name,
                    timeout=self.timeout,
                )
                if self._forkserver.start():
                    log.info("Forkserver: using C binary for %s", self.target)
                else:
                    self._forkserver = None

            if not self._persistent and not self._forkserver:
                # Per-call subprocess mode (fallback)
                fd, self._loader_path = tempfile.mkstemp(suffix=".py", prefix="fuzz_loader_")
                os.write(fd, _LOADER_SCRIPT.encode())
                os.close(fd)
                if cov:
                    fd, self._bitmap_out = tempfile.mkstemp(suffix=".cov", prefix="fuzz_cov_")
                    os.close(fd)

        self._is_c = True
        if self._forkserver:
            loader_type = "forkserver"
        elif self._persistent:
            loader_type = "persistent"
        elif self._loader_path:
            loader_type = "loader"
        else:
            loader_type = "none"
        log.info(
            "In-process C target: %s::%s (mode=%s, coverage=%s, loader=%s)",
            self.target,
            self.function_name,
            mode,
            self._shim.coverage_type if self._shim else "none",
            loader_type,
        )

    def _start_python(self):
        mod_path, _, func_name = self.target.rpartition(":")
        mod = importlib.import_module(mod_path)
        fn = getattr(mod, func_name)
        self._func = fn
        self._is_c = False
        log.info("In-process Python target: %s:%s", mod_path, func_name)

    # ------------------------------------------------------------------
    # Coverage
    # ------------------------------------------------------------------

    def read_bitmap(self) -> bytes | None:
        """Read the coverage bitmap.

        Checks sancov counters first, then SHM (for AFL shim targets).
        If sancov counters are all zeros (target uses SHM instead), falls
        through to read from SHM directly.
        """
        # Try sancov counters first
        if self.direct and self._bitmap_reader and self._bitmap_reader.valid:
            bm = self._bitmap_reader.read_bitmap()
            if bm and any(b != 0 for b in bm):
                return bm
            # sancov counters empty — target may use SHM instead
        if self._persistent and self._persistent._last_bitmap is not None:
            return self._persistent._last_bitmap
        if self._forkserver and self._forkserver._last_bitmap is not None:
            return self._forkserver._last_bitmap
        if self._bitmap_out and os.path.exists(self._bitmap_out):
            try:
                with open(self._bitmap_out, "rb") as f:
                    data = f.read()
                    if data:  # Only return if file has content
                        return data
            except OSError:
                pass
        # Read from SHM (AFL shim targets write here)
        if self.coverage_env_id:
            try:
                # Cache SHM attachment for performance
                if not hasattr(self, "_shm_ptr") or self._shm_ptr is None:
                    import ctypes.util as _ct_util

                    libc = ctypes.CDLL(_ct_util.find_library("c") or "libc.so.6")
                    libc.shmat.restype = ctypes.c_void_p
                    self._shm_ptr = libc.shmat(int(self.coverage_env_id), None, 0)
                if self._shm_ptr and self._shm_ptr != -1:
                    return (ctypes.c_uint8 * self.shm_size).from_address(self._shm_ptr)
            except Exception:
                log.warning(
                    "shmat read failed for coverage_env_id=%s", self.coverage_env_id, exc_info=True
                )
        return None

    def reset_bitmap(self):
        """Reset the coverage bitmap to zero."""
        if self.direct and self._bitmap_reader and self._bitmap_reader.valid:
            self._bitmap_reader.reset_bitmap()
        # Reset SHM for AFL shim targets (use cached pointer)
        if self.coverage_env_id:
            try:
                # Cache SHM attachment for performance
                if not hasattr(self, "_shm_ptr") or self._shm_ptr is None:
                    import ctypes.util as _ct_util

                    libc = ctypes.CDLL(_ct_util.find_library("c") or "libc.so.6")
                    libc.shmat.restype = ctypes.c_void_p
                    self._shm_ptr = libc.shmat(int(self.coverage_env_id), None, 0)
                if self._shm_ptr and self._shm_ptr != -1:
                    ctypes.memset(self._shm_ptr, 0, self.shm_size)
            except Exception:
                log.warning(
                    "shmat reset failed for coverage_env_id=%s", self.coverage_env_id, exc_info=True
                )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run_one(self, data: bytes) -> tuple[int, str]:
        """Execute target on one input. Returns (returncode, stderr_str)."""
        if self._is_c:
            if self.direct_lite:
                return self._run_c_direct_lite(data)
            if self.direct:
                return self._run_c_direct(data)
            if self._persistent:
                return self._run_c_persistent(data)
            if self._forkserver:
                return self._run_c_forkserver(data)
            return self._run_c_subprocess(data)
        return self._run_python(data)

    def _run_python(self, data: bytes) -> tuple[int, str]:
        if self._func is None:
            return -2, "runner not initialized"
        try:
            rc = self._func(data)
            return int(rc), ""
        except Exception as e:
            return -2, str(e)

    def _run_c_direct(self, data: bytes) -> tuple[int, str]:
        """Direct ctypes.CDLL call — zero overhead.

        Catches SIGSEGV/SIGABRT via signal handler so target crashes
        and ASAN errors don't kill the fuzzer process.
        Timeout via SIGALRM + setitimer.
        """
        if self._lib is None or self._func_ptr is None:
            return -2, "runner not initialized"
        if self._coverage_enabled():
            self.reset_bitmap()

        crashed = False
        crashed_sig = 0
        timed_out = False
        stderr_buf = []

        def _crash_handler(signum, frame):
            nonlocal crashed, crashed_sig
            crashed = True
            crashed_sig = signum

        def _alarm_handler(signum, frame):
            nonlocal timed_out
            timed_out = True

        old_segv = signal.signal(signal.SIGSEGV, _crash_handler)
        old_abrt = signal.signal(signal.SIGABRT, _crash_handler)
        old_alarm = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.setitimer(signal.ITIMER_REAL, self.timeout)
        try:
            # Capture stderr for ASAN output
            old_stderr_fd = os.dup(2)
            read_fd, write_fd = os.pipe()
            os.dup2(write_fd, 2)
            os.close(write_fd)

            buf = (ctypes.c_uint8 * len(data))(*data)
            rc = self._func_ptr(buf, len(data))

            # Restore stderr and read any captured output
            os.dup2(old_stderr_fd, 2)
            os.close(old_stderr_fd)
            os.set_blocking(read_fd, False)
            try:
                stderr_buf.append(os.read(read_fd, 65536))
            except OSError:
                pass
            os.close(read_fd)

            if timed_out:
                return -1, "timeout"
            return rc, b"".join(stderr_buf).decode(errors="replace")
        except Exception as e:
            # Restore stderr on exception
            try:
                os.dup2(old_stderr_fd, 2)
                os.close(old_stderr_fd)
            except Exception:
                pass
            return -2, str(e)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_alarm)
            signal.signal(signal.SIGSEGV, old_segv)
            signal.signal(signal.SIGABRT, old_abrt)
            if crashed:
                # 128 + signal number
                return 128 + crashed_sig, ""

    def _run_c_direct_lite(self, data: bytes) -> tuple[int, str]:
        """Lightweight direct ctypes call — minimal overhead.

        Catches SIGSEGV/SIGABRT via signal handler so target crashes
        and ASAN errors don't kill the fuzzer process.
        Uses cached ctypes buffer for maximum throughput.
        Timeout via SIGALRM + setitimer.
        """
        if self._lib is None or self._func_ptr is None:
            return -2, "runner not initialized"
        # Skip bitmap reset — runner already calls shm.reset_edge_map()
        # Cache ctypes buffer allocation
        if not hasattr(self, "_c_buf") or self._c_buf is None or len(self._c_buf) != len(data):
            self._c_buf = (ctypes.c_uint8 * len(data))(*data)
        else:
            ctypes.memmove(self._c_buf, data, len(data))

        crashed = False
        crashed_sig = 0
        timed_out = False

        def _crash_handler(signum, frame):
            nonlocal crashed, crashed_sig
            crashed = True
            crashed_sig = signum

        # Initialize signal handlers
        old_segv = signal.signal(signal.SIGSEGV, _crash_handler)
        old_abrt = signal.signal(signal.SIGABRT, _crash_handler)
        if not hasattr(self, "_alarm_handler"):
            self._timed_out = False
            def _alarm_handler(signum, frame):
                self._timed_out = True
            self._alarm_handler = _alarm_handler
            self._old_alarm_handler = signal.signal(signal.SIGALRM, self._alarm_handler)

        self._timed_out = False
        signal.setitimer(signal.ITIMER_REAL, self.timeout)
        try:
            rc = self._func_ptr(self._c_buf, len(data))
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGSEGV, old_segv)
            signal.signal(signal.SIGABRT, old_abrt)
            if crashed:
                return 128 + crashed_sig, ""

        if self._timed_out:
            return -1, "timeout"
        return rc, ""

    def _run_c_persistent(self, data: bytes) -> tuple[int, str]:
        """Persistent subprocess — one process, many calls."""
        rc, bitmap = self._persistent.run_one(data)
        self._persistent._last_bitmap = bitmap
        return rc, ""

    def _run_c_forkserver(self, data: bytes) -> tuple[int, str]:
        """Forkserver via compiled C binary."""
        rc, bitmap = self._forkserver.run_one(data)
        self._forkserver._last_bitmap = bitmap
        return rc, ""

    def _run_c_subprocess(self, data: bytes) -> tuple[int, str]:
        if self._loader_path is None:
            return -2, "loader not initialized"
        try:
            env = os.environ.copy()
            env["_TIMEOUT"] = str(int(self.timeout))

            if self._shim and self._shim.shim_path:
                env["_COV_SHM_PATH"] = self._shim.shim_path
            if self._bitmap_out:
                env["_COV_BITMAP_OUT"] = self._bitmap_out
            if self.coverage_env_id:
                env["__AFL_SHM_ID"] = self.coverage_env_id
                env["AFL_MAP_SIZE"] = str(self.shm_size)

            proc = subprocess.Popen(
                [sys.executable, self._loader_path, self.target, self.function_name],
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
                with contextlib.suppress(OSError):
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                for _ in range(10):
                    try:
                        proc.wait(timeout=0.5)
                        break
                    except subprocess.TimeoutExpired:
                        continue
                return -1, "timeout"
        except Exception as e:
            return -2, str(e)

    def _coverage_enabled(self) -> bool:
        return (self._shim is not None and self._shim.coverage_type != "none") or bool(
            self.coverage_env_id
        )

    def update_shm_after_resize(self, new_ptr: int, new_size: int, new_env_id: str = "") -> None:
        """Patch target's __afl_area and invalidate cached SHM pointer after resize.

        When SHM is resized, the old segment is detached and a new one allocated.
        In inprocess mode the target's constructor already attached to the old
        SHM — we must redirect it to the new one, or coverage writes go to freed memory.
        """
        # Invalidate cached SHM pointer so read_bitmap() re-attaches
        self._shm_ptr = None
        self.shm_size = new_size
        # Update coverage_env_id to the new SHM so read_bitmap() attaches
        # to the correct (new) segment, not the old (removed) one.
        if new_env_id:
            self.coverage_env_id = new_env_id

        # Re-run __afl_map_shm() to update __afl_area, __afl_map_size,
        # and __afl_map_mask to the new SHM. The env vars (__AFL_SHM_ID
        # and AFL_MAP_SIZE) are already updated by the caller.
        if self._lib is not None:
            try:
                getattr(self._lib, '__afl_map_shm')()
            except (OSError, AttributeError):
                # Fallback: patch __afl_area directly
                try:
                    afl_area = ctypes.c_void_p.in_dll(self._lib, "__afl_area")
                    afl_area.value = new_ptr
                except (ValueError, OSError):
                    pass

        # Also patch the separate shim library if loaded
        if self._shim_handle is not None:
            try:
                getattr(self._shim_handle, '__afl_map_shm')()
            except (OSError, AttributeError):
                try:
                    afl_area = ctypes.c_void_p.in_dll(self._shim_handle, "__afl_area")
                    afl_area.value = new_ptr
                except (ValueError, OSError):
                    pass

    def stop(self):
        self._func = None
        self._lib = None
        self._func_ptr = None
        self._is_c = False
        self._shim_handle = None
        if self._persistent:
            self._persistent.stop()
            self._persistent = None
        if self._forkserver:
            self._forkserver.stop()
            self._forkserver = None
        if self._shim and self._shim.shim_path:
            cleanup_shim(self._shim.shim_path)
            self._shim = None
        if self._loader_path and os.path.exists(self._loader_path):
            with contextlib.suppress(OSError):
                os.unlink(self._loader_path)
            self._loader_path = None
        if self._bitmap_out and os.path.exists(self._bitmap_out):
            with contextlib.suppress(OSError):
                os.unlink(self._bitmap_out)
            self._bitmap_out = None
