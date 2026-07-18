"""Target execution and crash detection.

Extracted from Fuzzer class (~lines 784-1115). Contains:
- _run_target() — dispatches to appropriate execution backend
- _run_target_ptrace() — ptrace-based execution with breakpoint instrumentation
- _ptrace_handle_breakpoint() — handles SIGTRAP during ptrace execution
- _verify_kernel_crash() — dmesg-based kernel crash verification
- _check_python_crashes() — detects Python process crashes in dmesg
- _is_interesting() — checks if execution result is interesting
- _is_crash() — checks if execution result is a crash
"""

import contextlib
import ctypes
import logging
import os
import signal
import struct
import threading
import time

from fuzzer_tool.adapters.process import (
    SIGNAL_CRASH_CODES,
    run_target_file,
    run_target_stdin,
)
from fuzzer_tool.core.sanitizer import SanitizerReport
from fuzzer_tool.services.ptrace_coverage import (
    PTRACE_CONT,
    PTRACE_GETREGS,
    PTRACE_SETREGS,
    PTRACE_TRACEME,
    PtraceCoverage,
)

log = logging.getLogger(__name__)


def _write_and_close(fd: int, data: bytes) -> None:
    """Write *data* to *fd* then close it — designed to run in a thread."""
    try:
        os.write(fd, data)
    finally:
        try:
            os.close(fd)
        except OSError:
            log.debug("Failed to close fd %d (already closed?)", fd)


class TargetRunner:
    """Manages target execution across multiple backends.

    Holds a reference to the Fuzzer instance for accessing shared state.
    """

    def __init__(self, fuzzer):
        self.f = fuzzer

    def run_target(self, data: bytes) -> tuple[int, str]:
        f = self.f
        # Resolve per-target SHM for multi-target mode
        shm = f._target_shm_covs.get(f.target, f.shm_cov) if f.multi_targets else f.shm_cov

        if f._inprocess_runner:
            if shm:
                shm.reset_edge_map()
            rc, err = f._inprocess_runner.run_one(data)
            if shm:
                bitmap = f._inprocess_runner.read_bitmap()
                if bitmap and len(bitmap) <= shm.size:
                    ctypes.memmove(shm._ptr, bitmap, len(bitmap))
            return rc, err

        if f._persistent_runner:
            return f._persistent_runner.run_one(data)

        if f.ptrace_cov:
            return self._run_target_ptrace(data)

        if f._forkserver and f._forkserver._ready:
            rc, bitmap = f._forkserver.run_one(data)
            if bitmap and shm and len(bitmap) <= shm.size:
                ctypes.memmove(shm._ptr, bitmap, len(bitmap))
            return rc, ""

        if shm:
            shm.reset_edge_map()

        env = os.environ.copy()
        if f.use_coverage:
            env["AFL_MAP_SIZE"] = str(f.map_size)
        if shm:
            env["__AFL_SHM_ID"] = shm.env_id
        if f._cmplog:
            env = f._cmplog.setup_env(env)

        if f.file_mode:
            rc, stderr, pid = run_target_file(
                f.target,
                data,
                f.timeout,
                str(f._tmp_dir),
                f.target_args,
                env=env,
            )
            f._last_child_pid = pid
            return rc, stderr
        rc, stderr, pid = run_target_stdin(f.target, data, f.timeout, env=env)
        f._last_child_pid = pid
        return rc, stderr

    def _ptrace_handle_breakpoint(self, pid: int, libc, cov: PtraceCoverage, regs_buf) -> bool:
        if not cov._is_x86_64:
            log.warning("ptrace coverage requires x86_64")
            return False
        libc.ptrace(PTRACE_GETREGS, pid, None, regs_buf)
        rip = struct.unpack_from("<Q", bytes(regs_buf), 128)[0]
        bp_addr = rip - 1

        if bp_addr not in cov.original_bytes:
            libc.ptrace(PTRACE_CONT, pid, None, None)
            return True

        orig = cov.original_bytes[bp_addr]
        val = cov._read_memory(pid, bp_addr)
        cov._write_memory(pid, bp_addr, (val & ~0xFF) | orig)
        del cov.original_bytes[bp_addr]

        rsp = struct.unpack_from("<Q", bytes(regs_buf), 128 + 48)[0]
        if rsp > 0x1000:
            cov._stack_initialized = True
            cov.record_edge(bp_addr)
            cov.discover_new_bbs(pid, bp_addr)
            regs_buf2 = (ctypes.c_char * (27 * 8))()
            libc.ptrace(PTRACE_GETREGS, pid, None, regs_buf2)
            regs = bytearray(regs_buf2)
            struct.pack_into("<Q", regs, 128, bp_addr)
            libc.ptrace(PTRACE_SETREGS, pid, None, bytes(regs))
        libc.ptrace(PTRACE_CONT, pid, None, None)
        return True

    def _run_target_ptrace(self, data: bytes) -> tuple[int, str]:
        f = self.f
        cov = f.ptrace_cov
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
        f._last_child_pid = pid
        if pid == 0:
            os.setsid()
            os.dup2(stdin_r, 0)
            os.close(stdin_r)
            os.close(stdin_w)
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, 1)
            os.dup2(devnull, 2)
            os.close(devnull)
            ld_preload = os.environ.get("LD_PRELOAD", "")
            if ld_preload:
                cleaned = [p for p in ld_preload.split(":") if "ksm_preload" not in p]
                if cleaned:
                    os.environ["LD_PRELOAD"] = ":".join(cleaned)
                else:
                    os.environ.pop("LD_PRELOAD", None)
            libc.ptrace(PTRACE_TRACEME, 0, None, None)
            signal.signal(signal.SIGTRAP, signal.SIG_IGN)
            os.execv(f.target, [f.target])
            os._exit(127)

        os.close(stdin_r)
        writer = threading.Thread(target=_write_and_close, args=(stdin_w, data))
        writer.start()

        try:
            _, status = os.waitpid(pid, 0)
            if os.WIFSTOPPED(status) and os.WSTOPSIG(status) == signal.SIGTRAP:
                pass
            elif os.WIFSTOPPED(status):
                sig = os.WSTOPSIG(status)
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
                return -sig, ""
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

            deadline = time.time() + f.timeout

            last_action = None
            last_sig = 0
            returncode = 0
            child_reaped = False
            while time.time() < deadline:
                _, status = os.waitpid(pid, os.WNOHANG | os.WUNTRACED)
                if status == 0:
                    time.sleep(0.0005)
                    continue

                if os.WIFEXITED(status):
                    returncode = os.WEXITSTATUS(status)
                    child_reaped = True
                    break
                if os.WIFSIGNALED(status):
                    returncode = -os.WTERMSIG(status)
                    child_reaped = True
                    break

                if os.WIFSTOPPED(status):
                    sig = os.WSTOPSIG(status)
                    last_sig = sig
                    if sig == signal.SIGTRAP:
                        regs_buf = (ctypes.c_char * (27 * 8))()
                        if self._ptrace_handle_breakpoint(pid, libc, cov, regs_buf):
                            last_action = "cont"
                        else:
                            break
                    else:
                        break

            if child_reaped:
                pass
            elif last_action == "cont" and last_sig == signal.SIGTRAP:
                _, status = os.waitpid(pid, os.WNOHANG | os.WUNTRACED)
                if status != 0 and os.WIFSTOPPED(status):
                    libc.ptrace(PTRACE_CONT, pid, None, None)
                    _, status = os.waitpid(pid, 0)
                elif status != 0:
                    if os.WIFSIGNALED(status):
                        returncode = -os.WTERMSIG(status)
                    elif os.WIFEXITED(status):
                        returncode = os.WEXITSTATUS(status)
            else:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)

            if returncode == 0 and not child_reaped:
                if os.WIFSIGNALED(status):
                    returncode = -os.WTERMSIG(status)
                elif os.WIFEXITED(status):
                    returncode = os.WEXITSTATUS(status)
                elif os.WIFSTOPPED(status):
                    returncode = -os.WSTOPSIG(status)
                    with contextlib.suppress(ProcessLookupError):
                        os.kill(pid, signal.SIGKILL)
                        os.waitpid(pid, 0)
            return returncode, ""

        except ChildProcessError:
            return -2, ""
        except Exception as e:
            try:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
            except Exception:
                log.debug("Failed to kill orphan pid %d", pid, exc_info=True)
            return -2, str(e)
        finally:
            if writer is not None:
                writer.join(timeout=f.timeout)

    def verify_kernel_crash(self, child_pid: int | None) -> bool:
        f = self.f
        if not child_pid:
            return False

        kernel_hits = f._dmesg.drain_stream(pid=child_pid)
        if not kernel_hits:
            import time as _time
            _time.sleep(0.05)
            kernel_hits = f._dmesg.drain_stream(pid=child_pid)
        if not kernel_hits:
            text_crashes = f._dmesg._poll_text(since=f._dmesg._last_ts)
            if text_crashes:
                kernel_hits = [kc for kc in text_crashes if kc.pid == child_pid]

        if kernel_hits:
            for kc in kernel_hits:
                f._kernel_crashes.append(kc)
                log.info(
                    "Kernel crash verified: %s at ip=%s (ts=%.3f)",
                    kc.crash_type,
                    kc.ip or "?",
                    kc.timestamp,
                )
            return True

        self._check_python_crashes()
        return False

    def _check_python_crashes(self):
        f = self.f
        all_crashes = f._dmesg._poll_text(since=f._dmesg._last_ts)
        for kc in all_crashes:
            if kc.process_name and "python" in kc.process_name.lower():
                if kc.crash_type == "segfault":
                    print(
                        f"\n[*] Python process crash detected: pid={kc.pid}, ip={kc.ip or '?'}, "
                        f"type={kc.crash_type} (may indicate fuzzer infrastructure issue)"
                    )
                    kc.crash_type = "python_segfault"
                    f._kernel_crashes.append(kc)

    def is_interesting(self, returncode: int, stderr: str) -> bool:
        f = self.f
        if returncode in SIGNAL_CRASH_CODES or returncode in f.extra_crash_codes:
            return True
        if returncode < 0 and returncode != -1:
            return True
        if returncode in (-1, 0) and "ASAN" in stderr:
            return True
        if "Segmentation fault" in stderr:
            return True
        return "Aborted" in stderr

    def is_crash(self, returncode: int, stderr: str) -> bool:
        f = self.f
        f.last_report = None
        if returncode in (-2, -1):
            return False

        report = SanitizerReport.parse(stderr)
        if report and report.is_valid():
            f.last_report = report
            return True

        if returncode in SIGNAL_CRASH_CODES or returncode in f.extra_crash_codes:
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
