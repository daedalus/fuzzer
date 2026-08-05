"""Crash trace reporter: GDB backtrace, registers, disassembly, strace.

Generates detailed trace reports for crash inputs by running the target
under GDB (and optionally strace). Reports are saved alongside crash files
for post-mortem analysis.

Usage:
    tracer = CrashTracer(target_path)
    report = tracer.trace(crash_input_data)
    tracer.save_report(report, crash_dir, crash_name)
"""

import contextlib
import logging
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


def _get_exported_functions(target: str, max_funcs: int = 20) -> list[str]:
    """Discover exported function names from a binary via nm -D."""
    try:
        result = subprocess.run(
            ["nm", "-D", "--defined-only", target],
            capture_output=True,
            text=True,
            timeout=5,
        )
        funcs = []
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[1] == "T":
                name = parts[2]
                if not name.startswith("_") and len(name) > 2:
                    funcs.append(name)
        return funcs[:max_funcs]
    except Exception:
        return []


_SO_HARNESS = """\
import ctypes, sys
so = ctypes.CDLL(sys.argv[1])
fn = getattr(so, sys.argv[2])
fn.restype = ctypes.c_int
fn.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
data = sys.stdin.buffer.read()
buf = (ctypes.c_uint8 * len(data))(*data)
fn(buf, len(data))
"""

# returncode -> (signal name, number) for the crash signals the fuzzer
# reports. Both conventions occur: negative -sig (ptrace/persistent
# WIFSIGNALED paths) and positive 128+sig (guarded-call 128+signum exit).
_SIGNAL_FROM_RETURNCODE = {
    -6: ("SIGABRT", 6),
    -7: ("SIGBUS", 7),
    -8: ("SIGFPE", 8),
    -11: ("SIGSEGV", 11),
    -4: ("SIGILL", 4),
    -5: ("SIGTRAP", 5),
    134: ("SIGABRT", 6),
    135: ("SIGBUS", 7),
    136: ("SIGFPE", 8),
    139: ("SIGSEGV", 11),
    132: ("SIGILL", 4),
    133: ("SIGTRAP", 5),
}


def _is_shared_object(target: str) -> bool:
    return target.lower().endswith((".so", ".dylib", ".dll"))


def _probe_so_function(target: str) -> str | None:
    """Pick the fuzz entry point for a shared-object target.

    Mirrors Fuzzer._probe_so_function's preference: fuzz_shm_run, else the
    first exported fuzz_* symbol. Returns None when nothing callable is found.
    """
    funcs = _get_exported_functions(target, max_funcs=100)
    for preferred in ("fuzz_shm_run", "fuzz_test"):
        if preferred in funcs:
            return preferred
    for name in funcs:
        if name.startswith("fuzz"):
            return name
    return None


@dataclass
class TraceReport:
    """Structured crash trace data."""

    # Backtrace
    backtrace: str = ""
    frames: list[dict] = field(default_factory=list)  # [{frame, addr, func, file, line}]

    # Registers
    registers: str = ""
    reg_values: dict[str, int] = field(default_factory=dict)

    # Disassembly
    disassembly: str = ""

    # Source context around crash
    source_context: str = ""

    # Strace (optional)
    strace: str = ""
    strace_summary: str = ""

    # Signal info
    signal: str = ""
    signal_num: int = 0
    # Instruction pointer at the crash site (where the crashing instruction
    # lives) — distinct from fault_addr, the memory address it was touching.
    crash_rip: str = ""
    fault_addr: str = ""

    # Error message from target (if any)
    error_msg: str = ""

    # Metadata
    target: str = ""
    input_size: int = 0
    repro_cmd: str = ""

    def format(self) -> str:
        """Format the full trace report as text."""
        sections = []

        sections.append("=" * 72)
        sections.append("CRASH TRACE REPORT")
        sections.append("=" * 72)
        sections.append(f"Target:  {self.target}")
        sections.append(f"Input:   {self.input_size} bytes")
        sections.append(f"Signal:  {self.signal} ({self.signal_num})")
        if self.crash_rip:
            sections.append(f"RIP:     {self.crash_rip}")
        if self.fault_addr:
            sections.append(f"Fault:   {self.fault_addr}")
        if self.error_msg:
            sections.append(f"Error:   {self.error_msg}")
        sections.append("")

        if self.registers:
            sections.append("--- Registers ---")
            sections.append(self.registers)
            sections.append("")

        if self.backtrace:
            sections.append("--- Backtrace ---")
            sections.append(self.backtrace)
            sections.append("")

        if self.source_context:
            sections.append("--- Source ---")
            sections.append(self.source_context)
            sections.append("")

        if self.disassembly:
            sections.append("--- Disassembly ---")
            sections.append(self.disassembly)
            sections.append("")

        if self.strace:
            sections.append("--- Strace (last 50 lines) ---")
            sections.append(self.strace)
            sections.append("")
            if self.strace_summary:
                sections.append("--- Strace Summary ---")
                sections.append(self.strace_summary)
                sections.append("")

        if self.repro_cmd:
            sections.append("--- Reproducer ---")
            sections.append(self.repro_cmd)
            sections.append("")

        sections.append("=" * 72)
        return "\n".join(sections)

    def sidecar_block(self) -> str:
        """Compact GDB crash-replay section for embedding in the .txt sidecar.

        Returns "" when GDB captured nothing useful (unavailable, or the
        target was not traceable), so callers can skip the section entirely.
        """
        if not (self.signal or self.registers or self.backtrace):
            return ""
        lines = ["=== GDB crash replay ==="]
        if self.signal:
            lines.append(f"Signal: {self.signal} ({self.signal_num})")
        if self.crash_rip:
            lines.append(f"RIP:    {self.crash_rip}")
        if self.fault_addr:
            lines.append(f"Fault:  {self.fault_addr}")
        if self.error_msg:
            lines.append(f"Error:  {self.error_msg}")
        if self.registers:
            lines.extend(["", "--- Registers ---", self.registers])
        if self.backtrace:
            lines.extend(["", "--- Backtrace ---", self.backtrace])
        if self.source_context:
            lines.extend(["", "--- Source ---", self.source_context])
        if self.disassembly:
            lines.extend(["", "--- Disassembly ---", self.disassembly])
        return "\n".join(lines)


class CrashTracer:
    """Generate trace reports for crash inputs using GDB/strace.

    Args:
        target_path: Path to the target binary.
        timeout: Max seconds per GDB/strace run.
    """

    def __init__(self, target_path: str, timeout: int = 10):
        self.target_path = os.path.abspath(target_path)
        self.timeout = timeout
        self._has_gdb = self._check_tool("gdb")
        self._has_strace = self._check_tool("strace")

    @staticmethod
    def _check_tool(name: str) -> bool:
        try:
            result = subprocess.run(["which", name], capture_output=True, timeout=2)
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def trace(self, data: bytes, returncode: int = 0) -> TraceReport:
        """Run target under GDB/strace and build a trace report.

        Args:
            data: The crash input bytes.
            returncode: The observed returncode (for signal info).

        Returns:
            Populated TraceReport.
        """
        report = self.gdb_replay(data, returncode)

        if self._has_strace:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
                f.write(data)
                tmp_path = f.name
            try:
                self._run_strace(tmp_path, report)
            finally:
                os.unlink(tmp_path)
        self._build_repro(data, report)

        return report

    def gdb_replay(self, data: bytes, returncode: int = 0) -> TraceReport:
        """Run ONLY the GDB crash replay (no strace) for the report sidecar.

        Best-effort: returns an empty report when gdb is unavailable or the
        target cannot be traced.
        """
        report = TraceReport(target=self.target_path, input_size=len(data))
        if returncode in _SIGNAL_FROM_RETURNCODE:
            report.signal, report.signal_num = _SIGNAL_FROM_RETURNCODE[returncode]

        if not self._has_gdb:
            return report
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            f.write(data)
            input_path = f.name
        try:
            self._run_gdb(input_path, report)
        finally:
            os.unlink(input_path)
        return report

    def _run_gdb(self, input_path: str, report: TraceReport):
        """Run GDB in batch mode to extract backtrace, registers, disassembly.

        Executables run with the crash input as argv[1] (legacy behavior).
        Shared objects cannot be exec'd, so they are driven through a ctypes
        harness that loads the library and calls the probed fuzz entry point
        with the crash input on stdin — mirroring how direct_lite and the
        loaders execute them.
        """
        harness_path = None
        so_stdin = None
        if _is_shared_object(self.target_path):
            func = _probe_so_function(self.target_path)
            if func is None:
                log.debug("No fuzz entry point found in %s — skipping GDB", self.target_path)
                return
            harness_path = self._write_so_harness()
            program_args = [sys.executable, harness_path, self.target_path, func]
            # The harness reads the crash input from stdin; gdb has no `run <`
            # redirection, so redirect gdb's own stdin — the inferior inherits it.
            so_stdin = input_path
        else:
            program_args = [self.target_path, input_path]

        cmds = [
            "set pagination off",
            "run",
            "info registers",
            "print $_siginfo",
            "bt full",
            "thread apply all bt",
            "disassemble $pc",
            # DWARF source context: the crashing frame is #0 (often address 0
            # for a NULL-jump), so select the target's own frame and list.
            "frame 1",
            "list",
        ]
        # Also disassemble exported functions from the target binary
        for func in _get_exported_functions(self.target_path):
            cmds.append(f"disassemble {func}")

        # Each command as separate -ex (newlines in one -ex break GDB)
        gdb_args = ["gdb", "-batch"]
        for cmd in cmds:
            gdb_args.extend(["-ex", cmd])
        gdb_args.extend(["--args", *program_args])

        try:
            with contextlib.ExitStack() as stack:
                stdin_file = stack.enter_context(open(so_stdin, "rb")) if so_stdin else None
                result = subprocess.run(
                    gdb_args,
                    capture_output=True,
                    timeout=self.timeout,
                    stdin=stdin_file,
                )
            output = result.stdout.decode(errors="replace")
            self._parse_gdb_output(output, report)
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            log.debug("GDB failed: %s", e)
        finally:
            if harness_path:
                with contextlib.suppress(OSError):
                    os.unlink(harness_path)

    def _write_so_harness(self) -> str:
        """Write the ctypes harness that drives a .so target under GDB."""
        fd, path = tempfile.mkstemp(suffix=".py", prefix="fuzz_gdb_harness_")
        with os.fdopen(fd, "w") as f:
            f.write(_SO_HARNESS)
        return path

    def _parse_gdb_output(self, output: str, report: TraceReport):
        """Parse GDB batch output into structured report fields."""
        lines = output.split("\n")

        # Extract signal
        for line in lines:
            m = re.match(r"Program received signal (\w+), (.+)", line)
            if m:
                report.signal = m.group(1)
                report.error_msg = m.group(2)
                break

        # Extract the real faulting address from GDB's $_siginfo. The regex is
        # anchored to the distinct si_addr token so register names (rax, ...)
        # can never match.
        m = re.search(r"si_addr\s*=\s*(0x[0-9a-fA-F]+)", output)
        if m:
            report.fault_addr = m.group(1)

        # Extract registers
        reg_lines = []
        in_regs = False
        for line in lines:
            if re.match(r"^\s*(rax|rbx|rcx|rdx|rsi|rdi|rbp|rsp|r\d+|rip|eflags)\s+0x", line):
                in_regs = True
            if in_regs:
                if line.strip() and "0x" in line:
                    reg_lines.append(line.rstrip())
                    m = re.match(r"(\w+)\s+0x([0-9a-f]+)", line)
                    if m:
                        report.reg_values[m.group(1)] = int(m.group(2), 16)
                elif reg_lines:
                    break
        if reg_lines:
            report.registers = "\n".join(reg_lines)
            if "rip" in report.reg_values:
                report.crash_rip = hex(report.reg_values["rip"])

        # Extract backtrace frames
        bt_lines = []
        in_bt = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#"):
                in_bt = True
                bt_lines.append(line.rstrip())
                # Parse frame: #0  0xaddr in func (args) at file:line
                m = re.match(
                    r"#(\d+)\s+0x([0-9a-f]+)\s+in\s+(.+?)(?:\s+\((.+?)\))?"
                    r"(?:\s+at\s+(.+?):(\d+))?",
                    stripped,
                )
                if m:
                    frame = {
                        "frame": int(m.group(1)),
                        "addr": f"0x{m.group(2)}",
                        "func": m.group(3).strip(),
                    }
                    if m.group(5):
                        frame["file"] = m.group(5)
                        frame["line"] = int(m.group(6))
                    report.frames.append(frame)
            elif in_bt and stripped == "":
                in_bt = False

        if bt_lines:
            report.backtrace = "\n".join(bt_lines)

        # Extract source context (from `list` or bt source lines)
        # GDB source lines follow the pattern: linenum  source_code  [filename:line]
        src_lines = []
        in_source = False
        for line in lines:
            # GDB source context lines start with whitespace + line number + whitespace + code
            # e.g. "   10   if (x > 0) {"
            # Avoid matching register output or backtrace lines
            stripped = line.strip()
            if (
                not stripped
                or stripped.startswith("#")
                or stripped.startswith("rax")
                or stripped.startswith("0x")
            ):
                in_source = False
                continue
            m = re.match(r"^\s*\d+\s+\S", line)
            if m and not re.match(r"^\s+(rax|rbx|rcx|rdx|rsi|rdi|rbp|rsp|r\d+|rip|eflags)\s", line):
                in_source = True
                src_lines.append(line.rstrip())
                if len(src_lines) >= 20:
                    break
            elif in_source:
                break
        if src_lines:
            report.source_context = "\n".join(src_lines[:20])

        # Extract disassembly
        disasm_lines = []
        in_disasm = False
        for line in lines:
            if "Dump of assembler" in line:
                in_disasm = True
                disasm_lines.append(line.rstrip())
            elif in_disasm:
                if line.strip() == "End of assembler dump.":
                    disasm_lines.append(line.rstrip())
                    in_disasm = False
                elif line.strip() and (
                    "0x" in line
                    or "push" in line
                    or "call" in line
                    or "mov" in line
                    or "ret" in line
                    or "jmp" in line
                    or "lea" in line
                    or "cmp" in line
                    or "xor" in line
                ):
                    disasm_lines.append(line.rstrip())
        if disasm_lines:
            report.disassembly = "\n".join(disasm_lines)

    def _run_strace(self, input_path: str, report: TraceReport):
        """Run strace to capture syscall trace."""
        try:
            result = subprocess.run(
                [
                    "strace",
                    "-f",
                    "-e",
                    "trace=read,write,mmap,mprotect,open,close,"
                    "madvise,brk,rt_sigaction,clone,futex",
                    self.target_path,
                    input_path,
                ],
                capture_output=True,
                timeout=self.timeout,
            )
            output = result.stderr.decode(errors="replace")
            lines = output.strip().split("\n")

            # Last 50 lines
            report.strace = "\n".join(lines[-50:])

            # Summarize: count syscalls, find crashes
            syscall_counts: dict[str, int] = {}
            crash_line = ""
            for line in lines:
                m = re.match(r"(\w+)\(", line)
                if m:
                    name = m.group(1)
                    syscall_counts[name] = syscall_counts.get(name, 0) + 1
                if "SIGABRT" in line or "SIGSEGV" in line or "SIGBUS" in line:
                    crash_line = line.strip()

            top = sorted(syscall_counts.items(), key=lambda x: -x[1])[:10]
            summary_parts = [f"{name}: {count}" for name, count in top]
            if crash_line:
                summary_parts.append(f"Crash: {crash_line}")
            report.strace_summary = " | ".join(summary_parts)

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            log.debug("strace failed: %s", e)

    def _build_repro(self, data: bytes, report: TraceReport):
        """Build a reproducer command."""
        import base64

        encoded = base64.b64encode(data).decode()
        report.repro_cmd = f"printf '%s' '{encoded}' | base64 -d | {self.target_path}"

    def save_report(self, report: TraceReport, crash_dir: str, name: str):
        """Save trace report as .trace file alongside crash files.

        Args:
            report: The populated trace report.
            crash_dir: Directory containing crash files.
            name: Base name for the trace file (without extension).
        """
        trace_path = os.path.join(crash_dir, f"{name}.trace")
        try:
            with open(trace_path, "w") as f:
                f.write(report.format())
            log.info("Trace report saved: %s", trace_path)
        except OSError as e:
            log.debug("Failed to save trace report: %s", e)
