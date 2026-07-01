"""Crash trace reporter: GDB backtrace, registers, disassembly, strace.

Generates detailed trace reports for crash inputs by running the target
under GDB (and optionally strace). Reports are saved alongside crash files
for post-mortem analysis.

Usage:
    tracer = CrashTracer(target_path)
    report = tracer.trace(crash_input_data)
    tracer.save_report(report, crash_dir, crash_name)
"""

import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


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
        report = TraceReport(target=self.target_path, input_size=len(data))

        # Determine signal from returncode
        signal_map = {
            -6: ("SIGABRT", 6),
            -7: ("SIGBUS", 7),
            -8: ("SIGFPE", 8),
            -11: ("SIGSEGV", 11),
            -4: ("SIGILL", 4),
            -5: ("SIGTRAP", 5),
        }
        if returncode in signal_map:
            report.signal, report.signal_num = signal_map[returncode]

        # Write crash input to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            f.write(data)
            tmp_path = f.name

        try:
            if self._has_gdb:
                self._run_gdb(tmp_path, report)
            if self._has_strace:
                self._run_strace(tmp_path, report)
            self._build_repro(data, report)
        finally:
            os.unlink(tmp_path)

        return report

    def _run_gdb(self, input_path: str, report: TraceReport):
        """Run GDB in batch mode to extract backtrace, registers, disassembly."""
        cmds = [
            "set pagination off",
            "run",
            "info registers",
            "bt full",
            "thread apply all bt",
            "disassemble $pc",
        ]
        # Also try to disassemble common libpng functions
        for func in ["png_read_row", "png_error", "png_process_data"]:
            cmds.append(f"disassemble {func}")

        # Each command as separate -ex (newlines in one -ex break GDB)
        gdb_args = ["gdb", "-batch"]
        for cmd in cmds:
            gdb_args.extend(["-ex", cmd])
        gdb_args.extend(["--args", self.target_path, input_path])

        try:
            result = subprocess.run(
                gdb_args,
                capture_output=True,
                timeout=self.timeout,
            )
            output = result.stdout.decode(errors="replace")
            self._parse_gdb_output(output, report)
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            log.debug("GDB failed: %s", e)

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
                report.fault_addr = hex(report.reg_values["rip"])

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
            m = re.match(r"^\s+\d+\s+\S", line)
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
