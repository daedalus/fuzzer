"""Shim-side crash symbolization, read back to hydrate the crash sidecar.

The shim (``adapters/afl_shim.c``) records the faulting PC when a guarded
signal fires and, after the jump out of the signal handler, asks the
sanitizer runtime to symbolize it (``__sanitizer_symbolize_pc``). One line
per crash is appended to the file named by ``$__AFL_CRASH_SYM_OUT``::

    SYM <sig> <pc-hex> <fault-hex> <fn>|<file>|<line>|<module>|<offset>

Inlined frames are joined with ``;``. The text field is ``-`` when no
sanitizer runtime is linked into the target; the PC is still reported.

This module owns the Python half: the sink file, parsing, and filling
``CrashMetadata``. Unset env var means the shim does nothing.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from dataclasses import dataclass, field

ENV_VAR = "__AFL_CRASH_SYM_OUT"
_MAX_BYTES = 1 << 16  # a sink this large only holds repeats of the same crash


@dataclass
class SymFrame:
    function: str = ""
    file: str = ""
    line: int = 0
    module: str = ""
    offset: int = 0

    def render(self, pc: int) -> str:
        """``0xPC in fn file:line (module+0xoff)``, omitting what is unknown."""
        parts = [f"0x{pc:x}"]
        if self.function:
            parts.append(f"in {self.function}")
        if self.file:
            parts.append(f"{self.file}:{self.line}" if self.line else self.file)
        if self.module:
            parts.append(f"({os.path.basename(self.module)}+0x{self.offset:x})")
        return " ".join(parts)


@dataclass
class CrashSymbol:
    signal: int = 0
    pc: int = 0
    fault_addr: int = 0
    symbolized: bool = False
    frames: list[SymFrame] = field(default_factory=list)

    def frame_strings(self) -> list[str]:
        if not self.frames:
            return [f"0x{self.pc:x}"] if self.pc else []
        return [f.render(self.pc) for f in self.frames]

    def to_dict(self) -> dict:
        return {
            "signal": self.signal,
            "pc": f"0x{self.pc:x}",
            "fault_addr": f"0x{self.fault_addr:x}",
            "symbolized": self.symbolized,
            "frames": [
                {
                    "function": f.function,
                    "file": f.file,
                    "line": f.line,
                    "module": f.module,
                    "offset": f.offset,
                }
                for f in self.frames
            ],
        }


def _int(text: str, base: int = 10) -> int:
    try:
        return int(text, base)
    except ValueError:
        return 0


def _parse_frame(text: str) -> SymFrame | None:
    parts = text.split("|")
    parts += [""] * (5 - len(parts))
    fn, path, line, module, off = (p.strip() for p in parts[:5])
    if fn in ("", "??") and path in ("", "??") and not module:
        return None
    return SymFrame(
        function="" if fn == "??" else fn,
        file="" if path == "??" else path,
        line=_int(line),
        module=module,
        offset=_int(off, 16) if off.lower().startswith("0x") else _int(off),
    )


def parse_line(line: str) -> CrashSymbol | None:
    """Parse one ``SYM`` record; None for anything malformed."""
    fields = line.strip().split(" ", 4)
    if len(fields) < 4 or fields[0] != "SYM":
        return None
    try:
        sym = CrashSymbol(
            signal=int(fields[1]),
            pc=int(fields[2], 16),
            fault_addr=int(fields[3], 16),
        )
    except ValueError:
        return None
    text = fields[4].strip() if len(fields) > 4 else "-"
    if text and text != "-":
        sym.frames = [fr for chunk in text.split(";") if (fr := _parse_frame(chunk))]
        sym.symbolized = bool(sym.frames)
    return sym


class CrashSymbolSink:
    """The file the shim appends to, and the reader that drains it."""

    def __init__(self, path: str | None = None) -> None:
        self._owned = path is None
        if path is None:
            fd, path = tempfile.mkstemp(prefix="fuzzer_crashsym_", suffix=".log")
            os.close(fd)
        self.path = path

    def enable(self) -> None:
        """Export the path; the shim reads it at crash time, so any target
        started (or already running in-process) from here on will report."""
        os.environ[ENV_VAR] = self.path

    def drain(self) -> CrashSymbol | None:
        """Newest record, then truncate. None when the shim wrote nothing."""
        try:
            with open(self.path, "rb") as fh:
                fh.seek(0, os.SEEK_END)
                size = fh.tell()
                fh.seek(max(0, size - _MAX_BYTES))
                raw = fh.read().decode("utf-8", errors="replace")
            if size:
                with open(self.path, "r+b") as fh:
                    fh.truncate(0)
        except OSError:
            return None
        for line in reversed(raw.splitlines()):
            sym = parse_line(line)
            if sym is not None:
                return sym
        return None

    def close(self) -> None:
        if self._owned:
            with contextlib.suppress(OSError):
                os.unlink(self.path)


def hydrate(meta, sym: CrashSymbol | None) -> bool:
    """Fill *meta* from a shim record. Never overwrites a value another
    source (sanitizer report, ptrace) already set. Returns True if applied."""
    if sym is None:
        return False
    meta.shim_symbol = sym.to_dict()
    if not meta.frames:
        meta.frames = sym.frame_strings()
    if not meta.rip and sym.pc:
        meta.rip = sym.pc
    if not meta.fault_addr and sym.signal in (7, 11):  # SIGBUS / SIGSEGV
        meta.fault_addr = f"0x{sym.fault_addr:x}"
    return True
