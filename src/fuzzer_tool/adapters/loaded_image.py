"""Symbols for a target image that can be rebuilt while it is loaded.

The sanitizer runtime symbolizes in-process frames by opening the module's
*path*. A rebuild that replaces the file mid-run keeps the old image mapped
but puts new code at that path, so every frame resolves against the wrong
binary (ffmpeg: a matroskadec.c overflow reported as avg_c/vc1_decode_frame).

    load ──► hold fd + (dev, ino, mtime, size)    rebuild ──► path = new inode
    report ─► "#0 0x7f.. in <wrong name>"         fix() ──► llvm-symbolizer on the fd

Only frames inside this image are rewritten. When the old bytes are gone
(rewritten in place) or no symbolizer exists, those frames go unsymbolized
("module+0xoff") rather than keep a wrong name.
"""

from __future__ import annotations

import enum
import logging
import os
import re
import shutil
import subprocess

log = logging.getLogger(__name__)

FRAME_MARK = "#0 0x"
SYMBOLIZER = "llvm-symbolizer"
SYMBOLIZE_TIMEOUT_S = 30
_UNKNOWN = "??"
_MAPS = "/proc/self/maps"

# "    #3 0x7f00deadbeef in func /src/a.c:1:2" -> indent, index, pc, rest.
_FRAME_RE = re.compile(r"^([ \t]*)#(\d+)[ \t]+(0x[0-9a-fA-F]+)(.*)$")


class ImageState(enum.Enum):
    """What the target path names now, relative to the loaded image."""

    SAME = "same"  # path still names the loaded file
    REPLACED = "replaced"  # new or deleted inode: the held fd has the old bytes
    REWRITTEN = "rewritten"  # same inode, new content: old bytes are gone


class LoadedImage:
    """The file a target was loaded from, kept readable across rebuilds."""

    def __init__(self, path: str, fd: int, ident: tuple[int, int, int, int]):
        self._path = path
        self._fd = fd
        self._ident = ident
        self._module = os.path.basename(path)
        self.base: int | None = None
        self._end = 0

    @classmethod
    def open(cls, path: str) -> LoadedImage:
        """Hold *path* open; call before loading it so the fd names that file."""
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
        st = os.fstat(fd)
        return cls(path, fd, (st.st_dev, st.st_ino, st.st_mtime_ns, st.st_size))

    def locate(self) -> None:
        """Find the image's load range in this process (after dlopen)."""
        dev, ino = self._ident[0], self._ident[1]
        starts, ends = [], []
        with open(_MAPS) as maps:
            for line in maps:
                rng, _perm, off, mdev, mino = line.split(maxsplit=5)[:5]
                if int(mino) != ino or _dev(mdev) != dev:
                    continue

                lo, hi = (int(x, 16) for x in rng.split("-"))
                ends.append(hi)
                if int(off, 16) == 0:
                    starts.append(lo)

        if starts:
            self.base, self._end = min(starts), max(ends)

    def state(self) -> ImageState:
        """Compare the path's current file with the one that was loaded."""
        try:
            st = os.stat(self._path)
        except FileNotFoundError:
            return ImageState.REPLACED

        if (st.st_dev, st.st_ino) != self._ident[:2]:
            return ImageState.REPLACED
        if (st.st_mtime_ns, st.st_size) != self._ident[2:]:
            return ImageState.REWRITTEN
        return ImageState.SAME

    def fix(self, text: str) -> str:
        """*text* with this image's frames symbolized from the loaded bytes.

        Unchanged unless *text* holds a stack and the path no longer names
        the loaded file.
        """
        if self.base is None or FRAME_MARK not in text:
            return text

        state = self.state()
        if state is ImageState.SAME:
            return text

        pcs = [int(m.group(3), 16) for m in map(_FRAME_RE.match, text.splitlines()) if m]
        mine = [pc for pc in dict.fromkeys(pcs) if self._owns(pc)]
        if not mine:
            return text

        names = self._symbolize(mine) if state is ImageState.REPLACED else {}
        return self._rewrite(text, names)

    def close(self) -> None:
        """Release the held fd."""
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    def _owns(self, pc: int) -> bool:
        return self.base is not None and self.base <= pc < self._end

    def _symbolize(self, pcs: list[int]) -> dict[int, list[tuple[str, str]]]:
        """``pc -> [(func, file:line:col), ...]`` innermost first; one batched call."""
        tool = shutil.which(SYMBOLIZER)
        if tool is None:
            log.warning("%s not found: stale frames left unsymbolized", SYMBOLIZER)
            return {}

        # /dev/fd/N in the child is our fd: no capability needed, unlike map_files.
        argv = [tool, f"--obj=/dev/fd/{self._fd}", "--inlining"]
        argv += [hex(pc - self.base) for pc in pcs]
        try:
            out = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                pass_fds=(self._fd,),
                timeout=SYMBOLIZE_TIMEOUT_S,
            ).stdout
        except (OSError, subprocess.SubprocessError) as e:
            log.warning("re-symbolizing %s failed (%s): frames left unsymbolized", self._path, e)
            return {}

        blocks = out.strip("\n").split("\n\n")
        return {pc: _entries(block) for pc, block in zip(pcs, blocks, strict=False)}

    def _rewrite(self, text: str, names: dict[int, list[tuple[str, str]]]) -> str:
        """Replace this image's frames; an inlined group (one pc) is redone whole."""
        out: list[str] = []
        n = 0
        prev: int | None = None
        for line in text.splitlines(keepends=True):
            m = _FRAME_RE.match(line.rstrip("\n"))
            if not m:
                out.append(line)
                prev = None
                continue

            indent, idx, pc = m.group(1), int(m.group(2)), int(m.group(3), 16)
            if idx == 0:
                n, prev = 0, None

            if not self._owns(pc):
                out.append(f"{indent}#{n}{line[m.end(2) :]}")
                n, prev = n + 1, None
                continue

            if pc == prev:
                continue

            prev = pc
            for func, loc in names.get(pc) or [("", "")]:
                out.append(self._line(indent, n, pc, func, loc))
                n += 1

        return "".join(out)

    def _line(self, indent: str, n: int, pc: int, func: str, loc: str) -> str:
        if not func:
            return f"{indent}#{n} {pc:#x} ({self._module}+{pc - self.base:#x})\n"
        if not loc:
            return f"{indent}#{n} {pc:#x} in {func}\n"
        return f"{indent}#{n} {pc:#x} in {func} {loc}\n"


class NoImage:
    """Null image for runners that loaded nothing: ``fix`` is the identity."""

    base = None

    def fix(self, text: str) -> str:
        return text

    def close(self) -> None:
        return None


NO_IMAGE = NoImage()


def _dev(field: str) -> int:
    """ "fd:01" (hex major:minor, as /proc/pid/maps prints it) -> st_dev."""
    major, minor = field.split(":")
    return os.makedev(int(major, 16), int(minor, 16))


def _entries(block: str) -> list[tuple[str, str]]:
    """llvm-symbolizer's func/location line pairs; unknown ones dropped."""
    lines = block.splitlines()
    out = []
    for func, loc in zip(lines[0::2], lines[1::2], strict=False):
        if func == _UNKNOWN:
            continue
        out.append((func, "" if loc.startswith(_UNKNOWN) else loc))
    return out
