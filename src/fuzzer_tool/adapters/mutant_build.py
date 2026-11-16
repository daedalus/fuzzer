"""Mutant binaries and their fork replay driver (mutant-kill metric, P2).

Builds a fixed, cached set of first-order mutants of one C source and
replays packed inputs through them. Plan:
``docs/action_plan_mutant_kill_metric.md``.

    src.c ──emit (asan, no passes)──> base.ll ──find/sample_sites──> sites
      │                                  │
      │                     apply_site ──┴──> (stdin) ──> k.o ──hash──┐
      ▼                                                            │
    base.o ──link + wrapper.o + mutant_replay.o──> original        │
                                                                   ▼
             same .o as base or an earlier mutant? ── yes ──> equivalent (dropped)
                                                   └─ no ───> link mutant_k

Binaries are oracles, not fuzz targets: ASAN yes, AFL shim no (the driver
stubs ``__afl_map_edge``). They link the shared ASAN runtime (72 KB per
binary instead of 1.7 MB; same replay speed).
"""

from __future__ import annotations

import enum
import functools
import logging
import os
import pickle
import struct
import subprocess
import tempfile
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from fuzzer_tool.adapters.filesystem import hash_data
from fuzzer_tool.core import ir_mutants
from fuzzer_tool.core.ir_mutants import MutOp, Site, apply_site, find_sites, sample_sites
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.state_store import safe_load

log = logging.getLogger(__name__)

# Record cap; mirrors MAX_RECORD in mutant_replay.c and the 64 KiB stdin
# read of targets/*_read.c.
MAX_INPUT_BYTES = 1 << 16

_DRIVER_SRC = Path(__file__).with_name("mutant_replay.c")
_MANIFEST = "manifest.pkl"
_MANIFEST_VERSION = 1
_BASE_IR = "base.ll"
_ORIGINAL = "original"
_MUTANT_FMT = "mutant_{:05d}"

_CC = "clang"
_CFLAGS = ("-O2", "-fsanitize=address", "-fno-delete-null-pointer-checks")
# Emit before any pass: -fsanitize at emit tags functions sanitize_address,
# otherwise the .ll -> binary step never instruments them.
_EMIT_FLAGS = ("-Xclang", "-disable-llvm-passes", "-emit-llvm", "-S")
_LINK_FLAGS = ("-shared-libasan",)
_LIBS = ("-lm",)
_IR_STDIN = ("-x", "ir", "-")
# targets/*_read.c define main(); the driver brings its own.
_WRAPPER_MAIN = "-Dmain=fuzz_wrapper_main"

# Fast, deterministic reports: symbolizing each crash cost 18x the replay.
_ASAN_OPTIONS = "symbolize=0:detect_leaks=0:print_summary=0:malloc_context_size=0"
_PACK_HDR = struct.Struct("<I")
_DRIVER_BAD_INPUT = 2


class Status(enum.IntEnum):
    """One replayed input's outcome (status byte from mutant_replay.c)."""

    CLEAN = 0
    CRASH = 1
    HANG = 2


class ReplayError(RuntimeError):
    """The driver could not run: missing binary, malformed pack, fork failure."""


class BuildError(RuntimeError):
    """The toolchain failed on the original (mutant failures are only logged)."""


@dataclass(frozen=True, slots=True)
class TargetSpec:
    """The source to mutate and the wrapper exposing ``fuzz_shm_run``."""

    src: Path
    wrapper: Path
    include_dirs: tuple[Path, ...] = ()


@functools.cache
def asan_available() -> bool:
    """clang can link a shared-runtime ASAN binary (libclang-rt-<N>-dev present)."""
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "probe.c"
        src.write_text("int main(void) { return 0; }\n")
        try:
            _cc(*_CFLAGS, *_LINK_FLAGS, str(src), "-o", str(Path(tmp) / "probe"))
        except (BuildError, FileNotFoundError):
            return False
    return True


def write_pack(inputs: Iterable[bytes], path: Path) -> int:
    """Write *inputs* as ``(u32 len, bytes)*``, each cut to MAX_INPUT_BYTES.

    Returns:
        The number of records written.
    """
    n = 0
    with open(path, "wb") as fh:
        for data in inputs:
            data = data[:MAX_INPUT_BYTES]
            fh.write(_PACK_HDR.pack(len(data)))
            fh.write(data)
            n += 1
    return n


def count_records(pack: Path) -> int:
    """Records in *pack*, walking headers only.

    Raises:
        ReplayError: a header or body runs past the end of the file.
    """
    size = pack.stat().st_size
    n = pos = 0
    with open(pack, "rb") as fh:
        while pos < size:
            hdr = fh.read(_PACK_HDR.size)
            if len(hdr) != _PACK_HDR.size:
                raise ReplayError(f"{pack}: truncated header at record {n}")

            (length,) = _PACK_HDR.unpack(hdr)
            pos += _PACK_HDR.size + length
            if pos > size:
                raise ReplayError(f"{pack}: record {n} runs past end of file")
            fh.seek(pos)
            n += 1
    return n


def pack_record(pack: Path, index: int) -> bytes:
    """Record *index* of *pack* (0-based), walking headers only.

    Raises:
        IndexError: the pack has no such record.
        ReplayError: the pack is truncated before it.
    """
    with open(pack, "rb") as fh:
        for i in range(index + 1):
            hdr = fh.read(_PACK_HDR.size)
            if not hdr:
                raise IndexError(f"{pack}: no record {index} (has {i})")
            if len(hdr) != _PACK_HDR.size:
                raise ReplayError(f"{pack}: truncated header at record {i}")

            (length,) = _PACK_HDR.unpack(hdr)
            if i < index:
                fh.seek(length, os.SEEK_CUR)
                continue

            data = fh.read(length)
            if len(data) != length:
                raise ReplayError(f"{pack}: record {i} runs past end of file")
            return data
    raise AssertionError("unreachable")


def replay(binary: Path, pack: Path, mask: bytes | None, timeout_ms: int) -> bytes:
    """Status bytes (``Status``) for each record of *pack* run on *binary*.

    With *mask* (one byte per record, 1 = eligible), stops after the first
    CRASH on an eligible record, so the result can be shorter than the pack.

    Raises:
        ReplayError: missing binary, malformed pack, or driver failure.
        ValueError: *mask* length differs from the record count.
    """
    if not binary.is_file():
        raise ReplayError(f"no such binary: {binary}")
    if mask is not None and len(mask) != count_records(pack):
        raise ValueError(f"mask has {len(mask)} entries for {count_records(pack)} records")

    env = {**os.environ, "ASAN_OPTIONS": _ASAN_OPTIONS}
    if mask is None:
        return _run_driver(binary, pack, "-", timeout_ms, env)

    with tempfile.NamedTemporaryFile(dir=pack.parent, prefix=".mask_") as fh:
        fh.write(mask)
        fh.flush()
        return _run_driver(binary, pack, fh.name, timeout_ms, env)


def _run_driver(binary: Path, pack: Path, mask: str, timeout_ms: int, env: dict[str, str]) -> bytes:
    res = subprocess.run(
        [str(binary), str(pack), mask, str(timeout_ms)], capture_output=True, env=env
    )
    if res.returncode != 0:
        msg = res.stderr.decode(errors="replace").strip()
        raise ReplayError(f"{binary.name}: driver exit {res.returncode}: {msg}")
    return res.stdout


def _cc(*args: str, stdin: str | None = None) -> None:
    """Run clang; BuildError carries its stderr."""
    res = subprocess.run([_CC, *args], input=stdin, capture_output=True, text=True)
    if res.returncode != 0:
        raise BuildError(res.stderr.strip())


def _compile_ir(text: str, out: Path) -> None:
    """IR text -> object, read from stdin.

    ASAN writes the input file name into .rodata: one shared name ("<stdin>")
    keeps byte-identical code byte-identical, which TCE relies on.
    """
    _cc(*_CFLAGS, *_IR_STDIN, "-c", "-o", str(out), stdin=text)


@functools.cache
def _runtime_dir() -> str:
    """Directory of the shared ASAN runtime, baked in as rpath."""
    res = subprocess.run([_CC, "-print-runtime-dir"], capture_output=True, text=True, check=True)
    return res.stdout.strip()


@functools.cache
def _cc_version() -> str:
    return subprocess.run([_CC, "--version"], capture_output=True, text=True, check=True).stdout


@dataclass
class MutantSet:
    """A built, cached mutant set: original + one binary per kept site.

    ``mutants[k]`` is the binary for ``sites[k]``.
    """

    out_dir: Path
    key: str
    spec_src: Path
    spec_wrapper: Path
    sites: list[Site]
    equivalent: list[Site] = field(default_factory=list)
    failed: list[Site] = field(default_factory=list)

    @property
    def original(self) -> Path:
        return self.out_dir / _ORIGINAL

    @property
    def mutants(self) -> list[Path]:
        return [self.out_dir / _MUTANT_FMT.format(k) for k in range(len(self.sites))]

    @property
    def base_ir(self) -> str:
        return (self.out_dir / _BASE_IR).read_text()

    @classmethod
    def load(cls, out_dir: Path) -> MutantSet:
        """Read *out_dir*'s manifest (pickle, state allowlist only)."""
        with open(out_dir / _MANIFEST, "rb") as fh:
            m = safe_load(fh)

        def sites(rows: list[tuple[int, int]]) -> list[Site]:
            return [Site(line, MutOp(op)) for line, op in rows]

        return cls(
            out_dir=out_dir,
            key=m["key"],
            spec_src=Path(m["src"]),
            spec_wrapper=Path(m["wrapper"]),
            sites=sites(m["sites"]),
            equivalent=sites(m["equivalent"]),
            failed=sites(m["failed"]),
        )

    @classmethod
    def build(cls, spec: TargetSpec, out_dir: Path, cap: int, seed: int, jobs: int) -> MutantSet:
        """Build (or reuse) the mutant set for *spec* in *out_dir*.

        A manifest with the same key means nothing changed: reuse it.
        The manifest is written last, so an interrupted build rebuilds.

        Raises:
            BuildError: the original does not compile or link.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        key = _build_key(spec, cap, seed)
        cached = _cached(out_dir, key)
        if cached is not None:
            return cached

        _MutantBuilder(spec, out_dir, jobs).run(key, cap, seed)
        return cls.load(out_dir)


def _build_key(spec: TargetSpec, cap: int, seed: int) -> str:
    """Everything that changes the mutants: sources, operators, flags, compiler."""
    parts = [
        spec.src.read_bytes(),
        spec.wrapper.read_bytes(),
        _DRIVER_SRC.read_bytes(),
        Path(ir_mutants.__file__).read_bytes(),
        repr((_CFLAGS, _EMIT_FLAGS, _LINK_FLAGS, _LIBS, spec.include_dirs, cap, seed)).encode(),
        _cc_version().encode(),
    ]
    return hash_data(b"\0".join(parts))


def _cached(out_dir: Path, key: str) -> MutantSet | None:
    if not (out_dir / _MANIFEST).is_file():
        return None
    found = MutantSet.load(out_dir)
    return found if found.key == key else None


class _MutantBuilder:
    """One build: original first, then mutants in parallel, manifest last."""

    def __init__(self, spec: TargetSpec, out_dir: Path, jobs: int) -> None:
        self._spec = spec
        self._out = out_dir
        self._jobs = max(1, jobs)
        self._work = out_dir / "work"
        self._incs = tuple(f"-I{d}" for d in spec.include_dirs)

    def run(self, key: str, cap: int, seed: int) -> None:
        self._clear()
        self._work.mkdir()

        base_ll = self._out / _BASE_IR
        _cc(*_CFLAGS, *_EMIT_FLAGS, *self._incs, str(self._spec.src), "-o", str(base_ll))
        wrapper_o = self._obj(self._spec.wrapper, "wrapper.o", _WRAPPER_MAIN)
        driver_o = self._obj(_DRIVER_SRC, "driver.o")
        self._shared = (wrapper_o, driver_o)

        text = base_ll.read_text()
        base_o = self._work / "base.o"
        _compile_ir(text, base_o)
        self._link(base_o, self._out / _ORIGINAL)

        sites = sample_sites(find_sites(text), cap, RandPool(seed))
        kept, equivalent, failed = self._compile_all(text, sites, base_o)
        self._link_all(kept)

        self._write_manifest(key, kept, equivalent, failed)
        for p in self._work.iterdir():
            p.unlink()
        self._work.rmdir()

    def _clear(self) -> None:
        """Remove a stale build's files; only names this builder creates."""
        if self._work.is_dir():
            for p in self._work.iterdir():
                p.unlink()
            self._work.rmdir()

        stale = [self._out / _MANIFEST, self._out / _BASE_IR, self._out / _ORIGINAL]
        stale += self._out.glob("mutant_[0-9]*")
        for p in stale:
            p.unlink(missing_ok=True)

    def _obj(self, src: Path, name: str, *extra: str) -> Path:
        out = self._work / name
        _cc(*_CFLAGS, *self._incs, *extra, "-c", str(src), "-o", str(out))
        return out

    def _link(self, obj: Path, out: Path) -> None:
        rpath = f"-Wl,-rpath,{_runtime_dir()}"
        _cc(
            *_CFLAGS, *_LINK_FLAGS, rpath, str(obj), *map(str, self._shared), *_LIBS, "-o", str(out)
        )

    def _compile_one(self, text: str, k: int, site: Site) -> Path | None:
        obj = self._work / f"{k}.o"
        try:
            _compile_ir(apply_site(text, site), obj)
        except BuildError as exc:
            log.warning("mutant %s does not compile: %s", site, exc)
            return None
        return obj

    def _compile_all(
        self, text: str, sites: list[Site], base_o: Path
    ) -> tuple[list[tuple[Site, Path]], list[Site], list[Site]]:
        """Compile every site; split into kept / equivalent / failed, in site order.

        Trivial compiler equivalence: an object identical to the original's
        (or to an earlier mutant's) is the same program after -O2.
        """
        with ThreadPoolExecutor(self._jobs) as pool:
            objs = list(pool.map(lambda kv: self._compile_one(text, *kv), enumerate(sites)))

        seen = {hash_data(base_o.read_bytes())}
        kept: list[tuple[Site, Path]] = []
        equivalent: list[Site] = []
        failed: list[Site] = []
        for site, obj in zip(sites, objs, strict=True):
            if obj is None:
                failed.append(site)
                continue

            digest = hash_data(obj.read_bytes())
            if digest in seen:
                equivalent.append(site)
                continue
            seen.add(digest)
            kept.append((site, obj))
        return kept, equivalent, failed

    def _link_all(self, kept: list[tuple[Site, Path]]) -> None:
        outs = [self._out / _MUTANT_FMT.format(k) for k in range(len(kept))]
        with ThreadPoolExecutor(self._jobs) as pool:
            list(pool.map(self._link, [obj for _, obj in kept], outs))

    def _write_manifest(
        self, key: str, kept: list[tuple[Site, Path]], equivalent: list[Site], failed: list[Site]
    ) -> None:
        """Plain builtins only, so ``safe_load`` reads it without extra globals."""

        def rows(sites: Iterable[Site]) -> list[tuple[int, int]]:
            return [(s.line, s.op.value) for s in sites]

        manifest = {
            "version": _MANIFEST_VERSION,
            "key": key,
            "src": str(self._spec.src),
            "wrapper": str(self._spec.wrapper),
            "sites": rows(s for s, _ in kept),
            "equivalent": rows(equivalent),
            "failed": rows(failed),
        }
        tmp = self._out / (_MANIFEST + ".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(manifest, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, self._out / _MANIFEST)
