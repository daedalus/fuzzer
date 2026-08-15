"""Regression: no registered mutation operator may be a pure no-op.

``byte_shuffle`` shuffled a throwaway bytearray slice copy and returned its
input byte-for-byte unchanged for every input (see
``test_mutations.TestByteShuffleRegression``). An operator like that is worse
than useless: the schedulers still select it, spend budget on it, and credit
it, so it dilutes every bandit/Elo signal while producing nothing.

This sweep drives *every* registered operator through the real dispatch table
(``REGISTRY.dispatch``) over a fixed, diverse input battery and requires that
each operator which declares itself *available* (``REGISTRY.available``) for an
input changes at least one input.

To stay deterministic and non-flaky, each operator is evaluated in isolation
with the fuzzer RNG reseeded per repetition, so one operator's RNG consumption
never cascades into another's. Operators gated on runtime scheduling state a
unit test does not build (dictionary scratch, grammar, cmplog pairs,
CEM/markov) report themselves unavailable and are skipped, so no
hand-maintained skip list is needed beyond one structural operator noted below.
"""

from __future__ import annotations

import os
import random
import struct
import tempfile
import zlib
from unittest.mock import patch

import fuzzer_tool.core.mutations as _mutations
from fuzzer_tool.core.operator_registry import REGISTRY
from fuzzer_tool.services.fuzzer import Fuzzer

_REPS = 12
_BASE_SEED = 4242


def _minimal_elf64() -> bytes:
    """A structurally valid ELF64 with one program header and two sections.

    ``elf_chunk_mutate``'s availability predicate only checks the magic and
    the class/endianness bytes, but the mutator itself parses the full
    header and patches real offset/count fields, so a bare ``\\x7fELF`` is
    not enough to exercise it -- hence this synthesized body.
    """
    phoff, phentsize, phnum = 64, 56, 1
    shoff, shentsize, shnum = phoff + phentsize * phnum, 64, 2

    ehdr = b"\x7fELF" + bytes([2, 1, 1, 0]) + bytes(8)  # 64-bit, LE, v1, SysV
    ehdr += struct.pack(
        "<HHIQQQIHHHHHH",
        2,  # e_type    = ET_EXEC
        0x3E,  # e_machine = x86-64
        1,  # e_version
        0x401000,  # e_entry
        phoff,
        shoff,
        0,  # e_flags
        64,  # e_ehsize
        phentsize,
        phnum,
        shentsize,
        shnum,
        1,  # e_shstrndx
    )
    phdr = struct.pack(
        "<IIQQQQQQ", 1, 5, 0, 0x400000, 0x400000, 0x100, 0x100, 0x1000
    )  # PT_LOAD, R+X
    sh_null = bytes(shentsize)
    sh_strtab = struct.pack(
        "<IIQQQQIIQQ", 1, 3, 0, 0, shoff + shentsize * shnum, 0x11, 0, 0, 1, 0
    )
    return ehdr + phdr + sh_null + sh_strtab + b"\x00.shstrtab\x00" + bytes(16)


def _binary_stl(triangles: int = 2) -> bytes:
    """Binary STL: 80-byte header, u32 triangle count, count * 50 bytes.

    Has no magic, so its sniffer checks that length invariant -- meaning a
    truncated or padded sample silently fails to register.
    """
    out = bytearray(b"synthetic binary STL".ljust(80, b"\x00"))
    out += struct.pack("<I", triangles)
    for _ in range(triangles):
        out += struct.pack("<12f", 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0)
        out += struct.pack("<H", 0)  # attribute byte count
    return bytes(out)


def _webp() -> bytes:
    vp8 = b"\x9d\x01\x2a" + bytes(29)
    body = b"WEBP" + b"VP8 " + struct.pack("<I", len(vp8)) + vp8
    return b"RIFF" + struct.pack("<I", len(body)) + body


def _pgs() -> bytes:
    """One PGS presentation segment: magic, pts, dts, type, length, payload."""
    payload = bytes(range(16))
    return b"PG" + struct.pack(">IIBH", 1, 0, 0x16, len(payload)) + payload


def _h264_annexb() -> bytes:
    """Two Annex-B NAL units (SPS then PPS start codes)."""
    return (
        b"\x00\x00\x00\x01\x67\x42\xc0\x1e"
        + bytes(24)
        + b"\x00\x00\x00\x01\x68\xce\x3c\x80"
        + bytes(16)
    )


def _battery() -> list[bytes]:
    """Fixed input battery: random inputs of several lengths (deterministic via
    a local seed) plus magic-prefixed samples so format-aware operators become
    available and get exercised."""
    rng = random.Random(999)
    inputs = [
        bytes(rng.randrange(256) for _ in range(n))
        for n in (8, 32, 128, 512)
        for _ in range(4)
    ]
    inputs += [
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + bytes(30),
        b"\xff\xd8\xff\xe0\x00\x10JFIF" + bytes(30),
        b"GIF89a" + bytes(30),
        b"RIFF\x24\x00\x00\x00WAVEfmt " + bytes(30),
        bytes(4) + b"ftypisom" + bytes(30),
        b"\x1aE\xdf\xa3" + bytes(30),
        b"PK\x03\x04" + bytes(30),
        b"\x1f\x8b\x08\x00" + bytes(30),
        b"Rar!\x1a\x07\x00" + bytes(30),
        b"BM\x8a\x00\x00\x00" + bytes(30),
        b"12345 6789 -3 0.5 abcdef ghij",
        _minimal_elf64(),
        _binary_stl(),
        _webp(),
        _pgs(),
        _h264_annexb(),
        # zlib stream: covers zlib_chunk_mutate and recompress_zlib, whose
        # sniffers check the CMF/FLG header rather than a magic string.
        zlib.compress(b"the quick brown fox jumps over the lazy dog" * 3, 6),
    ]
    return inputs


class TestNoMutationIsAPureNoOp:
    # Structural mutators need a well-formed file body, not just a magic
    # prefix, before they will do anything -- `available` is decided by cheap
    # sniffing that a bare 4-byte magic satisfies. `_battery()` therefore
    # synthesizes real bodies (see `_minimal_elf64`) so these operators are
    # covered like everything else.
    #
    # This set is now empty and should stay that way: an operator landing here
    # is excluded from the only check that proves it does anything at all.
    # Adding a valid sample to the battery is always the better fix.
    _NEEDS_VALID_FILE_BODY: set[str] = set()

    def _make_fuzzer(self) -> Fuzzer:
        tmp = tempfile.mkdtemp(prefix="noop_sweep_")
        os.makedirs(f"{tmp}/corpus", exist_ok=True)
        # /bin/true stands in for a target; the isfile/access patches let the
        # constructor accept it, and _setup_forkserver is neutralised so no
        # compiled fuzz_loader is required (this is a pure mutation unit test).
        with (
            patch.object(Fuzzer, "_setup_forkserver", lambda self: None),
            patch("os.path.isfile", return_value=True),
            patch("os.access", return_value=True),
        ):
            f = Fuzzer(
                target="/bin/true",
                corpus_dir=f"{tmp}/corpus",
                crashes_dir=f"{tmp}/crashes",
                max_len=4096,
                timeout=1,
                mutations_per_input=2,
            )
        # Dictionary/grammar/cmplog operators need per-seed scheduling state the
        # mutate loop builds; leave it empty so they report unavailable and are
        # skipped rather than flagged as false no-ops.
        f.dictionary = []
        # Cross-seed operators (splice/crossover) need >1 distinct corpus entry.
        f.corpus = [
            bytearray(s)
            for s in (
                b"the quick brown fox jumped 12345",
                b"a second, distinct corpus seed \xff\xfe\x01",
                b"\x89PNG\r\n\x1a\nIHDR....",
                b"third seed payload zzzzzzzz",
            )
        ]
        return f

    def _sweep(self, f: Fuzzer) -> tuple[set[str], set[str]]:
        """Return (available_somewhere, changed_somewhere) operator-name sets.

        Each operator is exercised on its own with the fuzzer RNG reseeded per
        repetition, so the result is deterministic and independent of operator
        ordering or of any other operator's RNG use.
        """
        table = REGISTRY.dispatch(f._operators)
        battery = _battery()
        names = sorted({n for inp in battery for n in REGISTRY.available(f, inp)})
        available: set[str] = set(names)
        changed: set[str] = set()
        for name in names:
            done = False
            for rep in range(_REPS):
                f._rand_pool.reseed(_BASE_SEED + rep)
                for inp in battery:
                    if name not in set(REGISTRY.available(f, inp)):
                        continue
                    buf = bytearray(inp)
                    idx = len(buf) // 2
                    try:
                        ret = table[name](buf, idx, bytes(inp))
                    except Exception:
                        # An operator raising is a separate concern; a no-op is
                        # about *silence*, not errors. Skip and let other tests
                        # cover exceptions.
                        continue
                    out = ret if isinstance(ret, (bytes, bytearray)) else buf
                    if bytes(out) != inp:
                        changed.add(name)
                        done = True
                        break
                if done:
                    break
        return available, changed

    def test_every_available_operator_changes_some_input(self):
        f = self._make_fuzzer()
        available, changed = self._sweep(f)

        # Guard the guard: if the registry/dispatch wiring breaks, `available`
        # collapses and the no-op check becomes vacuously true. Require a broad
        # operator set to actually have been exercised.
        assert len(available) > 80, f"only {len(available)} operators exercised"

        noops = (available - changed) - self._NEEDS_VALID_FILE_BODY
        assert not noops, (
            "operator(s) are available but never changed any input "
            f"(pure no-ops): {', '.join(sorted(noops))}"
        )

    def test_sweep_detects_a_planted_no_op(self):
        """Meta-test: the sweep must fail on an actual no-op, or it protects
        nothing. Plant a no-op ``byte_shuffle`` and confirm it is flagged."""
        original = _mutations.byte_shuffle
        _mutations.byte_shuffle = lambda data, rng=None: bytes(data)
        try:
            f = self._make_fuzzer()
            available, changed = self._sweep(f)
            assert "byte_shuffle" in available
            assert "byte_shuffle" in (available - changed), (
                "a planted no-op byte_shuffle went undetected — the sweep "
                "does not actually protect against no-op operators"
            )
        finally:
            _mutations.byte_shuffle = original
