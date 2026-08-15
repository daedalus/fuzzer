#!/usr/bin/env python3
"""Measure every mutation operator's yield without needing a built target.

A live fuzzing run reports each operator's *coverage* yield, which conflates
three separate things: how often the operator was offered, how often it
actually produced a different buffer when offered, and how often that buffer
found new edges. Only the third is about the target. The first two are
properties of the operator and the corpus, they are exactly where a silently
broken operator hides (see the byte_shuffle no-op, fixed in f4835f6), and
they can be measured here in seconds instead of a 10k-exec campaign.

What this reports per operator:

    avail%    fraction of (rep, input) pairs where the availability
              predicate offered the operator
    change%   of those, the fraction that returned a buffer differing from
              the input -- the no-op metric
    err%      fraction that raised
    dlen      mean length delta of a changed output
    us/call   mean wall time per call

A low change% is not automatically a bug: some operators are conditional by
design and decline on input they cannot parse. It is a bug when change% is
zero *and* the operator was offered on input it should have handled, which
is what the paired avail%/change% columns are for.

Usage:
    python3 tools/measure_operators.py [--reps N] [--corpus DIR] [--json OUT]

With --corpus, real files are used instead of the synthetic battery.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import struct
import sys
import tempfile
import time
import zlib
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fuzzer_tool.core.operator_registry import REGISTRY  # noqa: E402
from fuzzer_tool.services.fuzzer import Fuzzer  # noqa: E402


# ── synthetic corpus ────────────────────────────────────────────────────
# Deliberately diverse: a format-aware operator that is never offered a
# matching file measures as inert for a reason that has nothing to do with
# the operator.


def _minimal_elf64() -> bytes:
    phoff, phentsize, phnum = 64, 56, 1
    shoff, shentsize, shnum = phoff + phentsize * phnum, 64, 2
    ehdr = b"\x7fELF" + bytes([2, 1, 1, 0]) + bytes(8)
    ehdr += struct.pack(
        "<HHIQQQIHHHHHH", 2, 0x3E, 1, 0x401000, phoff, shoff, 0,
        64, phentsize, phnum, shentsize, shnum, 1,
    )
    phdr = struct.pack("<IIQQQQQQ", 1, 5, 0, 0x400000, 0x400000, 0x100, 0x100, 0x1000)
    sh_strtab = struct.pack("<IIQQQQIIQQ", 1, 3, 0, 0, shoff + shentsize * shnum, 0x11, 0, 0, 1, 0)
    return ehdr + phdr + bytes(shentsize) + sh_strtab + b"\x00.shstrtab\x00" + bytes(16)


def _binary_stl(triangles: int = 4) -> bytes:
    out = bytearray(b"synthetic binary STL".ljust(80, b"\x00"))
    out += struct.pack("<I", triangles)
    for _ in range(triangles):
        out += struct.pack("<12f", 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0) + struct.pack("<H", 0)
    return bytes(out)


def _png() -> bytes:
    def chunk(tag: bytes, payload: bytes) -> bytes:
        body = tag + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

    raw = b"".join(b"\x00" + bytes((x * 7) & 0xFF for x in range(16)) for _ in range(16))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 16, 16, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def _jpeg() -> bytes:
    qt = bytes(range(1, 65))
    return (
        b"\xff\xd8"
        + b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        + b"\xff\xdb" + struct.pack(">H", 67) + b"\x00" + qt
        + b"\xff\xc0" + struct.pack(">H", 11) + b"\x08\x00\x10\x00\x10\x01\x01\x11\x00"
        + b"\xff\xda" + struct.pack(">H", 8) + b"\x01\x01\x00\x00\x3f\x00"
        + bytes(64) + b"\xff\xd9"
    )


def _riff_wave() -> bytes:
    pcm = b"".join(struct.pack("<h", (i * 137) % 3000 - 1500) for i in range(256))
    fmt = struct.pack("<HHIIHH", 1, 1, 8000, 16000, 2, 16)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
    body += b"data" + struct.pack("<I", len(pcm)) + pcm
    return b"RIFF" + struct.pack("<I", len(body)) + body


def _mp4() -> bytes:
    def box(tag: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload) + 8) + tag + payload

    return (
        box(b"ftyp", b"isom\x00\x00\x02\x00isomiso2mp41")
        + box(b"moov", box(b"mvhd", bytes(100)) + box(b"trak", box(b"tkhd", bytes(84))))
        + box(b"mdat", bytes(range(256)))
    )


def _webp() -> bytes:
    vp8 = b"\x9d\x01\x2a" + bytes(61)
    body = b"WEBP" + b"VP8 " + struct.pack("<I", len(vp8)) + vp8
    return b"RIFF" + struct.pack("<I", len(body)) + body


def _matroska() -> bytes:
    return b"\x1aE\xdf\xa3" + b"\x01\x00\x00\x00\x00\x00\x00\x20" + bytes(96)


def _ogg() -> bytes:
    return b"OggS\x00\x02" + bytes(20) + b"\x01vorbis" + bytes(64)


def _mp3_id3() -> bytes:
    return b"ID3\x03\x00\x00\x00\x00\x02\x76" + bytes(0x176) + b"\xff\xfb\x90\x00" + bytes(200)


def _gif() -> bytes:
    return (
        b"GIF89a" + struct.pack("<HH", 16, 16) + b"\xf7\x00\x00"
        + bytes(768) + b"\x2c" + struct.pack("<HHHH", 0, 0, 16, 16) + b"\x00\x08" + bytes(40) + b"\x3b"
    )


def _pgs() -> bytes:
    payload = bytes(range(24))
    return b"PG" + struct.pack(">IIBH", 1, 0, 0x16, len(payload)) + payload


def _h264() -> bytes:
    return (
        b"\x00\x00\x00\x01\x67\x42\xc0\x1e" + bytes(24)
        + b"\x00\x00\x00\x01\x68\xce\x3c\x80" + bytes(16)
        + b"\x00\x00\x00\x01\x65" + bytes(64)
    )


def _der() -> bytes:
    inner = b"\x02\x01\x05" + b"\x04\x08" + bytes(8) + b"\x0c\x05hello"
    return b"\x30" + bytes([len(inner)]) + inner


def _zip() -> bytes:
    name, data = b"a.txt", b"hello world" * 4
    lfh = (
        b"PK\x03\x04" + struct.pack("<HHHHHIIIHH", 20, 0, 0, 0, 0, zlib.crc32(data),
                                    len(data), len(data), len(name), 0) + name + data
    )
    return lfh + b"PK\x05\x06" + bytes(18)


def _synthetic_corpus() -> list[bytes]:
    rng = random.Random(20260815)
    corpus = [
        _png(), _jpeg(), _riff_wave(), _mp4(), _webp(), _matroska(), _ogg(),
        _mp3_id3(), _gif(), _pgs(), _h264(), _der(), _zip(), _minimal_elf64(),
        _binary_stl(),
        zlib.compress(b"the quick brown fox jumps over the lazy dog" * 8, 6),
        b"\x1f\x8b\x08\x00" + bytes(6) + zlib.compress(b"gzip body" * 16, 6)[2:-4],
        b"BM" + struct.pack("<IHHI", 138, 0, 0, 54) + bytes(84) + bytes(256),
        b"\x00\x00\x01\xba" + bytes(120),
        b"solid mesh\n" + b"facet normal 0 0 1\n" * 8 + b"endsolid mesh\n",
        b'<?xml version="1.0"?><svg width="10" height="10"><path d="M0,0 L1,1"/></svg>',
        b"key=value\nother=123\n# comment\nlist=[1,2,3]\n",
        # x86-64 prologue/epilogue, for the code-aware mutators
        bytes.fromhex("554889e54883ec20488975e0488955d8b8000000004883c4205dc3") * 3,
        b"12345 6789 -3 0.5 abcdef ghij",
        b"the quick brown fox jumped 12345",
    ]
    corpus += [
        bytes(rng.randrange(256) for _ in range(n))
        for n in (16, 64, 256, 1024)
        for _ in range(2)
    ]
    return corpus


def _load_corpus(path: str) -> list[bytes]:
    out = []
    for root, _dirs, files in os.walk(path):
        for fn in sorted(files):
            fp = os.path.join(root, fn)
            try:
                with open(fp, "rb") as fh:
                    data = fh.read(1 << 20)
                if data:
                    out.append(data)
            except OSError:
                continue
    return out


# ── harness ─────────────────────────────────────────────────────────────

_DICT = [
    b"ftyp", b"moov", b"mdat", b"WEBP", b"RIFF", b"IHDR", b"IDAT",
    b"JFIF", b"Exif", b"vorbis", b"OpusHead", b"\x00\x00\x01",
]


def _make_fuzzer(corpus: list[bytes]) -> Fuzzer:
    tmp = tempfile.mkdtemp(prefix="measure_ops_")
    os.makedirs(f"{tmp}/corpus", exist_ok=True)
    with (
        patch.object(Fuzzer, "_setup_forkserver", lambda self: None),
        patch("os.path.isfile", return_value=True),
        patch("os.access", return_value=True),
    ):
        f = Fuzzer(
            target="/bin/true",
            corpus_dir=f"{tmp}/corpus",
            crashes_dir=f"{tmp}/crashes",
            max_len=1 << 16,
            timeout=1,
            mutations_per_input=2,
        )
    # A populated dictionary makes the dict-gated operators available; leaving
    # it empty (as the no-op regression test does) hides eight of them.
    f.dictionary = list(_DICT)
    f.corpus = [bytearray(c) for c in corpus]

    # Flag-gated operators. These are off by default in a plain run, so they
    # measure as never-offered rather than as inert.
    f.enable_x86_mutator = True
    f.enable_arm_mutator = True
    f.enable_regex_bomb = True

    # cmplog-gated operators need a pair pool. Real pairs come from the shim
    # at runtime; these stand in so the operators are at least exercised.
    # Tokens are drawn from the corpus so the "solve" has somewhere to land.
    from fuzzer_tool.core.cmplog import CmplogCollector

    pool = CmplogCollector()
    pool.pairs = [
        (b"ftyp", b"isom"), (b"RIFF", b"WEBP"), (b"\x89PNG", b"\x89png"),
        (b"IHDR", b"IDAT"), (b"\xff\xd8", b"\xff\xd9"), (b"OggS", b"Oggs"),
        (b"fmt ", b"data"), (b"moov", b"mdat"),
    ]
    f._cmplog = pool

    # redqueen is gated on per-seed metadata recorded by a live run.
    f.seed_meta = {
        bytes(c): {"redqueen_matches": [(0, b"ftyp")], "redqueen_offsets": [0, 4]}
        for c in corpus
    }

    return f


def _prime_dict_scratch(f: Fuzzer, n: int = 4096) -> None:
    """Refill the dictionary index scratch.

    The mutate loop refills this once per seed; dispatching operators
    directly bypasses that, and the dict operators then decline every call
    (they gate on ``_dict_scratch_idx < len(_dict_scratch)`` while their
    availability predicate only checks that a dictionary exists). Without
    this the six index-consuming dict operators measure as 0% change, which
    is a property of the harness, not of the operators.
    """
    if not getattr(f, "dictionary", None):
        return
    f._dict_scratch = f._rand_pool.randint_list(0, len(f.dictionary) - 1, n)
    f._dict_scratch_idx = 0


def measure(corpus: list[bytes], reps: int, seed: int) -> list[dict]:
    f = _make_fuzzer(corpus)
    table = REGISTRY.dispatch(f._operators)
    names = sorted(table)
    stats = {
        n: {"offered": 0, "changed": 0, "errors": 0, "dlen": 0, "time": 0.0, "none": 0}
        for n in names
    }

    for rep in range(reps):
        f._rand_pool.reseed(seed + rep)
        for inp in corpus:
            _prime_dict_scratch(f)
            avail = set(REGISTRY.available(f, inp))
            for name in names:
                if name not in avail:
                    continue
                s = stats[name]
                s["offered"] += 1
                buf = bytearray(inp)
                idx = len(buf) // 2
                t0 = time.perf_counter()
                try:
                    ret = table[name](buf, idx, bytes(inp))
                except Exception:
                    s["errors"] += 1
                    s["time"] += time.perf_counter() - t0
                    continue
                s["time"] += time.perf_counter() - t0
                if ret is None:
                    s["none"] += 1
                out = bytes(ret) if isinstance(ret, (bytes, bytearray)) else bytes(buf)
                if out != inp:
                    s["changed"] += 1
                    s["dlen"] += len(out) - len(inp)

    total_slots = reps * len(corpus)
    rows = []
    for name, s in stats.items():
        off = s["offered"]
        rows.append({
            "operator": name,
            "category": REGISTRY.category_of(name),
            "offered": off,
            "avail_pct": 100.0 * off / total_slots if total_slots else 0.0,
            "changed": s["changed"],
            "change_pct": 100.0 * s["changed"] / off if off else 0.0,
            "err_pct": 100.0 * s["errors"] / off if off else 0.0,
            "none_pct": 100.0 * s["none"] / off if off else 0.0,
            "mean_dlen": s["dlen"] / s["changed"] if s["changed"] else 0.0,
            "us_per_call": 1e6 * s["time"] / off if off else 0.0,
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--corpus", type=str, default=None)
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--sort", default="change_pct", choices=["change_pct", "avail_pct", "us_per_call", "operator"])
    args = ap.parse_args()

    corpus = _load_corpus(args.corpus) if args.corpus else _synthetic_corpus()
    if not corpus:
        print("empty corpus", file=sys.stderr)
        return 2

    rows = measure(corpus, args.reps, args.seed)
    rows.sort(key=lambda r: (r[args.sort] if args.sort != "operator" else r["operator"]))

    print(f"corpus={len(corpus)} inputs  reps={args.reps}  slots={args.reps*len(corpus)}  seed={args.seed}\n")
    print(f"{'operator':<26}{'category':<14}{'avail%':>8}{'change%':>9}{'err%':>7}{'none%':>7}{'dlen':>9}{'us/call':>9}")
    print("-" * 89)
    for r in rows:
        print(
            f"{r['operator']:<26}{r['category']:<14}{r['avail_pct']:>8.1f}{r['change_pct']:>9.1f}"
            f"{r['err_pct']:>7.1f}{r['none_pct']:>7.1f}{r['mean_dlen']:>9.1f}{r['us_per_call']:>9.1f}"
        )

    dead = [r["operator"] for r in rows if r["offered"] and r["change_pct"] == 0.0]
    unmeasured = [r["operator"] for r in rows if not r["offered"]]
    print(f"\n{len(rows) - len(unmeasured)} operators exercised, {len(unmeasured)} never offered")
    for n in dead:
        print(f"  ZERO CHANGE  {n}")
    for n in unmeasured:
        print(f"  NOT OFFERED  {n}  (needs live state this harness cannot fake)")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
