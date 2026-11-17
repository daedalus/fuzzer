#!/usr/bin/env python3
"""Carve media from every Chrome/Chromium profile cache and send each file as a TCP seed.

Feeds a run started with ``fuzzer-tool fuzz ... --tcp-listen-port PORT``: one
connection per file (send, close). Bodies are found by magic bytes and cut at
the simple-cache EOF trailer:

    <profile>/Cache/Cache_Data/<hash>_0
    +--------+-----+--------------------+---------------+--------------+
    | header | key | body (jpg/png/...) | EOF d8410d97..| HTTP headers |
    +--------+-----+--------------------+---------------+--------------+
      24 B     URL   ^ sniffed            ^ cut here

Searched roots (every profile under each: Default, Profile N, ...):
    Linux   $XDG_CACHE_HOME|~/.cache/{google-chrome,chromium}, snap, flatpak
    macOS   ~/Library/Caches/{Google/Chrome,Chromium}
    Windows %LOCALAPPDATA%/{Google/Chrome,Chromium}/User Data

Usage:
    python tools/chrome_cache_seeds.py --port 4444 --types png,jpg
    python tools/chrome_cache_seeds.py --dry-run            # count only
    python tools/chrome_cache_seeds.py --root ~/cache_copy  # explicit root

Read-only on the cache; close the browser or copy the cache for a stable view.
Bodies stored gzip/brotli-encoded carry no magic and are skipped.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import enum
import hashlib
import os
import socket
import struct
import sys
import time
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path


class Media(enum.Enum):
    JPG = "jpg"
    PNG = "png"
    GIF = "gif"
    WEBP = "webp"
    MP4 = "mp4"
    WEBM = "webm"


# Fixed-offset signatures: (media, magic).
SIGNATURES = (
    (Media.JPG, b"\xff\xd8\xff"),
    (Media.PNG, b"\x89PNG\r\n\x1a\n"),
    (Media.GIF, b"GIF8"),
    (Media.WEBM, b"\x1a\x45\xdf\xa3"),
)
RIFF = b"RIFF"
WEBP_TAG = b"WEBP"
WEBP_TAG_OFF = 8  # RIFF <u32 size> WEBP
FTYP = b"ftyp"
BOX_SIZE_LEN = 4  # ISO-BMFF box: <u32 size> ftyp

# Chromium simple-cache entry layout (net/disk_cache/simple/simple_entry_format.h).
ENTRY_MAGIC = 0xFCFB6D1BA7725C30
ENTRY_HEADER = struct.Struct("<QIIII")  # magic, version, key_length, key_hash, pad
TRAILER = struct.pack("<Q", 0xF4FA6F45970D41D8)  # SimpleFileEOF magic

# Bytes read to find a body start; bodies begin right after header + key.
SNIFF_WINDOW = 4096
# Per-seed cap; matches the fuzzer inbox's default cap.
DEFAULT_MAX_BYTES = 1 << 20
DEFAULT_PORT = 4444
DEFAULT_HOST = "127.0.0.1"
# Pause between seeds: the fuzzer pops one per round and drops past 1024 queued.
DEFAULT_DELAY_S = 0.05
CONNECT_TIMEOUT_S = 5.0
# Seeds between "sent N so far" lines.
PROGRESS_EVERY = 100
# Resends of one seed on a network error; waits 1+2+4+8+16 s by default,
# enough to ride out a fuzzer restart.
DEFAULT_RETRIES = 5
DEFAULT_BACKOFF_S = 1.0
MAX_BACKOFF_S = 30.0

PLATFORM_MAC = "darwin"
PLATFORM_WIN = "win"  # sys.platform prefix (win32)


@dataclasses.dataclass
class ScanStats:
    """Scan counters for the summary line."""

    unreadable: int = 0
    duplicates: int = 0
    by_media: collections.Counter = dataclasses.field(default_factory=collections.Counter)


def sniff(buf: bytes) -> tuple[Media, int] | None:
    """Earliest media signature in *buf*: (media, offset), or None."""
    hits = [(i, media) for media, magic in SIGNATURES if (i := buf.find(magic)) >= 0]

    i = buf.find(RIFF)
    if i >= 0 and buf[i + WEBP_TAG_OFF : i + WEBP_TAG_OFF + len(WEBP_TAG)] == WEBP_TAG:
        hits.append((i, Media.WEBP))

    # The box size precedes "ftyp", so the file starts 4 bytes earlier.
    i = buf.find(FTYP, BOX_SIZE_LEN)
    if i >= 0:
        hits.append((i - BOX_SIZE_LEN, Media.MP4))

    if not hits:
        return None
    off, media = min(hits, key=lambda h: h[0])
    return media, off


def _body_start(head: bytes) -> int:
    """Offset past a simple-cache header + key; 0 for other layouts (blockfile f_*)."""
    if len(head) < ENTRY_HEADER.size:
        return 0
    magic, _, key_len, _, _ = ENTRY_HEADER.unpack_from(head)
    if magic != ENTRY_MAGIC:
        return 0
    return min(ENTRY_HEADER.size + key_len, len(head))


def carve(path: Path, max_bytes: int = DEFAULT_MAX_BYTES) -> tuple[Media, bytes] | None:
    """(media, body) from one cache file, at most *max_bytes*; None if no media."""
    with path.open("rb") as fh:
        head = fh.read(SNIFF_WINDOW)
        start = _body_start(head)
        hit = sniff(head[start:])
        if hit is None:
            return None

        media, rel = hit
        fh.seek(start + rel)
        # Read past the cap by the trailer length so a trailer at the cap is still cut.
        body = fh.read(max_bytes + len(TRAILER))

    end = body.find(TRAILER)
    if end >= 0:
        body = body[:end]
    return media, body[:max_bytes]


def browser_roots(platform: str, home: Path, env: dict) -> list[Path]:
    """Chrome and Chromium cache roots for *platform*; each holds one dir per profile."""
    if platform == PLATFORM_MAC:
        base = home / "Library" / "Caches"
        return [base / "Google" / "Chrome", base / "Chromium"]

    if platform.startswith(PLATFORM_WIN):
        local = Path(env.get("LOCALAPPDATA") or home / "AppData" / "Local")
        return [local / "Google" / "Chrome" / "User Data", local / "Chromium" / "User Data"]

    cache = Path(env.get("XDG_CACHE_HOME") or home / ".cache")
    flatpak = home / ".var" / "app"
    return [
        cache / "google-chrome",
        cache / "chromium",
        home / "snap" / "chromium" / "common" / ".cache" / "chromium",
        flatpak / "org.chromium.Chromium" / "cache" / "chromium",
        flatpak / "com.google.Chrome" / "cache" / "google-chrome",
    ]


def _profile_cache(profile: Path) -> Path | None:
    """The profile's HTTP cache dir: Cache/Cache_Data, or legacy Cache."""
    for cand in (profile / "Cache" / "Cache_Data", profile / "Cache"):
        if cand.is_dir():
            return cand
    return None


def find_caches(roots: Iterable[Path]) -> list[Path]:
    """Every profile's cache dir under *roots*, in root then profile-name order."""
    found = []
    for root in roots:
        if not root.is_dir():
            continue
        for profile in sorted(root.iterdir()):
            cache = _profile_cache(profile)
            if cache is not None:
                found.append(cache)
    return found


def iter_seeds(
    caches: Iterable[Path],
    wanted: set[Media],
    max_bytes: int = DEFAULT_MAX_BYTES,
    stats: ScanStats | None = None,
) -> Iterator[tuple[Media, bytes]]:
    """Distinct (media, body) of the *wanted* types across *caches*."""
    stats = stats if stats is not None else ScanStats()
    # 16-byte digests: one profile's body is often cached by another.
    seen: set[bytes] = set()

    for cache in caches:
        for path in sorted(cache.iterdir()):
            if path.is_dir():
                continue
            try:
                got = carve(path, max_bytes)
            except OSError:
                # Evicted or locked mid-scan: counted, not fatal.
                stats.unreadable += 1
                continue
            if got is None or got[0] not in wanted or not got[1]:
                continue

            digest = hashlib.blake2b(got[1], digest_size=16).digest()
            if digest in seen:
                stats.duplicates += 1
                continue
            seen.add(digest)
            stats.by_media[got[0]] += 1
            yield got


def tcp_sink(host: str, port: int, delay: float) -> Callable[[bytes], None]:
    """Sender: one connection per seed, closed to end the message."""

    def send(data: bytes) -> None:
        with socket.create_connection((host, port), timeout=CONNECT_TIMEOUT_S) as s:
            s.sendall(data)
        time.sleep(delay)

    return send


def with_retry(
    send: Callable[[bytes], None],
    retries: int,
    backoff: float,
    sleep: Callable[[float], None] = time.sleep,
) -> Callable[[bytes], None]:
    """Wrap *send*: resend the same seed on network errors, waiting backoff * 2**k.

    E.g. retries=3, backoff=1: waits 1, 2, 4 s, then re-raises. Waits are capped
    at MAX_BACKOFF_S. Only OSError (refused, reset, timeout) is retried; a bug
    surfaces at once. A seed cut mid-send may arrive twice (partial + full).
    """

    def send_retrying(data: bytes) -> None:
        for attempt in range(retries + 1):
            try:
                send(data)
                return
            except OSError as e:
                if attempt == retries:
                    raise
                wait = min(backoff * 2**attempt, MAX_BACKOFF_S)
                print(
                    f"[!] send failed: {e}; retry {attempt + 1}/{retries} in {wait:g}s",
                    file=sys.stderr,
                )
                sleep(wait)

    return send_retrying


def _non_negative(kind: type) -> Callable[[str], int | float]:
    """argparse type: *kind* parsed from text, at least 0."""

    def parse(text: str) -> int | float:
        value = kind(text)
        if value < 0:
            raise argparse.ArgumentTypeError(f"must be >= 0, got {text}")
        return value

    return parse


def _media_set(spec: str) -> set[Media]:
    """argparse type: comma list of media names (jpg,png,...)."""
    try:
        return {Media(name.strip().lower()) for name in spec.split(",") if name.strip()}
    except ValueError:
        names = ",".join(m.value for m in Media)
        raise argparse.ArgumentTypeError(f"types are {names}, got {spec!r}") from None


def _parse(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--root", type=Path, action="append", help="cache root (repeatable); default: all known"
    )
    ap.add_argument(
        "--types", type=_media_set, default=set(Media), help="e.g. png,jpg (default: all)"
    )
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--delay", type=float, default=DEFAULT_DELAY_S, help="seconds between seeds")
    ap.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES, help="per-seed cap")
    ap.add_argument(
        "--retries",
        type=_non_negative(int),
        default=DEFAULT_RETRIES,
        help=f"resends per seed on a network error (default {DEFAULT_RETRIES})",
    )
    ap.add_argument(
        "--retry-backoff",
        type=_non_negative(float),
        default=DEFAULT_BACKOFF_S,
        help=f"first wait in s, doubled per retry, capped at {MAX_BACKOFF_S:g} (default {DEFAULT_BACKOFF_S:g})",
    )
    ap.add_argument("--dry-run", action="store_true", help="count, send nothing")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)

    roots = args.root or browser_roots(sys.platform, Path.home(), dict(os.environ))
    caches = find_caches(roots)
    if not caches:
        print("[-] no Chrome/Chromium cache found; pass --root", file=sys.stderr)
        return 1
    for cache in caches:
        print(f"[*] cache: {cache}")

    # Ifs up: pick the sink once.
    send = (
        (lambda _data: None)
        if args.dry_run
        else with_retry(
            tcp_sink(args.host, args.port, args.delay), args.retries, args.retry_backoff
        )
    )
    verb = "would send" if args.dry_run else "sent"
    stats = ScanStats()

    # Count only completed sends: a failed one is reported, not counted.
    sent = 0
    try:
        for _media, body in iter_seeds(caches, args.types, args.max_bytes, stats):
            send(body)
            sent += 1
            if sent % PROGRESS_EVERY == 0:
                print(f"[*] {verb} {sent} seeds so far", flush=True)
    except OSError as e:
        print(f"[-] send to {args.host}:{args.port} failed: {e}", file=sys.stderr)
        print(f"[-] sent {sent} seeds before failure", file=sys.stderr)
        return 1

    counts = ", ".join(
        f"{m.value}: {n}" for m, n in sorted(stats.by_media.items(), key=lambda kv: kv[0].value)
    )
    print(
        f"[*] {verb} {sent} seeds ({counts or 'none'}); dup {stats.duplicates}, unreadable {stats.unreadable}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
