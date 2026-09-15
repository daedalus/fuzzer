"""Structure-aware ffconcat (HLS/M3U) mutations.

Parses ffconcat-style playlist text structure (file/duration/target
entries separated by newlines). Targets CVE-2026-65704: integer underflow
in size decrement wraps to near-SIZE_MAX, and CVE-2016-1897: malformed
stream URL parsing.

FFmpeg ffconcat format (see ffconcat(1)):

    ffconcat version 1.0
    file 'path/to/input.mp4'
    duration 5.000000
    ...
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fuzzer_tool.core.rand_pool import RandPool


@dataclass
class FfconcatEntry:
    """A single ffconcat entry."""

    key: str
    value: str
    raw: bytes


def parse_ffconcat(data: bytes) -> list[FfconcatEntry] | None:
    """Parse ffconcat playlist from data.

    Returns list of entries or None if not a valid ffconcat stream.
    """
    if b"ffconcat" not in data[:64].lower():
        return None

    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return None

    lines = text.strip().split("\n")
    entries: list[FfconcatEntry] = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            entries.append(FfconcatEntry(key.strip(), value.strip(), line.encode("utf-8")))
        # Keep ffconcat and file/duration lines as-is without key-value
        if (
            line.lower().startswith("ffconcat")
            or line.lower().startswith("file")
            or line.lower().startswith("duration")
        ):
            entries.append(FfconcatEntry("", "", line.encode("utf-8")))

    return entries if entries else None


class FfconcatMutator:
    """Structure-aware ffconcat mutator.

    Targets:
    - CVE-2026-65704: integer underflow in duration/length fields
    - CVE-2016-1897: malformed stream URL/relative path parsing
    """

    def __init__(self, seed=None):
        rng = RandPool(seed=seed)
        self._rng = rng

    def mutate(self, data: bytes, max_len: int = 65536, rng: Any = None) -> bytes:
        self._rng = rng or self._rng
        entries = parse_ffconcat(data)
        if not entries:
            return self._generate_random_ffconcat(max_len=max_len, rng=self._rng)

        op = self._rng.randint(0, 4)
        mutators = [
            self._mutate_duration,
            self._mutate_file_size,
            self._corrupt_path,
            self._truncate_entry,
            lambda _data, _entries, max_len: self._generate_random_ffconcat(
                max_len=max_len, rng=self._rng
            ),
        ]
        result = mutators[op](data, entries, max_len)
        return result[:max_len]

    def _mutate_duration(self, data: bytes, entries: list[FfconcatEntry], max_len: int) -> bytes:
        """Corrupt duration fields to trigger integer underflow."""
        raw = bytearray(data)
        for line in entries:
            if line.key == "duration":
                raw_str = line.raw.decode("utf-8", errors="replace")
                new_durations = [
                    "duration -1",
                    "duration 0",
                    "duration 0.000001",
                    "duration 999999999999999999999.000000",
                    "duration -999999999",
                ]
                new_duration = self._rng.choice(new_durations)
                raw[:] = data.replace(raw_str.encode(), new_duration.encode())[:max_len]
                break
        return bytes(raw[:max_len])

    def _mutate_file_size(self, data: bytes, entries: list[FfconcatEntry], max_len: int) -> bytes:
        """Corrupt file size values to trigger overflow in size tracking."""
        raw = bytearray(data)
        for line in entries:
            if line.key.startswith("file"):
                raw_str = line.raw.decode("utf-8", errors="replace")
                new_files = [
                    "file 'size:0xFFFFFFFFFFFFFFFF'",
                    "file 'size:0'",
                    "file 'size:-1'",
                    "file 'size:4294967296'",
                ]
                new_file = self._rng.choice(new_files)
                raw[:] = data.replace(raw_str.encode(), new_file.encode())[:max_len]
                break
        return bytes(raw[:max_len])

    def _corrupt_path(self, data: bytes, entries: list[FfconcatEntry], max_len: int) -> bytes:
        """Corrupt stream URL/relative path entries."""
        raw = bytearray(data)
        for line in entries:
            if line.key.startswith("file") or line.key.startswith("http"):
                raw_str = line.raw.decode("utf-8", errors="replace")
                new_paths = [
                    "file 'file:file:file:'",
                    "file 'http://localhost/../../../etc/passwd'",
                    "file 'concat:null'",
                    "file ''",
                    "file '//host/share/file.mp4'",
                ]
                new_file = self._rng.choice(new_paths)
                raw[:] = data.replace(raw_str.encode(), new_file.encode())[:max_len]
                break
        return bytes(raw[:max_len])

    def _truncate_entry(self, data: bytes, entries: list[FfconcatEntry], max_len: int) -> bytes:
        """Truncate a random entry mid-value."""
        if not entries:
            return data
        target = self._rng.choice(entries)
        if len(target.raw) > 4:
            truncated = target.raw[: self._rng.randint(1, len(target.raw) - 1)]
            raw = data.replace(target.raw, truncated)
            return bytes(raw[:max_len])
        return data

    def _generate_random_ffconcat(self, max_len: int = 65536, rng: Any = None) -> bytes:
        """Generate a minimal ffconcat playlist with corrupt entries."""
        self._rng = rng or self._rng
        result = bytearray()
        result.extend(b"ffconcat version 1.0\n")
        result.extend(b"file 'size:0xFFFFFFFFFFFFFFFF'\n")
        result.extend(b"duration -1\n")
        result.extend(b"file 'http://localhost/../../../etc/passwd'\n")

        return bytes(result[:max_len])


__all__ = ["parse_ffconcat", "FfconcatMutator", "FfconcatEntry"]
