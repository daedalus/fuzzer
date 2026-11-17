#!/usr/bin/env python3
"""Listen on TCP and turn every received message into a fuzzer corpus seed.

Ingestion only: nothing is executed and nothing is forwarded to a fuzzer.
One connection = one message (send, close), the same protocol as
``fuzzer-tool fuzz --tcp-listen-port``, so every feeder works unchanged
(chrome_cache_seeds.py, blockfile_seeds.py, dir_seeds.py, ``nc``).

    feeder --TCP--> [TcpSeedInbox] --pop--> CorpusSink --> CORPUS/seeds/<h[:2]>/id_<h>
                                                |
                                     --bloom FILE: already ingested? skip

The layout is the fuzzer's own (``save_to_corpus``, ``fuzzer-tool import``),
so ``fuzzer-tool fuzz TARGET -d CORPUS`` uses the result as is. A message is
skipped when the bloom file says an earlier run (or this one) took it, or
when the same bytes are already on disk; bytes that only share the id are
kept beside as ``id_<h>~1``. Each file is written atomically (temp + rename)
so a fuzzer reading the same corpus never sees half a seed. The bloom file
is saved every SAVE_EVERY new seeds and on exit (merged, never overwritten).
Ctrl-C stops listening and writes whatever was already received.

Usage:
    python tools/tcp_corpus.py --port 4444 --corpus ~/fuzzing/corpuses/web --bloom ~/fuzzing/web_corpus.pkl.gz
    python tools/dir_seeds.py ~/samples --port 4444      # any feeder, unchanged
"""

from __future__ import annotations

import argparse
import enum
import os
import sys
import threading
from pathlib import Path

import chrome_cache_seeds as ccs

from fuzzer_tool.adapters.filesystem import free_slot
from fuzzer_tool.adapters.tcp_seed import DEFAULT_HOST, DEFAULT_MAX_BYTES, TcpSeed, TcpSeedInbox
from fuzzer_tool.core.negative_bloom import NegativeBloomError
from fuzzer_tool.core.state_store import UnsafeStateError

# New seeds between bloom-file saves (also saved on exit).
SAVE_EVERY = 100
# Idle wait between queue checks: bounds Ctrl-C / stop latency.
POLL_S = 0.05
# The fuzzer's corpus layout: CORPUS/seeds/<h[:2]>/id_<h>.
SEEDS_DIR = "seeds"
ID_PREFIX = "id_"
TMP_SUFFIX = ".tmp"


class Taken(enum.Enum):
    """What CorpusSink.take() did with a message."""

    SAVED = "saved"
    DUPLICATE = "duplicate"


class CorpusSink:
    """Writes messages into a corpus directory, skipping ones already taken."""

    def __init__(self, corpus: Path, log: ccs.SentLog, save_every: int = SAVE_EVERY) -> None:
        self._corpus = corpus
        self._log = log
        self._save_every = save_every
        self.saved = 0
        self.duplicates = 0

    def take(self, seed: TcpSeed) -> Taken:
        """Store *seed* unless the bloom or the disk already has it."""
        data = seed.data
        if self._log.seen(data):
            return self._duplicate(seed, "seen before")

        sub = self._corpus / SEEDS_DIR / seed.seed_id[:2]
        sub.mkdir(parents=True, exist_ok=True)
        # The id's file, or its first ~n sibling that is free or holds these bytes.
        dest = free_slot(sub / f"{ID_PREFIX}{seed.seed_id}", data)
        if dest.exists():
            self._log.record(data)  # on disk from an earlier import; remember it
            return self._duplicate(seed, "already in corpus")

        _write_atomic(dest, data)
        self._log.record(data)
        self.saved += 1
        _say(f"[*] Saved id_{seed.seed_id} ({len(data)} bytes) -> {dest}")
        if self.saved % self._save_every == 0:
            self._log.save()
        return Taken.SAVED

    def _duplicate(self, seed: TcpSeed, why: str) -> Taken:
        self.duplicates += 1
        _say(f"[=] Skipped id_{seed.seed_id}: {why}")
        return Taken.DUPLICATE


def _say(line: str) -> None:
    """One write per line: the listener thread prints "Received" lines concurrently."""
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def _write_atomic(dest: Path, data: bytes) -> None:
    """Write via a temp file and rename: a concurrent reader sees all or nothing."""
    tmp = dest.with_name(dest.name + TMP_SUFFIX)
    tmp.write_bytes(data)
    os.replace(tmp, dest)


def serve(inbox: TcpSeedInbox, sink: CorpusSink, stop: threading.Event) -> None:
    """Move messages from *inbox* to *sink* until *stop*; then write what is still queued."""
    while not stop.is_set():
        seed = inbox.pop_seed()
        if seed is None:
            stop.wait(POLL_S)
            continue
        sink.take(seed)

    # Received before the stop: written, not dropped.
    while (seed := inbox.pop_seed()) is not None:
        sink.take(seed)


def _parse(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, required=True, help="TCP port to listen on")
    ap.add_argument(
        "--corpus", type=Path, required=True, help="corpus directory (created if missing)"
    )
    ap.add_argument(
        "--bloom",
        type=Path,
        required=True,
        metavar="FILE",
        help="bloom file of messages already ingested (created if missing, merged on save)",
    )
    ap.add_argument("--host", default=DEFAULT_HOST, help=f"bind address (default {DEFAULT_HOST})")
    ap.add_argument(
        "--max-bytes",
        type=int,
        default=DEFAULT_MAX_BYTES,
        help="per-message cap; longer ones are cut",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None, stop: threading.Event | None = None) -> int:
    """Listen until Ctrl-C (or *stop*, for tests); 0 on a clean stop."""
    args = _parse(argv)
    if args.corpus.exists() and not args.corpus.is_dir():
        print(f"[-] --corpus {args.corpus}: not a directory", file=sys.stderr)
        return 1
    try:
        log = ccs.SentLog(args.bloom, ccs.Persist.WRITE)
    except (NegativeBloomError, UnsafeStateError) as e:
        print(f"[-] --bloom: {e}", file=sys.stderr)
        return 1

    sink = CorpusSink(args.corpus, log)
    inbox = TcpSeedInbox(args.port, args.host, max_bytes=args.max_bytes)
    print(f"[*] Listening on {args.host}:{inbox.port} -> {args.corpus} (Ctrl-C to stop)")
    stop = stop if stop is not None else threading.Event()
    try:
        serve(inbox, sink, stop)
    except KeyboardInterrupt:
        stop.set()
        inbox.close()
        serve(inbox, sink, stop)  # write what was received before Ctrl-C
    finally:
        inbox.close()
        log.save()

    print(
        f"[*] saved {sink.saved}, duplicate {sink.duplicates}; "
        f"received {inbox.received}, dropped {inbox.dropped} (inbox full)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
