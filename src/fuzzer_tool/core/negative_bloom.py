"""Negative bloom: corpus hashes in a sized bloom file; fuzzing skips members.

    seeds --build--> BloomFilter(10 x len(seeds)) --save--> neg.pkl.gz
    neg.pkl.gz --load--> Fuzzer._neg_bloom: mutant in filter -> skipped
    neg.pkl.gz exists --load--> + new seeds --save--> neg.pkl.gz (CLI extends)

The file is a gzip'd pickle of ``BloomFilter.to_dict()``, read back through
the sanitised unpickler of ``state_store`` (a tampered file cannot run code).
"""

import gzip
import hashlib
import os
import pickle
import tempfile
import zlib
from collections.abc import Iterable
from pathlib import Path

from fuzzer_tool.core.bloom import BloomFilter
from fuzzer_tool.core.state_store import _safe_loads

#: Filter capacity per corpus entry: the headroom keeps the realised FPR far
#: below ``NEG_BLOOM_ERROR_RATE`` and leaves room for growth.
NEG_BLOOM_FACTOR = 10
NEG_BLOOM_ERROR_RATE = 1e-3
#: Seeds read per batch by the CLI (``--batch-size``); 0 there loads all at once.
NEG_BLOOM_BATCH = 4096
_DIGEST_LEN = hashlib.sha256().digest_size
_GZIP_LEVEL = 6
#: Decompressed-size cap (Rule 54): a gzip bomb must not exhaust memory.
MAX_RAW_BYTES = 1 << 30
_LOAD_ERRORS = (
    OSError,
    EOFError,
    zlib.error,
    pickle.UnpicklingError,
    ValueError,
    KeyError,
    TypeError,
    OverflowError,
)


class NegativeBloomError(Exception):
    """Unusable negative-bloom input or file."""


def build(seeds: Iterable[bytes], base: BloomFilter | None = None) -> BloomFilter:
    """Filter holding every seed, sized ``NEG_BLOOM_FACTOR`` x the corpus.

    Duplicate seeds count once (``n_added``); ``n_queries`` / ``n_accepted``
    / ``n_rejected`` give the ingest stats: probes, accepted (new), rejected
    (already present).  A *base* filter is extended in place, not resized.
    """
    items = list(seeds)
    if not items:
        raise NegativeBloomError("empty corpus: nothing to put in the negative bloom")

    bf = base if base is not None else _sized(len(items))
    for item in items:
        bf.update_bytes(bytes(item))
    return bf


def build_batched(batches: Iterable[list[bytes]], base: BloomFilter | None = None) -> BloomFilter:
    """:func:`build` over seed batches, one batch in memory at a time.

    The count sizes the filter, so digests are kept (32 B per seed, 6.4 MB
    for 200k) until the last batch, then inserted.
    """
    digests = bytearray()
    for batch in batches:
        for seed in batch:
            digests += hashlib.sha256(seed).digest()
    if not digests:
        raise NegativeBloomError("empty corpus: nothing to put in the negative bloom")

    bf = base if base is not None else _sized(len(digests) // _DIGEST_LEN)
    view = memoryview(digests)
    for off in range(0, len(digests), _DIGEST_LEN):
        bf.update_digest(int.from_bytes(view[off : off + _DIGEST_LEN], "big"))
    return bf


def _sized(count: int) -> BloomFilter:
    return BloomFilter(capacity=NEG_BLOOM_FACTOR * count, error_rate=NEG_BLOOM_ERROR_RATE)


def print_stats(bf: BloomFilter) -> None:
    """Print the ingest stats for *bf* to stdout (used by the CLI)."""
    fpr = bf.expected_fpr
    print(
        f"  Negative bloom:    {bf.n_queries} update-queries, "
        f"{bf.n_accepted} accepted (new), {bf.n_rejected} rejected (already present), "
        f"FP rate: {fpr:.2e}"
    )


def save(path: str | Path, bf: BloomFilter) -> None:
    """Write *bf* as gzip'd pickle; atomic (temp file + rename)."""
    path = Path(path)
    blob = gzip.compress(pickle.dumps(bf.to_dict()), compresslevel=_GZIP_LEVEL)

    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(blob)
        Path(tmp).replace(path)
    finally:
        Path(tmp).unlink(missing_ok=True)
    print(f"  Saved negative bloom: {path}  FP rate at save: {bf.expected_fpr:.2e}")


def load(path: str | Path) -> BloomFilter:
    """Read a file written by :func:`save`.

    Raises:
        NegativeBloomError: missing, truncated, oversized or inconsistent file.
        UnsafeStateError: the pickle references a disallowed global.
    """
    try:
        with gzip.open(path, "rb") as fh:
            raw = fh.read(MAX_RAW_BYTES + 1)
        if len(raw) > MAX_RAW_BYTES:
            raise NegativeBloomError(f"{path}: decompressed size over {MAX_RAW_BYTES} bytes")

        state = _safe_loads(raw)
        if not isinstance(state, dict):
            raise NegativeBloomError(f"{path}: not a negative-bloom file")

        bf = BloomFilter.from_dict(state)
        print(f"  Loaded negative bloom: {path}  FP rate at load: {bf.expected_fpr:.2e}")
        return bf
    except _LOAD_ERRORS as exc:
        raise NegativeBloomError(f"{path}: {exc}") from exc
