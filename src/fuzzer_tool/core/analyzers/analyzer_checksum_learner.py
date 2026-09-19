"""Checksum learner: recovers unknown linear checksum polynomials from
observed ``(data, checksum)`` pairs and exposes them to the fuzzer's
mutation operators.

Attaches to the fuzzer instance as ``f.checksum_learner``.  Feeds on
three pair sources:

1. **Format-aware extraction** — deterministic extraction from known
   container formats (PNG, ZIP, GZIP) where the checksum field location
   and coverage are specified by the format.
2. **cmplog pair heuristics** — scans the global ``f._cmplog.pairs`` pool
   for 4-byte operands that appear near data in the input buffer,
   yielding candidate checksum pairs.
3. **Seed-meta / redqueen** — reuses existing ``seed_meta`` entries
   where one operand is a plausible 4-byte checksum.

Recovery covers three disjoint checksum families:

- **GF(2)-affine** (CRC-32/16/8): Berlekamp-Massey (sequential outputs) or
  GCD-of-syndromes (independent pairs), whichever the evidence supports.
- **Integer-linear mod N** (Adler-32, Fletcher-16/32, bespoke
  ``sum(data[i] * k^i) mod N``): GCD-of-differences over ``Z``, in
  :mod:`fuzzer_tool.core.int_checksum_solver`.  The GF(2) machinery cannot
  represent these at all, so a target gating on one had that field
  permanently rejected and everything downstream of validation unreachable.
- **GF(2) XOR-bitmask** (XOR-of-selected-bits): Z3-backed incremental
  recovery in :mod:`fuzzer_tool.core.xor_map_solver`.  Fills the gap for
  fields whose value is a fixed-but-unknown XOR combination of input bits
  not expressible as a single LFSR tap polynomial (packed bitmask/flags
  fields, non-adjacent byte-range XOR checksums).  Tried only after both
  GF(2) and integer paths fail to verify.

The integer path is attempted only after the GF(2) paths fail to verify;
the two models are kept separate rather than merged, because they are
used differently — see :meth:`ChecksumLearner.compute_checksum` versus
:meth:`ChecksumLearner.compute_int_checksum`.

The recovered polynomial is cached on ``f.checksum_poly`` and persisted
to ``state.json``.
"""

from __future__ import annotations

import contextlib
import struct
import zlib
from collections.abc import Callable
from typing import Any

from fuzzer_tool.core.berlekamp_massey import (
    compute_checksum as _compute_checksum,
)
from fuzzer_tool.core.berlekamp_massey import (
    recover_lfsr,
    recover_polynomial_gcd,
)
from fuzzer_tool.core.crc32 import _STANDARD_POLY, crc32, set_active_model
from fuzzer_tool.core.int_checksum import (
    IntModel,
    eval_model,
    model_from_dict,
    model_to_dict,
    set_active_int_model,
)
from fuzzer_tool.core.int_checksum_solver import recover_int_model, verify_model
from fuzzer_tool.core.xor_map_solver import (
    XorBitmaskModel,
    compute_xor_checksum,
    recover_xor_model,
    set_active_xor_model,
    verify_xor_model,
    xor_model_from_dict,
    xor_model_to_dict,
)

# recover_polynomial_gcd builds one Python int per pair (data bit-shifted
# by the checksum width) and reduces them pairwise via poly_gcd/_poly_mod,
# whose cost scales with operand bit-length -- for a PNG IDAT chunk that's
# several KB of "data" per pair, i.e. tens of thousands of bits, and
# _recover() runs this GCD reduction over the FULL pair set twice (once
# non-reflected, once reflected) plus a BM fallback. With no cap here,
# self._pairs grew for the life of the run (every new chunk/cmplog pair
# extended it, nothing ever evicted), so cost per recovery attempt grew
# without bound as the campaign progressed. Measured: a single _recover()
# call blocked fuzz_one() for 30+ seconds once the pair set reached a few
# hundred entries -- eps collapsed to single digits.
CHECKSUM_PAIRS_MAX = 128  # cap the pair set GCD reduction runs over
# Byte budget for the pairs carried in the run state file. The count cap
# above bounds the list length but not its size -- one pair's data half is a
# whole checksummed region -- and the state file is rewritten on every save.
CHECKSUM_STATE_BYTES_MAX = 256 * 1024

# _maybe_recover() previously re-ran full recovery whenever pair count
# changed AT ALL -- during active fuzzing with cmplog/format-extraction
# on, that's virtually every iteration, making the "only retry when
# something changed" guard a no-op in practice for an unverifiable pair
# set (which keeps producing "new" pairs forever without ever verifying).
# Require a real batch of new evidence before re-attempting.
RECOVERY_RETRY_BATCH = 32

# recover_polynomial_gcd's syndrome cost scales with data bit-length; a
# single pair with several KB of data (a PNG IDAT chunk) is expensive on
# its own regardless of CHECKSUM_PAIRS_MAX. GCD-of-syndromes only needs a
# couple of independent pairs in principle, so restrict the GCD path to
# modestly-sized pairs -- real checksums this tool targets (PNG IHDR/
# tEXt/etc, ZIP filenames, generic 4-byte cmplog operands) are typically
# well under this. recover_lfsr (checksum values only, no data-length
# dependency) is unaffected and still sees every pair.
_GCD_MAX_PAIR_DATA_BYTES = 256

# The integer path must NOT reuse _GCD_MAX_PAIR_DATA_BYTES -- it needs the
# opposite bias. Modulus recovery works by observing the raw sum wrap N, so a
# pair only carries modulus information once sum(data) > N: for an Adler-32
# style k=1 sum that is ~2*65521/255 ~= 512 bytes minimum. A 256-byte cap
# filters out precisely the pairs that carry the signal and leaves the GCD at
# zero forever. The integer path instead prefers the LONGEST pairs, and its
# cost is bounded differently: with multiplier 1 the raw value is a single
# small int (a byte sum) no matter how long the data is, so length is nearly
# free -- unlike the GF(2) syndromes, whose bit-length tracks the data.
# int_checksum_solver caps the multiplier > 1 case, where cost does scale.
_ZLIB_MAX_STREAM_BYTES = 1 << 18  # compressed bytes fed to the inflater
_ZLIB_MAX_INFLATED_BYTES = 1 << 20  # bound on inflate output per stream


class ChecksumLearner:
    """Learns and caches checksum polynomials from observed pairs."""

    def __init__(self, fuzzer: Any, min_pairs: int = 64, poly_width: int = 32):
        self.f = fuzzer
        self.min_pairs = min_pairs
        self._pairs: list[tuple[bytes, int]] = []
        self._poly: int | None = None  # connection polynomial (model form)
        self._poly_width: int = poly_width
        self._reflect: bool = False  # True when recovered via BM (reflected domain)
        # Pair count at the last recovery attempt. Recovery only re-runs when
        # the pair set has grown by at least RECOVERY_RETRY_BATCH since —
        # see ensure_poly()/_maybe_recover() for the why.
        self._pairs_attempted_at = -1
        # Monotonic total across add_pairs() calls, independent of the
        # CHECKSUM_PAIRS_MAX FIFO eviction in self._pairs -- the retry
        # gate needs "how much new evidence arrived", which len(self._pairs)
        # alone can't answer once eviction keeps that length pinned at the
        # cap.
        self._total_pairs_seen = 0
        # Recovered integer-modulus model (Adler/Fletcher/weighted sum), kept
        # alongside _poly rather than merged with it: the two families have
        # incompatible parameter sets and different consumers.
        self._int_model: IntModel | None = None
        # Recovered XOR-bitmask model, kept alongside _poly and _int_model
        # for the same reason: a different family with incompatible parameters.
        self._xor_model: XorBitmaskModel | None = None
        self._format_extractors: list[Callable[[bytes], list[tuple[bytes, int]]]] = [
            self._extract_png_pairs,
            self._extract_zip_pairs,
            self._extract_gzip_pairs,
            self._extract_zlib_adler_pairs,
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def has_enough_pairs(self) -> bool:
        """Return True when enough pairs have been collected to attempt recovery."""
        return len(self._pairs) >= self.min_pairs

    def add_pairs(self, pairs: list[tuple[bytes, int]]) -> None:
        """Add newly-observed (data, checksum) pairs and trigger recovery.

        Capped at CHECKSUM_PAIRS_MAX (FIFO): recovery's cost scales with
        the pair count, so an unbounded list means an unverifiable pair
        set (one that keeps producing "new" pairs without ever
        satisfying _verify()) makes every subsequent attempt more
        expensive than the last, for the life of the run.
        """
        if not pairs:
            return
        self._pairs.extend(pairs)
        self._total_pairs_seen += len(pairs)
        if len(self._pairs) > CHECKSUM_PAIRS_MAX:
            self._pairs = self._pairs[-CHECKSUM_PAIRS_MAX:]
        self._maybe_recover()

    def extract_format_pairs(self, data: bytes) -> list[tuple[bytes, int]]:
        """Run all format-aware extractors on *data*."""
        result: list[tuple[bytes, int]] = []
        for extractor in self._format_extractors:
            with contextlib.suppress(Exception):
                result.extend(extractor(data))
        return result

    def ensure_poly(self) -> int | None:
        """Return the recovered polynomial, attempting recovery if needed.

        Recovery is attempted only when at least RECOVERY_RETRY_BATCH new
        pairs have arrived since the last attempt. ``ensure_poly()`` runs
        once per fuzz iteration via the ``crc_learn`` availability gate,
        so re-running the full GCD/BM recovery on every call when it
        cannot verify (e.g. PNG CRCs with mismatched init/final_xor)
        collapses throughput — measured ~56 s of a 103 s fuzz profile,
        eps to single digits. A pure "did the count change at all" gate
        turned out not to help in practice: active fuzzing with cmplog
        and format extraction on finds a "new" pair almost every
        iteration, so that gate was true almost every time too. Requiring
        a real batch of new evidence is what actually bounds retry
        frequency; CHECKSUM_PAIRS_MAX bounds the cost of each retry.
        """
        self._maybe_recover()
        return self._poly

    def ensure_int_model(self) -> IntModel | None:
        """Return the recovered integer-modulus model, attempting recovery if needed.

        Same retry discipline as :meth:`ensure_poly` — the two families share
        one attempt counter rather than each keeping their own. They are
        recovered in a single ``_recover()`` call (integer is tried only once
        GF(2) has failed to verify), so an independent counter would only
        schedule extra attempts at extra cost without making the integer path
        any more likely to succeed.
        """
        self._maybe_recover()
        return self._int_model

    def has_model(self) -> bool:
        """True when any checksum family has a verified, active model."""
        return self._poly is not None or self._int_model is not None or self._xor_model is not None

    def ensure_model(self) -> bool:
        """Attempt recovery if needed; True when any model is available."""
        self._maybe_recover()
        return self.has_model()

    def ensure_xor_model(self) -> XorBitmaskModel | None:
        """Return the recovered XOR-bitmask model, attempting recovery if needed.

        Same retry discipline as :meth:`ensure_poly` and
        :meth:`ensure_int_model` — the three families share one attempt
        counter.  The XOR path is tried only after both GF(2) and integer
        paths have failed to verify.
        """
        self._maybe_recover()
        return self._xor_model

    def _maybe_recover(self) -> None:
        """Run recovery when unverified and a batch of new pairs arrived."""
        if (
            not self.has_model()
            and self.has_enough_pairs()
            and (
                self._pairs_attempted_at < 0
                or self._total_pairs_seen - self._pairs_attempted_at >= RECOVERY_RETRY_BATCH
            )
        ):
            self._recover()

    @property
    def pair_count(self) -> int:
        return len(self._pairs)

    # ------------------------------------------------------------------
    # Format-aware extractors
    # ------------------------------------------------------------------

    def _extract_png_pairs(self, data: bytes) -> list[tuple[bytes, int]]:
        """Extract (chunk_type+data, crc) pairs from a PNG file."""
        if len(data) < 8 or data[:8] != b"\x89PNG\r\n\x1a\n":
            return []
        result: list[tuple[bytes, int]] = []
        pos = 8
        n = len(data)
        while pos + 12 <= n:
            length = struct.unpack_from(">I", data, pos)[0]
            chunk_type = data[pos + 4 : pos + 8]
            chunk_data = data[pos + 8 : pos + 8 + length]
            crc_bytes = data[pos + 8 + length : pos + 12 + length]
            if len(crc_bytes) != 4:
                break
            crc = struct.unpack(">I", crc_bytes)[0]
            result.append((bytes(chunk_type) + bytes(chunk_data), crc))
            pos += 12 + length
            if chunk_type == b"IEND":
                break
        return result

    def _extract_zip_pairs(self, data: bytes) -> list[tuple[bytes, int]]:
        """Extract (filename+data, crc32) pairs from a ZIP local file header."""
        result: list[tuple[bytes, int]] = []
        pos = 0
        n = len(data)
        while pos + 30 <= n:
            sig = struct.unpack_from("<I", data, pos)[0]
            if sig != 0x04034B50:  # PK\\x03\\x04
                break
            crc = struct.unpack_from("<I", data, pos + 14)[0]
            fname_len = struct.unpack_from("<H", data, pos + 26)[0]
            extra_len = struct.unpack_from("<H", data, pos + 28)[0]
            fname = data[pos + 30 : pos + 30 + fname_len]
            # Use filename as the "data" for pair extraction
            result.append((bytes(fname), crc))
            comp_size = struct.unpack_from("<I", data, pos + 18)[0]
            pos += 30 + fname_len + extra_len + comp_size
        return result

    def _extract_gzip_pairs(self, data: bytes) -> list[tuple[bytes, int]]:
        """Extract (payload, original_crc) pairs from a GZIP member.

        Known limitation: the compressed payload is not decompressed, so
        pairs carry an empty payload placeholder — a gzip member's real
        (payload, crc) relationship requires inflate, which is out of
        scope for format extraction.  The empty-payload pairs never pass
        ``_verify`` (they reproduce no observed checksum), so they cannot
        corrupt recovery.
        """
        if len(data) < 18 or data[:2] != b"\x1f\x8b":
            return []
        # Trailer: CRC32(4) + ISIZE(4) at the end of the member.
        crc = struct.unpack_from("<I", data, len(data) - 8)[0]
        return [(b"", crc)]

    def _extract_zlib_adler_pairs(self, data: bytes) -> list[tuple[bytes, int]]:
        """Extract (inflated payload, Adler-32) pairs from an embedded zlib stream.

        A zlib stream carries an Adler-32 of the *decompressed* data in its
        4-byte big-endian trailer.  In a PNG that stream sits inside the
        concatenated IDAT payloads — one layer below the chunk CRC-32 the
        learner already extracts — so a real integer-checksum target is
        sitting inside a format the learner already supports.

        The trailer is read from the stream rather than recomputed: the point
        is to *observe* the format's checksum, not to assert our own.  These
        pairs are also the only routinely-large ones the learner sees, which
        matters because modulus recovery needs the raw sum to wrap ``N``
        (see ``_ZLIB_MAX_STREAM_BYTES`` and the note above it).
        """
        stream = self._zlib_stream(data)
        if stream is None:
            return []
        obj = zlib.decompressobj()
        # max_length bounds inflate output: this runs inside fuzz_one() on
        # attacker-shaped input, so an unbounded inflate is a zip-bomb.
        inflated = obj.decompress(stream, _ZLIB_MAX_INFLATED_BYTES)
        if not obj.eof:
            return []  # truncated, corrupt, or over the output bound
        consumed = len(stream) - len(obj.unused_data)
        if consumed < 4 or not inflated:
            return []
        adler = int.from_bytes(stream[consumed - 4 : consumed], "big")
        return [(inflated, adler)]

    @staticmethod
    def _zlib_stream(data: bytes) -> bytes | None:
        """Return the embedded zlib stream in *data*, or None.

        Handles PNG (concatenated IDAT payloads) and bare zlib streams.
        """
        n = len(data)
        if n >= 8 and data[:8] == b"\x89PNG\r\n\x1a\n":
            parts: list[bytes] = []
            total = 0
            pos = 8
            while pos + 12 <= n:
                length = struct.unpack_from(">I", data, pos)[0]
                if length > n:  # corrupt length field
                    return None
                if data[pos + 4 : pos + 8] == b"IDAT":
                    parts.append(data[pos + 8 : pos + 8 + length])
                    total += length
                    if total > _ZLIB_MAX_STREAM_BYTES:
                        return None
                pos += 12 + length
            return b"".join(parts) if parts else None
        # Bare zlib: CMF/FLG with deflate method and a valid header checksum.
        if n >= 6 and (data[0] & 0x0F) == 8 and ((data[0] << 8) | data[1]) % 31 == 0:
            return data[:_ZLIB_MAX_STREAM_BYTES]
        return None

    # ------------------------------------------------------------------
    # cmplog pair extraction (heuristic)
    # ------------------------------------------------------------------

    def extract_cmplog_pairs(self, input_data: bytes) -> list[tuple[bytes, int]]:
        """Heuristic extraction of (data, checksum) pairs from cmplog.

        Scans ``f._cmplog.pairs`` for 4-byte operands that appear after
        data of a plausible length in the input buffer.
        """
        pairs: list[tuple[bytes, int]] = []
        cmplog = getattr(self.f, "_cmplog", None)
        if not cmplog or not cmplog.pairs:
            return pairs

        for op_a, op_b in cmplog.pairs:
            len_a = len(op_a)
            len_b = len(op_b)
            if len_a == 4 and len_b == 4:
                # The checksum operand appears AFTER the data operand in
                # the input buffer; the earlier one is the "data".
                pos_a = input_data.find(op_a)
                pos_b = input_data.find(op_b)
                if pos_a >= 0 and pos_b >= 0 and pos_a != pos_b:
                    data_op = op_a if pos_a < pos_b else op_b
                    checksum_op = op_b if pos_a < pos_b else op_a
                    pairs.append((data_op, int.from_bytes(checksum_op, "big")))
        return pairs

    # ------------------------------------------------------------------
    # Recovery
    # ------------------------------------------------------------------

    def _recover(self) -> None:
        """Attempt polynomial recovery from collected pairs.

        A candidate polynomial is only accepted when it reproduces the
        observed checksums for at least two distinct pairs — this rejects
        the GCD residue that independent pairs with a mismatched
        ``(init, final_xor)`` configuration produce (e.g. standard PNG
        CRCs, which use ``final_xor=0xFFFFFFFF``), preventing a garbage
        polynomial from ever being activated.
        """
        if not self._pairs:
            return

        self._pairs_attempted_at = self._total_pairs_seen

        # Deduplicate pairs
        unique = list({(d, c) for d, c in self._pairs})

        # recover_polynomial_gcd's cost scales with data bit-length (each
        # pair becomes a Python int shifted by the checksum width, reduced
        # via poly_gcd/_poly_mod). CHECKSUM_PAIRS_MAX bounds pair *count*,
        # but a single large-data pair (a PNG IDAT chunk can be several KB)
        # is still expensive on its own, and GCD-of-syndromes doesn't need
        # every pair -- 2 independent pairs are enough in principle, more
        # just improves confidence. Cap which pairs feed the GCD path to
        # ones with a modest data span; recover_lfsr (checksums only, no
        # data-length dependency) still sees the full set.
        gcd_pairs = [p for p in unique if len(p[0]) <= _GCD_MAX_PAIR_DATA_BYTES]

        # GCD-of-syndromes first: works for independent (data, checksum)
        # pairs (different files/chunks).  Recovers in the non-reflected
        # domain (normal-form polynomial).
        poly = recover_polynomial_gcd(gcd_pairs, width=self._poly_width)
        if poly and poly != 0 and self._verify(poly, unique, reflect=False):
            self._reflect = False
            self._set_model(poly)
            return

        # Reflected GCD-of-syndromes: same technique, but for the
        # reflected (LSB-first) shift convention used by virtually every
        # real-world CRC-32 — zlib, gzip, PNG, ZIP, Ethernet. This is the
        # common case in practice, so it's tried before the BM fallback,
        # which needs sequential register states independent pairs rarely
        # provide.
        poly = recover_polynomial_gcd(gcd_pairs, width=self._poly_width, reflected=True)
        if poly and poly != 0 and self._verify(poly, unique, reflect=True):
            self._reflect = True
            self._set_model(poly)
            return

        # BM fallback: recovers sequential LFSR output streams (reflected
        # domain, reversed-form polynomial).  Requires the checksum values
        # to be one-step-apart register states, which realistic corpora
        # rarely provide — hence the strict verification below.
        checksums = sorted({c for _, c in unique})
        poly = recover_lfsr(checksums, width=self._poly_width)
        if poly and poly != 0 and self._verify(poly, unique, reflect=True):
            self._reflect = True
            self._set_model(poly)
            return

        # Integer-modulus path (Adler-32, Fletcher, weighted sums). Attempted
        # only after every GF(2) path has failed to verify: the two families
        # are disjoint, so a verified GF(2) model is definitive and there is
        # no reason to spend the integer search on top of it. Unlike the GCD
        # path above, this one is fed the FULL pair set -- large pairs are
        # what carry the modulus, not what makes it expensive.
        model = recover_int_model(unique)
        if model is not None and self._verify_int(model, unique):
            self._set_int_model(model)
            return

        # XOR-bitmask path (XOR-of-selected-bits). Attempted only after both
        # GF(2) and integer paths fail: the three families are disjoint, so
        # a verified model from either earlier path is definitive.
        #
        # Not cost-gated any more. recover_xor_model solves by elimination
        # over F2, not SAT, so the full 8/16/32 ladder runs in well under a
        # millisecond and the 32-bit rung is no longer skipped. What bounds
        # it now is *evidence*: the recovery abstains unless the pair set
        # makes the system full rank, which needs at least `width`
        # independent pairs — so a 32-bit model only appears once ~32+ have
        # accumulated, and until then this returns None rather than a guess.
        # CHECKSUM_PAIRS_MAX still caps the pair count fed in.
        xor_model = recover_xor_model(unique)
        if xor_model is not None and self._verify_xor(xor_model, unique):
            self._set_xor_model(xor_model)

    def _verify_int(self, model: IntModel, pairs: list[tuple[bytes, int]]) -> bool:
        """True when *model* reproduces the checksum of >= 2 distinct pairs.

        ``recover_int_model`` already verifies internally; this is the same
        gate applied at the learner boundary, so that no path can activate an
        integer model without it — the identical discipline that keeps a GCD
        residue with a mismatched ``(init, final_xor)`` from being activated
        on the GF(2) side.
        """
        return verify_model(model, pairs)

    def _verify(self, poly: int, pairs: list[tuple[bytes, int]], reflect: bool) -> bool:
        """True when *poly* reproduces the checksum of >= 2 distinct pairs."""
        matches = 0
        for data, crc in pairs:
            if reflect:
                got = _compute_checksum(
                    data,
                    poly=poly,
                    width=self._poly_width,
                    init=0,
                    final_xor=0,
                    reflect_in=True,
                    reflect_out=False,
                )
            else:
                got = _compute_checksum(data, poly=poly, width=self._poly_width)
            if got == crc:
                matches += 1
                if matches >= 2:
                    return True
        return False

    def _set_model(self, poly: int) -> None:
        """Cache the recovered polynomial and activate it in the crc32 module."""
        self._poly = poly
        if poly == _STANDARD_POLY:
            return  # the crc32 module already handles the standard poly
        set_active_model(
            poly,
            width=self._poly_width,
            init=0,
            final_xor=0,
            reflect_in=self._reflect,
            reflect_out=False,
        )

    def _set_int_model(self, model: IntModel) -> None:
        """Cache the recovered integer model and activate it module-wide."""
        self._int_model = model
        set_active_int_model(model)

    def _verify_xor(self, model: XorBitmaskModel, pairs: list[tuple[bytes, int]]) -> bool:
        """True when *model* reproduces the checksum of >= 2 distinct pairs."""
        return verify_xor_model(model, pairs)

    def _set_xor_model(self, model: XorBitmaskModel) -> None:
        """Cache the recovered XOR model and activate it module-wide."""
        self._xor_model = model
        set_active_xor_model(model)

    def compute_int_checksum(self, data: bytes) -> int | None:
        """Compute *data*'s checksum under the recovered integer model.

        Returns ``None`` when no integer model has been recovered — there is
        deliberately no fallback to a guessed model, and deliberately no
        fallback to ``crc32`` either: the two families are not
        interchangeable, and silently substituting one for the other is the
        exact silent-corruption failure mode the verification gate exists to
        prevent.
        """
        model = self.ensure_int_model()
        return None if model is None else eval_model(model, data)

    def compute_xor_checksum(self, data: bytes) -> int | None:
        """Compute the XOR-bitmask checksum of *data*.

        Returns ``None`` when no XOR model has been recovered — there is
        deliberately no fallback to another family.
        """
        model = self.ensure_xor_model()
        return None if model is None else compute_xor_checksum(data, model)

    def compute_checksum(self, data: bytes) -> int:
        """Compute a CRC-style checksum for *data* using the recovered polynomial.

        Falls back to the standard ``crc32`` wrapper when no polynomial has
        been recovered yet.

        This stays on the GF(2) family even when an integer model is active.
        Callers such as ``_patch_png_crc``/``_patch_zip_crc`` mean "the
        CRC-32 this format specifies" — PNG chunk CRCs are CRC-32 by
        specification, and the Adler-32 the learner may have recovered lives
        one layer *below* them inside the IDAT zlib stream. Routing an
        integer model through here would make those patchers write an
        Adler-32 into a CRC-32 field and silently corrupt every mutated PNG.
        Use :meth:`compute_int_checksum` for the integer family.
        """
        poly = self.ensure_poly()
        if poly is None:
            return crc32(data)
        return _compute_checksum(
            data,
            poly=poly,
            width=self._poly_width,
            init=0,
            final_xor=0,
            reflect_in=self._reflect,
            reflect_out=False,
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the learner for the run state file.

        The observed pairs are persisted, not just their count. Recovery
        needs ``min_pairs`` (64) of them before it will run at all, so a
        resume that dropped the evidence restarted collection from zero --
        a run repeatedly stopped and resumed could never reach the threshold
        no matter how many pairs it had seen in total.

        Pairs are kept newest-first under a byte budget rather than by count:
        a pair's data half is a whole checksummed region (a PNG IDAT chunk
        can be megabytes), so ``CHECKSUM_PAIRS_MAX`` alone bounds the list
        but not its size, and the state file is written on every save.
        """
        kept: list[tuple[bytes, int]] = []
        budget = CHECKSUM_STATE_BYTES_MAX
        for data, checksum in reversed(self._pairs):
            budget -= len(data) + 8
            if budget < 0:
                break
            kept.append((data, checksum))
        kept.reverse()
        return {
            "poly": self._poly,
            "poly_width": self._poly_width,
            "reflect": self._reflect,
            "pair_count": self.pair_count,
            "pairs": kept,
            "total_pairs_seen": self._total_pairs_seen,
            "pairs_attempted_at": self._pairs_attempted_at,
            "int_model": model_to_dict(self._int_model),
            "xor_model": xor_model_to_dict(self._xor_model),
        }

    @classmethod
    def from_dict(cls, fuzzer: Any, data: dict[str, Any] | None) -> ChecksumLearner:
        learner = cls(fuzzer)
        if data:
            learner._poly = data.get("poly")
            learner._poly_width = data.get("poly_width", 32)
            learner._reflect = data.get("reflect", False)
            # Restore the evidence, not just its count. Anything malformed is
            # dropped pair by pair: a corrupt state file must not take down
            # the run, the same contract model_from_dict already follows.
            restored: list[tuple[bytes, int]] = []
            for item in data.get("pairs") or ():
                try:
                    pair_data, checksum = item
                    if isinstance(pair_data, bytes | bytearray) and isinstance(checksum, int):
                        restored.append((bytes(pair_data), checksum))
                except (TypeError, ValueError):
                    continue
            learner._pairs = restored[-CHECKSUM_PAIRS_MAX:]
            # total_pairs_seen gates the retry throttle, and pairs_attempted_at
            # records where the last recovery attempt ran. Restoring the first
            # without the second would make the next add_pairs() look like a
            # full RECOVERY_RETRY_BATCH had arrived and re-run recovery
            # immediately; restoring neither re-runs it on the 64th new pair.
            seen = data.get("total_pairs_seen")
            learner._total_pairs_seen = seen if isinstance(seen, int) else len(learner._pairs)
            attempted = data.get("pairs_attempted_at")
            learner._pairs_attempted_at = attempted if isinstance(attempted, int) else -1
            # model_from_dict returns None for malformed input rather than
            # raising -- a corrupt state.json must not take down the run.
            learner._int_model = model_from_dict(data.get("int_model"))
            if learner._int_model is not None:
                set_active_int_model(learner._int_model)
            learner._xor_model = xor_model_from_dict(data.get("xor_model"))
            if learner._xor_model is not None:
                set_active_xor_model(learner._xor_model)
            if learner._poly and learner._poly != _STANDARD_POLY:
                # Re-activate the recovered model in the crc32 module.
                set_active_model(
                    learner._poly,
                    width=learner._poly_width,
                    init=0,
                    final_xor=0,
                    reflect_in=learner._reflect,
                    reflect_out=False,
                )
        return learner
