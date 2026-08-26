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

Those skips were a blind spot, not a safe default: an operator that is only
ever *reachable* with runtime state is also only ever *checkable* with it, and
``crc_learn`` was a pure no-op there for exactly that reason -- its
availability gate is ``ChecksumLearner.ensure_model()``, true for any of three
checksum families, while the handler consumed only two of them. A target whose
checksum recovered as XOR-bitmask offered ``crc_learn`` on every selection and
got nothing back.

``TestStateGatedOperatorsAreNotNoOps`` below closes that hole: it builds the
state each gate wants and sweeps the operators that only appear once it exists.
Its ``test_every_selectable_operator_is_reachable`` is the guard-the-guard --
adding a new state-gated operator without teaching ``_build_gated_state`` how
to reach it fails there rather than silently reopening the blind spot.
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
    sh_strtab = struct.pack("<IIQQQQIIQQ", 1, 3, 0, 0, shoff + shentsize * shnum, 0x11, 0, 0, 1, 0)
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


def _mpegts() -> bytes:
    """Two synced 188-byte MPEG-TS packets: a PAT on PID 0, then one packet
    carrying an adaptation field, so mpegts_chunk_mutate's sniffer (sync byte
    at both offset 0 and 188) and its adaptation-field mutation both have
    something to act on."""
    pat_payload = bytes([0x00, 0x00, 0xB0, 0x0D, 0x00, 0x01, 0xC1, 0x00, 0x00])
    pat_payload += bytes([0x00, 0x01, 0xE1, 0x00]) + bytes(4)
    pat_header = bytes([0x47, 0x40, 0x00, 0x10])
    pat = pat_header + pat_payload + b"\xff" * (188 - len(pat_header) - len(pat_payload))

    af = bytes([0x01, 0x00])  # adaptation_field_length=1, no flags set
    af_header = bytes([0x47, 0x01, 0x01, 0x30])  # PID 0x0101, adaptation+payload
    af_pkt = af_header + af + b"\xff" * (188 - len(af_header) - len(af))
    return pat + af_pkt


def _h264_annexb() -> bytes:
    """Two Annex-B NAL units (SPS then PPS start codes)."""
    return (
        b"\x00\x00\x00\x01\x67\x42\xc0\x1e"
        + bytes(24)
        + b"\x00\x00\x00\x01\x68\xce\x3c\x80"
        + bytes(16)
    )


def _ogg() -> bytes:
    """One minimal Ogg BOS page: "OggS" header, no payload beyond a single
    lacing byte, kept short so it doesn't skew corpus_invariants' shared-
    byte accounting the way a full page with random payload would."""
    header = (
        b"OggS"
        + bytes([0, 0x02])  # version, header_type=BOS
        + bytes(8)  # granule_position
        + bytes(4)  # serial_number
        + bytes(4)  # sequence_number
        + bytes(4)  # checksum
        + bytes([1])  # page_segments
        + bytes([4])  # segment_table: one 4-byte segment
    )
    return header + bytes(4)


def _flv() -> bytes:
    """Minimal FLV: 9-byte header, one small audio tag, trailing size."""
    header = b"FLV" + bytes([1, 0x04]) + (9).to_bytes(4, "big")
    data = bytes(8)
    tag = bytes([8]) + len(data).to_bytes(3, "big") + bytes(4) + bytes(3) + data
    return header + bytes(4) + tag + (11 + len(data)).to_bytes(4, "big")


def _asf() -> bytes:
    """Minimal ASF: empty Header Object followed by a small Data Object,
    kept short for the same corpus_invariants reason as `_ogg`."""
    header_obj_guid = bytes.fromhex("3026B2758E66CF11A6D900AA0062CE6C")
    data_obj_guid = bytes.fromhex("3626B2758E66CF11A6D900AA0062CE6C")
    header_obj = header_obj_guid + (24).to_bytes(8, "little")
    data_body = bytes(8)
    data_obj = data_obj_guid + (24 + len(data_body)).to_bytes(8, "little") + data_body
    return header_obj + data_obj


def _battery() -> list[bytes]:
    """Fixed input battery: random inputs of several lengths (deterministic via
    a local seed) plus magic-prefixed samples so format-aware operators become
    available and get exercised."""
    rng = random.Random(999)
    inputs = [
        bytes(rng.randrange(256) for _ in range(n)) for n in (8, 32, 128, 512) for _ in range(4)
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
        _mpegts(),
        _ogg(),
        _flv(),
        _asf(),
        # zlib stream: covers zlib_chunk_mutate and recompress_zlib, whose
        # sniffers check the CMF/FLG header rather than a magic string.
        zlib.compress(b"the quick brown fox jumps over the lazy dog" * 3, 6),
    ]
    return inputs


def _make_fuzzer() -> Fuzzer:
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


def _sweep(f: Fuzzer) -> tuple[set[str], set[str]]:
    """Return (available_somewhere, changed_somewhere) operator-name sets.

    Exceptions raised by operators are collected on ``_sweep.last_raised``
    rather than discarded, so a "no-op" that is really a crash can be told
    apart from one that is really silence.

    Each operator is exercised on its own with the fuzzer RNG reseeded per
    repetition, so the result is deterministic and independent of operator
    ordering or of any other operator's RNG use.
    """
    table = REGISTRY.dispatch(f._operators)
    battery = _battery()
    names = sorted({n for inp in battery for n in REGISTRY.available(f, inp)})
    available: set[str] = set(names)
    changed: set[str] = set()
    raised: dict[str, list[BaseException]] = {}
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
                except Exception as exc:
                    # An operator raising is a separate concern from a no-op,
                    # so it is still skipped here -- but it is recorded, and
                    # test_no_operator_raises_on_every_input below turns
                    # "raises on everything" into its own failure.
                    #
                    # Swallowing silently is what hid the bytearray
                    # TypeError in extract_corpus_literals and _build_verse:
                    # both operators raised on every single input and were
                    # reported as pure no-ops, which sent two rounds of
                    # investigation looking for a missing state gate that
                    # did not exist.
                    raised.setdefault(name, []).append(exc)
                    continue
                out = ret if isinstance(ret, bytes | bytearray) else buf
                if bytes(out) != inp:
                    changed.add(name)
                    done = True
                    break
            if done:
                break
    _sweep.last_raised = raised  # type: ignore[attr-defined]
    return available, changed


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

    def test_every_available_operator_changes_some_input(self):
        f = _make_fuzzer()
        available, changed = _sweep(f)

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
            f = _make_fuzzer()
            available, changed = _sweep(f)
            assert "byte_shuffle" in available
            assert "byte_shuffle" in (available - changed), (
                "a planted no-op byte_shuffle went undetected — the sweep "
                "does not actually protect against no-op operators"
            )
        finally:
            _mutations.byte_shuffle = original


class TestStateGatedOperatorsAreNotNoOps:
    """The same no-op sweep, over the operators that need runtime state.

    ``TestNoMutationIsAPureNoOp`` deliberately leaves that state unbuilt so
    the gated operators report unavailable and are skipped. That kept the
    sweep clean at the cost of never checking 25 of 124 operators -- and
    ``crc_learn`` was a pure no-op in that gap. Here the state is built.

    Every registered operator must now be both reachable and non-inert.
    ``colorization`` used to be the one exception, gated on
    ``operator_registry._never``; it is gated on cmplog pairs instead, so the
    exception list is empty and the reachability test covers all 124.
    """

    #: Never returned by ``REGISTRY.available``. Empty now that colorization
    #: is gated on cmplog pairs rather than on ``_never``; kept as the seam
    #: for any future operator that is dispatchable but not selectable.
    _NEVER_SELECTABLE: set[str] = set()

    def _build_gated_state(self, f: Fuzzer) -> set[str]:
        """Populate the runtime state the gated predicates ask for.

        Returns the names of operators intentionally left unreachable in
        this environment (currently only the z3-dependent ones when the
        optional ``smt`` extra is absent).
        """
        unreachable: set[str] = set()

        # --- dictionary band (8 operators) -------------------------------
        # The gate only checks that a dictionary exists, but six of the
        # eight also consume a per-seed index scratch the mutate loop
        # refills. Dispatching directly bypasses that refill, so without it
        # they decline every call and measure as no-ops that are really
        # harness artifacts.
        f.dictionary = [b"IHDR", b"IDAT", b"<script>", b"%n", b"\xff\xff", b"AAAA"]
        f._dict_scratch = f._rand_pool.randint_list(0, len(f.dictionary) - 1, 4096)
        f._dict_scratch_idx = 0

        # --- flag-gated band ----------------------------------------------
        f.enable_regex_bomb = True
        f.enable_x86_mutator = True
        f.enable_arm_mutator = True

        # --- cmplog band (6 operators + path_negate) -----------------------
        from fuzzer_tool.core.cmplog import CmplogCollector

        pool = CmplogCollector()
        pool.pairs = [
            (b"IHDR", b"IDAT"),
            (b"\x89PNG", b"\x89png"),
            (b"ftyp", b"isom"),
            (b"RIFF", b"WEBP"),
        ]
        # Real shape is dict[(a, b)] -> (result, width); a list of tuples
        # makes branch_records() raise on every call, which measures this
        # fixture rather than the operator.
        pool._pair_cmp = {
            (b"IHDR", b"IDAT"): (-1, 4),
            (b"ftyp", b"isom"): (1, 4),
        }
        pool._pair_pc = {(b"IHDR", b"IDAT"): 0x401000, (b"ftyp", b"isom"): 0x401020}
        f._cmplog = pool

        # --- redqueen ------------------------------------------------------
        # Entries are (offset, operand_a, operand_b), and the offset must
        # point at a real occurrence of operand_a or the operator's own
        # guard declines.
        f.seed_meta = {}
        for inp in _battery():
            matches = [
                (inp.find(a), a, b)
                for a, b in ((b"IHDR", b"IDAT"), (b"RIFF", b"RIFX"), (b"\x7fELF", b"\x7felf"))
                if inp.find(a) != -1
            ]
            f.seed_meta[inp] = {"redqueen_matches": matches, "redqueen_offsets": [0, 4]}

        # --- grammar band --------------------------------------------------
        from fuzzer_tool.core.grammar import Grammar

        grammar = Grammar()
        grammar.parse(
            "start  = header body footer\n"
            'header = "GET " path " HTTP/1.1\\r\\n"\n'
            'path   = "/" | "/index.html" | "/a/b/c"\n'
            'body   = "Host: example.com\\r\\n" | "X-Test: 1\\r\\n"\n'
            'footer = "\\r\\n"\n'
        )
        f.grammar = grammar

        # --- CEM -----------------------------------------------------------
        from fuzzer_tool.core.schedulers.monte_carlo import MonteCarloScheduler

        mc = MonteCarloScheduler()
        for i, seed in enumerate([bytes(c) for c in f.corpus] + _battery()[:12]):
            mc.add_elite(seed, score=100 - i)
        mc.maybe_refit()
        f.mc = mc
        f.mc_cem = True
        f.markov_trained = True

        # --- invariant_break -----------------------------------------------
        # corpus_invariants() needs >= 16 samples before it will call an
        # offset invariant, but size alone is not enough: the mask is
        # ``&= ~(first ^ current)`` across entries, so a corpus with no shared
        # structure yields an all-zero mask and the operator has nothing to
        # break. `_battery()` is deliberately diverse and produces exactly
        # that (33 samples, zero invariant bytes), which reads as a no-op but
        # is a property of the corpus. A real fuzzing corpus concentrates on
        # one format, so add entries that share a header.
        shared_header = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        f.corpus = [bytearray(inp) for inp in _battery()]
        f.corpus += [
            bytearray(shared_header + bytes((i * 37 + j) & 0xFF for j in range(48)))
            for i in range(20)
        ]
        from fuzzer_tool.core.randomness import corpus_invariants

        assert any(corpus_invariants([bytes(c) for c in f.corpus]).mask), (
            "fixture corpus has no invariant bits, so invariant_break would "
            "measure as a no-op for a reason that is not about the operator"
        )

        # --- checksum learner (crc_learn) ------------------------------------
        # An XOR-bitmask model with no GF(2) and no integer model is the exact
        # state that made crc_learn a pure no-op: ensure_model() is true, so
        # the operator is offered on every selection, while the handler had no
        # branch for this family and returned the buffer untouched.
        #
        # The model is installed directly rather than recovered: recovery goes
        # through the z3 XOR-map solver, and this must run without the
        # optional smt extra. Installing it directly is a real reachable
        # state, and the same one ChecksumLearner.load() restores.
        from fuzzer_tool.core.checksum_learner import ChecksumLearner
        from fuzzer_tool.core.xor_map_solver import XorBitmaskModel

        learner = ChecksumLearner(f)
        learner._set_xor_model(
            XorBitmaskModel(
                masks=(
                    (1, 2),
                    (6, 7),
                    (3, 5, 7),
                    (1, 3),
                    (3, 6),
                    (0, 2, 3, 5),
                    (0, 2, 3, 4),
                    (0, 7),
                ),
                out_bits=8,
            )
        )
        assert learner.ensure_poly() is None
        assert learner.ensure_int_model() is None
        assert learner.ensure_model(), "fixture failed to reach the XOR-only model state"
        f.checksum_learner = learner

        # --- path_negate ----------------------------------------------------
        # Needs recorded outcomes *and* an enabled solver. Without z3 the
        # fuzzer leaves _path_solver as None and the operator is genuinely
        # unreachable, so report it rather than flagging a false no-op.
        f._path_solver = None
        try:
            from fuzzer_tool.core.path_constraints import PathConstraintSolver, _z3

            if _z3() is not None:
                f._path_solver = PathConstraintSolver()
        except Exception:  # pragma: no cover - environment-dependent
            pass
        if f._path_solver is None:
            unreachable.add("path_negate")

        return unreachable

    def _make_gated_fuzzer(self) -> tuple[Fuzzer, set[str]]:
        f = _make_fuzzer()
        return f, self._build_gated_state(f)

    def teardown_method(self):
        # _set_xor_model also publishes to a module-global active model; leave
        # it as found so later tests are not reading this fixture's model.
        from fuzzer_tool.core.xor_map_solver import clear_active_xor_model

        clear_active_xor_model()

    def test_no_operator_raises_on_every_single_input(self):
        """An operator that throws on every input is broken, not silent.

        This is the test that should have caught the bytearray TypeError in
        `extract_corpus_literals` and `_build_verse`. Both raised on every
        input in the battery; the no-op sweep swallowed the exception and
        reported them as operators that never changed anything, which is a
        true statement about a completely wrong cause.

        Raising on *some* inputs is fine and expected -- plenty of operators
        assume structure a random battery entry will not have. Raising on
        *all* of them means the operator cannot work at all.
        """
        f, unreachable = self._make_gated_fuzzer()
        available, changed = _sweep(f)
        raised = getattr(_sweep, "last_raised", {})

        always_raised = {
            name: excs
            for name, excs in raised.items()
            if name in available and name not in changed and name not in unreachable
        }
        assert not always_raised, (
            "operator(s) raised on every input they were offered: "
            + "; ".join(
                f"{n} -> {type(e[0]).__name__}: {e[0]}" for n, e in sorted(always_raised.items())
            )
        )

    def test_state_gated_operators_change_some_input(self):
        f, unreachable = self._make_gated_fuzzer()
        available, changed = _sweep(f)

        # Building the state must actually widen the sweep, or this whole
        # class is a slower copy of the one above.
        assert len(available) >= 115, f"only {len(available)} operators exercised"

        noops = available - changed - unreachable
        assert not noops, (
            "state-gated operator(s) are available but never changed any "
            f"input (pure no-ops): {', '.join(sorted(noops))}"
        )

    def test_every_selectable_operator_is_reachable(self):
        """No operator may sit outside the sweep without saying why.

        This is the guard against the failure that hid ``crc_learn``: an
        operator that is never offered is never checked, and a growing set
        of them makes the no-op sweep quietly narrower over time. A new
        state-gated operator fails here until ``_build_gated_state`` learns
        to reach it.
        """
        f, unreachable = self._make_gated_fuzzer()
        available, _ = _sweep(f)

        missing = set(REGISTRY.names()) - available - self._NEVER_SELECTABLE - unreachable
        assert not missing, (
            "operator(s) registered but never offered by the sweep, so never "
            f"checked for being a no-op: {', '.join(sorted(missing))}"
        )

    def test_crc_learn_is_not_a_no_op_under_an_xor_only_model(self):
        """Direct regression for the bug this class was added to catch.

        ``ensure_model()`` is true for the XOR-bitmask family, so the gate
        offers ``crc_learn`` on every selection; before the fix the handler
        fell through both its branches and returned the buffer untouched.
        """
        f, _ = self._make_gated_fuzzer()
        table = REGISTRY.dispatch(f._operators)
        battery = _battery()

        offered = changed = 0
        for rep in range(_REPS):
            f._rand_pool.reseed(_BASE_SEED + rep)
            for inp in battery:
                if "crc_learn" not in set(REGISTRY.available(f, inp)):
                    continue
                offered += 1
                buf = bytearray(inp)
                ret = table["crc_learn"](buf, len(buf) // 2, bytes(inp))
                out = ret if isinstance(ret, bytes | bytearray) else buf
                if bytes(out) != inp:
                    changed += 1

        assert offered, "crc_learn was never offered — the XOR-only fixture is wrong"
        assert changed, (
            f"crc_learn offered on {offered} inputs and changed none: it is a "
            "pure no-op whenever the recovered model is XOR-bitmask"
        )
