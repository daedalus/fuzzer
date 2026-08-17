"""Tests for core/root_cause.py (edit-script ddmin) and
services/root_cause.py (crash reproduction + baseline selection)."""

from unittest.mock import patch

from fuzzer_tool.core.gf2_common import bitmask_from_indices, invert_bitmask_map
from fuzzer_tool.core.root_cause import (
    apply_edit_subset,
    build_edit_script,
    ddmin_edits,
    describe_scrambled_field,
    edit_indices,
    field_value,
    format_root_cause_report,
    invert_scrambled_field,
)
from fuzzer_tool.core.xor_map_solver import XorBitmaskModel
from fuzzer_tool.services.root_cause import root_cause


class TestEditScript:
    def test_identical_inputs_produce_no_edits(self):
        script = build_edit_script(b"hello", b"hello")
        assert edit_indices(script) == []

    def test_reconstructs_target_from_full_subset(self):
        base = b"hello world this is fine"
        crash = b"hello wOrld this iz fine EXTRA"
        script = build_edit_script(base, crash)
        idx = set(edit_indices(script))
        assert apply_edit_subset(base, script, idx) == crash

    def test_empty_subset_reconstructs_base(self):
        base = b"hello world"
        crash = b"HELLO WORLD"
        script = build_edit_script(base, crash)
        assert apply_edit_subset(base, script, set()) == base


class TestDdminEdits:
    def test_isolates_single_causal_replace(self):
        """8 total edits, but only replacing byte 7 ('o'->'O') matters."""
        base = b"hello world this is fine"
        crash = b"hello wOrld this iz fine EXTRA"
        script = build_edit_script(base, crash)
        assert len(edit_indices(script)) > 1

        def interesting(candidate: bytes) -> bool:
            return len(candidate) > 7 and candidate[7] == ord("O")

        minimal_bytes, minimal_idx = ddmin_edits(base, script, interesting, max_stages=200)
        assert len(minimal_idx) == 1
        op, pos, data = script[minimal_idx[0]]
        assert (op, pos, data) == ("replace", 7, b"O")
        assert interesting(minimal_bytes)

    def test_requires_both_edits_together(self):
        """Crash needs the first AND last byte changed simultaneously —
        ddmin must not shrink below either one."""
        base = b"AAAABBBBCCCCDDDD"
        crash = b"XAAABBBBCCCCDDDY"
        script = build_edit_script(base, crash)
        changes = edit_indices(script)
        assert len(changes) == 2

        def interesting(candidate: bytes) -> bool:
            return candidate[:1] == b"X" and candidate[-1:] == b"Y"

        minimal_bytes, minimal_idx = ddmin_edits(base, script, interesting, max_stages=200)
        assert sorted(minimal_idx) == sorted(changes)
        assert minimal_bytes == crash

    def test_no_edits_returns_base_unchanged(self):
        base = b"same"
        script = build_edit_script(base, base)
        minimal_bytes, minimal_idx = ddmin_edits(base, script, lambda c: True, max_stages=50)
        assert minimal_bytes == base
        assert minimal_idx == []

    def test_nothing_interesting_falls_back_to_full_diff(self):
        """If interesting_fn never returns True for any subset, ddmin can't
        reduce and the result is the full edit set (still reconstructs crash
        bytes since that's what the caller already verified separately)."""
        base = b"AAAA"
        crash = b"BBBB"
        script = build_edit_script(base, crash)
        minimal_bytes, minimal_idx = ddmin_edits(base, script, lambda c: False, max_stages=50)
        assert sorted(minimal_idx) == edit_indices(script)

    def test_respects_max_stages_budget(self):
        """A pathological interesting_fn that's always False should not
        cause unbounded target executions."""
        base = bytes(range(50))
        crash = bytes((b + 1) % 256 for b in base)
        script = build_edit_script(base, crash)
        calls = {"n": 0}

        def counting_fn(candidate: bytes) -> bool:
            calls["n"] += 1
            return False

        ddmin_edits(base, script, counting_fn, max_stages=10)
        assert calls["n"] <= 10


class TestFormatReport:
    def test_empty_minimal_set_reports_no_isolation(self):
        base, crash = b"abc", b"abd"
        script = build_edit_script(base, crash)
        report = format_root_cause_report(base, crash, script, [])
        assert "no subset" in report

    def test_report_lists_replace_offsets(self):
        base, crash = b"abc", b"abd"
        script = build_edit_script(base, crash)
        idx = edit_indices(script)
        report = format_root_cause_report(base, crash, script, idx)
        assert "REPLACE" in report
        assert "0x0002" in report


class TestRootCauseService:
    def test_crash_file_missing(self):
        assert root_cause("/bin/true", "/nonexistent/crash.bin", corpus_dir="/tmp") is None

    def test_empty_crash_file(self, tmp_path):
        crash = tmp_path / "crash.bin"
        crash.write_bytes(b"")
        assert root_cause("/bin/true", str(crash), corpus_dir=str(tmp_path)) is None

    def test_crash_not_reproduced(self, tmp_path):
        crash = tmp_path / "crash.bin"
        crash.write_bytes(b"AAAA")
        with patch("fuzzer_tool.adapters.process.run_target_stdin", return_value=(0, "", 1)):
            result = root_cause("/bin/true", str(crash), corpus_dir=str(tmp_path))
        assert result is None

    def test_no_baseline_source_given(self, tmp_path):
        crash = tmp_path / "crash.bin"
        crash.write_bytes(b"AAAA")
        with patch("fuzzer_tool.adapters.process.run_target_stdin", return_value=(-11, "", 1)):
            result = root_cause("/bin/true", str(crash))
        assert result is None

    def test_empty_corpus_dir(self, tmp_path):
        crash = tmp_path / "crash.bin"
        crash.write_bytes(b"AAAA")
        empty_corpus = tmp_path / "corpus"
        empty_corpus.mkdir()
        with patch("fuzzer_tool.adapters.process.run_target_stdin", return_value=(-11, "", 1)):
            result = root_cause("/bin/true", str(crash), corpus_dir=str(empty_corpus))
        assert result is None

    def test_explicit_baseline_that_also_crashes_rejected(self, tmp_path):
        crash = tmp_path / "crash.bin"
        crash.write_bytes(b"AAAA")
        baseline = tmp_path / "baseline.bin"
        baseline.write_bytes(b"BBBB")

        # Everything crashes -> baseline is rejected as not-actually-safe
        with patch("fuzzer_tool.adapters.process.run_target_stdin", return_value=(-11, "", 1)):
            result = root_cause("/bin/true", str(crash), baseline_file=str(baseline))
        assert result is None

    def test_end_to_end_isolates_causal_byte(self, tmp_path):
        """Full pipeline: explicit baseline, real ddmin round-trip through
        a fake target that only crashes when byte 2 is 'X'."""
        crash_input = b"AAXAA"
        baseline_input = b"AAAAA"
        crash = tmp_path / "crash.bin"
        crash.write_bytes(crash_input)
        baseline = tmp_path / "baseline.bin"
        baseline.write_bytes(baseline_input)

        def fake_run(target, data, timeout, env=None):
            if len(data) > 2 and data[2] == ord("X"):
                return (-11, "", 1)
            return (0, "", 1)

        with patch("fuzzer_tool.adapters.process.run_target_stdin", side_effect=fake_run):
            result = root_cause(
                "/bin/true",
                str(crash),
                baseline_file=str(baseline),
                max_stages=100,
            )

        assert result is not None
        assert len(result["minimal_indices"]) == 1
        op, pos, data = result["script"][result["minimal_indices"][0]]
        assert (op, pos, data) == ("replace", 2, b"X")
        assert "REPLACE" in result["report"]

    def test_picks_nearest_corpus_seed_as_baseline(self, tmp_path):
        """Two corpus seeds: pick whichever is textually nearer to the crash,
        confirm it as non-crashing, and diff against it."""
        crash_input = b"hello wOrld"
        crash = tmp_path / "crash.bin"
        crash.write_bytes(crash_input)

        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()
        near = corpus_dir / "near.bin"
        near.write_bytes(b"hello world")
        far = corpus_dir / "far.bin"
        far.write_bytes(b"completely unrelated text of a different shape entirely")

        def fake_run(target, data, timeout, env=None):
            if b"O" in data:
                return (-11, "", 1)
            return (0, "", 1)

        with patch("fuzzer_tool.adapters.process.run_target_stdin", side_effect=fake_run):
            result = root_cause(
                "/bin/true",
                str(crash),
                corpus_dir=str(corpus_dir),
                max_stages=100,
            )

        assert result is not None
        assert result["baseline_name"] == "near.bin"
        assert len(result["minimal_indices"]) == 1


def _random_square_model(n_bits: int, rng) -> XorBitmaskModel:
    """Build a random invertible n_bits x n_bits square XorBitmaskModel by
    composing row swaps/XOR-additions onto the identity (elementary ops,
    always full rank -- no rejection loop needed)."""
    from fuzzer_tool.core.gf2_common import indices_from_bitmask

    rows = [1 << i for i in range(n_bits)]
    for _ in range(n_bits * 4):
        a, b = rng.randrange(n_bits), rng.randrange(n_bits)
        if a != b:
            rows[a] ^= rows[b]
    return XorBitmaskModel(masks=tuple(indices_from_bitmask(r) for r in rows), out_bits=n_bits)


class TestFieldValue:
    def test_little_endian_matches_xor_map_solver_convention(self):
        # byte 0 supplies input bits 0..7, byte 1 supplies bits 8..15, LSB-first
        data = bytes([0b00000001, 0b00000010])
        assert field_value(data, 0, 2) == 0x0201

    def test_offset(self):
        data = b"\x00\x00\xff\xff"
        assert field_value(data, 2, 2) == 0xFFFF


class TestInvertScrambledField:
    def test_square_model_round_trips(self):
        import random

        rng = random.Random(21)
        model = _random_square_model(8, rng)
        forward = [bitmask_from_indices(m) for m in model.masks]
        for v in range(0, 256, 17):
            checksum = 0
            for j, bits in enumerate(forward):
                if (bits & v).bit_count() & 1:
                    checksum |= 1 << j
            recovered = invert_scrambled_field(model, checksum)
            assert recovered == v

    def test_non_square_model_returns_none(self):
        # out_bits (8) narrower than the referenced input domain (up to 31)
        model = XorBitmaskModel(
            masks=tuple((i, i + 8, i + 16, i + 24) for i in range(8)), out_bits=8
        )
        assert invert_scrambled_field(model, 0x42) is None

    def test_singular_square_model_returns_none(self):
        model = XorBitmaskModel(masks=((0,), (0,)), out_bits=2)
        assert invert_scrambled_field(model, 0b01) is None


class TestDescribeScrambledField:
    def test_reports_original_domain_diff(self):
        import random

        rng = random.Random(4)
        model = _random_square_model(8, rng)
        forward = [bitmask_from_indices(m) for m in model.masks]
        inv = invert_bitmask_map(forward, 8)
        assert inv is not None

        def scramble(v: int) -> int:
            r = 0
            for j, bits in enumerate(forward):
                if (bits & v).bit_count() & 1:
                    r |= 1 << j
            return r

        base_orig, minimal_orig = 0x12, 0x99
        base = bytes([scramble(base_orig), 0xAA])
        minimal = bytes([scramble(minimal_orig), 0xAA])

        desc = describe_scrambled_field(base, minimal, 0, model)
        assert desc is not None
        assert f"{base_orig:#04x}" in desc
        assert f"{minimal_orig:#04x}" in desc

    def test_returns_none_when_field_outside_range(self):
        model = XorBitmaskModel(masks=((0,), (1,)), out_bits=2)
        assert describe_scrambled_field(b"\x00", b"\x00", 5, model) is None

    def test_returns_none_when_not_invertible(self):
        model = XorBitmaskModel(masks=((0,), (0,)), out_bits=2)
        assert describe_scrambled_field(b"\x01", b"\x02", 0, model) is None


class TestFormatReportWithScrambledField:
    def test_includes_field_descrambling_when_provided(self):
        import random

        rng = random.Random(8)
        model = _random_square_model(8, rng)
        forward = [bitmask_from_indices(m) for m in model.masks]

        def scramble(v: int) -> int:
            r = 0
            for j, bits in enumerate(forward):
                if (bits & v).bit_count() & 1:
                    r |= 1 << j
            return r

        base = bytes([scramble(0x10), ord("x")])
        crash = bytes([scramble(0x10), ord("y")])
        script = build_edit_script(base, crash)
        minimal_indices = edit_indices(script)
        minimal_bytes = apply_edit_subset(base, script, set(minimal_indices))

        report = format_root_cause_report(
            base,
            crash,
            script,
            minimal_indices,
            scrambled_field=(0, model),
            minimal_bytes=minimal_bytes,
        )
        assert "scrambled field" in report

    def test_omits_field_section_when_not_applicable(self):
        base, crash = b"ab", b"aB"
        script = build_edit_script(base, crash)
        minimal_indices = edit_indices(script)
        minimal_bytes = apply_edit_subset(base, script, set(minimal_indices))
        singular_model = XorBitmaskModel(masks=((0,), (0,)), out_bits=2)

        report = format_root_cause_report(
            base,
            crash,
            script,
            minimal_indices,
            scrambled_field=(0, singular_model),
            minimal_bytes=minimal_bytes,
        )
        assert "scrambled field" not in report

    def test_no_scrambled_field_arg_behaves_as_before(self):
        base, crash = b"ab", b"aB"
        script = build_edit_script(base, crash)
        minimal_indices = edit_indices(script)
        report = format_root_cause_report(base, crash, script, minimal_indices)
        assert "scrambled field" not in report
