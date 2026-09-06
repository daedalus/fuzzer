"""Smoke tests for FormatFuzzer MutatorBase integration.

These tests do **not** require a real FormatFuzzer binary. They verify:

* registration into REGISTRY
* availability gating on the feature flag
* graceful decline when the binary is missing
* MutationContext carries formatfuzzer_enabled
"""

from __future__ import annotations

import pytest

from fuzzer_tool.core.mutations.formatfuzzer import (
    FormatFuzzerMutator,
    _looks_like,
    register_formatfuzzer_mutators,
    report_availability,
)
from fuzzer_tool.core.mutator_interface import MutationContext
from fuzzer_tool.core.operator_registry import REGISTRY

PNG_MAGIC = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
ZIP_MAGIC = b"PK\x03\x04" + b"\x00" * 16
JPEG_MAGIC = b"\xff\xd8\xff" + b"\x00" * 16
RANDOM = b"\x00\x01\x02\x03" + b"\xff" * 20


class TestLooksLike:
    def test_png(self):
        assert _looks_like("png", PNG_MAGIC) is True
        assert _looks_like("png", RANDOM) is False

    def test_zip(self):
        assert _looks_like("zip", ZIP_MAGIC) is True
        assert _looks_like("zip", PNG_MAGIC) is False

    def test_unknown_template_accepts_all(self):
        assert _looks_like("nosuchformat", RANDOM) is True

    def test_alias_resolves_to_upstream_magic(self):
        # jpeg is not an upstream template name; jpg is.
        assert _looks_like("jpeg", JPEG_MAGIC) is True
        assert _looks_like("jpeg", RANDOM) is False


class TestMutatorAvailability:
    def test_disabled_without_flag(self):
        m = FormatFuzzerMutator(template="png")
        ctx = MutationContext(formatfuzzer_enabled=False)
        assert m.is_available(ctx, PNG_MAGIC) is False

    def test_enabled_flag_but_no_binary(self):
        m = FormatFuzzerMutator(template="png", bin_dir="/nonexistent")
        ctx = MutationContext(formatfuzzer_enabled=True)
        # Binary missing → unavailable even with flag
        assert m.is_available(ctx, PNG_MAGIC) is False

    def test_mutate_returns_none_without_binary(self):
        m = FormatFuzzerMutator(template="png", bin_dir="/nonexistent")
        rng = __import__("random").Random(0)
        assert m.mutate(PNG_MAGIC, rng, max_len=4096) is None


@pytest.fixture
def registry_restored():
    """Undo registrations made during a test.

    ``REGISTRY`` is process-global. Registering into it without removing the
    entry afterwards leaks the operator into every later test in the session,
    and the ones that assert an exact inventory then fail a long way from the
    cause -- ``test_regression_operator_registry`` compares the live registry
    against the import-time ``OPERATOR_CATEGORIES`` snapshot, which cannot
    contain a name registered after it was taken. This mirrors the
    register/restore pattern in ``test_regression_scheduler_operator_reach``.
    """
    before = set(REGISTRY.names())
    yield
    for leaked in set(REGISTRY.names()) - before:
        REGISTRY._ops.pop(leaked, None)
    REGISTRY._categories_cache = None


class TestRegistration:
    def test_register_creates_named_operators(self, tmp_path, registry_restored):
        # Use a unique template name so we don't clash with import-time registration
        name = "ff_testfmt"
        # Ensure clean slate for this name
        if name in REGISTRY.names():
            pytest.skip("name already present")

        muts = register_formatfuzzer_mutators(
            templates=["testfmt"],
            bin_dir=tmp_path,  # empty dir → no binary
        )
        assert len(muts) == 1
        assert muts[0].name == name
        assert name in REGISTRY.names()
        assert REGISTRY.category_of(name) == "format"

    def test_default_templates_registered_on_import(self):
        # Import-time registration should have added the defaults
        names = set(REGISTRY.names())
        # At least one of the default templates should be present
        assert any(n.startswith("ff_") for n in names)


class TestMutationContext:
    def test_default_false(self):
        ctx = MutationContext()
        assert ctx.formatfuzzer_enabled is False

    def test_explicit_true(self):
        ctx = MutationContext(formatfuzzer_enabled=True)
        assert ctx.formatfuzzer_enabled is True

    def test_from_fuzzer_reads_attribute(self):
        class Fake:
            max_len = 4096
            formatfuzzer = True

        ctx = MutationContext.from_fuzzer(Fake())
        assert ctx.formatfuzzer_enabled is True


# ---------------------------------------------------------------------------
# Upstream CLI contract (github.com/uds-se/FormatFuzzer)
# ---------------------------------------------------------------------------

FAKE_FF = r"""#!/usr/bin/env python3
import sys, os, hashlib
argv = sys.argv[1:]
if not argv:
    sys.exit(1)
cmd, argv = argv[0], argv[1:]
dec = None
if argv[:1] == ["--decisions"]:
    dec, argv = argv[1], argv[2:]
if cmd == "parse":
    if not argv:
        sys.exit(1)
    data = open(argv[0], "rb").read()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        sys.exit(1)
    if dec:
        open(dec, "wb").write(bytes(b % 4 for b in data[:64]) or b"\x00")
    sys.exit(0)
if cmd == "fuzz":
    body = open(dec, "rb").read() if dec and os.path.exists(dec) else b"\x00" * 32
    tag = hashlib.sha256(body).digest()[:8]
    for out in argv:
        open(out, "wb").write(b"\x89PNG\r\n\x1a\n" + tag + body)
    sys.exit(0)
sys.stderr.write("Unknown command " + cmd + "\n")
sys.exit(1)
"""


@pytest.fixture
def ff_bin(tmp_path):
    """A stand-in honouring the real CLI: command first, output to a file."""
    d = tmp_path / "ffbin"
    d.mkdir()
    exe = d / "png-fuzzer"      # upstream spells it with a HYPHEN
    exe.write_text(FAKE_FF)
    exe.chmod(0o755)
    return d


class TestBinaryDiscovery:
    def test_hyphenated_upstream_name_is_found(self, ff_bin):
        m = FormatFuzzerMutator(template="png", bin_dir=ff_bin)
        assert m._bin is not None
        assert m._bin.name == "png-fuzzer"

    def test_bare_format_name_is_not_searched_on_path(self, monkeypatch, tmp_path):
        """PATH lookup must not pick up /usr/bin/zip and drive it as a generator."""

        calls = []

        def fake_which(name):
            calls.append(name)
            return None

        monkeypatch.setattr(
            "fuzzer_tool.core.mutations.formatfuzzer.shutil.which", fake_which
        )
        FormatFuzzerMutator(template="zip", bin_dir=tmp_path)
        assert "zip" not in calls, f"bare format name searched on PATH: {calls}"
        assert "zip-fuzzer" in calls

    def test_alias_looks_for_the_upstream_binary(self, tmp_path):
        m = FormatFuzzerMutator(template="isobmff", bin_dir=tmp_path)
        assert m.name == "ff_isobmff"       # registry inventory unchanged
        assert m.format == "mp4"            # but mp4-fuzzer is what exists


class TestProbe:
    def test_non_generator_is_rejected(self, tmp_path):
        d = tmp_path / "decoy"
        d.mkdir()
        exe = d / "gif-fuzzer"
        exe.write_text("#!/bin/sh\nexit 0\n")
        exe.chmod(0o755)
        m = FormatFuzzerMutator(template="gif", bin_dir=d)
        assert m._bin is not None, "resolved by name"
        ctx = MutationContext(formatfuzzer_enabled=True)
        assert m.is_available(ctx, b"GIF89a" + b"\x00" * 20) is False

    def test_probe_is_cached(self, ff_bin):
        m = FormatFuzzerMutator(template="png", bin_dir=ff_bin)
        ctx = MutationContext(formatfuzzer_enabled=True)
        assert m.is_available(ctx, PNG_MAGIC) is True
        assert m._probed is True
        # Removing the binary must not re-probe.
        (ff_bin / "png-fuzzer").unlink()
        assert m.is_available(ctx, PNG_MAGIC) is True


class TestMutateAgainstRealContract:
    def test_produces_a_structurally_valid_neighbour(self, ff_bin):
        import random

        m = FormatFuzzerMutator(template="png", bin_dir=ff_bin)
        rng = random.Random(11)
        data = b"\x89PNG\r\n\x1a\n" + bytes(range(64))
        out = m.mutate(data, rng, max_len=4096)
        assert out is not None, "the placeholder CLI read stdout and always declined"
        assert out.startswith(b"\x89PNG\r\n\x1a\n")

    def test_mutants_vary(self, ff_bin):
        import random

        m = FormatFuzzerMutator(template="png", bin_dir=ff_bin)
        rng = random.Random(3)
        data = b"\x89PNG\r\n\x1a\n" + bytes(range(64))
        outs = {m.mutate(data, rng, 4096) for _ in range(8)}
        assert len(outs) > 1

    def test_max_len_is_honoured(self, ff_bin):
        import random

        m = FormatFuzzerMutator(template="png", bin_dir=ff_bin)
        out = m.mutate(b"\x89PNG\r\n\x1a\n" + bytes(range(64)), random.Random(2), 16)
        assert out is not None and len(out) == 16

    def test_unparseable_input_falls_back_to_generation(self, ff_bin):
        import random

        m = FormatFuzzerMutator(template="png", bin_dir=ff_bin)
        out = m.mutate(b"not a png at all", random.Random(5), 4096)
        assert out is not None
        assert out.startswith(b"\x89PNG\r\n\x1a\n")

    def test_no_temp_files_are_left_behind(self, ff_bin, tmp_path, monkeypatch):
        import random
        import tempfile

        scratch = tmp_path / "scratch"
        scratch.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(scratch))
        m = FormatFuzzerMutator(template="png", bin_dir=ff_bin)
        for _ in range(5):
            m.mutate(b"\x89PNG\r\n\x1a\n" + bytes(range(32)), random.Random(1), 4096)
        assert list(scratch.iterdir()) == []


class TestBinDirRebind:
    def test_ff_bin_dir_rebinds_an_already_registered_operator(
        self, ff_bin, registry_restored
    ):
        """--ff-bin-dir was a no-op for every default template.

        The module self-registers the defaults at import time with the
        default directory; the later call skipped existing names and
        returned the stale instances, so the flag never took effect.
        """
        # A fresh name, so the shared ff_png instance is not left rebound to
        # a tmp_path that this test is about to delete.
        (ff_bin / "rebindfmt-fuzzer").write_text((ff_bin / "png-fuzzer").read_text())
        (ff_bin / "rebindfmt-fuzzer").chmod(0o755)

        first = register_formatfuzzer_mutators(
            templates=["rebindfmt"], bin_dir="/nonexistent"
        )
        assert first[0]._bin_available is False

        second = register_formatfuzzer_mutators(templates=["rebindfmt"], bin_dir=ff_bin)
        assert second[0] is first[0], "same registered instance"
        assert second[0]._bin_available is True
        assert second[0]._bin.name == "rebindfmt-fuzzer"

    def test_rebind_resets_the_probe(self, ff_bin, registry_restored):
        m = FormatFuzzerMutator(template="png", bin_dir=ff_bin)
        ctx = MutationContext(formatfuzzer_enabled=True)
        assert m.is_available(ctx, PNG_MAGIC) is True
        m.rebind("/nonexistent")
        assert m._probed is None
        assert m.is_available(ctx, PNG_MAGIC) is False


class TestAvailabilityReport:
    def test_warns_when_nothing_is_installed(self, caplog, tmp_path):
        import logging

        muts = [FormatFuzzerMutator(template="png", bin_dir=tmp_path)]
        with caplog.at_level(logging.WARNING):
            assert report_availability(muts) == 0
        assert any("no binary found" in r.message for r in caplog.records)

    def test_silent_warning_free_when_all_ready(self, caplog, ff_bin):
        import logging

        muts = [FormatFuzzerMutator(template="png", bin_dir=ff_bin)]
        with caplog.at_level(logging.WARNING):
            assert report_availability(muts) == 1
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
