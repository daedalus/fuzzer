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
        assert _looks_like("isobmff", RANDOM) is True


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
