"""tools/find_dup_tests.py: identical-body test detection."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import find_dup_tests as fdt  # noqa: E402


def _write(root: Path, name: str, src: str) -> Path:
    path = root / name
    path.write_text(textwrap.dedent(src))
    return path


def _names(groups) -> list[set[str]]:
    return [{h.name for h in g} for g in groups]


def test_docstring_and_comment_do_not_split_a_group(tmp_path):
    """Falsification: only the docstring/comments differ -> one group."""
    _write(
        tmp_path,
        "test_a.py",
        '''
        def test_one():
            """Doc A."""
            assert 1 + 1 == 2  # comment

        class TestX:
            def test_two(self):
                """Doc B."""
                assert 1 + 1 == 2
        ''',
    )

    groups = fdt.find_dups([tmp_path])

    assert _names(groups) == [{"test_one", "test_two"}]


def test_different_literal_is_not_a_duplicate(tmp_path):
    """Falsification: one changed constant must split the pair."""
    _write(tmp_path, "test_a.py", "def test_one():\n    assert f(1)\n")
    _write(tmp_path, "test_b.py", "def test_one():\n    assert f(2)\n")

    assert fdt.find_dups([tmp_path]) == []


def test_params_and_decorators_are_part_of_identity(tmp_path):
    """Same body, different fixtures/parametrize -> different tests."""
    _write(
        tmp_path,
        "test_a.py",
        """
        import pytest

        def test_one(tmp_path):
            assert g()

        @pytest.mark.slow
        def test_two():
            assert g()

        def test_three():
            assert g()
        """,
    )

    assert fdt.find_dups([tmp_path]) == []


def test_adversarial_stubs_helpers_and_bad_files(tmp_path):
    """Adversarial: pass-only stubs, non-test helpers and an unparsable file
    produce no groups and no crash."""
    _write(
        tmp_path,
        "test_a.py",
        """
        def test_stub_a():
            '''Stub.'''
            pass

        def test_stub_b():
            pass

        def helper_a():
            return 1

        def helper_b():
            return 1
        """,
    )
    _write(tmp_path, "test_broken.py", "def test_x(:\n")
    _write(tmp_path, "conftest.py", "def test_c():\n    assert h()\n")

    assert fdt.find_dups([tmp_path]) == []


def test_hits_carry_location_and_largest_group_first(tmp_path):
    _write(
        tmp_path,
        "test_a.py",
        """
        def test_small_a():
            assert a()

        def test_small_b():
            assert a()

        def test_big_a():
            x = b()
            assert x

        def test_big_b():
            x = b()
            assert x
        """,
    )

    groups = fdt.find_dups([tmp_path])

    assert _names(groups) == [{"test_big_a", "test_big_b"}, {"test_small_a", "test_small_b"}]
    first = groups[0][0]
    assert first.path.name == "test_a.py"
    assert first.line == 8
    assert first.size == 2
