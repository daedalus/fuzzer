"""Regressions from the 2026-11-13 audit: low-severity CLI, mail, grammar, resolver, state items."""

import argparse
import gzip
import struct

import pytest

from fuzzer_tool.cli.commands import cmd_ppmd, cmd_rank
from fuzzer_tool.core import elf
from fuzzer_tool.core.grammar import Grammar
from fuzzer_tool.core.state_store import StateStore
from fuzzer_tool.services.sendmail import MailConfig

_INT_MAX_STR_DIGITS = 4300


@pytest.fixture
def target(tmp_path):
    t = tmp_path / "target"
    t.write_bytes(b"\x7fELF")
    t.chmod(0o755)
    return str(t)


@pytest.fixture
def empty_corpus(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    return str(d)


class TestEmptyCorpusReports:
    """``rank`` / ``ppmd`` on an empty corpus reported the synthetic ``AAAAAAAA`` seed."""

    def test_regression_rank_says_empty(self, target, empty_corpus, capsys):
        args = argparse.Namespace(
            corpus=empty_corpus, top=10, dump=None, format="text", target=target
        )

        assert cmd_rank(args) == 1
        out, err = capsys.readouterr()
        assert "Empty corpus" in err
        assert "AAAAAAAA" not in out

    def test_regression_ppmd_says_empty(self, empty_corpus, capsys):
        args = argparse.Namespace(corpus=empty_corpus, graph=None, top=10)

        assert cmd_ppmd(args) == 1
        assert "No seeds found" in capsys.readouterr().out


class TestMailSubject:
    """``{target.foo}`` raised ``AttributeError``: every crash mail failed."""

    @pytest.mark.parametrize("template", ["{target.foo} x", "{base_name[9]}", "{nope}", "{"])
    def test_regression_bad_template_sent_as_is(self, template):
        cfg = MailConfig(to="a@example.invalid", subject=template)

        assert cfg.resolve_subject(target="/bin/t", base_name="crash_1") == template

    def test_valid_template_is_formatted(self):
        """Falsification: placeholders still expand."""
        cfg = MailConfig(to="a@example.invalid", subject="{target_base}:{base_name}:{returncode}")

        assert (
            cfg.resolve_subject(target="/bin/t", base_name="crash_1", returncode=-11)
            == "t:crash_1:-11"
        )


class TestGrammarRepeatDigits:
    """A repeat count past 4300 digits raised ``ValueError`` out of the grammar loader."""

    _HUGE = "9" * (_INT_MAX_STR_DIGITS + 1)

    @pytest.mark.parametrize("spec", ["b{%s}", "b{1,%s}", "b{%s,%s}"])
    def test_regression_huge_count_is_clamped(self, spec):
        grammar = Grammar()
        grammar.parse("a = " + spec.replace("%s", self._HUGE) + '\nb = "x"\n')

        assert "a" in grammar.rules

    def test_small_count_unchanged(self):
        """Falsification: an ordinary count generates exactly that many."""
        grammar = Grammar()
        grammar.parse('a = b{3}\nb = "x"\n')

        assert grammar.generate("a") == b"xxx"


class TestCollectWords:
    """``_collect_words`` looped on the declared size, not the bytes present."""

    def test_regression_declared_size_past_the_section(self):
        words: set[int] = set()
        section = struct.pack("<QQ", 0x1122334455667788, 0x99AABBCCDDEEFF00)

        elf._collect_words(section, len(section) * 1000, words)

        assert 0x1122334455667788 in words

    def test_honest_size_unchanged(self):
        """Falsification: the same words come out when the size is truthful."""
        honest: set[int] = set()
        lying: set[int] = set()
        section = struct.pack("<QQ", 0x1122334455667788, 0x99AABBCCDDEEFF00)

        elf._collect_words(section, len(section), honest)
        elf._collect_words(section, len(section) * 1000, lying)

        assert honest == lying


class TestStateAllocationBomb:
    """An allowlisted ``bytearray(n)`` in a tampered state file raised ``MemoryError`` out of ``load``."""

    # GLOBAL builtins.bytearray, MARK, LONG 2**62, TUPLE, REDUCE, STOP
    _BOMB = b"\x80\x04cbuiltins\nbytearray\n(L4611686018427387904L\ntR."

    def test_regression_bomb_is_ignored(self, tmp_path):
        (tmp_path / "state.pkl.gz").write_bytes(gzip.compress(self._BOMB))

        assert StateStore(tmp_path).load() == {}

    def test_good_state_round_trips(self, tmp_path):
        """Falsification: a real state file still loads."""
        store = StateStore(tmp_path)
        store.set("corpus", {"exec_count": 5})
        store.save()

        assert StateStore(tmp_path).load()["corpus"] == {"exec_count": 5}
