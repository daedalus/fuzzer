"""Registry catalog: list/query ops, analyzers and schedulers (services + CLI)."""

import argparse

import pytest

from fuzzer_tool.cli.commands import cmd_registry
from fuzzer_tool.core.analyzer_registry import REGISTRY as ANALYZERS
from fuzzer_tool.core.operator_registry import REGISTRY as OPS
from fuzzer_tool.services import fuzzer as fuzzer_mod
from fuzzer_tool.services.position_arena import POSITION_STRATEGY_NAMES
from fuzzer_tool.services.registry_catalog import (
    RegistryKind,
    categories,
    entries,
    query,
)
from fuzzer_tool.services.target_arena import TARGET_STRATEGY_NAMES


def _ns(kind, pattern=None, category=None, verbose=False, cats=False):
    return argparse.Namespace(
        kind=kind, pattern=pattern, category=category, verbose=verbose, categories=cats
    )


def test_ops_match_operator_registry():
    names = [e.name for e in entries(RegistryKind.OPS)]
    assert names == OPS.names()
    assert all(e.category == OPS.category_of(e.name) for e in entries(RegistryKind.OPS))


def test_analyzers_match_analyzer_registry():
    assert [e.name for e in entries(RegistryKind.ANALYZERS)] == ANALYZERS.names()


def test_schedulers_cover_every_family():
    got = {(e.category, e.name) for e in entries(RegistryKind.SCHEDULERS)}
    families = {
        "op": fuzzer_mod._OPERATOR_STRATEGY_NAMES,
        "seed": fuzzer_mod._SEED_STRATEGY_NAMES,
        "position": POSITION_STRATEGY_NAMES,
        "target": TARGET_STRATEGY_NAMES,
    }
    want = {(fam, n) for fam, names in families.items() for n in names}
    assert got == want


def test_categories_count_entries():
    counts = categories(RegistryKind.OPS)
    expected = {c: len(ns) for c, ns in OPS.categories().items()}
    assert counts == expected


def test_query_substring_and_glob():
    name = OPS.names()[0]
    assert name in [e.name for e in query(RegistryKind.OPS, name)]
    glob_hits = query(RegistryKind.OPS, name[:2] + "*")
    assert all(e.name.startswith(name[:2]) for e in glob_hits)


def test_query_hyphen_equals_underscore():
    hits = query(RegistryKind.SCHEDULERS, "kl-ducb")
    assert {e.name for e in hits} == {"kl_ducb"}


def test_query_category_filter():
    cat = OPS.category_of(OPS.names()[0])
    hits = query(RegistryKind.OPS, category=cat)
    assert {e.name for e in hits} == OPS.categories()[cat]


def test_falsification_unknown_name_matches_nothing():
    assert query(RegistryKind.OPS, "no_such_operator_zz") == []
    assert query(RegistryKind.ANALYZERS, category="no_such_category") == []


def test_adversarial_glob_metachars_do_not_raise():
    # Unclosed bracket / regex syntax are literal, not errors.
    for pat in ("[", "(.*", "\\", "*[!", "?"):
        query(RegistryKind.OPS, pat)
    assert query(RegistryKind.OPS, "(.*") == []


def test_cli_lists_and_exits_nonzero_on_miss(capsys):
    assert cmd_registry(_ns("analyzers")) == 0
    out = capsys.readouterr().out
    assert all(n in out for n in ANALYZERS.names())

    assert cmd_registry(_ns("ops", pattern="no_such_operator_zz")) == 1


def test_cli_verbose_and_categories(capsys):
    first = ANALYZERS.names()[0]
    assert cmd_registry(_ns("analyzers", pattern=first, verbose=True)) == 0
    assert "phase=" in capsys.readouterr().out

    assert cmd_registry(_ns("schedulers", cats=True)) == 0
    out = capsys.readouterr().out
    assert {"op", "seed", "position", "target"} <= {ln.split()[0] for ln in out.splitlines()}


def test_unknown_kind_rejected():
    with pytest.raises(ValueError):
        RegistryKind("mutatorz")
