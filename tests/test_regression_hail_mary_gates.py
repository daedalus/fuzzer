"""Regression: every opt-in argparse gate on the fuzz subcommand must be
wired into the --hail-mary force-enable list.

--hail-mary is the kitchen-sink switch that turns on every optional
scheduler, mutation strategy, and diagnostic that is off by default. When a
new ``store_true`` / ``BooleanOptionalAction`` (default False/None) flag is
added to the fuzz parser but forgotten in ``_HAIL_MARY_FLAGS``, the flag is
silently left off under --hail-mary and the "everything on" contract is
broken.

This test derives the set of opt-in gates from the *shipped* source of
``cli/commands.py`` (not a hand-maintained mirror) and asserts membership.
Deliberate exclusions documented next to ``_HAIL_MARY_FLAGS`` are allow-listed.
An end-to-end check runs the real parser with ``--hail-mary`` and verifies
every listed dest is force-enabled.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from fuzzer_tool.cli import commands

_COMMANDS_PATH = Path(commands.__file__).resolve()

# Documented exclusions next to _HAIL_MARY_FLAGS in commands.py:
#   * the meta flag itself
#   * --resume (fails without prior state)
#   * --refresh-profile / --profile-hotpath (hail-mary already slow; profiling
#     makes exploratory run untrackable)
# Special-cased inside _apply_hail_mary (not plain bool dests in the tuple):
#   * elo (string value "all")
#   * cmplog (tri-state None/True/False)
#   * anneal_budget (int)
# Default-on BooleanOptionalAction features are not additive opt-ins.
_EXCLUDED_OPT_IN = frozenset(
    {
        "hail_mary",
        "resume",
        "cmplog",  # special-cased
        "refresh_profile",  # profiling: hail-mary already slow, full re-analysis per run is overkill
        "profile_hotpath",  # cProfile overhead makes exploratory run untrackable
    }
)


def _fuzz_parser_section() -> str:
    text = _COMMANDS_PATH.read_text(encoding="utf-8")
    start = text.find('fuzz_parser = subparsers.add_parser("fuzz"')
    assert start >= 0, "fuzz_parser construction not found"
    # Next sibling subparser after the fuzz block.
    end_markers = (
        "\n    tmin_parser = subparsers.add_parser",
        "\n    estimate_parser = subparsers.add_parser",
        "\n    analyze_parser = subparsers.add_parser",
    )
    end = len(text)
    for marker in end_markers:
        pos = text.find(marker, start)
        if pos >= 0:
            end = min(end, pos)
    return text[start:end]


def _opt_in_bool_dests_from_source() -> set[str]:
    """Parse fuzz_parser.add_argument blocks for store_true / BooleanOptional
    flags whose default is off (False / None / omitted for store_true).
    """
    section = _fuzz_parser_section()
    dests: set[str] = set()
    i = 0
    needle = "fuzz_parser.add_argument("
    while True:
        j = section.find(needle, i)
        if j < 0:
            break
        k = j + len(needle)
        depth = 1
        while k < len(section) and depth:
            if section[k] == "(":
                depth += 1
            elif section[k] == ")":
                depth -= 1
            k += 1
        block = section[j:k]
        i = k
        if "store_true" not in block and "BooleanOptionalAction" not in block:
            continue
        dest_m = re.search(r'dest\s*=\s*["\'](\w+)["\']', block)
        if dest_m:
            dest = dest_m.group(1)
        else:
            longs = re.findall(r'["\']--([a-z0-9-]+)["\']', block)
            pos = [name for name in longs if not name.startswith("no-")]
            if not pos:
                continue
            dest = pos[0].replace("-", "_")
        default_m = re.search(r"default\s*=\s*([^,\n)]+)", block)
        if default_m:
            default_raw = default_m.group(1).strip()
        elif "store_true" in block:
            default_raw = "False"
        else:
            default_raw = "None"
        # Skip default-on features (coverage, resize_map_on_stall, ...).
        if default_raw in ("True", "true"):
            continue
        dests.add(dest)
    return dests


class TestHailMaryWiresEveryOptInGate:
    """Every additive fuzz gate must appear in _HAIL_MARY_FLAGS (or be special-cased)."""

    def test_hail_mary_flags_cover_all_opt_in_bool_actions(self):
        opt_in = _opt_in_bool_dests_from_source()
        required = opt_in - _EXCLUDED_OPT_IN
        hail = set(commands._HAIL_MARY_FLAGS)
        missing = sorted(required - hail)
        assert not missing, (
            "opt-in fuzz argparse gates not listed in _HAIL_MARY_FLAGS "
            f"(add them or document an exclusion next to the tuple): {missing}"
        )

    def test_hail_mary_flags_exist_on_parser(self):
        """Every entry in the tuple must still be a real argparse dest (no drift)."""
        section = _fuzz_parser_section()
        all_bool: set[str] = set()
        i = 0
        needle = "fuzz_parser.add_argument("
        while True:
            j = section.find(needle, i)
            if j < 0:
                break
            k = j + len(needle)
            depth = 1
            while k < len(section) and depth:
                if section[k] == "(":
                    depth += 1
                elif section[k] == ")":
                    depth -= 1
                k += 1
            block = section[j:k]
            i = k
            if "store_true" not in block and "BooleanOptionalAction" not in block:
                continue
            dest_m = re.search(r'dest\s*=\s*["\'](\w+)["\']', block)
            if dest_m:
                dest = dest_m.group(1)
            else:
                longs = re.findall(r'["\']--([a-z0-9-]+)["\']', block)
                pos = [name for name in longs if not name.startswith("no-")]
                if not pos:
                    continue
                dest = pos[0].replace("-", "_")
            all_bool.add(dest)
        unknown = sorted(set(commands._HAIL_MARY_FLAGS) - all_bool)
        assert not unknown, (
            f"_HAIL_MARY_FLAGS entries that are not boolean destinations on fuzz_parser: {unknown}"
        )

    def test_apply_hail_mary_sets_every_listed_flag(self, monkeypatch):
        """End-to-end: --hail-mary alone force-enables every listed dest."""
        captured: dict[str, object] = {}

        def _spy(args):
            captured["args"] = args
            return 0

        monkeypatch.setattr(commands, "cmd_fuzz", _spy)
        monkeypatch.setattr(sys, "argv", ["fuzzer-tool", "fuzz", "/bin/true", "--hail-mary"])
        rc = commands.main()
        assert rc == 0
        assert "args" in captured
        args = captured["args"]
        for dest in commands._HAIL_MARY_FLAGS:
            assert getattr(args, dest) is True, f"--hail-mary left {dest!r} off"
        assert args.elo == "all"
        assert args.cmplog is True
        assert args.anneal_budget == 10000
