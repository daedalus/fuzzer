"""docs/AB_LEDGER.md stays well formed (Hard Rule 59)."""

import re
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "docs" / "AB_LEDGER.md"

COLUMNS = 8
VERDICTS = ("WIN", "LOSS", "NULL-BOUNDED", "NULL", "LEAN+", "LEAN-", "EXPLORATORY", "VOID")
NO_CONTROL = "none"
UNCONTROLLED_BAN = ("WIN", "LOSS")
SOURCE_RE = re.compile(r"`(docs/[^`\s]+\.md)`")


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _rows(text: str) -> list[list[str]]:
    """Entry rows: table lines whose first cell is an integer."""
    rows = []
    for line in text.splitlines():
        if not line.startswith("|"):
            continue

        cells = _cells(line)
        if cells[0].isdigit():
            rows.append(cells)

    return rows


def _verdict(cell: str) -> str:
    # Longest match first: NULL-BOUNDED before NULL.
    for v in sorted(VERDICTS, key=len, reverse=True):
        if cell.startswith(v):
            return v

    return ""


def check(text: str, root: Path = ROOT) -> list[str]:
    """Return one error string per violation."""
    errors = []
    rows = _rows(text)
    if not rows:
        return ["no entries"]

    prev = date.min
    for i, cells in enumerate(rows, 1):
        tag = f"row {cells[0]}"
        if len(cells) != COLUMNS:
            errors.append(f"{tag}: {len(cells)} columns")
            continue

        num, day, _arm, _design, result, control, verdict, source = cells
        if int(num) != i:
            errors.append(f"{tag}: expected #{i}")

        when = date.fromisoformat(day)
        if when < prev:
            errors.append(f"{tag}: date before previous row")
        prev = when

        if not result:
            errors.append(f"{tag}: empty result")

        v = _verdict(verdict)
        if not v:
            errors.append(f"{tag}: verdict {verdict!r}")
        if control == NO_CONTROL and v in UNCONTROLLED_BAN:
            errors.append(f"{tag}: {v} without A/A")

        paths = SOURCE_RE.findall(source)
        if not paths:
            errors.append(f"{tag}: no source doc")
        errors += [f"{tag}: missing {p}" for p in paths if not (root / p).exists()]

    return errors


def test_ledger_is_well_formed():
    assert check(LEDGER.read_text()) == []


def test_ab_pending_points_at_ledger():
    assert "AB_LEDGER.md" in (ROOT / "docs" / "AB_PENDING.md").read_text()


# Falsification: each corruption of a valid row must be caught.
_GOOD = (
    "| 1 | 2026-10-01 | a vs b | t | 5/5 | 4/6 | NULL | `docs/AB_LEDGER.md` |\n"
    "| 2 | 2026-10-02 | a vs b | t | 9/1 | 5/5 | WIN | `docs/AB_LEDGER.md` |\n"
)


def test_good_sample_passes():
    assert check(_GOOD) == []


@pytest.mark.parametrize(
    ("old", "new", "needle"),
    [
        ("| 2 | 2026-10-02", "| 3 | 2026-10-02", "expected #2"),
        ("2026-10-02", "2026-09-30", "date before"),
        ("| 9/1 | 5/5 | WIN", "| 9/1 | none | WIN", "WIN without A/A"),
        ("| NULL |", "| MAYBE |", "verdict"),
        ("`docs/AB_LEDGER.md` |\n| 2", "`docs/NOPE.md` |\n| 2", "missing docs/NOPE.md"),
        ("| 5/5 | 4/6 |", "| 5/5 |", "columns"),
    ],
)
def test_corruption_is_caught(old, new, needle):
    bad = _GOOD.replace(old, new, 1)
    assert bad != _GOOD
    assert any(needle in e for e in check(bad))


# Adversarial: prose and header rows must not count as entries.
def test_header_and_prose_ignored():
    text = "| # | Date |\n|---|---|\nsome | pipe text\n" + _GOOD
    assert check(text) == []


def test_empty_ledger_fails():
    assert check("# A/B ledger\n") == ["no entries"]
