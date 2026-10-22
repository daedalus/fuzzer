#!/usr/bin/env python3
"""Move flat seed files into the canonical ``seeds/<hh>/id_<hash>`` shards.

The fuzzer writes seeds two-digit-sharded (``seeds/<hh>/id_<hash>``) and
looks them up by hash through ``filesystem.rehydrate_by_hash``, which only
consults ``<root>/<hh>/id_<hash>``. Seeds dumped flat -- ``seeds/maybe_prune/``,
``seeds/pruned_0/`` -- are therefore invisible to that lookup, and pile up in
single directories holding hundreds of thousands of entries.

This script relocates every flat seed into its shard -- the ``maybe_prune/``
and ``pruned_0/`` dumps, and any loose ``seeds/id_<hash>`` left at the top
level -- and records each move in a TSV manifest so it can be reversed.

When the shard already holds that id, the two files are compared: the id is
the content hash, so a byte-identical copy is redundancy and goes to
``seeds/pruned/<hh>/`` -- on disk, out of the live corpus, never deleted. A
copy with different bytes under the same id is a conflict and stays put.

Examples:
    # Report what would happen; nothing is written.
    python reorganize_seeds.py --corpus ~/fuzzing/corpuses/ffmpeg/seeds

    # Apply, then inspect the manifest.
    python reorganize_seeds.py --corpus ~/fuzzing/corpuses/ffmpeg/seeds --apply
    python reorganize_seeds.py --rollback ~/fuzzing/logs/reorganize_seeds.<ts>.tsv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
import time
from pathlib import Path

SEED_NAME = re.compile(r"^id_[0-9a-f]{16}$")
SHARD_NAME = re.compile(r"^[0-9a-f]{2}$")

# Flat directories holding seeds dumped outside the sharded layout.
FLAT_SOURCES = ("maybe_prune", "pruned_0")

DEFAULT_LOG_DIR = Path.home() / "fuzzing" / "logs"

HASH_OFFSET = len("id_")

# One planned or performed step: action, source, destination, success.
Result = tuple[str, Path, Path | None, bool]

MOVE, COLLISION, DUPLICATE, CONFLICT, BADNAME = (
    "move",
    "collision",
    "duplicate",
    "conflict",
    "badname",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(
        description="Move flat seed files into the canonical seeds/<hh>/id_<hash> shards.",
    )
    p.add_argument(
        "--corpus",
        type=Path,
        help="corpus seeds root (the directory holding the <hh>/ shards)",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="perform the moves (default: dry run)",
    )
    p.add_argument(
        "--keep-sources",
        action="store_true",
        help="keep the emptied source directories instead of removing them",
    )
    p.add_argument(
        "--fix-mis-sharded",
        action="store_true",
        help="relocate seeds whose shard dir does not match their hash prefix",
    )
    p.add_argument(
        "--keep-loose",
        action="store_true",
        help="leave loose top-level seeds/id_<hash> files in place",
    )
    p.add_argument(
        "--rollback",
        type=Path,
        metavar="MANIFEST",
        help="undo the moves recorded in MANIFEST",
    )
    p.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help="where the manifest is written (default: %(default)s)",
    )
    args = p.parse_args(argv)
    # --rollback replays a manifest and needs no corpus; every other mode does.
    if args.corpus is None and not args.rollback:
        p.error("--corpus is required unless --rollback is given")
    return args


def iter_files(directory: Path):
    """Yield regular files directly inside *directory*, non-recursively."""
    if not directory.is_dir():
        return
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_file(follow_symlinks=False):
                yield Path(entry.path)


def shard_dir(name: str) -> str:
    """Return the two-hex shard a seed name belongs to."""
    return name[HASH_OFFSET : HASH_OFFSET + 2]


def plan_flat(source: Path, corpus: Path) -> list[tuple[str, Path, Path | None]]:
    """Plan the relocation of the flat seeds in *source* into *corpus* shards."""
    plan: list[tuple[str, Path, Path | None]] = []
    for src in sorted(iter_files(source)):
        if not SEED_NAME.match(src.name):
            plan.append((BADNAME, src, None))
            continue
        dst = corpus / shard_dir(src.name) / src.name
        # os.rename replaces the destination on POSIX, so the check comes first.
        action = COLLISION if dst.exists() else MOVE
        plan.append((action, src, dst))
    return plan


def plan_loose(corpus: Path) -> list[tuple[str, Path, Path | None]]:
    """Plan the relocation of loose top-level ``seeds/id_<hash>`` files."""
    plan: list[tuple[str, Path, Path | None]] = []
    for src in sorted(iter_files(corpus)):
        if not SEED_NAME.match(src.name):
            continue
        dst = corpus / shard_dir(src.name) / src.name
        action = COLLISION if dst.exists() else MOVE
        plan.append((action, src, dst))
    return plan


def plan_mis_sharded(corpus: Path) -> list[tuple[str, Path, Path | None]]:
    """Plan the relocation of shard files whose dir does not match their hash."""
    plan: list[tuple[str, Path, Path | None]] = []
    for entry in sorted(corpus.iterdir()):
        if not entry.is_dir() or not SHARD_NAME.match(entry.name):
            continue
        for src in sorted(iter_files(entry)):
            if not SEED_NAME.match(src.name) or shard_dir(src.name) == entry.name:
                continue
            dst = corpus / shard_dir(src.name) / src.name
            action = COLLISION if dst.exists() else MOVE
            plan.append((action, src, dst))
    return plan


def same_bytes(a: Path, b: Path, chunk: int = 1 << 20) -> bool:
    """Compare two files byte for byte without loading either fully."""
    if a.stat().st_size != b.stat().st_size:
        return False
    with a.open("rb") as fa, b.open("rb") as fb:
        while True:
            ba, bb = fa.read(chunk), fb.read(chunk)
            if ba != bb:
                return False
            if not ba:
                return True


def resolve_collision(src: Path, dst: Path, corpus: Path) -> tuple[str, Path]:
    """Decide what to do when the destination already holds this id.

    The id is the content hash, so a byte-identical copy is pure redundancy:
    the loose one goes to the pruned area, where it stays on disk and out of
    the live corpus. A copy with different bytes under the same id is a
    genuine conflict and stays exactly where it is.
    """
    if not same_bytes(src, dst):
        return CONFLICT, dst
    pruned = corpus / "pruned" / shard_dir(src.name) / src.name
    if pruned.exists():
        return CONFLICT, pruned
    return DUPLICATE, pruned


def relocate(src: Path, dst: Path) -> bool:
    """Move *src* to *dst*, creating the shard dir. Returns success."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.rename(src, dst)
    except OSError:
        # Different filesystem: copy then unlink.
        shutil.move(str(src), str(dst))
    return True


def resolve(
    plan: list[tuple[str, Path, Path | None]], corpus: Path
) -> list[tuple[str, Path, Path]]:
    """Turn planned collisions into duplicate/conflict decisions.

    Read-only: it compares bytes but moves nothing, so the dry run reports
    exactly what an apply would do.
    """
    resolved = []
    for action, src, dst in plan:
        if action == COLLISION:
            action, dst = resolve_collision(src, dst, corpus)
        resolved.append((action, src, dst))
    return resolved


def execute(resolved: list[tuple[str, Path, Path]]) -> list[Result]:
    """Run the resolved plan, returning one result row per planned entry."""
    results = []
    for action, src, dst in resolved:
        if action in (CONFLICT, BADNAME):
            results.append((action, src, dst, True))
            continue
        try:
            ok = relocate(src, dst)
        except OSError as exc:
            print(f"error: {src} -> {dst}: {exc}", file=sys.stderr)
            ok = False
        results.append((action, src, dst, ok))
    return results


def write_manifest(results, log_dir: Path) -> Path:
    """Write the manifest of planned and performed moves."""
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    path = log_dir / f"reorganize_seeds.{stamp}.tsv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(("action", "src", "dst", "done"))
        for action, src, dst, done in results:
            writer.writerow((action, src, dst or "", int(done)))
    return path


def rollback(manifest: Path) -> int:
    """Move every recorded seed back to where it came from."""
    if not manifest.is_file():
        print(f"error: manifest not found: {manifest}", file=sys.stderr)
        return 2

    restored = skipped = failed = 0
    with manifest.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row["action"] not in (MOVE, DUPLICATE) or row.get("done") != "1":
                continue
            src, dst = Path(row["src"]), Path(row["dst"])
            if not dst.exists():
                skipped += 1
                continue
            if src.exists():
                skipped += 1
                continue
            try:
                relocate(dst, src)
            except OSError as exc:
                print(f"error: {dst} -> {src}: {exc}", file=sys.stderr)
                failed += 1
                continue
            restored += 1

    print(f"restored {restored}, skipped {skipped}, failed {failed}")
    return 1 if failed else 0


def cleanup_sources(corpus: Path, sources: tuple[str, ...]) -> list[str]:
    """Remove the source directories once they are empty."""
    removed = []
    for name in sources:
        directory = corpus / name
        if not directory.is_dir():
            continue
        if any(directory.iterdir()):
            continue
        directory.rmdir()
        removed.append(name)
    return removed


def summarize(results: list[Result], corpus: Path, manifest: Path, applied: bool) -> None:
    """Print the plan or the outcome."""
    counts = {MOVE: 0, DUPLICATE: 0, CONFLICT: 0, BADNAME: 0}
    for action, _, _, _ in results:
        counts[action] = counts.get(action, 0) + 1

    shards = {dst.parent.name for action, _, dst, _ in results if action == MOVE and dst}

    print(f"corpus:   {corpus}")
    print(f"mode:     {'applied' if applied else 'dry run (pass --apply to write)'}")
    print(f"shards:   {len(shards)} touched")
    print()
    print(f"  move       {counts[MOVE]:>7}   into seeds/<hh>/")
    print(f"  duplicate  {counts[DUPLICATE]:>7}   byte-identical, moved to seeds/pruned/<hh>/")
    print(f"  conflict   {counts[CONFLICT]:>7}   destination holds different bytes, left in place")
    print(f"  badname    {counts[BADNAME]:>7}   not an id_<16 hex> name, skipped")
    print(f"  failed     {sum(1 for r in results if not r[3]):>7}")
    print()
    print(f"manifest: {manifest}")


def main(argv: list[str] | None = None) -> int:
    """Re-shard the flat seed files of a corpus."""
    args = parse_args(argv)

    if args.rollback:
        return rollback(args.rollback)

    corpus = args.corpus
    if not corpus.is_dir():
        print(f"error: corpus root not found: {corpus}", file=sys.stderr)
        return 2

    plan: list[tuple[str, Path, Path | None]] = []
    for name in FLAT_SOURCES:
        plan.extend(plan_flat(corpus / name, corpus))
    if not args.keep_loose:
        plan.extend(plan_loose(corpus))
    if args.fix_mis_sharded:
        plan.extend(plan_mis_sharded(corpus))

    resolved = resolve(plan, corpus)

    if args.apply:
        results = execute(resolved)
        removed = [] if args.keep_sources else cleanup_sources(corpus, FLAT_SOURCES)
    else:
        results = [(action, src, dst, True) for action, src, dst in resolved]
        removed = []

    manifest = write_manifest(results, args.log_dir)
    summarize(results, corpus, manifest, args.apply)

    if removed:
        print(f"removed:  {', '.join(removed)}")

    failed = sum(1 for r in results if not r[3])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
