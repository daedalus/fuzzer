#!/usr/bin/env python3
"""Group FATE / seed-corpus samples onto per-fuzzer directories (OSS-Fuzz port).

Port of google/oss-fuzz ``projects/ffmpeg/group_seed_corpus.py``.

OSS-Fuzz used this to build a dedicated ``*_seed_corpus.zip`` per
``ffmpeg_*_fuzzer`` binary by matching short tags derived from the
fuzzer name against sample path fragments.  It is disabled upstream
because the combined zips exceeded ClusterFuzz size limits; here we
keep the matching logic but write *loose files* under
``<out>/seeds/<fuzzer>/`` so Daedalus ``load_corpus`` can consume them
directly (optional ``--zip`` still emits the classic zip layout).

Typical workflow with the existing FATE downloader::

    # 1. Fetch baseline samples (already in extract_ffmpeg_seeds.py)
    python tools/extract_ffmpeg_seeds.py --source fate --out ~/fate-corpus

    # 2. Group them onto per-codec / per-demuxer seed dirs
    python tools/group_seed_corpus.py \\
        --corpus ~/fate-corpus/seeds \\
        --fuzzers-from-names ffmpeg_AV_CODEC_ID_H264_fuzzer,ffmpeg_DEMUXER_fuzzer \\
        --out ~/fate-corpus

Or point ``--corpus`` at a full local FATE suite (rsync copy)::

    rsync -a rsync://fate-suite.ffmpeg.org/fate-suite/ ~/fate-suite/
    python tools/group_seed_corpus.py --corpus ~/fate-suite --out ~/fate-grouped \\
        --fuzzers-file fuzzers.txt

``fuzzers.txt`` is one fuzzer basename per line (e.g.
``ffmpeg_AV_CODEC_ID_HEVC_fuzzer``).
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="INFO: %(message)s")
log = logging.getLogger("group_seed_corpus")

CODEC_NAME_REGEXP = re.compile(r"codec_id_(.+?)_fuzzer", re.IGNORECASE)

# Skip these when walking a corpus tree (FATE reference dumps, checksums).
_SKIP_NAME_SUBSTR = ("md5sum",)
_SKIP_SUFFIXES = (".s16", ".dec", ".pcm", ".md5")


def get_fuzzer_tags(fuzzer_name: str) -> list[str]:
    """Derive short path-matching tags from a fuzzer binary name.

    Mirrors OSS-Fuzz ``group_seed_corpus.get_fuzzer_tags``:
    - subtitle targets get an explicit ``sub`` tag
    - codec id is split on ``_`` and stripped of common suffixes
      (video/audio/subtitle/text); long tokens are truncated to 3 chars
    """
    tags: list[str] = []
    name = fuzzer_name.lower()
    if "subtitle" in name:
        tags.append("sub")

    m = CODEC_NAME_REGEXP.search(name)
    if m:
        codec_name = m.group(1)
        for part in codec_name.split("_"):
            codec = part
            for noise in ("video", "audio", "subtitle", "text"):
                codec = codec.split(noise)[0]
            if not codec:
                continue
            # Trailing-char codecs (VP6F, FLV1, JPEGLS): keep first 3 chars
            # when the token is long enough.
            tags.append(codec[:3] if len(codec) > 3 else codec)
        return tags

    # Demuxer / BSF / other: use the significant token after the prefix.
    # e.g. ffmpeg_dem_mov_fuzzer -> mov, ffmpeg_BSF_H264_MP4TOANNEXB_fuzzer
    # -> h26 / mp4 / ..., ffmpeg_DEMUXER_fuzzer -> dem (weak, still useful)
    for prefix in (
        "ffmpeg_dem_",
        "ffmpeg_bsf_",
        "ffmpeg_av_codec_id_",
        "ffmpeg_",
    ):
        if name.startswith(prefix):
            rest = name[len(prefix) :]
            rest = rest.removesuffix("_fuzzer").removesuffix("_dec").removesuffix("_enc")
            for part in rest.split("_"):
                if part and part not in ("id", "av", "codec"):
                    tags.append(part[:3] if len(part) > 3 else part)
            break

    return tags


def parse_corpus(corpus_directory: Path) -> list[Path]:
    """Recursively list sample files; skip checksums and raw FATE refs."""
    files: list[Path] = []
    for root, _dirs, names in os.walk(corpus_directory):
        for filename in names:
            lower = filename.lower()
            if any(s in lower for s in _SKIP_NAME_SUBSTR):
                continue
            if any(lower.endswith(suf) for suf in _SKIP_SUFFIXES):
                continue
            files.append(Path(root) / filename)
    log.info("Parsed %d corpus files from %s", len(files), corpus_directory)
    return files


def parse_fuzzers_from_dir(fuzzers_directory: Path) -> list[str]:
    """List ``ffmpeg_*_fuzzer`` basenames present in a directory."""
    names: list[str] = []
    if not fuzzers_directory.is_dir():
        return names
    for entry in fuzzers_directory.iterdir():
        name = entry.name
        if name.startswith("ffmpeg_") and name.endswith("_fuzzer"):
            names.append(name)
        # Also accept already-grouped seed subdirs named like the fuzzer.
        elif entry.is_dir() and name.startswith("ffmpeg_") and "fuzzer" in name:
            names.append(name)
    log.info("Parsed %d fuzzers from %s", len(names), fuzzers_directory)
    return names


def parse_fuzzers_arg(names: str | None, path: Path | None, discover: Path | None) -> list[str]:
    """Resolve the fuzzer-name list from CLI flags."""
    result: list[str] = []
    if names:
        result.extend(n.strip() for n in names.split(",") if n.strip())
    if path is not None:
        text = path.read_text()
        result.extend(line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#"))
    if discover is not None:
        result.extend(parse_fuzzers_from_dir(discover))
    # de-dupe, preserve order
    seen: set[str] = set()
    ordered: list[str] = []
    for n in result:
        if n not in seen:
            seen.add(n)
            ordered.append(n)
    return ordered


def relevant_files(corpus_files: list[Path], fuzzer_tags: list[str]) -> list[Path]:
    """Select samples whose path contains any of the fuzzer's tags.

    Falls back to stripping the last character of each tag when the
    strict match yields nothing (OSS-Fuzz behaviour for RV40→RV, PCX→PC).
    """
    if not fuzzer_tags:
        return []

    hits: set[Path] = set()
    for path in corpus_files:
        # Drop the literal "ffmpeg" substring so the MPEG tag does not
        # match every path under a fate-suite tree rooted at ffmpeg/.
        sanitized = str(path).replace("ffmpeg", "").lower()
        for tag in fuzzer_tags:
            if tag in sanitized:
                hits.add(path)

    if not hits:
        for path in corpus_files:
            sanitized = str(path).replace("ffmpeg", "").lower()
            for tag in fuzzer_tags:
                if len(tag) > 1 and tag[:-1] in sanitized:
                    hits.add(path)

    return sorted(hits)


def write_grouped(
    corpus_files: list[Path],
    fuzzers: list[str],
    out_dir: Path,
    *,
    do_zip: bool = False,
    symlink: bool = False,
    max_per_fuzzer: int = 0,
) -> dict[str, int]:
    """Copy (or symlink) matching samples into ``out/seeds/<fuzzer>/``.

    Returns a map of fuzzer name → number of samples written.
    """
    seeds_root = out_dir / "seeds"
    seeds_root.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    for fuzzer in fuzzers:
        tags = get_fuzzer_tags(fuzzer)
        matched = relevant_files(corpus_files, tags)
        if max_per_fuzzer > 0:
            matched = matched[:max_per_fuzzer]

        log.info(
            "Found %d relevant samples for %s (tags=%s)",
            len(matched),
            fuzzer,
            tags,
        )
        if not matched:
            counts[fuzzer] = 0
            continue

        dest_dir = seeds_root / fuzzer
        dest_dir.mkdir(parents=True, exist_ok=True)

        written = 0
        for src in matched:
            # Preserve a stable, collision-resistant name: relative path
            # with separators flattened when the source tree is deep.
            try:
                rel = src.resolve().relative_to(Path("/"))
            except ValueError:
                rel = Path(src.name)
            dest_name = str(rel).replace(os.sep, "__")
            dest = dest_dir / dest_name
            if dest.exists():
                written += 1
                continue
            if symlink:
                try:
                    dest.symlink_to(src.resolve())
                except OSError:
                    shutil.copy2(src, dest)
            else:
                shutil.copy2(src, dest)
            written += 1

        counts[fuzzer] = written

        if do_zip and written:
            zip_path = out_dir / f"{fuzzer}_seed_corpus.zip"
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for src in matched:
                    zf.write(src, arcname=src.name)
            log.info("Wrote %s (%d entries)", zip_path, written)

    return counts


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help="Directory of seed samples (FATE suite tree or extract_ffmpeg_seeds output)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output root; writes <out>/seeds/<fuzzer>/…",
    )
    ap.add_argument(
        "--fuzzers-from-names",
        default=None,
        help="Comma-separated fuzzer basenames",
    )
    ap.add_argument(
        "--fuzzers-file",
        type=Path,
        default=None,
        help="Text file with one fuzzer basename per line",
    )
    ap.add_argument(
        "--fuzzers-dir",
        type=Path,
        default=None,
        help="Discover ffmpeg_*_fuzzer names from this directory",
    )
    ap.add_argument(
        "--zip",
        action="store_true",
        help="Also emit <fuzzer>_seed_corpus.zip next to seeds/ (OSS-Fuzz layout)",
    )
    ap.add_argument(
        "--symlink",
        action="store_true",
        help="Symlink samples instead of copying (falls back to copy on error)",
    )
    ap.add_argument(
        "--max-per-fuzzer",
        type=int,
        default=0,
        help="Cap samples per fuzzer (0 = unlimited)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print tag matches; do not write files",
    )
    args = ap.parse_args()

    if not args.corpus.is_dir():
        print(f"[!] corpus directory not found: {args.corpus}", file=sys.stderr)
        return 1

    fuzzers = parse_fuzzers_arg(args.fuzzers_from_names, args.fuzzers_file, args.fuzzers_dir)
    if not fuzzers:
        print(
            "[!] No fuzzers specified. Use --fuzzers-from-names, --fuzzers-file, "
            "or --fuzzers-dir.",
            file=sys.stderr,
        )
        return 1

    corpus_files = parse_corpus(args.corpus)
    if not corpus_files:
        print(f"[!] No sample files under {args.corpus}", file=sys.stderr)
        return 1

    if args.dry_run:
        for fuzzer in fuzzers:
            tags = get_fuzzer_tags(fuzzer)
            matched = relevant_files(corpus_files, tags)
            if args.max_per_fuzzer > 0:
                matched = matched[: args.max_per_fuzzer]
            print(f"{fuzzer}: tags={tags} matches={len(matched)}")
            for p in matched[:10]:
                print(f"    {p}")
            if len(matched) > 10:
                print(f"    … +{len(matched) - 10} more")
        return 0

    counts = write_grouped(
        corpus_files,
        fuzzers,
        args.out,
        do_zip=args.zip,
        symlink=args.symlink,
        max_per_fuzzer=args.max_per_fuzzer,
    )

    total = sum(counts.values())
    nonempty = sum(1 for n in counts.values() if n > 0)
    print(f"[*] Grouped {total} sample placements across {nonempty}/{len(fuzzers)} fuzzers")
    print(f"[*] Output: {args.out / 'seeds'}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
