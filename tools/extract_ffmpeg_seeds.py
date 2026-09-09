#!/usr/bin/env python3
# ruff: noqa: B023
"""Extract FFmpeg seeds from OSS-Fuzz, FATE suite, and CVE PoCs.

Downloads crash reproductions from OSS-Fuzz (via the upstream
target_dec_fate.list), FATE suite baseline samples, and PoC files
from known CVE repositories. Seeds land in <out>/seeds/ so the
fuzzer's load_corpus can consume them directly.

Usage:
    python tools/extract_ffmpeg_seeds.py [--out DIR] [--source oss-fuzz|fate|cve|all]
    python tools/extract_ffmpeg_seeds.py --source oss-fuzz --max 50   # test with N crashes
    python tools/extract_ffmpeg_seeds.py --source fate --codecs h264,png,jpeg
    python tools/extract_ffmpeg_seeds.py --source all --analyze

Seeds are fetched via urllib (no external deps). A pickle cache at
/tmp/ffmpeg_seeds.pkl avoids re-downloading on repeated runs.
"""

import argparse
import os
import pickle
import re
import sys
import time
import urllib.request
from urllib.parse import urljoin, urlparse

# ---------------------------------------------------------------------------
# Source URLs
# ---------------------------------------------------------------------------
FATE_LIST_URL = "https://raw.githubusercontent.com/FFmpeg/FFmpeg/master/tools/target_dec_fate.list"
FATE_SUITE_BASE = "https://fate-suite.ffmpeg.org"

# Known CVE PoC sources from survey.txt
CVE_POCS = {
    "CVE-2024-7055": {
        "repo": "https://github.com/CookedMelon/ReportCVE",
        "path": "FFmpeg/poc3",
        "format": "PNM",
        "component": "libavcodec/pnmdec.c",
    },
    "CVE-2024-7272": {
        "repo": "https://github.com/CookedMelon/ReportCVE",
        "path": "FFmpeg/poc5",
        "format": "Audio",
        "component": "libswresample/swresample.c",
    },
    "CVE-2025-9951": {
        "repo": "https://github.com/fm0ss/cve-2025-9951-ffmpeg-jp2-poc",
        "path": ".",
        "format": "JPEG2000",
        "component": "libavcodec/jpeg2000dec.c",
    },
    "CVE-2026-8461": {
        "repo": "https://github.com/Y5neKO/CVE-2026-8461-EXP",
        "path": ".",
        "format": "MagicYUV",
        "component": "libavcodec/magicyuv.c",
    },
}

# Cache path
CACHE_PATH = "/tmp/ffmpeg_seeds.pkl"

# Minimum file size to keep (skip tiny/empty downloads)
MIN_SIZE = 8


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
def load_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)
    return {"oss_fuzz": {}, "fate": {}, "cve": {}, "analysis": {}}


def save_cache(cache: dict) -> None:
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------
def download(
    url: str, dest: str, retries: int = 4, backoff: float = 1.0, max_size: int = 4096
) -> bool:
    """Download URL to dest. Returns True on success."""
    req = urllib.request.Request(url)
    req.add_header("Accept", "*/*")
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                # Check Content-Length header if present
                if "Content-Length" in resp.headers and max_size > 0:
                    try:
                        cl = int(resp.headers["Content-Length"])
                        if cl > max_size:
                            return False
                    except ValueError:
                        pass
                data = resp.read()
            if max_size > 0 and len(data) > max_size:
                return False
            if len(data) < MIN_SIZE:
                return False
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as f:
                f.write(data)
            return True
        except Exception as e:
            last_err = e
            sleep = backoff * (2**attempt)
            print(f"  [warn] download attempt {attempt + 1} failed for {url}: {e}", file=sys.stderr)
            time.sleep(sleep)
    print(f"  [warn] download failed for {url}: {last_err}", file=sys.stderr)
    return False


def url_basename(url: str) -> str:
    """Extract safe filename from URL."""
    path = urlparse(url).path
    name = os.path.basename(path)
    return re.sub(r"[^A-Za-z0-9._-]", "_", name) or "seed"


# ---------------------------------------------------------------------------
# OSS-Fuzz: parse target_dec_fate.list and download crash reproductions
# ---------------------------------------------------------------------------
def parse_fate_list(text: str) -> list[dict]:
    """Parse target_dec_fate.list into structured entries.

    Format: <issue_num>/clusterfuzz-testcase-<testcase_id>  target_dec_<codec>_fuzzer
    Lines starting with # are comments.
    """
    entries = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Split on whitespace — fuzzer name is the last token
        parts = line.split()
        if len(parts) < 2:
            continue
        spec = parts[0]
        fuzzer = parts[-1]
        # Match: <issue_num>/clusterfuzz-testcase-<testcase_id>
        m = re.match(r"(\d+)/clusterfuzz-testcase-(\d+)", spec)
        if not m:
            continue
        issue_num, testcase_id = m.groups()
        entries.append(
            {
                "issue_num": int(issue_num),
                "testcase_id": testcase_id,
                "fuzzer": fuzzer,
                "url": f"https://oss-fuzz.com/download?testcase_id={testcase_id}",
            }
        )
    return entries


def fetch_fate_list() -> list[dict]:
    """Download and parse target_dec_fate.list from upstream."""
    print(f"[*] Fetching {FATE_LIST_URL}")
    req = urllib.request.Request(FATE_LIST_URL)
    with urllib.request.urlopen(req, timeout=30) as resp:
        text = resp.read().decode("utf-8")
    entries = parse_fate_list(text)
    print(f"[*] Parsed {len(entries)} entries from target_dec_fate.list")
    return entries


def download_oss_fuzz_seeds(
    out_dir: str,
    entries: list[dict] | None = None,
    max_seeds: int = 0,
    cache: dict | None = None,
    max_size: int = 4096,
) -> int:
    """Download OSS-Fuzz crash reproductions.

    Organizes seeds into <out_dir>/seeds/<fuzzer>/ to match the
    fuzzer's target layout.
    """
    if cache is None:
        cache = {}
    if entries is None:
        entries = fetch_fate_list()

    seeds_dir = os.path.join(out_dir, "seeds")
    os.makedirs(seeds_dir, exist_ok=True)

    saved = 0
    total = len(entries)
    if max_seeds > 0:
        entries = entries[:max_seeds]
        total = len(entries)

    for i, entry in enumerate(entries):
        testcase_id = entry["testcase_id"]
        fuzzer = entry["fuzzer"]
        url = entry["url"]

        # Skip if already cached
        if testcase_id in cache.get("oss_fuzz", {}):
            print(f"  [{i + 1}/{total}] [{fuzzer}] cache hit {testcase_id}")
            saved += 1
            continue

        # Organize by fuzzer type
        fuzzer_dir = os.path.join(seeds_dir, fuzzer)
        os.makedirs(fuzzer_dir, exist_ok=True)
        dest = os.path.join(fuzzer_dir, testcase_id)

        if os.path.exists(dest):
            cache.setdefault("oss_fuzz", {})[testcase_id] = dest
            print(f"  [{i + 1}/{total}] [{fuzzer}] exists {testcase_id}")
            saved += 1
            continue

        print(f"  [{i + 1}/{total}] [{fuzzer}] downloading {testcase_id} ...")
        if download(url, dest, max_size=max_size):
            cache.setdefault("oss_fuzz", {})[testcase_id] = dest
            saved += 1
        else:
            with open(dest, "wb"):
                pass  # touch empty file so we don't retry
            print(f"  [warn] failed {testcase_id}", file=sys.stderr)

    return saved


# ---------------------------------------------------------------------------
# FATE suite: baseline clean samples
# ---------------------------------------------------------------------------
FATE_FORMATS = [
    "mov",
    "mkv",
    "wav",
    "png",
    "jpeg",
    "h264",
    "swf",
    "mp3",
    "aac",
    "flac",
    "ogg",
    "webm",
    "avi",
]


def download_fate_seeds(
    out_dir: str,
    codecs: list[str] | None = None,
    cache: dict | None = None,
    max_size: int = 4096,
) -> int:
    """Download FATE suite baseline samples via HTTP.

    FATE suite layout: http://fate-suite.ffmpeg.org/<format>/<file>
    """
    if cache is None:
        cache = {}
    if codecs is None:
        codecs = FATE_FORMATS

    seeds_dir = os.path.join(out_dir, "seeds")
    os.makedirs(seeds_dir, exist_ok=True)

    saved = 0
    for fmt in codecs:
        fmt_dir = os.path.join(seeds_dir, f"fate_{fmt}")
        os.makedirs(fmt_dir, exist_ok=True)

        # List directory via the FATE suite HTTP index
        index_url = f"{FATE_SUITE_BASE}/{fmt}/"
        print(f"[*] Fetching FATE index: {index_url}")
        try:
            req = urllib.request.Request(index_url)
            with urllib.request.urlopen(req, timeout=30) as resp:
                html = resp.read().decode("utf-8")
        except Exception as e:
            print(f"  [warn] cannot list {fmt}: {e}", file=sys.stderr)
            continue

        # Extract filenames from HTML index
        links = re.findall(r'href=["\']([^"\'>]+)["\']', html)
        for link in links:
            name = os.path.basename(link)
            if not name or name in (".", ".."):
                continue
            # Skip directories and parent links
            if link.endswith("/"):
                continue

            dest = os.path.join(fmt_dir, name)
            if os.path.exists(dest):
                cache.setdefault("fate", {})[name] = dest
                continue

            file_url = urljoin(index_url, link)
            print(f"  downloading fate/{fmt}/{name} ...")
            if download(file_url, dest, max_size=max_size):
                cache.setdefault("fate", {})[name] = dest
                saved += 1
            time.sleep(0.1)  # be polite to the server

    return saved


# ---------------------------------------------------------------------------
# CVE PoCs: fetch PoC files from ReportCVE repo
# ---------------------------------------------------------------------------
def download_cve_pocs(
    out_dir: str,
    cves: dict[str, dict] | None = None,
    cache: dict | None = None,
    max_size: int = 4096,
) -> int:
    """Download CVE PoC files from ReportCVE / known repos."""
    if cache is None:
        cache = {}
    if cves is None:
        cves = CVE_POCS

    seeds_dir = os.path.join(out_dir, "seeds")
    os.makedirs(seeds_dir, exist_ok=True)

    saved = 0
    for cve_id, info in cves.items():
        repo = info["repo"]
        path = info["path"]
        fmt_dir = os.path.join(seeds_dir, f"cve_{cve_id}")
        os.makedirs(fmt_dir, exist_ok=True)

        # Try to fetch README (usually contains PoC bytes as base64)
        readme_url = f"{repo}/main/{path}/README.md"
        readme_dest = os.path.join(fmt_dir, "README.md")
        if not os.path.exists(readme_dest):
            print(f"  [{cve_id}] downloading README ...")
            if download(readme_url, readme_dest, max_size=max_size):
                saved += 1
                cache.setdefault("cve", {})[cve_id] = readme_dest

        # Try to fetch poc files (typically poc*.bin, poc*.jp2, etc.)
        for fname in [
            "poc.bin",
            "poc0.bin",
            "poc1.bin",
            f"{cve_id.lower()}.bin",
            "payload.bin",
            "poc",
        ]:
            poc_url = f"{repo}/main/{path}/{fname}"
            poc_dest = os.path.join(fmt_dir, fname)
            if os.path.exists(poc_dest):
                continue
            print(f"  [{cve_id}] trying {fname} ...")
            if download(poc_url, poc_dest, max_size=max_size):
                saved += 1
                cache.setdefault("cve", {})[f"{cve_id}/{fname}"] = poc_dest
                break

        time.sleep(0.2)

    return saved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract FFmpeg seeds from OSS-Fuzz, FATE suite, and CVE PoCs"
    )
    parser.add_argument(
        "--out",
        default="corpus_ffmpeg_seeds",
        help="Output corpus directory (default: corpus_ffmpeg_seeds)",
    )
    parser.add_argument(
        "--source",
        choices=["oss-fuzz", "fate", "cve", "all"],
        default="all",
        help="Which sources to fetch from (default: all)",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=0,
        help="Max OSS-Fuzz seeds to download (0=all 656)",
    )
    parser.add_argument(
        "--max-size",
        dest="max_size",
        type=int,
        default=4096,
        help="Max download size per file in bytes (0=unlimited, default 4096)",
    )
    parser.add_argument(
        "--codecs",
        default=None,
        help="Comma-separated FATE codecs (default: all)",
    )
    parser.add_argument("--analyze", action="store_true", help="Print analysis of extracted seeds")
    args = parser.parse_args()

    cache = load_cache()
    total_saved = 0
    max_size = args.max_size

    if args.source in ("oss-fuzz", "all"):
        print("=" * 60)
        print("[*] OSS-Fuzz crash reproductions")
        print("=" * 60)
        n = download_oss_fuzz_seeds(args.out, max_seeds=args.max, cache=cache, max_size=max_size)
        total_saved += n
        print(f"[*] OSS-Fuzz: {n} seeds saved")

    if args.source in ("fate", "all"):
        print("=" * 60)
        print("[*] FATE suite baseline samples")
        print("=" * 60)
        codecs = args.codecs.split(",") if args.codecs else None
        n = download_fate_seeds(args.out, codecs=codecs, cache=cache, max_size=max_size)
        total_saved += n
        print(f"[*] FATE: {n} seeds saved")

    if args.source in ("cve", "all"):
        print("=" * 60)
        print("[*] CVE PoC files")
        print("=" * 60)
        n = download_cve_pocs(args.out, cache=cache, max_size=max_size)
        total_saved += n
        print(f"[*] CVE PoCs: {n} seeds saved")

    save_cache(cache)

    # Count total files
    seeds_dir = os.path.join(args.out, "seeds")
    total_files = sum(len(files) for _, _, files in os.walk(seeds_dir))
    print(f"[*] Total corpus: {total_files} files in {seeds_dir}/")

    if args.analyze:
        print("\n" + "=" * 60)
        print("[*] SEED ANALYSIS REPORT")
        print("=" * 60)

        # Analyze OSS-Fuzz seeds
        oss_fuzz_cache = cache.get("oss_fuzz", {})
        fate_cache = cache.get("fate", {})
        cve_cache = cache.get("cve", {})

        print(f"\n1. OSS-Fuzz Seeds: {len(oss_fuzz_cache)} crash reproductions")
        if oss_fuzz_cache:
            print("   Sample entries:")
            for _i, (testcase_id, path) in enumerate(list(oss_fuzz_cache.items())[:5]):
                print(f"     - {testcase_id} -> {path}")

        print(f"\n2. FATE Suite Seeds: {len(fate_cache)} baseline samples")
        if fate_cache:
            print("   Sample entries:")
            for _i, (filename, path) in enumerate(list(fate_cache.items())[:5]):
                print(f"     - {filename} -> {path}")

        print(f"\n3. CVE PoC Seeds: {len(cve_cache)} files")
        if cve_cache:
            print("   Sample entries:")
            for _i, (key, path) in enumerate(list(cve_cache.items())[:5]):
                print(f"     - {key} -> {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
