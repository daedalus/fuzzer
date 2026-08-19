#!/usr/bin/env python3
# ruff: noqa: B023
"""Generate an FFmpeg corpus from Forgejo issue attachments and linked samples.

Fetches issues from https://code.ffmpeg.org/FFmpeg/FFmpeg/issues via the
Forgejo REST API, scans issue bodies, comments, and inline uploads for
downloadable seeds, and writes them into a local corpus directory.

A persistent pickle cache at /tmp/corpus_ffmpeg_issues.pkl stores per-issue
data keyed by a hash of the issue query, so repeated runs skip issues that
have already been fetched.

Usage:
    python tools/corpus_ffmpeg_issues.py [--out DIR] [--max-issues N] [--token TOKEN] [--skip-size BYTES] [--filetype EXT1,EXT2]

Output is written to <out>/seeds/ (the layout load_corpus reads), suitable
as a fuzzer seed corpus for targets/ffmpeg_read.c.
"""

import argparse
import contextlib
import hashlib
import http
import json
import os
import pickle
import re
import sys
import time
import urllib.request
from urllib.parse import urljoin

CACHE_PATH = "/tmp/corpus_ffmpeg_issues.pkl"

# Media extensions that are useful as FFmpeg fuzz seeds. Anything outside
# this set is skipped to avoid pulling avatars, logs, or patch files.
MEDIA_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".flv",
    ".wmv",
    ".webm",
    ".m4v",
    ".mpg",
    ".mpeg",
    ".ts",
    ".m2ts",
    ".vob",
    ".3gp",
    ".3g2",
    ".ogv",
    ".m4a",
    ".mp3",
    ".aac",
    ".flac",
    ".ogg",
    ".opus",
    ".wav",
    ".wma",
    ".amr",
    ".au",
    ".mka",
    ".y4m",
    ".yuv",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".tiff",
    ".webp",
    ".heic",
    ".avif",
    ".mxf",
    ".nut",
    ".asf",
    ".rm",
    ".ra",
    ".ivf",
    ".dts",
    ".ac3",
    ".eac3",
    ".truehd",
    ".pcm",
    ".srt",
    ".ass",
    ".ssa",
    ".sub",
    ".bin",
    ".dat",
    ".cue",
    ".ifo",
    ".iso",
    ".img",
    ".m3u",
    ".m3u8",
    ".m2v",
    ".264",
    ".hevc",
    ".vc1",
    ".brstm",
    ".fsb",
}


def is_media_url(url: str, allowed_extensions: set[str] | None = None) -> bool:
    """Return True if the URL looks like a downloadable media/test seed."""
    if allowed_extensions is None:
        allowed_extensions = MEDIA_EXTENSIONS
    path = url.split("?", 1)[0].lower()
    return any(path.endswith(ext) for ext in allowed_extensions)


def is_media_attachment(name: str, allowed_extensions: set[str] | None = None) -> bool:
    """Return True if the attachment filename looks like a useful seed."""
    if allowed_extensions is None:
        allowed_extensions = MEDIA_EXTENSIONS
    base = (name or "").lower()
    return any(base.endswith(ext) for ext in allowed_extensions)


def sanitize_filename(name: str) -> str:
    """Make a safe filename from an issue title or URL path."""
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_") or "seed"


def cache_key_for_issue(owner: str, repo: str, number: int) -> str:
    """Return a stable cache key for a single issue query."""
    query = f"/api/v1/repos/{owner}/{repo}/issues/{number}"
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def load_cache() -> dict:
    """Load the persistent issue cache from disk."""
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def save_cache(cache: dict) -> None:
    """Persist the issue cache to disk."""
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)


def api_get(
    base_url: str, path: str, token: str | None = None, retries: int = 5, backoff_base: float = 1.0
) -> list | dict:
    """GET a Forgejo API endpoint and return parsed JSON."""
    url = urljoin(base_url, path)
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/json")
    if token:
        req.add_header("Authorization", f"token {token}")

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last_error = e
            if e.code < 500 and e.code != 429:
                raise
            sleep = backoff_base * (2**attempt)
            print(f"  [warn] HTTP {e.code} for {url}, retrying in {sleep:.1f}s", file=sys.stderr)
            time.sleep(sleep)
        except urllib.error.URLError as e:
            last_error = e
            sleep = backoff_base * (2**attempt)
            print(
                f"  [warn] network error for {url}: {e}, retrying in {sleep:.1f}s", file=sys.stderr
            )
            time.sleep(sleep)
    raise last_error or RuntimeError(f"Failed to GET {url} after {retries} retries")


def fetch_issue_comments(
    base_url: str, owner: str, repo: str, issue_number: int, token: str | None = None
) -> list[dict]:
    """Fetch comments for a single issue, paginated."""
    comments: list[dict] = []
    page = 1
    while True:
        path = (
            f"/api/v1/repos/{owner}/{repo}/issues/{issue_number}/comments?page={page}&per_page=100"
        )
        data = api_get(base_url, path, token)
        if not data:
            break
        comments.extend(data)
        if len(data) < 100:
            break
        page += 1
        time.sleep(0.1)
    return comments


def extract_urls(text: str) -> list[str]:
    """Pull absolute URLs from a blob of text."""
    if not text:
        return []
    return re.findall(r"https?://[^\s)>\]]+", text)


def download_file(url: str, dest_path: str, retries: int = 4, backoff_base: float = 1.0) -> bool:
    """Download a URL to dest_path. Returns True on success."""
    req = urllib.request.Request(url)
    req.add_header("Accept", "*/*")

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            if len(data) < 64:
                return False
            with open(dest_path, "wb") as f:
                f.write(data)
            return True
        except urllib.error.HTTPError as e:
            last_error = e
            if e.code < 500 and e.code != 429:
                return False
            sleep = backoff_base * (2**attempt)
            print(
                f"  [warn] download HTTP {e.code} for {url}, retrying in {sleep:.1f}s",
                file=sys.stderr,
            )
            time.sleep(sleep)
        except urllib.error.URLError as e:
            last_error = e
            sleep = backoff_base * (2**attempt)
            print(
                f"  [warn] download network error for {url}: {e}, retrying in {sleep:.1f}s",
                file=sys.stderr,
            )
            time.sleep(sleep)
        except http.client.IncompleteRead as e:
            last_error = e
            sleep = backoff_base * (2**attempt)
            print(
                f"  [warn] download truncated for {url}: {e}, retrying in {sleep:.1f}s",
                file=sys.stderr,
            )
            time.sleep(sleep)
    print(f"  [warn] download failed for {url}: {last_error}", file=sys.stderr)
    return False


def get_content_length(url: str, timeout: int = 30) -> int | None:
    """Return the Content-Length for a URL, or None if unavailable."""
    req = urllib.request.Request(url)
    req.add_header("Accept", "*/*")
    req.get_method = lambda: "HEAD"  # type: ignore[method-assign]
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return int(resp.headers.get("Content-Length", "0") or 0) or None
    except Exception:
        return None


def collect_seeds_from_issue(
    base_url: str,
    owner: str,
    repo: str,
    issue: dict,
    token: str | None,
    out_dir: str,
    issue_cache: dict,
    skip_size: int = 0,
    allowed_extensions: set[str] | None = None,
) -> int:
    """Scan one issue for downloadable media seeds. Returns count saved."""
    if allowed_extensions is None:
        allowed_extensions = MEDIA_EXTENSIONS
    saved = 0
    number = issue["number"]
    body = issue.get("body", "") or ""

    key = cache_key_for_issue(owner, repo, number)
    if key in issue_cache:
        cached = issue_cache[key]
        comments = cached.get("comments") or []
        print(f"  [{number}] cache hit")
    else:
        comments = fetch_issue_comments(base_url, owner, repo, number, token)
        issue_cache[key] = {
            "issue": issue,
            "comments": comments,
        }

    seen_urls: set[str] = set()

    def _try_download(url: str, fname_hint: str, force_by_name: bool = False) -> None:
        nonlocal saved
        if url in seen_urls:
            return
        seen_urls.add(url)
        if not force_by_name and not is_media_url(url, allowed_extensions):
            return
        if skip_size > 0:
            length = get_content_length(url)
            if length is not None and length > skip_size:
                print(f"  [{number}] skipping {url} ({length} bytes > --skip-size {skip_size})")
                return
        if force_by_name:
            fname = sanitize_filename(fname_hint)
        else:
            fname = sanitize_filename(url.rsplit("/", 1)[-1])
        if not fname:
            fname = sanitize_filename(fname_hint)
        if not force_by_name and not fname.lower().endswith(tuple(allowed_extensions)):
            fname += ".bin"
        dest = os.path.join(out_dir, fname)
        if os.path.exists(dest):
            return
        print(f"  [{number}] downloading {url}")
        if download_file(url, dest):
            saved += 1
        else:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(dest)

    # Inline attachments uploaded to the issue itself.
    for asset in issue.get("assets") or []:
        if is_media_attachment(asset.get("name", ""), allowed_extensions):
            _try_download(
                asset.get("browser_download_url", ""),
                asset.get("name", "asset"),
                force_by_name=True,
            )

    # Links in the issue body.
    for url in extract_urls(body):
        _try_download(url, f"issue_{number}_body_link")

    # Links in comments.
    for comment in comments:
        for url in extract_urls(comment.get("body", "") or ""):
            _try_download(url, f"issue_{number}_comment_{comment.get('id', '')}")

    return saved


def main() -> int:
    parser = argparse.ArgumentParser(description="Download FFmpeg issue seeds from Forgejo")
    parser.add_argument(
        "--out",
        default="corpus_ffmpeg_issues",
        help="Corpus directory (default: corpus_ffmpeg_issues)",
    )
    parser.add_argument("--max-issues", type=int, default=0, help="Max issues to scan (0=all)")
    parser.add_argument(
        "--token", default=None, help="Forgejo access token for authenticated requests"
    )
    parser.add_argument(
        "--skip-size",
        type=int,
        default=0,
        help="Skip downloads larger than this many bytes (0=no limit)",
    )
    parser.add_argument(
        "--filetype",
        default=None,
        help="Comma-separated file extensions to include (e.g. mp4,mkv,png). Default: all supported media types.",
    )
    parser.add_argument(
        "--base-url",
        default="https://code.ffmpeg.org",
        help="Forgejo instance base URL",
    )
    parser.add_argument(
        "--owner",
        default="FFmpeg",
        help="Repository owner",
    )
    parser.add_argument(
        "--repo",
        default="FFmpeg",
        help="Repository name",
    )
    args = parser.parse_args()

    allowed_extensions: set[str] | None = None
    if args.filetype:
        allowed_extensions = {
            f".{ext.strip().lower()}"
            if not ext.strip().lower().startswith(".")
            else ext.strip().lower()
            for ext in args.filetype.split(",")
            if ext.strip()
        }

    seeds_dir = os.path.join(args.out, "seeds")
    os.makedirs(seeds_dir, exist_ok=True)

    issue_cache = load_cache()

    page = 1
    total_issues = 0
    total_saved = 0

    print(f"[*] Scanning {args.base_url}/{args.owner}/{args.repo}/issues for media seeds")
    try:
        while True:
            path = (
                f"/api/v1/repos/{args.owner}/{args.repo}/issues?page={page}&per_page=30&state=all"
            )
            issues = api_get(args.base_url, path, args.token)
            if not issues:
                break

            for issue in issues:
                if args.max_issues and total_issues >= args.max_issues:
                    break
                total_issues += 1
                saved = collect_seeds_from_issue(
                    args.base_url,
                    args.owner,
                    args.repo,
                    issue,
                    args.token,
                    seeds_dir,
                    issue_cache,
                    args.skip_size,
                    allowed_extensions,
                )
                total_saved += saved

            if args.max_issues and total_issues >= args.max_issues:
                break
            if len(issues) < 30:
                break
            page += 1
            time.sleep(0.25)
    finally:
        save_cache(issue_cache)

    total_files = len(os.listdir(seeds_dir))
    print(f"[*] Scanned {total_issues} issues, saved {total_saved} new seeds")
    print(f"[*] Corpus ready: {total_files} files in {seeds_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
