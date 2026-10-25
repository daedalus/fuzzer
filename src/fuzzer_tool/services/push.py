"""Crash push notifications: ntfy, Gotify, Pushover, JSON webhook.

Used by ``--push-on-crash URL``: each *novel* crash sends a short alert
(target, crash name, signal, sanitizer SUMMARY line). No crash bytes leave
the machine.

URL forms (the scheme picks the service)::

    https://ntfy.sh/TOPIC            ntfy; user:pass@ -> Basic auth
    gotify://HOST[:PORT][/PATH]/TOKEN  Gotify over HTTPS (gotify+http:// for HTTP)
    pushover://USER_KEY@APP_TOKEN    Pushover
    json+https://HOST/PATH           JSON POST {title, message, text, content}
                                     (text: Slack, content: Discord)

Delivery runs in a child process (``python -I push.py``; stdlib only)::

    save_crash -> PushNotifier.notify -> Popen(child) --HTTPS--> service
                       |                     ^
                       +-- reap / drain -----+

so a slow or dead server never stalls the exec loop, and the fuzzer stays
single-threaded for its os.fork() paths (runner.py, persistent_subprocess.py).
"""

from __future__ import annotations

import base64
import json
import os
import pickle
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import Enum

# Read when --push-on-crash is absent; keeps tokens out of argv and history.
PUSH_ENV = "FUZZER_PUSH_URL"

HTTP_TIMEOUT_S = 10
# A sender older than this is killed: urllib's timeout does not cover DNS.
SENDER_DEADLINE_S = 60
DRAIN_TIMEOUT_S = 5
# Senders in flight; a crash burst beyond this is counted, not sent.
MAX_INFLIGHT = 4
TITLE_MAX = 200
BODY_MAX = 1000
DISCORD_CONTENT_MAX = 2000

PUSHOVER_API = "https://api.pushover.net/1/messages.json"
NTFY_PRIORITY_HIGH = 4
GOTIFY_PRIORITY_HIGH = 8
PUSHOVER_PRIORITY_HIGH = 1

# Most specific first: ASAN's SUMMARY names bug class and frame in one line.
_SUMMARY_MARKERS = ("SUMMARY: ", "ERROR: ", "runtime error:")
_CTRL = re.compile(r"[\x00-\x09\x0b-\x1f\x7f]")


class PushService(Enum):
    NTFY = "ntfy"
    GOTIFY = "gotify"
    PUSHOVER = "pushover"
    WEBHOOK = "webhook"


class PushError(RuntimeError):
    """Delivery failed. The message names the service and host, never the token."""


@dataclass(frozen=True)
class PushTarget:
    service: PushService
    endpoint: str
    host: str
    # ntfy/webhook: "user:pass" (Basic). Gotify: app token. Pushover: app token.
    secret: str | None = None
    # ntfy: topic. Pushover: user key.
    key: str | None = None

    @property
    def label(self) -> str:
        return f"{self.service.value}://{self.host}"


@dataclass(frozen=True)
class PushMessage:
    title: str
    body: str


def _split_netloc(netloc: str) -> tuple[str | None, str]:
    """``user:pass@host:port`` -> (``user:pass``, ``host:port``); case kept."""
    userinfo, at, hostport = netloc.rpartition("@")
    if not at:
        return None, netloc
    return urllib.parse.unquote(userinfo), hostport


def _parse_ntfy(parts: urllib.parse.SplitResult) -> PushTarget:
    auth, hostport = _split_netloc(parts.netloc)
    prefix, _, topic = parts.path.rstrip("/").rpartition("/")
    if not hostport or not topic:
        raise ValueError("ntfy URL needs a topic: https://HOST/TOPIC")
    # JSON publish goes to the server root, so no title header to inject into.
    endpoint = f"{parts.scheme}://{hostport}{prefix}/"
    return PushTarget(PushService.NTFY, endpoint, hostport, auth, topic)


def _parse_gotify(parts: urllib.parse.SplitResult) -> PushTarget:
    _, hostport = _split_netloc(parts.netloc)
    prefix, _, token = parts.path.rstrip("/").rpartition("/")
    if not hostport or not token:
        raise ValueError("Gotify URL needs an app token: gotify://HOST/TOKEN")
    proto = "http" if parts.scheme == "gotify+http" else "https"
    endpoint = f"{proto}://{hostport}{prefix}/message"
    return PushTarget(PushService.GOTIFY, endpoint, hostport, token)


def _parse_pushover(parts: urllib.parse.SplitResult) -> PushTarget:
    user, app = _split_netloc(parts.netloc)
    if not user or not app:
        raise ValueError("Pushover URL must be pushover://USER_KEY@APP_TOKEN")
    host = urllib.parse.urlsplit(PUSHOVER_API).netloc
    return PushTarget(PushService.PUSHOVER, PUSHOVER_API, host, app, user)


def _parse_webhook(parts: urllib.parse.SplitResult) -> PushTarget:
    auth, hostport = _split_netloc(parts.netloc)
    if not hostport:
        raise ValueError("webhook URL needs a host: json+https://HOST/PATH")
    proto = parts.scheme.removeprefix("json+")
    endpoint = urllib.parse.urlunsplit((proto, hostport, parts.path, parts.query, ""))
    return PushTarget(PushService.WEBHOOK, endpoint, hostport, auth)


_PARSERS = {
    "http": _parse_ntfy,
    "https": _parse_ntfy,
    "gotify": _parse_gotify,
    "gotify+http": _parse_gotify,
    "pushover": _parse_pushover,
    "json+http": _parse_webhook,
    "json+https": _parse_webhook,
}


def parse_push_url(url: str) -> PushTarget:
    """Parse one ``--push-on-crash`` URL; ValueError names the scheme, not the URL."""
    parts = urllib.parse.urlsplit(url.strip())
    parser = _PARSERS.get(parts.scheme.lower())
    if parser is None:
        raise ValueError(
            f"unsupported push URL scheme {parts.scheme!r}; "
            "use https:// (ntfy), gotify://, pushover:// or json+https://"
        )
    return parser(parts._replace(scheme=parts.scheme.lower()))


def push_targets(urls: Sequence[str] | None) -> tuple[PushTarget, ...]:
    """CLI URLs, else whitespace-separated ``$FUZZER_PUSH_URL``; () when neither."""
    if not urls:
        urls = os.environ.get(PUSH_ENV, "").split()
    return tuple(parse_push_url(u) for u in urls)


def _clean(text: str, limit: int, *, oneline: bool) -> str:
    """Strip control chars (and newlines when *oneline*), cap at *limit*."""
    if oneline:
        text = text.replace("\n", " ")
    text = _CTRL.sub(" ", text).strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _exit_desc(returncode: int) -> str:
    if returncode >= 0:
        return f"exit {returncode}"
    try:
        return f"{signal.Signals(-returncode).name} (rc {returncode})"
    except ValueError:
        return f"signal {-returncode}"


def _summary_line(stderr: str) -> str:
    for marker in _SUMMARY_MARKERS:
        for line in stderr.splitlines():
            if marker in line:
                return line.strip()
    return ""


def crash_message(
    *,
    target: str,
    base_name: str,
    returncode: int = 0,
    exec_count: int = 0,
    stderr: str = "",
) -> PushMessage:
    """Short novel-crash alert: who crashed, how, where."""
    target_base = os.path.basename(target) if target else "target"
    lines = [
        f"target: {target_base}",
        f"crash: {base_name}",
        f"result: {_exit_desc(returncode)}",
        f"execs: {exec_count}",
        f"host: {socket.gethostname()}",
    ]
    summary = _summary_line(stderr)
    if summary:
        lines.append(summary)
    return PushMessage(
        title=_clean(f"[fuzzer-tool] crash {base_name} ({target_base})", TITLE_MAX, oneline=True),
        body=_clean("\n".join(lines), BODY_MAX, oneline=False),
    )


def _basic(userinfo: str) -> str:
    return "Basic " + base64.b64encode(userinfo.encode()).decode("ascii")


def _request(target: PushTarget, msg: PushMessage) -> urllib.request.Request:
    """Build the service-specific POST. Secrets go in headers/body, never the URL."""
    headers = {"Content-Type": "application/json", "User-Agent": "fuzzer-tool"}
    service = target.service

    if service is PushService.PUSHOVER:
        form = {
            "token": target.secret,
            "user": target.key,
            "title": msg.title,
            "message": msg.body,
            "priority": PUSHOVER_PRIORITY_HIGH,
        }
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        data = urllib.parse.urlencode(form).encode()
        return urllib.request.Request(target.endpoint, data, headers, method="POST")

    if service is PushService.NTFY:
        payload: dict[str, object] = {
            "topic": target.key,
            "title": msg.title,
            "message": msg.body,
            "priority": NTFY_PRIORITY_HIGH,
            "tags": ["warning"],
        }
    elif service is PushService.GOTIFY:
        payload = {"title": msg.title, "message": msg.body, "priority": GOTIFY_PRIORITY_HIGH}
        headers["X-Gotify-Key"] = target.secret or ""
    else:
        text = f"{msg.title}\n{msg.body}"
        payload = {
            "title": msg.title,
            "message": msg.body,
            "text": text,
            "content": text[:DISCORD_CONTENT_MAX],
        }

    if service in (PushService.NTFY, PushService.WEBHOOK) and target.secret:
        headers["Authorization"] = _basic(target.secret)
    data = json.dumps(payload).encode()
    return urllib.request.Request(target.endpoint, data, headers, method="POST")


def send_push(target: PushTarget, msg: PushMessage) -> None:
    """POST *msg* to *target* synchronously; raises PushError."""
    try:
        with urllib.request.urlopen(_request(target, msg), timeout=HTTP_TIMEOUT_S) as resp:
            resp.read()
    except urllib.error.HTTPError as exc:
        raise PushError(f"{target.label}: HTTP {exc.code} {exc.reason}") from None
    except (urllib.error.URLError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise PushError(f"{target.label}: {reason}") from None


def send_test_push(targets: Sequence[PushTarget], *, target: str = "") -> list[str]:
    """Probe every target in-process (``--push-test``); returns failure messages."""
    target_base = os.path.basename(target) if target else "-"
    msg = PushMessage(
        title="[fuzzer-tool] push test",
        body=f"target: {target_base}\nhost: {socket.gethostname()}\n"
        "Crash alerts will use this channel.",
    )
    failures = []
    for t in targets:
        try:
            send_push(t, msg)
        except PushError as exc:
            failures.append(str(exc))
    return failures


@dataclass
class _Sender:
    proc: subprocess.Popen[bytes]
    started: float


class PushNotifier:
    """Fire-and-forget crash alerts, one child process per crash.

    At most MAX_INFLIGHT senders run; crashes past that are counted and
    reported in the next alert that goes out (or at close()).
    """

    def __init__(self, targets: Sequence[PushTarget]):
        self._targets = tuple(targets)
        self._senders: list[_Sender] = []
        self._dropped = 0

    def notify(self, msg: PushMessage) -> None:
        self._reap()
        if len(self._senders) >= MAX_INFLIGHT:
            self._dropped += 1
            return

        if self._dropped:
            note = f"\n(+{self._dropped} earlier crashes not pushed)"
            msg = replace(msg, body=msg.body + note)
            self._dropped = 0

        # Popen is fork+exec in C: no Python thread, no os.fork() hazard.
        # -I: run this file by path without importing the package (which may
        # not be installed) and without its directory shadowing the stdlib.
        proc = subprocess.Popen(
            [sys.executable, "-I", os.path.abspath(__file__)],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        assert proc.stdin is not None
        proc.stdin.write(_encode(self._targets, msg))
        proc.stdin.close()
        self._senders.append(_Sender(proc, time.monotonic()))

    def close(self, timeout: float = DRAIN_TIMEOUT_S) -> None:
        """Wait up to *timeout* for in-flight alerts, then kill the rest."""
        deadline = time.monotonic() + timeout
        for sender in self._senders:
            try:
                sender.proc.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                sender.proc.kill()
                sender.proc.wait()
                print("[!] crash push abandoned at shutdown (server too slow)")
        self._reap()
        if self._dropped:
            print(f"[!] {self._dropped} crash pushes not sent (more than {MAX_INFLIGHT} in flight)")
            self._dropped = 0

    def _reap(self) -> None:
        """Collect finished senders, report failures, kill overdue ones."""
        now = time.monotonic()
        alive = []
        for sender in self._senders:
            if sender.proc.poll() is None and now - sender.started > SENDER_DEADLINE_S:
                sender.proc.kill()
                sender.proc.wait()
                print(f"[!] crash push killed after {SENDER_DEADLINE_S}s")
                continue
            if sender.proc.poll() is None:
                alive.append(sender)
                continue
            self._report(sender.proc)
        self._senders = alive

    @staticmethod
    def _report(proc: subprocess.Popen[bytes]) -> None:
        assert proc.stderr is not None
        err = proc.stderr.read().decode("utf-8", errors="replace").strip()
        proc.stderr.close()
        if proc.returncode != 0 and err:
            print(err)


def _encode(targets: Sequence[PushTarget], msg: PushMessage) -> bytes:
    """Plain tuples: the child runs this file as __main__, not as the package module."""
    rows = [(t.service.value, t.endpoint, t.host, t.secret, t.key) for t in targets]
    return pickle.dumps((rows, msg.title, msg.body))


def _decode(blob: bytes) -> tuple[list[PushTarget], PushMessage]:
    rows, title, body = pickle.loads(blob)
    targets = [PushTarget(PushService(s), e, h, sec, k) for s, e, h, sec, k in rows]
    return targets, PushMessage(title, body)


def _main() -> int:
    """Child entry: decode (targets, message) from stdin, send to each."""
    targets, msg = _decode(sys.stdin.buffer.read())
    failed = 0
    for target in targets:
        try:
            send_push(target, msg)
        except PushError as exc:
            print(f"[!] crash push failed: {exc}", file=sys.stderr)
            failed += 1
    return 1 if failed else 0


__all__ = [
    "PUSH_ENV",
    "PushError",
    "PushMessage",
    "PushNotifier",
    "PushService",
    "PushTarget",
    "crash_message",
    "parse_push_url",
    "push_targets",
    "send_push",
    "send_test_push",
]


if __name__ == "__main__":
    sys.exit(_main())
