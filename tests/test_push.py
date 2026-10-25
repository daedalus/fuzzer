"""Tests for services/push.py — novel-crash push notifications."""

from __future__ import annotations

import argparse
import base64
import json
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from fuzzer_tool.services import push
from fuzzer_tool.services.push import (
    PushError,
    PushMessage,
    PushNotifier,
    PushService,
    crash_message,
    parse_push_url,
    push_targets,
    send_push,
)

ASAN_STDERR = (
    "==1==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x602\n"
    "    #0 0x4011 in parse_chunk png.c:88\n"
    "SUMMARY: AddressSanitizer: heap-buffer-overflow png.c:88 in parse_chunk\n"
)


class _Recorder(BaseHTTPRequestHandler):
    """Records every POST; status and delay come from the server object."""

    def do_POST(self):  # noqa: N802 — http.server API
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        time.sleep(self.server.delay)
        self.server.hits.append((self.path, dict(self.headers), body))
        self.send_response(self.server.status)
        self.end_headers()

    def log_message(self, *_):
        pass


@pytest.fixture
def server(monkeypatch):
    """Local HTTP endpoint; proxies off so requests (and the child's) reach it."""
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("NO_PROXY", "*")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Recorder)
    srv.hits, srv.status, srv.delay = [], 200, 0.0
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    srv.server_close()
    thread.join()


def _url(srv, path: str = "/alerts") -> str:
    return f"http://127.0.0.1:{srv.server_address[1]}{path}"


def _wait_hits(srv, n: int, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while len(srv.hits) < n and time.monotonic() < deadline:
        time.sleep(0.02)


class TestParse:
    def test_ntfy_topic_and_basic_auth(self):
        t = parse_push_url("https://bob:p%40ss@ntfy.example:8443/sub/crashes")
        assert t.service is PushService.NTFY
        assert t.key == "crashes"
        assert t.secret == "bob:p@ss"
        assert t.endpoint == "https://ntfy.example:8443/sub/"

    def test_gotify_token_and_prefix(self):
        t = parse_push_url("gotify://push.example/gotify/AbCdEf123")
        assert t.endpoint == "https://push.example/gotify/message"
        assert t.secret == "AbCdEf123"

    def test_gotify_plain_http(self):
        assert parse_push_url("gotify+http://h:8080/T").endpoint == "http://h:8080/message"

    def test_pushover_keeps_token_case(self):
        # urlsplit().hostname lowercases; Pushover tokens are case-sensitive.
        t = parse_push_url("pushover://UserKEY123@AppTOKEN456")
        assert (t.key, t.secret) == ("UserKEY123", "AppTOKEN456")
        assert t.endpoint == push.PUSHOVER_API

    def test_webhook_strips_prefix(self):
        t = parse_push_url("json+https://hooks.example/services/T0/B1/xyz?a=1")
        assert t.service is PushService.WEBHOOK
        assert t.endpoint == "https://hooks.example/services/T0/B1/xyz?a=1"

    @pytest.mark.parametrize(
        "url", ["https://ntfy.sh/", "gotify://host/", "pushover://onlytoken", "json+https:///x"]
    )
    def test_incomplete_urls_rejected(self, url):
        with pytest.raises(ValueError):
            parse_push_url(url)

    def test_adversarial_bad_scheme_error_does_not_echo_url(self):
        with pytest.raises(ValueError) as err:
            parse_push_url("ftp://user:SECRET@host/x")
        assert "SECRET" not in str(err.value)
        assert "'ftp'" in str(err.value)

    def test_env_fallback_and_cli_precedence(self, monkeypatch):
        monkeypatch.setenv(push.PUSH_ENV, "https://ntfy.sh/a  gotify://g/T")
        assert [t.service for t in push_targets(None)] == [PushService.NTFY, PushService.GOTIFY]
        assert [t.key for t in push_targets(["https://ntfy.sh/cli"])] == ["cli"]

    def test_nothing_configured_is_empty(self, monkeypatch):
        monkeypatch.delenv(push.PUSH_ENV, raising=False)
        assert push_targets(None) == ()


class TestMessage:
    def test_summary_line_preferred_and_signal_named(self):
        msg = crash_message(
            target="/b/png_read", base_name="crash_3", returncode=-11, stderr=ASAN_STDERR
        )
        assert msg.title == "[fuzzer-tool] crash crash_3 (png_read)"
        assert "SIGSEGV (rc -11)" in msg.body
        assert msg.body.splitlines()[-1].startswith("SUMMARY: AddressSanitizer")

    def test_positive_exit_code(self):
        assert "result: exit 1" in crash_message(target="t", base_name="c", returncode=1).body

    def test_adversarial_control_chars_stripped(self):
        # A target path is attacker-adjacent text: no CR/LF or escapes in the title.
        msg = crash_message(target="/x/a\r\nX-Evil: 1\x1b[2J", base_name="c\n")
        assert not any(ch in msg.title for ch in "\r\n\x1b")
        assert "\r" not in msg.body and "\x1b" not in msg.body

    def test_adversarial_huge_stderr_capped(self):
        msg = crash_message(target="t", base_name="c", stderr="SUMMARY: " + "A" * 50_000)
        assert len(msg.body) <= push.BODY_MAX
        assert len(msg.title) <= push.TITLE_MAX


class TestRequest:
    MSG = PushMessage("T", "B")

    def test_ntfy_json_and_auth_header(self):
        req = push._request(parse_push_url("https://u:pw@n.example/top"), self.MSG)
        body = json.loads(req.data)
        assert body["topic"] == "top" and body["priority"] == push.NTFY_PRIORITY_HIGH
        assert req.get_header("Authorization") == "Basic " + base64.b64encode(b"u:pw").decode()

    def test_pushover_form(self):
        req = push._request(parse_push_url("pushover://USR@APP"), self.MSG)
        form = urllib.parse.parse_qs(req.data.decode())
        assert form["token"] == ["APP"] and form["user"] == ["USR"]

    def test_webhook_slack_and_discord_fields(self):
        body = json.loads(push._request(parse_push_url("json+https://h/p"), self.MSG).data)
        assert body["text"] == body["content"] == "T\nB"

    @pytest.mark.parametrize(
        "url,secret",
        [
            ("https://u:S3CRET@n.example/top", "S3CRET"),
            ("gotify://g.example/S3CRET", "S3CRET"),
            ("pushover://USR@S3CRET", "S3CRET"),
        ],
    )
    def test_adversarial_secret_never_in_request_url(self, url, secret):
        req = push._request(parse_push_url(url), self.MSG)
        assert secret not in req.full_url


class TestSend:
    def test_ntfy_round_trip(self, server):
        send_push(parse_push_url(_url(server, "/crashes")), PushMessage("T", "B"))
        path, _, body = server.hits[0]
        assert path == "/"
        assert json.loads(body)["topic"] == "crashes"

    def test_gotify_key_header(self, server):
        port = server.server_address[1]
        send_push(parse_push_url(f"gotify+http://127.0.0.1:{port}/TOK"), PushMessage("T", "B"))
        path, headers, _ = server.hits[0]
        assert path == "/message"
        assert headers["X-Gotify-Key"] == "TOK"

    def test_falsification_http_error_raises_without_token(self, server):
        # A sender that ignored 4xx/5xx would report delivery of a lost alert.
        server.status = 401
        port = server.server_address[1]
        with pytest.raises(PushError) as err:
            send_push(
                parse_push_url(f"gotify+http://127.0.0.1:{port}/TOPSECRET"), PushMessage("T", "B")
            )
        assert "401" in str(err.value)
        assert "TOPSECRET" not in str(err.value)

    def test_unreachable_raises(self, server):
        port = server.server_address[1]
        server.shutdown()
        server.server_close()
        with pytest.raises(PushError):
            send_push(parse_push_url(f"http://127.0.0.1:{port}/t"), PushMessage("T", "B"))
        server.serve_forever = lambda: None  # fixture teardown on a closed server


class TestNotifier:
    def test_child_delivers(self, server):
        notifier = PushNotifier([parse_push_url(_url(server))])
        notifier.notify(crash_message(target="t", base_name="crash_9", stderr=ASAN_STDERR))
        notifier.close(timeout=15)
        assert len(server.hits) == 1
        assert "crash_9" in json.loads(server.hits[0][2])["title"]

    def test_falsification_notify_adds_no_thread(self, server):
        # The fuzzer must stay single-threaded around os.fork(): delivery is a process.
        before = threading.active_count()
        notifier = PushNotifier([parse_push_url(_url(server))])
        notifier.notify(PushMessage("T", "B"))
        assert threading.active_count() == before
        notifier.close(timeout=15)

    def test_falsification_notify_does_not_block_on_slow_server(self, server):
        server.delay = 2.0
        notifier = PushNotifier([parse_push_url(_url(server))])
        t0 = time.monotonic()
        notifier.notify(PushMessage("T", "B"))
        assert time.monotonic() - t0 < 1.0
        notifier.close(timeout=15)
        assert len(server.hits) == 1

    def test_child_failure_reported_on_close(self, server, capsys):
        server.status = 500
        notifier = PushNotifier([parse_push_url(_url(server))])
        notifier.notify(PushMessage("T", "B"))
        notifier.close(timeout=15)
        assert "crash push failed" in capsys.readouterr().out

    def test_adversarial_burst_is_capped_and_counted(self, server, monkeypatch, capsys):
        monkeypatch.setattr(push, "MAX_INFLIGHT", 1)
        server.delay = 1.0
        notifier = PushNotifier([parse_push_url(_url(server))])
        notifier.notify(PushMessage("A", "a"))
        notifier.notify(PushMessage("B", "b"))  # dropped: one in flight
        notifier.notify(PushMessage("C", "c"))  # dropped
        _wait_hits(server, 1)
        while notifier._senders:  # wait for A's child to exit, then reap it
            notifier._reap()
            time.sleep(0.02)
        notifier.notify(PushMessage("D", "d"))
        notifier.close(timeout=15)

        titles = [json.loads(b)["title"] for _, _, b in server.hits]
        assert titles == ["A", "D"]
        assert "(+2 earlier crashes not pushed)" in json.loads(server.hits[1][2])["message"]
        assert "not sent" not in capsys.readouterr().out

    def test_close_reports_undelivered_drops(self, monkeypatch, capsys):
        monkeypatch.setattr(push, "MAX_INFLIGHT", 0)
        notifier = PushNotifier([parse_push_url("https://ntfy.invalid/t")])
        notifier.notify(PushMessage("A", "a"))
        notifier.close(timeout=0)
        assert "1 crash pushes not sent" in capsys.readouterr().out


class TestWiring:
    def test_push_crash_builds_message(self):
        from fuzzer_tool.services.corpus_manager import CorpusManager

        notifier = MagicMock()
        f = SimpleNamespace(target="/b/tgt", exec_count=42, _push=notifier)
        cm = CorpusManager.__new__(CorpusManager)
        cm.f = f
        cm._push_crash("crash_1", -6, ASAN_STDERR)

        msg = notifier.notify.call_args[0][0]
        assert "crash_1" in msg.title and "SIGABRT" in msg.body and "execs: 42" in msg.body

    def test_adversarial_spawn_failure_does_not_raise(self, capsys):
        from fuzzer_tool.services.corpus_manager import CorpusManager

        notifier = MagicMock()
        notifier.notify.side_effect = OSError("fork failed")
        cm = CorpusManager.__new__(CorpusManager)
        cm.f = SimpleNamespace(target="t", exec_count=0, _push=notifier)
        cm._push_crash("c", -11, "")
        assert "crash push failed" in capsys.readouterr().out


class TestCliPushTest:
    def _args(self, **kw):
        base = {"push_test": True, "push_on_crash": ["https://ntfy.sh/t"], "targets": ["/b/tgt"]}
        base.update(kw)
        return argparse.Namespace(**base)

    def test_success_returns_zero(self):
        from fuzzer_tool.cli.commands import cmd_fuzz

        with patch("fuzzer_tool.services.push.send_push") as send:
            assert cmd_fuzz(self._args()) == 0
        assert "tgt" in send.call_args[0][1].body

    def test_failure_returns_one(self):
        from fuzzer_tool.cli.commands import cmd_fuzz

        with patch("fuzzer_tool.services.push.send_push", side_effect=PushError("ntfy://x: 500")):
            assert cmd_fuzz(self._args()) == 1

    def test_bad_url_returns_two(self):
        from fuzzer_tool.cli.commands import cmd_fuzz

        assert cmd_fuzz(self._args(push_on_crash=["ftp://x/y"])) == 2

    def test_unconfigured_returns_two(self, monkeypatch):
        from fuzzer_tool.cli.commands import cmd_fuzz

        monkeypatch.delenv(push.PUSH_ENV, raising=False)
        assert cmd_fuzz(self._args(push_on_crash=None)) == 2

    def test_flags_parse(self, monkeypatch):
        import sys

        from fuzzer_tool.cli import commands

        captured = {}
        monkeypatch.setattr(commands, "cmd_fuzz", lambda a: captured.setdefault("a", a) and 0)
        argv = ["fuzzer-tool", "fuzz", "/bin/true", "--push-on-crash", "https://a/x"]
        monkeypatch.setattr(sys, "argv", [*argv, "--push-on-crash", "gotify://g/T", "--push-test"])
        commands.main()
        assert captured["a"].push_on_crash == ["https://a/x", "gotify://g/T"]
        assert captured["a"].push_test is True


class TestSaveCrashHook:
    """save_crash pushes novel crashes only; duplicates stay silent."""

    def test_novel_pushed_duplicate_not(self, tmp_path, monkeypatch):
        from fuzzer_tool.services import corpus_manager as cm_service
        from fuzzer_tool.services.corpus_manager import CorpusManager
        from tests.test_regression_crash_triage_cost import _Fuzzer

        monkeypatch.setattr(cm_service, "_gdb_crash_replay", lambda *a: "")
        f = _Fuzzer(tmp_path)
        f.email_on_crash = None
        f._push = MagicMock()
        mgr = CorpusManager(f)

        assert mgr.save_crash(b"crash-input-1", -11, ASAN_STDERR)
        assert mgr.save_crash(b"crash-input-1", -11, ASAN_STDERR) is False
        assert f._push.notify.call_count == 1


class TestFuzzerLifecycle:
    def test_off_by_default(self, tmp_path):
        from tests.test_regression_end_of_run_persistence import _make_fuzzer

        assert _make_fuzzer(tmp_path)._push is None

    def test_run_teardown_drains_notifier(self, tmp_path):
        # Without close(), alerts for crashes found just before stop are lost.
        from tests.test_regression_end_of_run_persistence import _make_fuzzer

        f = _make_fuzzer(tmp_path, push_on_crash=(parse_push_url("https://ntfy.sh/t"),))
        assert isinstance(f._push, PushNotifier)
        with (
            patch.object(f, "_run_target", return_value=(0, "")),
            patch.object(PushNotifier, "close") as close,
        ):
            f.run(max_execs=3)
        close.assert_called_once()
