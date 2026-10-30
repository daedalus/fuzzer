"""`edge-dump --push-on-finish [URL]`: one push when the run ends, success or not.

--push-on-finish URL   push to URL (repeatable)
--push-on-finish       push to $FUZZER_PUSH_URL
(absent)               no push, even with $FUZZER_PUSH_URL set
"""

import sys

import pytest

from fuzzer_tool.cli import commands
from fuzzer_tool.services import push
from fuzzer_tool.services.push import PUSH_ENV, PushError, edge_dump_message
from tests.test_edge_export import _setup

NTFY = "https://ntfy.example/topic"
NTFY2 = "https://ntfy.example/other"


@pytest.fixture
def sent(monkeypatch):
    """Capture send_push calls instead of hitting the network."""
    calls: list = []
    monkeypatch.setattr(push, "send_push", lambda t, m: calls.append((t, m)))
    monkeypatch.setattr(commands, "_validate_target", lambda t: None)
    monkeypatch.delenv(PUSH_ENV, raising=False)
    return calls


def _run(monkeypatch, corpus, out, *extra) -> int:
    argv = ["fuzzer-tool", "edge-dump", "t", "-d", str(corpus), "-o", str(out), *extra]
    monkeypatch.setattr(sys, "argv", argv)
    return commands.main()


def test_message_fields():
    msg = edge_dump_message(
        target="/x/t.so", corpus="/c", outputs=["/o/e.pkl"], seeds=7, rc=0, elapsed=3.2
    )
    assert "edge-dump" in msg.title and "t.so" in msg.title and "done" in msg.title
    for field in ("seeds: 7", "e.pkl", "3.2s", "host:"):
        assert field in msg.body


def test_message_failure_and_sanitized():
    """Adversarial: control chars in paths never reach the title; failure is named."""
    msg = edge_dump_message(
        target="t\n\x1b[31m", corpus="/c", outputs=["e\x00.pkl"], seeds=0, rc=1, elapsed=0.0
    )
    assert "failed" in msg.title
    assert "\n" not in msg.title and "\x1b" not in msg.title and "\x00" not in msg.body


def test_push_on_success(tmp_path, monkeypatch, sent):
    corpus, out = _setup(tmp_path, monkeypatch, {"a": b"ab"})
    assert _run(monkeypatch, corpus, out, "--push-on-finish", NTFY, "--push-on-finish", NTFY2) == 0

    assert [t.key for t, _ in sent] == ["topic", "other"]
    assert all("done" in m.title and "seeds: 1" in m.body for _, m in sent)


def test_push_on_failure(tmp_path, monkeypatch, sent):
    corpus, out = _setup(tmp_path, monkeypatch, {"a": b"ab"}, blind=True)
    assert _run(monkeypatch, corpus, out, "--push-on-finish", NTFY) == 1
    assert len(sent) == 1 and "failed" in sent[0][1].title


def test_no_flag_no_push_even_with_env(tmp_path, monkeypatch, sent):
    """Falsification: the env var alone must not push."""
    monkeypatch.setenv(PUSH_ENV, NTFY)
    corpus, out = _setup(tmp_path, monkeypatch, {"a": b"ab"})
    assert _run(monkeypatch, corpus, out) == 0
    assert sent == []


def test_bare_flag_uses_env(tmp_path, monkeypatch, sent):
    monkeypatch.setenv(PUSH_ENV, NTFY)
    corpus, out = _setup(tmp_path, monkeypatch, {"a": b"ab"})
    assert _run(monkeypatch, corpus, out, "--push-on-finish") == 0
    assert [t.key for t, _ in sent] == ["topic"]


def test_push_error_keeps_exit_code(tmp_path, monkeypatch, sent, capsys):
    """Adversarial: a dead push server does not change the run's result."""

    def boom(t, m):
        raise PushError("ntfy://ntfy.example: down")

    monkeypatch.setattr(push, "send_push", boom)
    corpus, out = _setup(tmp_path, monkeypatch, {"a": b"ab"})
    assert _run(monkeypatch, corpus, out, "--push-on-finish", NTFY) == 0
    assert "push failed" in capsys.readouterr().out


def test_bad_url_rejected_before_replay(tmp_path, monkeypatch, sent):
    """Adversarial: an unusable URL fails fast; nothing is replayed or written."""
    corpus, out = _setup(tmp_path, monkeypatch, {"a": b"ab"})
    assert _run(monkeypatch, corpus, out, "--push-on-finish", "ftp://x/y") == 2
    assert not out.exists()
    assert sent == []
