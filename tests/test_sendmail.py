"""Unit tests for services/sendmail.py — crash notification mail."""

from __future__ import annotations

import smtplib
from email.message import EmailMessage
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fuzzer_tool.services.sendmail import (
    MailConfig,
    build_crash_message,
    crash_attachments,
    format_crash_body,
    send_crash_email,
    send_message,
    _parse_smtp_server,
)


class TestMailConfig:
    def test_from_cli_basic(self):
        cfg = MailConfig.from_cli(to="a@b.c")
        assert cfg.to == "a@b.c"
        assert cfg.smtp_server is None
        assert cfg.auth_user is None
        assert cfg.require_tls is False

    def test_from_cli_auth_split(self):
        cfg = MailConfig.from_cli(
            to="a@b.c",
            smtp_server="smtp.example:587",
            auth="user@ex:s3cret",
            require_tls=True,
        )
        assert cfg.auth_user == "user@ex"
        assert cfg.auth_password == "s3cret"
        assert cfg.require_tls is True
        assert cfg.smtp_server == "smtp.example:587"

    def test_from_cli_auth_requires_colon(self):
        with pytest.raises(ValueError, match="USER:PASSWORD"):
            MailConfig.from_cli(to="a@b.c", auth="nocolon")

    def test_resolve_from_defaults_to_hostname(self):
        cfg = MailConfig(to="a@b.c")
        assert cfg.resolve_from().startswith("fuzzer-tool@")

    def test_resolve_from_prefers_auth_user_with_at(self):
        cfg = MailConfig(to="a@b.c", auth_user="bot@example.com", auth_password="x")
        assert cfg.resolve_from() == "bot@example.com"


class TestParseSmtpServer:
    def test_host_only(self):
        assert _parse_smtp_server("mail.example") == ("mail.example", 25)

    def test_host_port(self):
        assert _parse_smtp_server("mail.example:587") == ("mail.example", 587)

    def test_ipv6(self):
        assert _parse_smtp_server("[2001:db8::1]:465") == ("2001:db8::1", 465)


class TestMessageBuild:
    def test_attachments_and_headers(self, tmp_path: Path):
        bin_path = tmp_path / "crash_1.bin"
        bin_path.write_bytes(b"\x00\x01")
        txt_path = tmp_path / "crash_1.txt"
        txt_path.write_text("triage")
        cfg = MailConfig(to="dest@example.com", from_addr="src@example.com")
        msg = build_crash_message(
            cfg,
            subject="subj",
            body="body text",
            attachments=[bin_path, txt_path],
        )
        assert msg["To"] == "dest@example.com"
        assert msg["From"] == "src@example.com"
        assert msg["Subject"] == "subj"
        # one body part + two attachments
        assert len(list(msg.iter_attachments())) == 2

    def test_crash_attachments_collects_sidecars(self, tmp_path: Path):
        base = "crash_99_abc"
        for ext, data in (
            (".bin", b"x"),
            (".txt", "t"),
            (".sh", "#!/bin/sh\n"),
            (".hex", "00"),
        ):
            (tmp_path / f"{base}{ext}").write_bytes(
                data if isinstance(data, bytes) else data.encode()
            )
        paths = crash_attachments(tmp_path, base)
        assert [p.suffix for p in paths] == [".bin", ".txt", ".sh", ".hex"]

    def test_format_body_includes_sidecar(self, tmp_path: Path):
        body = format_crash_body(
            target="/bin/t",
            base_name="crash_1",
            returncode=-11,
            exec_count=42,
            crashes_dir=tmp_path,
            sidecar_text="ERROR: heap-buffer-overflow",
        )
        assert "heap-buffer-overflow" in body
        assert "execs:      42" in body


class TestSendMessage:
    def test_smtp_path_uses_starttls_when_required(self):
        cfg = MailConfig(
            to="a@b.c",
            smtp_server="smtp.example:587",
            auth_user="u",
            auth_password="p",
            require_tls=True,
            from_addr="u@b.c",
        )
        msg = build_crash_message(cfg, subject="s", body="b")
        fake = MagicMock()
        with patch("fuzzer_tool.services.sendmail.smtplib.SMTP", return_value=fake) as smtp_cls:
            send_message(cfg, msg)
        smtp_cls.assert_called_once_with("smtp.example", 587, timeout=30)
        fake.ehlo.assert_called()
        fake.starttls.assert_called_once()
        fake.login.assert_called_once_with("u", "p")
        fake.send_message.assert_called_once()
        fake.quit.assert_called_once()

    def test_smtp_ssl_on_port_465(self):
        cfg = MailConfig(
            to="a@b.c",
            smtp_server="smtp.example:465",
            from_addr="u@b.c",
        )
        msg = build_crash_message(cfg, subject="s", body="b")
        fake = MagicMock()
        with patch(
            "fuzzer_tool.services.sendmail.smtplib.SMTP_SSL", return_value=fake
        ) as ssl_cls:
            send_message(cfg, msg)
        ssl_cls.assert_called_once_with("smtp.example", 465, timeout=30)
        fake.starttls.assert_not_called()

    def test_sendmail_path_invokes_subprocess(self):
        cfg = MailConfig(to="a@b.c", from_addr="src@b.c")
        msg = build_crash_message(cfg, subject="s", body="b")
        completed = MagicMock(returncode=0, stderr=b"")
        with (
            patch("fuzzer_tool.services.sendmail.os.path.isfile", return_value=True),
            patch("fuzzer_tool.services.sendmail.os.access", return_value=True),
            patch(
                "fuzzer_tool.services.sendmail.subprocess.run", return_value=completed
            ) as run,
        ):
            send_message(cfg, msg)
        run.assert_called_once()
        args = run.call_args[0][0]
        assert args[0] == "/usr/sbin/sendmail"
        assert "-t" in args

    def test_send_crash_email_end_to_end_mocked(self, tmp_path: Path):
        base = "crash_1_sig"
        (tmp_path / f"{base}.bin").write_bytes(b"\x41\x42")
        (tmp_path / f"{base}.txt").write_text("ERROR: SEGV\n")
        cfg = MailConfig(to="a@b.c", from_addr="src@b.c")
        with patch("fuzzer_tool.services.sendmail.send_message") as send:
            send_crash_email(
                cfg,
                target="/tmp/target",
                base_name=base,
                crashes_dir=tmp_path,
                returncode=-11,
                exec_count=9,
            )
        send.assert_called_once()
        msg: EmailMessage = send.call_args[0][1]
        assert "crash_1_sig" in msg["Subject"]
        assert "SEGV" in msg.get_body().get_content()
        assert len(list(msg.iter_attachments())) == 2
