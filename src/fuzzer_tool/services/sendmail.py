"""Crash notification email via system MTA or SMTP.

Used by ``--send-email-on-crash``: when a *novel* crash is saved, a short
triage report is mailed to the configured address with the crash artefacts
(``.bin``, ``.txt``, ``.sh``, ``.hex``) attached.

Transport selection:
  * If ``smtp_server`` is set → connect with ``smtplib`` (optional AUTH/TLS).
  * Otherwise → pipe the message to the system MTA via ``/usr/sbin/sendmail -t``
    (falls back to ``sendmail`` on ``PATH``).
"""

from __future__ import annotations

import mimetypes
import os
import smtplib
import socket
import subprocess
from dataclasses import dataclass, field
from email.message import EmailMessage
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class MailConfig:
    """Destination and transport settings for crash notification mail."""

    to: str
    smtp_server: str | None = None
    auth_user: str | None = None
    auth_password: str | None = None
    require_tls: bool = False
    from_addr: str | None = None
    extra_to: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_cli(
        cls,
        to: str,
        smtp_server: str | None = None,
        auth: str | None = None,
        require_tls: bool = False,
        from_addr: str | None = None,
    ) -> MailConfig:
        user = password = None
        if auth:
            if ":" not in auth:
                raise ValueError(
                    "--send-mail-auth must be USER:PASSWORD (colon-separated)"
                )
            user, password = auth.split(":", 1)
            if not user:
                raise ValueError("--send-mail-auth USER part is empty")
        return cls(
            to=to.strip(),
            smtp_server=(smtp_server.strip() if smtp_server else None),
            auth_user=user,
            auth_password=password,
            require_tls=bool(require_tls),
            from_addr=(from_addr.strip() if from_addr else None),
        )

    def resolve_from(self) -> str:
        if self.from_addr:
            return self.from_addr
        if self.auth_user and "@" in self.auth_user:
            return self.auth_user
        host = socket.gethostname() or "localhost"
        return f"fuzzer-tool@{host}"


def _parse_smtp_server(server: str) -> tuple[str, int]:
    """Split ``host`` or ``host:port`` into (host, port). Default port 25."""
    server = server.strip()
    if not server:
        raise ValueError("empty SMTP server")
    if server.startswith("[") and "]" in server:
        # IPv6 literal: [2001:db8::1]:587
        host, _, rest = server[1:].partition("]")
        if rest.startswith(":"):
            return host, int(rest[1:])
        return host, 25
    if server.count(":") == 1:
        host, port_s = server.rsplit(":", 1)
        if port_s.isdigit():
            return host, int(port_s)
    return server, 25


def _attach_file(msg: EmailMessage, path: Path) -> None:
    if not path.is_file():
        return
    ctype, encoding = mimetypes.guess_type(str(path))
    if ctype is None or encoding is not None:
        ctype = "application/octet-stream"
    maintype, subtype = ctype.split("/", 1)
    msg.add_attachment(
        path.read_bytes(),
        maintype=maintype,
        subtype=subtype,
        filename=path.name,
    )


def build_crash_message(
    config: MailConfig,
    *,
    subject: str,
    body: str,
    attachments: Sequence[Path] = (),
) -> EmailMessage:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = config.resolve_from()
    recipients = [config.to, *config.extra_to]
    msg["To"] = ", ".join(recipients)
    msg.set_content(body)
    for path in attachments:
        _attach_file(msg, Path(path))
    return msg


def _send_via_sendmail(msg: EmailMessage) -> None:
    candidates = (
        "/usr/sbin/sendmail",
        "/usr/lib/sendmail",
        "sendmail",
    )
    sendmail = None
    for cand in candidates:
        if cand.startswith("/") and os.path.isfile(cand) and os.access(cand, os.X_OK):
            sendmail = cand
            break
        if not cand.startswith("/"):
            from shutil import which

            found = which(cand)
            if found:
                sendmail = found
                break
    if sendmail is None:
        raise RuntimeError(
            "no system MTA found (tried /usr/sbin/sendmail, /usr/lib/sendmail, "
            "sendmail on PATH); set --send-mail-smtp-server to use SMTP instead"
        )
    # -t: read recipients from message headers; -oi: don't treat '.' alone as EOF
    proc = subprocess.run(
        [sendmail, "-t", "-oi"],
        input=msg.as_bytes(),
        capture_output=True,
        timeout=60,
        check=False,
    )
    if proc.returncode != 0:
        err = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"sendmail exited {proc.returncode}" + (f": {err}" if err else "")
        )


def _send_via_smtp(config: MailConfig, msg: EmailMessage) -> None:
    assert config.smtp_server is not None
    host, port = _parse_smtp_server(config.smtp_server)
    recipients = [config.to, *config.extra_to]

    # Port 465 is implicit SSL; otherwise STARTTLS when required (or port 587).
    use_ssl = port == 465
    if use_ssl:
        smtp: smtplib.SMTP = smtplib.SMTP_SSL(host, port, timeout=30)
    else:
        smtp = smtplib.SMTP(host, port, timeout=30)
    try:
        smtp.ehlo()
        if config.require_tls or port == 587:
            if not use_ssl:
                smtp.starttls()
                smtp.ehlo()
        if config.auth_user is not None:
            smtp.login(config.auth_user, config.auth_password or "")
        smtp.send_message(msg, to_addrs=recipients)
    finally:
        try:
            smtp.quit()
        except Exception:
            pass


def send_message(config: MailConfig, msg: EmailMessage) -> None:
    """Deliver *msg* using the transport selected by *config*."""
    if config.smtp_server:
        _send_via_smtp(config, msg)
    else:
        _send_via_sendmail(msg)


def crash_attachments(crashes_dir: Path, base_name: str) -> list[Path]:
    """Collect the standard sidecar set written by ``adapters.filesystem.save_crash``."""
    out: list[Path] = []
    for ext in (".bin", ".txt", ".sh", ".hex"):
        path = Path(crashes_dir) / f"{base_name}{ext}"
        if path.is_file():
            out.append(path)
    return out


def format_crash_body(
    *,
    target: str,
    base_name: str,
    returncode: int,
    exec_count: int,
    crashes_dir: Path,
    stderr: str = "",
    sidecar_text: str = "",
) -> str:
    lines = [
        "fuzzer-tool: novel crash detected",
        "",
        f"  target:     {target}",
        f"  crash:      {base_name}",
        f"  returncode: {returncode}",
        f"  execs:      {exec_count}",
        f"  crashes_dir:{crashes_dir}",
        "",
    ]
    if sidecar_text.strip():
        lines.append("--- triage sidecar (.txt) ---")
        lines.append(sidecar_text.rstrip())
        lines.append("")
    elif stderr.strip():
        # Cap stderr so the body stays reasonable for MTA size limits.
        clipped = stderr if len(stderr) <= 8000 else stderr[:8000] + "\n...[truncated]...\n"
        lines.append("--- stderr ---")
        lines.append(clipped.rstrip())
        lines.append("")
    lines.append("Crash artefacts are attached when available (.bin/.txt/.sh/.hex).")
    return "\n".join(lines) + "\n"


def send_crash_email(
    config: MailConfig,
    *,
    target: str,
    base_name: str,
    crashes_dir: Path | str,
    returncode: int = 0,
    exec_count: int = 0,
    stderr: str = "",
) -> None:
    """Build and send a novel-crash notification with standard attachments."""
    crashes_dir = Path(crashes_dir)
    attachments = crash_attachments(crashes_dir, base_name)
    sidecar_path = crashes_dir / f"{base_name}.txt"
    sidecar_text = ""
    if sidecar_path.is_file():
        try:
            sidecar_text = sidecar_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            sidecar_text = ""
    body = format_crash_body(
        target=target,
        base_name=base_name,
        returncode=returncode,
        exec_count=exec_count,
        crashes_dir=crashes_dir,
        stderr=stderr,
        sidecar_text=sidecar_text,
    )
    target_base = os.path.basename(target) if target else "target"
    subject = f"[fuzzer-tool] crash {base_name} ({target_base})"
    msg = build_crash_message(
        config, subject=subject, body=body, attachments=attachments
    )
    send_message(config, msg)


__all__ = [
    "MailConfig",
    "build_crash_message",
    "crash_attachments",
    "format_crash_body",
    "send_crash_email",
    "send_message",
]
