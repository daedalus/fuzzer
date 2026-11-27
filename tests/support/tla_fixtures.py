"""Small state machines for tests/test_tla_extract.py.

Each pair is a buggy class and its fixed twin: the fix is the control that
must pass before the bug's counterexample means anything (Hard Rule 46).
"""

import contextlib
import os
import signal
import subprocess
import tempfile
import threading

from fuzzer_tool.core import tla

_PROC_STATE = {"_proc": tla.Kind.PROC, "_ready": tla.Kind.BOOL}
_SPAWN = {"subprocess.Popen": tla.Op.SPAWN}


@tla.machine(state=_PROC_STATE, model=_SPAWN)
class LeakyRunner:
    """stop() drops the handle without killing: the loader runs forever."""

    def __init__(self):
        self._proc = None
        self._ready = False

    @tla.entry(tla.Phase.SETUP)
    def start(self):
        self._proc = subprocess.Popen(["true"])
        self._ready = True
        return True

    @tla.entry()
    def stop(self):
        self._proc = None
        self._ready = False


@tla.machine(state=_PROC_STATE, model=_SPAWN)
class CleanRunner:
    """LeakyRunner with the kill/reap stop() needs."""

    def __init__(self):
        self._proc = None
        self._ready = False

    @tla.entry(tla.Phase.SETUP)
    def start(self):
        self._proc = subprocess.Popen(["true"])
        self._ready = True
        return True

    @tla.entry()
    def stop(self):
        proc = self._proc
        self._proc = None
        self._ready = False
        if proc is None:
            return
        proc.kill()
        proc.wait()


_READER_STATE = {"_proc": tla.Kind.PROC, "_reader": tla.Kind.READER, "_ready": tla.Kind.BOOL}
_READER_MODEL = {"subprocess.Popen": tla.Op.SPAWN, "PipeReader": tla.Op.READER}


@tla.machine(state=_READER_STATE, model=_READER_MODEL, replies={"PING": "PONG"})
class StaleReplyRunner:
    """A timed-out PING keeps the loader: its late PONG answers the next PING."""

    def __init__(self):
        self._proc = None
        self._reader = None
        self._ready = False

    @tla.entry(tla.Phase.SETUP)
    def start(self):
        self._proc = subprocess.Popen(["true"])
        self._reader = PipeReader(self._proc.stdout.fileno())
        self._ready = True
        return True

    @tla.entry(requires="self._ready")
    def ping(self):
        try:
            self._proc.stdin.write(b"PING\n")
        except OSError:
            return False
        reply = self._reader.readline(1.0)
        return reply is not None


@tla.machine(state=_READER_STATE, model=_READER_MODEL, replies={"PING": "PONG"})
class FreshReplyRunner:
    """StaleReplyRunner that kills the loader on timeout, dropping the late reply."""

    def __init__(self):
        self._proc = None
        self._reader = None
        self._ready = False

    @tla.entry(tla.Phase.SETUP)
    def start(self):
        self._proc = subprocess.Popen(["true"])
        self._reader = PipeReader(self._proc.stdout.fileno())
        self._ready = True
        return True

    @tla.entry(requires="self._ready")
    def ping(self):
        try:
            self._proc.stdin.write(b"PING\n")
        except OSError:
            return False
        reply = self._reader.readline(1.0)
        if reply is None:
            self._proc.kill()
            self._proc.wait()
            self._ready = False
            return False
        return True


@tla.machine(state={"_ready": tla.Kind.BOOL})
class RecursiveRunner:
    """retry() calls itself with no guard."""

    def __init__(self):
        self._ready = False

    @tla.entry(tla.Phase.SETUP)
    def start(self):
        self._ready = True
        return True

    @tla.entry()
    def retry(self):
        return self.retry()


@tla.machine(state={"_ready": tla.Kind.BOOL})
class LoopRunner:
    """Loops are outside the subset."""

    def __init__(self):
        self._ready = False

    @tla.entry(tla.Phase.SETUP)
    def start(self):
        while not self._ready:
            self._ready = True
        return True


@tla.machine(state={"_ready": tla.Kind.BOOL})
class UnmodeledRunner:
    """The shell call is neither modeled nor ignored."""

    def __init__(self):
        self._ready = False

    @tla.entry(tla.Phase.SETUP)
    def start(self):
        os.system("true")
        self._ready = True
        return True


_JOB_STATE = {"_proc": tla.Kind.PROC, "_out": tla.Kind.FILE}
_JOB_MODEL = {
    "subprocess.Popen": tla.Op.SPAWN,
    "tempfile.TemporaryFile": tla.Op.OPEN_FILE,
    "os.killpg": tla.Op.KILLPG,
}


@tla.machine(state=_JOB_STATE, model=_JOB_MODEL)
class LeakyJob:
    """A Popen failure in __init__ leaves the output file open."""

    def __init__(self):
        self._out = tempfile.TemporaryFile()  # noqa: SIM115  (closed by kill/collect)
        self._proc = subprocess.Popen(["true"], stdout=self._out)

    @tla.entry()
    def kill(self):
        if self._proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self._proc.pid, signal.SIGKILL)
            self._proc.wait()
        self._out.close()


@tla.machine(state=_JOB_STATE, model=_JOB_MODEL)
class CleanJob:
    """LeakyJob that closes the file when Popen raises."""

    def __init__(self):
        self._out = tempfile.TemporaryFile()  # noqa: SIM115  (closed by kill/collect)
        try:
            self._proc = subprocess.Popen(["true"], stdout=self._out)
        except OSError:
            self._out.close()
            raise

    @tla.entry()
    def kill(self):
        if self._proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self._proc.pid, signal.SIGKILL)
            self._proc.wait()
        self._out.close()


@tla.machine(state={"_out": tla.Kind.FILE}, model={"tempfile.TemporaryFile": tla.Op.OPEN_FILE})
class RereadJob:
    """read() after close(): the second call raises ValueError."""

    def __init__(self):
        self._out = tempfile.TemporaryFile()  # noqa: SIM115  (closed by kill/collect)

    @tla.entry()
    def collect(self):
        self._out.seek(0)
        self._out.close()


_HOLD_STATE = {"_senders": tla.Kind.PROCSET}
_HOLD_MODEL = {"subprocess.Popen": tla.Op.SPAWN, "_Sender": tla.Op.WRAP}


@tla.machine(state=_HOLD_STATE, model=_HOLD_MODEL, ignore=("self._lock", "threading.Lock"))
class WriteFirstNotifier:
    """The write raises before the proc is held: nobody ever reaps it."""

    def __init__(self):
        self._senders = []
        self._lock = threading.Lock()

    @tla.entry(raises=("OSError",))  # the caller catches OSError
    def notify(self):
        proc = subprocess.Popen(["true"], stdin=subprocess.PIPE)
        with self._lock:
            proc.stdin.write(b"msg")
            proc.stdin.close()
        self._senders.append(_Sender(proc))


@tla.machine(state=_HOLD_STATE, model=_HOLD_MODEL, ignore=("self._lock", "threading.Lock"))
class HoldFirstNotifier:
    """WriteFirstNotifier that holds the proc before writing."""

    def __init__(self):
        self._senders = []
        self._lock = threading.Lock()

    @tla.entry(raises=("OSError",))  # the caller catches OSError
    def notify(self):
        proc = subprocess.Popen(["true"], stdin=subprocess.PIPE)
        self._senders.append(_Sender(proc))
        for _ in range(2):
            pass
        with self._lock:
            proc.stdin.write(b"msg")
            proc.stdin.close()


@tla.machine(state=_READER_STATE, model=_READER_MODEL, replies={"RUN": "RC"})
class ShortReadRunner:
    """A short payload read keeps the loader: its tail is read as the next header."""

    def __init__(self):
        self._proc = None
        self._reader = None
        self._ready = False

    @tla.entry(tla.Phase.SETUP)
    def start(self):
        self._proc = subprocess.Popen(["true"])
        self._reader = PipeReader(self._proc.stdout.fileno())
        self._ready = True
        return True

    @tla.entry(requires="self._ready")
    def run(self):
        try:
            self._proc.stdin.write(b"RUN 1\n")
        except OSError:
            return False
        header = self._reader.readline(1.0)
        if header is None:
            self._proc.kill()
            self._proc.wait()
            self._ready = False
            return False
        payload = self._reader.read_exact(4, 1.0)
        return payload is not None


@tla.machine(state=_READER_STATE, model=_READER_MODEL, replies={"RUN": "RC"})
class FullReadRunner:
    """ShortReadRunner that drops the loader after a short payload read."""

    def __init__(self):
        self._proc = None
        self._reader = None
        self._ready = False

    @tla.entry(tla.Phase.SETUP)
    def start(self):
        self._proc = subprocess.Popen(["true"])
        self._reader = PipeReader(self._proc.stdout.fileno())
        self._ready = True
        return True

    @tla.entry(requires="self._ready")
    def run(self):
        try:
            self._proc.stdin.write(b"RUN 1\n")
        except OSError:
            return False
        header = self._reader.readline(1.0)
        if header is None:
            self._proc.kill()
            self._proc.wait()
            self._ready = False
            return False
        payload = self._reader.read_exact(4, 1.0)
        if payload is None:
            self._proc.kill()
            self._proc.wait()
            self._ready = False
            return False
        return True


@tla.machine(state={"_proc": tla.Kind.PROC}, model=_SPAWN)
class EffectLoopRunner:
    """A loop that touches a modeled handle is outside the subset."""

    def __init__(self):
        self._proc = None

    @tla.entry(tla.Phase.SETUP)
    def start(self):
        for _ in range(2):
            self._proc = subprocess.Popen(["true"])
        return True


class _Sender:
    """Stand-in record; the model treats it as its proc (Op.WRAP)."""

    def __init__(self, proc):
        self.proc = proc


class PipeReader:
    """Stand-in so the fixtures import; the extractor never runs it."""

    def __init__(self, fd):
        self.fd = fd

    def readline(self, deadline):
        return None

    def read_exact(self, n, deadline):
        return None
