"""Small state machines for tests/test_tla_extract.py.

Each pair is a buggy class and its fixed twin: the fix is the control that
must pass before the bug's counterexample means anything (Hard Rule 46).
"""

import os
import subprocess

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


class PipeReader:
    """Stand-in so the fixtures import; the extractor never runs it."""

    def __init__(self, fd):
        self.fd = fd

    def readline(self, deadline):
        return None
