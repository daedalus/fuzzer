"""Small state machines for tests/test_tla_extract.py.

Each pair is a buggy class and its fixed twin: the fix is the control that
must pass before the bug's counterexample means anything (Hard Rule 46).
"""

import collections
import contextlib
import os
import signal
import subprocess
import tempfile
import threading
import time

from fuzzer_tool.adapters import libc_shm
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
    """`match` is outside the subset."""

    def __init__(self):
        self._ready = False

    @tla.entry(tla.Phase.SETUP)
    def start(self):
        match self._ready:
            case _:
                self._ready = True
        return True


_COUNTDOWN = {f: tla.Kind.BOOL for f in ("_w", "_x", "_y", "_z")}


@tla.machine(state=_COUNTDOWN, init=tla.Init.ASSUMED)
class ThreeStepLoop:
    """Exactly _LOOP_UNROLL iterations, then the condition fails: no LoopBound."""

    @tla.entry()
    def drain(self):
        self._x = True
        self._y = True
        self._z = True
        while self._x:
            if self._z:
                self._z = False
            elif self._y:
                self._y = False
            else:
                self._x = False


@tla.machine(state=_COUNTDOWN, init=tla.Init.ASSUMED)
class FourStepLoop:
    """One iteration past the unroll bound: LoopBound."""

    @tla.entry()
    def drain(self):
        self._w = True
        self._x = True
        self._y = True
        self._z = True
        while self._w:
            if self._z:
                self._z = False
            elif self._y:
                self._y = False
            elif self._x:
                self._x = False
            else:
                self._w = False


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


_PID_STATE = {"pid": tla.Kind.PID, "_started": tla.Kind.BOOL}
_CHILD_ONLY = ("os.execve", "os._exit")


@tla.machine(state=_PID_STATE, ignore=_CHILD_ONLY)
class StaleKillRunner:
    """stop() reaps the child, then signals its (possibly reused) pid again."""

    def __init__(self):
        self.pid = None
        self._started = False

    @tla.entry(tla.Phase.SETUP)
    def start(self):
        pid = os.fork()
        if pid == 0:
            os.execve("/bin/true", ["true"], {})
            os._exit(127)
        self.pid = pid
        self._started = True
        return True

    @tla.entry()
    def stop(self):
        if self.pid is not None:
            with contextlib.suppress(ProcessLookupError, ChildProcessError):
                os.kill(self.pid, signal.SIGKILL)
                os.waitpid(self.pid, 0)
        if self.pid is not None:
            with contextlib.suppress(ProcessLookupError):
                os.kill(self.pid, signal.SIGKILL)
        self.pid = None


@tla.machine(state=_PID_STATE, ignore=_CHILD_ONLY)
class ReapForgetRunner:
    """StaleKillRunner that forgets the pid once it is reaped."""

    def __init__(self):
        self.pid = None
        self._started = False

    @tla.entry(tla.Phase.SETUP)
    def start(self):
        pid = os.fork()
        if pid == 0:
            os.execve("/bin/true", ["true"], {})
            os._exit(127)
        self.pid = pid
        self._started = True
        return True

    @tla.entry()
    def stop(self):
        if self.pid is not None:
            with contextlib.suppress(ProcessLookupError, ChildProcessError):
                os.kill(self.pid, signal.SIGKILL)
                os.waitpid(self.pid, 0)
            self.pid = None
        if self.pid is not None:
            with contextlib.suppress(ProcessLookupError):
                os.kill(self.pid, signal.SIGKILL)
        self.pid = None


_POLL_IGNORE = (*_CHILD_ONLY, "time.")


@tla.machine(state=_PID_STATE, ignore=_POLL_IGNORE)
class ForgetStoppedRunner:
    """A child stopped by a foreign signal is forgotten while still alive."""

    def __init__(self):
        self.pid = None
        self._started = False

    @tla.entry(tla.Phase.SETUP)
    def start(self):
        pid = os.fork()
        if pid == 0:
            os._exit(127)
        self.pid = pid
        return True

    @tla.entry(requires="self.pid is not None")
    def poll(self):
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            waited, status = os.waitpid(self.pid, os.WNOHANG | os.WUNTRACED)
            if waited == 0:
                continue
            if os.WIFSTOPPED(status):
                if os.WSTOPSIG(status) in (signal.SIGSTOP, signal.SIGTRAP):
                    os.kill(self.pid, signal.SIGCONT)
                    return True
                self.pid = None
                return False
            self.pid = None
            return False
        return False


@tla.machine(state=_PID_STATE, ignore=_POLL_IGNORE)
class ReapStoppedRunner:
    """ForgetStoppedRunner that kills and reaps a foreign-stopped child first."""

    def __init__(self):
        self.pid = None
        self._started = False

    @tla.entry(tla.Phase.SETUP)
    def start(self):
        pid = os.fork()
        if pid == 0:
            os._exit(127)
        self.pid = pid
        return True

    @tla.entry(requires="self.pid is not None")
    def poll(self):
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            waited, status = os.waitpid(self.pid, os.WNOHANG | os.WUNTRACED)
            if waited == 0:
                continue
            if os.WIFSTOPPED(status):
                if os.WSTOPSIG(status) in (signal.SIGSTOP, signal.SIGTRAP):
                    os.kill(self.pid, signal.SIGCONT)
                    return True
                os.kill(self.pid, signal.SIGKILL)
                os.waitpid(self.pid, 0)
                self.pid = None
                return False
            self.pid = None
            return False
        return False


_TICK_STATE = {"_running": tla.Kind.BOOL, "_prev": tla.Kind.HANDLER, "_ticked": tla.Kind.BOOL}


@tla.machine(state=_TICK_STATE)
class TimerFirstSampler:
    """Arms the timer before installing the handler: a tick hits the default action."""

    def __init__(self):
        self._running = False
        self._prev = None
        self._ticked = False

    @tla.entry(requires="not self._running")
    def start(self):
        signal.setitimer(signal.ITIMER_PROF, 0.01, 0.01)
        self._prev = signal.signal(signal.SIGPROF, self._on_tick)
        self._running = True

    @tla.entry(requires="self._running")
    def stop(self):
        signal.setitimer(signal.ITIMER_PROF, 0)
        if self._prev is not None:
            signal.signal(signal.SIGPROF, self._prev)
        self._running = False

    def _on_tick(self, signum, frame):
        self._ticked = True


@tla.machine(state=_TICK_STATE)
class HandlerFirstSampler:
    """TimerFirstSampler with handler first and timer disarmed before restoring."""

    def __init__(self):
        self._running = False
        self._prev = None
        self._ticked = False

    @tla.entry(requires="not self._running")
    def start(self):
        self._prev = signal.signal(signal.SIGPROF, self._on_tick)
        signal.setitimer(signal.ITIMER_PROF, 0.01, 0.01)
        self._running = True

    @tla.entry(requires="self._running")
    def stop(self):
        signal.setitimer(signal.ITIMER_PROF, 0)
        if self._prev is not None:
            signal.signal(signal.SIGPROF, self._prev)
        self._running = False

    def _on_tick(self, signum, frame):
        self._ticked = True


_ALARM_STATE = {"_handler": tla.Kind.HANDLER, "_old": tla.Kind.HANDLER, "_timed_out": tla.Kind.BOOL}


@tla.machine(state=_ALARM_STATE, ignore=("os.getpid",))  # stands in for the target call
class InstallOnceRunner:
    """hasattr() install-once guard: after stop() sets the field to None it never reinstalls."""

    def __init__(self):
        self._timed_out = False

    @tla.entry()
    def run(self):
        if not hasattr(self, "_handler"):

            def _alarm(signum, frame):
                self._timed_out = True

            self._handler = _alarm
            self._old = signal.signal(signal.SIGALRM, self._handler)
        signal.setitimer(signal.ITIMER_REAL, 1.0)
        try:
            os.getpid()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)

    @tla.entry()
    def stop(self):
        if getattr(self, "_handler", None) is not None:
            if signal.getsignal(signal.SIGALRM) is self._handler and self._old is not None:
                signal.signal(signal.SIGALRM, self._old)
            self._handler = None
            self._old = None


@tla.machine(state=_ALARM_STATE, ignore=("os.getpid",))  # stands in for the target call
class InstallCheckRunner:
    """InstallOnceRunner whose guard also reinstalls after stop()."""

    def __init__(self):
        self._timed_out = False

    @tla.entry()
    def run(self):
        if getattr(self, "_handler", None) is None:

            def _alarm(signum, frame):
                self._timed_out = True

            self._handler = _alarm
            self._old = signal.signal(signal.SIGALRM, self._handler)
        signal.setitimer(signal.ITIMER_REAL, 1.0)
        try:
            os.getpid()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)

    @tla.entry()
    def stop(self):
        if getattr(self, "_handler", None) is not None:
            if signal.getsignal(signal.SIGALRM) is self._handler and self._old is not None:
                signal.signal(signal.SIGALRM, self._old)
            self._handler = None
            self._old = None


@tla.machine(state={"_timed_out": tla.Kind.BOOL})
class NonlocalHandlerRunner:
    """A handler that rebinds an enclosing local is outside the subset."""

    def __init__(self):
        self._timed_out = False

    @tla.entry()
    def run(self):
        hit = False

        def _alarm(signum, frame):
            nonlocal hit
            hit = True

        signal.signal(signal.SIGALRM, _alarm)
        signal.setitimer(signal.ITIMER_REAL, 1.0)


_GATED_STATE = {
    "_is_c": tla.Kind.BOOL,
    "_alarm_handler": tla.Kind.HANDLER,
    "_old_alarm_handler": tla.Kind.HANDLER,
    "_timed_out": tla.Kind.BOOL,
}
_GATED = {"init": tla.Init.ASSUMED, "assume": {"_is_c": True}, "ignore": ("os.getpid",)}


@tla.machine(state=_GATED_STATE, **_GATED)
class GatedOnceRunner:
    """InProcessRunner's shape: install-once handler, gated by _is_c, no way back after stop()."""

    def __init__(self):
        os.system("heavy setup the model never compiles")

    @tla.entry(requires="self._is_c")  # run_one dispatches here only while _is_c
    def run(self):
        if not hasattr(self, "_alarm_handler"):

            def _alarm(signum, frame):
                self._timed_out = True

            self._alarm_handler = _alarm
            self._old_alarm_handler = signal.signal(signal.SIGALRM, self._alarm_handler)
        signal.setitimer(signal.ITIMER_REAL, 1.0)
        try:
            os.getpid()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)

    @tla.entry()
    def stop(self):
        if getattr(self, "_alarm_handler", None) is not None:
            ours = signal.getsignal(signal.SIGALRM) is self._alarm_handler
            if ours and self._old_alarm_handler is not None:
                signal.signal(signal.SIGALRM, self._old_alarm_handler)
            self._alarm_handler = None
            self._old_alarm_handler = None
        self._is_c = False


@tla.machine(state=_GATED_STATE, **_GATED)
class ReusedOnceRunner:
    """GatedOnceRunner plus a reload(): reuse after stop() reaches the stale guard."""

    def __init__(self):
        os.system("heavy setup the model never compiles")

    @tla.entry(requires="self._is_c")  # run_one dispatches here only while _is_c
    def run(self):
        if not hasattr(self, "_alarm_handler"):

            def _alarm(signum, frame):
                self._timed_out = True

            self._alarm_handler = _alarm
            self._old_alarm_handler = signal.signal(signal.SIGALRM, self._alarm_handler)
        signal.setitimer(signal.ITIMER_REAL, 1.0)
        try:
            os.getpid()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)

    @tla.entry()
    def stop(self):
        if getattr(self, "_alarm_handler", None) is not None:
            ours = signal.getsignal(signal.SIGALRM) is self._alarm_handler
            if ours and self._old_alarm_handler is not None:
                signal.signal(signal.SIGALRM, self._old_alarm_handler)
            self._alarm_handler = None
            self._old_alarm_handler = None
        self._is_c = False

    @tla.entry()
    def reload(self):
        self._is_c = True


_ORACLE_STATE = {
    "_lock": tla.Kind.LOCK,
    "_wake": tla.Kind.COND,
    "_queue": tla.Kind.QUEUE,
    "_busy": tla.Kind.BOOL,
    "_closing": tla.Kind.BOOL,
    "_done": tla.Kind.BOOL,
    "_thread": tla.Kind.THREAD,
}
_ORACLE_MODEL = {"self._cov.measure": tla.Op.MAY_FAIL}


@tla.machine(state=_ORACLE_STATE, model=_ORACLE_MODEL)
class LostWakeupOracle:
    """drain() waits on the queue even after the worker exited: it never returns."""

    def __init__(self):
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._queue = collections.deque()
        self._busy = False
        self._closing = False
        self._done = False
        self._fails = 0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    @tla.entry()
    def submit(self):
        with self._lock:
            self._queue.append(1)
            self._wake.notify()

    @tla.entry()
    def close(self):
        with self._lock:
            self._closing = True
            self._wake.notify_all()
        self._thread.join()

    @tla.entry()
    def drain(self):
        with self._lock:
            while self._queue or self._busy:
                self._wake.wait()

    def _loop(self):
        try:
            while True:
                with self._lock:
                    while not self._queue and not self._closing:
                        self._wake.wait()
                    if self._closing:
                        return
                    self._queue.popleft()
                    self._busy = True
                try:
                    self._cov.measure()
                except RuntimeError:
                    self._fails += 1
                with self._lock:
                    self._busy = False
                    self._wake.notify_all()
        finally:
            with self._lock:
                self._done = True
                self._wake.notify_all()


@tla.machine(state=_ORACLE_STATE, model=_ORACLE_MODEL)
class FixedOracle:
    """LostWakeupOracle whose drain() also stops once the worker is done."""

    def __init__(self):
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._queue = collections.deque()
        self._busy = False
        self._closing = False
        self._done = False
        self._fails = 0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    @tla.entry()
    def submit(self):
        with self._lock:
            self._queue.append(1)
            self._wake.notify()

    @tla.entry()
    def close(self):
        with self._lock:
            self._closing = True
            self._wake.notify_all()
        self._thread.join()

    @tla.entry()
    def drain(self):
        with self._lock:
            while (self._queue or self._busy) and not self._done:
                self._wake.wait()

    def _loop(self):
        try:
            while True:
                with self._lock:
                    while not self._queue and not self._closing:
                        self._wake.wait()
                    if self._closing:
                        return
                    self._queue.popleft()
                    self._busy = True
                try:
                    self._cov.measure()
                except RuntimeError:
                    self._fails += 1
                with self._lock:
                    self._busy = False
                    self._wake.notify_all()
        finally:
            with self._lock:
                self._done = True
                self._wake.notify_all()


@tla.machine(state=_ORACLE_STATE, model=_ORACLE_MODEL)
class NestedBlockingOracle:
    """A blocking method called inside a larger expression is outside the subset."""

    def __init__(self):
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._queue = collections.deque()
        self._busy = False
        self._closing = False
        self._done = False
        self._fails = 0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    @tla.entry()
    def submit(self):
        with self._lock:
            self._queue.append(1)
            self._wake.notify()

    @tla.entry()
    def close(self):
        with self._lock:
            self._closing = True
            self._wake.notify_all()
        self._thread.join()

    @tla.entry()
    def drain(self):
        return 1 + self._wait_empty()

    def _wait_empty(self):
        with self._lock:
            while self._queue:
                self._wake.wait()
        return 0

    def _loop(self):
        try:
            while True:
                with self._lock:
                    while not self._queue and not self._closing:
                        self._wake.wait()
                    if self._closing:
                        return
                    self._queue.popleft()
                    self._busy = True
                try:
                    self._cov.measure()
                except RuntimeError:
                    self._fails += 1
                with self._lock:
                    self._busy = False
                    self._wake.notify_all()
        finally:
            with self._lock:
                self._done = True
                self._wake.notify_all()


def _reap_child(pid):
    """Module-level helper: the extractor inlines it."""
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, signal.SIGKILL)
    with contextlib.suppress(ChildProcessError):
        os.waitpid(pid, 0)


@tla.machine(state={}, ignore=_CHILD_ONLY)
class FalseExitRunner:
    """_ptrace_settle's shape: (0, 0) from WNOHANG reads as a clean exit; the child leaks."""

    @tla.entry(raises=("OSError",))  # fork failure is the caller's
    def run(self):
        pid = os.fork()
        if pid == 0:
            os._exit(0)
        last = None
        status = 0
        while True:
            waited, status = os.waitpid(pid, os.WNOHANG | os.WUNTRACED)
            if waited == 0:
                continue
            if os.WIFEXITED(status) or os.WIFSIGNALED(status):
                return 0
            last = "cont"
            os.kill(pid, signal.SIGCONT)
            break
        if last == "cont":
            waited, status = os.waitpid(pid, os.WNOHANG)
            if waited == 0:
                last = None  # nothing to do: status 0 reads as exit 0
        if os.WIFEXITED(status) or os.WIFSIGNALED(status):
            return 0  # reaped
        _reap_child(pid)
        return -1


@tla.machine(state={}, ignore=_CHILD_ONLY)
class ReapRunningRunner:
    """FalseExitRunner that kills and reaps a child still running after the resume."""

    @tla.entry(raises=("OSError",))  # fork failure is the caller's
    def run(self):
        pid = os.fork()
        if pid == 0:
            os._exit(0)
        last = None
        status = 0
        while True:
            waited, status = os.waitpid(pid, os.WNOHANG | os.WUNTRACED)
            if waited == 0:
                continue
            if os.WIFEXITED(status) or os.WIFSIGNALED(status):
                return 0
            last = "cont"
            os.kill(pid, signal.SIGCONT)
            break
        if last == "cont":
            waited, status = os.waitpid(pid, os.WNOHANG)
            if waited == 0:
                _reap_child(pid)
                return -2
        if os.WIFEXITED(status) or os.WIFSIGNALED(status):
            return 0  # reaped
        _reap_child(pid)
        return -1


@tla.machine(state={}, ignore=_CHILD_ONLY)
class EarlyTupleRunner:
    """_ptrace_first_stop's shape: a (rc, msg) tuple once reaped, None while running."""

    def _first(self, pid):
        waited, _status = os.waitpid(pid, os.WNOHANG)
        if waited == 0:
            return None
        return -1, ""

    @tla.entry(raises=("OSError",))
    def run(self):
        pid = os.fork()
        if pid == 0:
            os._exit(0)
        early = self._first(pid)
        if early is not None:
            return early
        _reap_child(pid)
        return 0


@tla.machine(state={}, ignore=_CHILD_ONLY)
class EarlyNoneLeakRunner:
    """EarlyTupleRunner that returns on None: the running child leaks."""

    def _first(self, pid):
        waited, _status = os.waitpid(pid, os.WNOHANG)
        if waited == 0:
            return None
        return -1, ""

    @tla.entry(raises=("OSError",))
    def run(self):
        pid = os.fork()
        if pid == 0:
            os._exit(0)
        early = self._first(pid)
        if early is None:
            return 0
        return early


_SHM_STATE = {"seg": tla.Kind.SEGMENT, "addr": tla.Kind.SEGMENT}
_SHM_OPS = {
    "libc_shm.shmget": tla.Op.SHM_GET,
    "libc_shm.shmat": tla.Op.SHM_AT,
    "libc_shm.shmdt": tla.Op.SHM_DT,
    "libc_shm.shmctl_rmid": tla.Op.SHM_RMID,
}


def _alloc(size):
    """_alloc_segment's shape: a failed attach removes the fresh segment."""
    seg = libc_shm.shmget(size)
    if seg is None:
        raise OSError("shmget")
    addr = libc_shm.shmat(seg)
    if addr is None:
        libc_shm.shmctl_rmid(seg)
        raise OSError("shmat")
    return seg, addr


@tla.machine(state=_SHM_STATE, model=_SHM_OPS)
class CleanResize:
    """ShmCoverage.resize's order: allocate new, detach + remove old, rebind."""

    def __init__(self):
        self.seg, self.addr = _alloc(1)

    @tla.entry(raises=("OSError",))
    def resize(self):
        new_seg, new_addr = _alloc(2)
        libc_shm.shmdt(self.addr)
        libc_shm.shmctl_rmid(self.seg)
        self.seg = new_seg
        self.addr = new_addr


@tla.machine(state=_SHM_STATE, model=_SHM_OPS)
class LeakyResize:
    """Detaches the old segment but never removes it: it outlives the process."""

    def __init__(self):
        self.seg, self.addr = _alloc(1)

    @tla.entry(raises=("OSError",))
    def resize(self):
        new_seg, new_addr = _alloc(2)
        libc_shm.shmdt(self.addr)
        self.seg = new_seg
        self.addr = new_addr


@tla.machine(state=_SHM_STATE, model=_SHM_OPS)
class DanglingResize:
    """Frees the old segment but keeps its address: later views read freed memory."""

    def __init__(self):
        self.seg, self.addr = _alloc(1)

    @tla.entry(raises=("OSError",))
    def resize(self):
        new_seg, _new_addr = _alloc(2)
        libc_shm.shmdt(self.addr)
        libc_shm.shmctl_rmid(self.seg)
        self.seg = new_seg


class _TickBase:
    """Shared start/stop for the havoc-dedup fixtures; tick() differs."""

    def __init__(self):
        self._proc = None
        self._ready = False
        self._times = []
        self._eps = 0.0

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


@tla.machine(state=_PROC_STATE, model=_SPAWN, ignore=("self._times.pop",))
class InertHavocRunner(_TickBase):
    """tick()'s havoc branches touch only unmodeled state: one path, not four."""

    @tla.entry()
    def tick(self):
        if len(self._times) > 4:
            self._times.pop(0)
        if self._eps == 0.0:
            avg = sum(self._times)
            if avg > 0:
                self._eps = 1.0 / avg
        return self._ready


@tla.machine(state=_PROC_STATE, model=_SPAWN)
class BareTickRunner(_TickBase):
    """InertHavocRunner without the inert branches: the same reachable states."""

    @tla.entry()
    def tick(self):
        return self._ready


@tla.machine(state=_PROC_STATE, model=_SPAWN)
class HavocWriteRunner(_TickBase):
    """The havoc branch writes a modeled field: both branches stay."""

    @tla.entry()
    def tick(self):
        if len(self._times) > 4:
            self._ready = False
        return self._ready


@tla.machine(state=_PROC_STATE, model=_SPAWN)
class HavocLocalRunner(_TickBase):
    """The havoc branch sets a modeled local read later: both branches stay."""

    @tla.entry()
    def tick(self):
        ok = True
        if len(self._times) > 4:
            ok = False
        if not ok:
            self._ready = False
        return ok


@tla.machine(state=_PROC_STATE, model=_SPAWN)
class HavocReturnRunner(_TickBase):
    """The havoc branch returns early: both branches stay."""

    @tla.entry()
    def tick(self):
        if len(self._times) > 4:
            return False
        self._ready = False
        return True


@tla.machine(state={"_proc": tla.Kind.PROC}, model=_SPAWN, init=tla.Init.ASSUMED)
class CallerLocalRunner:
    """A local read after an inlined call and an `if` keeps its value (not havoc)."""

    @tla.entry()
    def start(self):
        if self._proc is not None:
            return
        proc = self._spawn()
        if proc is not None:
            self._proc = proc

    @tla.entry()
    def stop(self):
        if self._proc is not None:
            self._proc.kill()
            self._proc.wait()
        self._proc = None

    def _spawn(self):
        with contextlib.suppress(OSError):
            return subprocess.Popen(["true"])
        return None


@tla.machine(state={"_proc": tla.Kind.PROC}, model=_SPAWN, init=tla.Init.ASSUMED)
class CallerLeakRunner:
    """CallerLocalRunner with the test inverted: the started proc is dropped."""

    @tla.entry()
    def start(self):
        if self._proc is not None:
            return
        proc = self._spawn()
        if proc is None:
            self._proc = proc

    def _spawn(self):
        with contextlib.suppress(OSError):
            return subprocess.Popen(["true"])
        return None


# CrashGdbQueue's shape: one live job (tuple of job and item), a deque behind it.
# `job.wait` is GdbJob.wait, bounded by the job's own deadline.
_GDBQ_SPEC = {
    "state": {"_live": tla.Kind.PROC, "_queue": tla.Kind.QUEUE},
    "model": {
        "subprocess.Popen": tla.Op.SPAWN,
        "job.wait": tla.Op.BOUNDED_WAIT,
        "_refresh": tla.Op.MAY_FAIL,
    },
    "init": tla.Init.ASSUMED,
}


def _refresh(item):
    """Stand-in report refresh; may raise a non-OSError (Op.MAY_FAIL)."""


@tla.machine(**_GDBQ_SPEC)
class CleanGdbQueue:
    """Publish failures of any class are contained; close() drains every job."""

    @tla.entry()
    def enqueue(self, item):
        self._queue.append(item)
        if self._live is None:
            self._start_next()

    @tla.entry()
    def tick(self):
        if self._live is None:
            return
        job, item = self._live
        if job.poll() is None:
            return
        self._publish(item)
        self._start_next()

    @tla.entry()
    def close(self):
        while self._live is not None:
            job, item = self._live
            job.wait()
            self._publish(item)
            self._start_next()

    def _start_next(self):
        self._live = None
        while self._queue:
            item = self._queue.popleft()
            with contextlib.suppress(OSError):
                job = subprocess.Popen(["gdb"])
                self._live = (job, item)
                return

    def _publish(self, item):
        with contextlib.suppress(Exception):
            _refresh(item)


@tla.machine(**_GDBQ_SPEC)
class OsErrorOnlyGdbQueue:
    """CrashGdbQueue._publish before the fix: only OSError is caught, so a failed
    refresh escapes tick() and stops close() draining."""

    @tla.entry()
    def enqueue(self, item):
        self._queue.append(item)
        if self._live is None:
            self._start_next()

    @tla.entry()
    def tick(self):
        if self._live is None:
            return
        job, item = self._live
        if job.poll() is None:
            return
        self._publish(item)
        self._start_next()

    @tla.entry()
    def close(self):
        while self._live is not None:
            job, item = self._live
            job.wait()
            self._publish(item)
            self._start_next()

    def _start_next(self):
        self._live = None
        while self._queue:
            item = self._queue.popleft()
            with contextlib.suppress(OSError):
                job = subprocess.Popen(["gdb"])
                self._live = (job, item)
                return

    def _publish(self, item):
        with contextlib.suppress(OSError):
            _refresh(item)


@tla.machine(**_GDBQ_SPEC)
class NoWaitGdbQueue:
    """close() publishes and moves on without waiting: the live gdb is dropped."""

    @tla.entry()
    def enqueue(self, item):
        self._queue.append(item)
        if self._live is None:
            self._start_next()

    @tla.entry()
    def close(self):
        while self._live is not None:
            _job, item = self._live
            self._publish(item)
            self._start_next()

    def _start_next(self):
        self._live = None
        while self._queue:
            item = self._queue.popleft()
            with contextlib.suppress(OSError):
                job = subprocess.Popen(["gdb"])
                self._live = (job, item)
                return

    def _publish(self, item):
        with contextlib.suppress(Exception):
            _refresh(item)


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
