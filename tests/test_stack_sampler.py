"""SIGPROF stack sampler for ``--profile-mode sample`` (algorithmica §4.4)."""

from __future__ import annotations

import signal
import sys
import threading
import time

import pytest

from fuzzer_tool.adapters.stack_sampler import OVERFLOW, StackSampler


def _frames() -> dict:
    """This thread's frame, keyed as ``sys._current_frames()`` keys it."""
    return {threading.get_ident(): sys._getframe(1)}


def _leaf(s: StackSampler) -> dict:
    s._record(_frames())
    return s


def _outer(s: StackSampler) -> StackSampler:
    return _leaf(s)


def _recurse(s: StackSampler, depth: int) -> StackSampler:
    if depth == 0:
        s._record(_frames())
        return s
    return _recurse(s, depth - 1)


def _name(label: str) -> str:
    return label.split(" ", 1)[0]


class TestRecord:
    def test_leaf_is_self_time(self):
        s = _outer(StackSampler())

        assert s.samples == 1
        assert _name(s.top_self(1)[0][0]) == "_leaf"

    def test_total_counts_every_caller(self):
        s = StackSampler()
        for _ in range(3):
            _outer(s)

        total = {_name(k): v for k, v in s.top_total(50)}
        assert total["_leaf"] == 3
        assert total["_outer"] == 3

    def test_recursion_counted_once_per_sample(self):
        """Falsification: inclusive time cannot exceed the sample count."""
        s = _recurse(StackSampler(), 5)

        total = {_name(k): v for k, v in s.top_total(50)}
        assert total["_recurse"] == 1

    def test_thread_root_kept_out_of_tables(self):
        s = _outer(StackSampler())

        assert all(not k.startswith("thread:") for k, _ in s.top_total(50))

    def test_depth_truncated(self):
        s = _recurse(StackSampler(max_depth=8), 50)

        (stack,) = s._stacks
        assert len(stack) <= 8 + 1  # thread root + frames

    def test_regression_stack_table_bounded(self):
        """Adversarial: distinct stacks past the cap go to one overflow bucket."""
        s = StackSampler(max_stacks=1)
        _outer(s)
        _recurse(s, 1)
        _recurse(s, 2)

        assert len(s._stacks) == 2  # one kept + overflow
        assert s._stacks[OVERFLOW] == 2
        assert s.samples == 3

    def test_write_collapsed(self, tmp_path):
        s = StackSampler()
        _outer(s)
        _outer(s)
        out = tmp_path / "stacks.txt"

        s.write_collapsed(out)

        (line,) = out.read_text().splitlines()
        stack, count = line.rsplit(" ", 1)
        assert count == "2"
        assert stack.startswith("thread:")
        assert _name(stack.split(";")[-1]) == "_leaf"


class TestTimer:
    def test_samples_cpu_and_restores(self):
        before = signal.getsignal(signal.SIGPROF)
        s = StackSampler(interval_s=0.001)

        s.start()
        end = time.process_time() + 0.2
        while time.process_time() < end:
            pass
        s.stop()

        assert s.samples > 0
        assert signal.getitimer(signal.ITIMER_PROF) == (0.0, 0.0)
        assert signal.getsignal(signal.SIGPROF) is before

    def test_double_start_raises(self):
        s = StackSampler()
        s.start()
        try:
            with pytest.raises(RuntimeError):
                s.start()
        finally:
            s.stop()

    def test_stop_without_start_is_noop(self):
        s = StackSampler()

        s.stop()

        assert s.samples == 0

    def test_regression_target_syscall_not_interrupted(self):
        """Adversarial: a direct_lite target's raw libc read must not see EINTR.

        Python retries EINTR itself (PEP 475); C code in the target does not.
        The ticks land while the main thread is blocked in libc ``read``.
        """
        import ctypes
        import os

        libc = ctypes.CDLL(None, use_errno=True)
        rfd, wfd = os.pipe()

        def burn_then_write():
            # Blocked here, the kernel must deliver SIGPROF to the main thread,
            # which sits in read(); otherwise it lands on this busy thread.
            signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGPROF})
            end = time.process_time() + 0.2
            while time.process_time() < end:
                pass
            os.write(wfd, b"x")

        s = StackSampler(interval_s=0.001)
        t = threading.Thread(target=burn_then_write)
        buf = ctypes.create_string_buffer(1)
        s.start()
        try:
            t.start()
            n = libc.read(rfd, buf, 1)
        finally:
            s.stop()
            t.join()
            os.close(rfd)
            os.close(wfd)

        assert n == 1, f"errno={ctypes.get_errno()}"
        assert buf.raw == b"x"


def _tick_from_here(s: StackSampler) -> None:
    s._on_tick(signal.SIGPROF, sys._getframe())


def _burn(seconds: float) -> None:
    end = time.thread_time() + seconds
    while time.thread_time() < end:
        pass


class TestTick:
    def test_regression_handler_frame_not_leaf(self):
        """The interrupted frame is the leaf, not the SIGPROF handler itself."""
        s = StackSampler()
        _burn(0.01)
        _tick_from_here(s)  # baseline
        _burn(0.01)
        _tick_from_here(s)

        leaves = {_name(k) for k, _ in s.top_self(10)}
        assert leaves == {"_tick_from_here"}

    def test_regression_idle_thread_not_sampled(self):
        """Adversarial: a thread blocked in a wait used no CPU and gets no sample."""
        stop = threading.Event()
        idle = threading.Thread(target=stop.wait, name="idle-waiter")
        idle.start()
        try:
            s = StackSampler()
            _tick_from_here(s)  # baseline
            _burn(0.01)
            _tick_from_here(s)
        finally:
            stop.set()
            idle.join()

        roots = {stack[0] for stack in s._stacks}
        assert roots == {"thread:" + threading.main_thread().name}
        assert s.samples == 1

    def test_busy_thread_sampled(self):
        """Falsification: a thread that burned CPU between ticks is sampled."""
        go, done = threading.Event(), threading.Event()

        def worker():
            go.wait()
            _burn(0.02)
            done.set()
            go.clear()
            go.wait()  # parked, but its CPU delta is already recorded

        t = threading.Thread(target=worker, name="busy-worker")
        t.start()
        s = StackSampler()
        try:
            _tick_from_here(s)  # baseline
            go.set()
            done.wait()
            _tick_from_here(s)
        finally:
            go.set()
            t.join()

        roots = {stack[0] for stack in s._stacks}
        assert "thread:busy-worker" in roots
