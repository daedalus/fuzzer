"""Tests for adapters/persistent.py — PersistentRunner init and error paths."""

from fuzzer_tool.adapters.persistent import PersistentRunner


class TestPersistentRunner:
    def test_init(self):
        pr = PersistentRunner(target="/nonexistent", timeout=1)
        assert pr.target == "/nonexistent"
        assert pr.timeout == 1
        assert not pr._started

    def test_init_defaults(self):
        pr = PersistentRunner(target="/bin/true")
        assert pr.timeout == 5
        assert pr.HEADER_SIZE == 8

    def test_start_nonexistent_target(self):
        pr = PersistentRunner(target="/nonexistent/binary", timeout=1)
        result = pr.start()
        assert result is False

    def test_stop_when_not_running(self):
        pr = PersistentRunner(target="/nonexistent")
        pr.stop()  # should not raise

    def test_run_one_not_started(self):
        pr = PersistentRunner(target="/nonexistent")
        rc, data = pr.run_one(b"test")
        assert rc == -2
