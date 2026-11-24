"""run_target_file must not write predictable input paths (2026-11-24 review).

The target input went to ``tmp_dir/fuzz_<pid>`` — a fixed name, while every
other temp path in the tree is mkstemp/mkdtemp. Safe in the current
single-caller, private-directory usage, but one shared-directory refactor
away from a symlink race. mkstemp names are unique per call.
"""

import os

from fuzzer_tool.adapters import process


def test_temp_input_names_are_unique_and_unpredictable(monkeypatch, tmp_path):
    seen: list[str] = []

    def fake_spawn(cmd, _stdin, _timeout, _env, _sessions):
        seen.append(cmd[-1])  # no target_args: argv is [target, path]
        return (0, "", 0)

    monkeypatch.setattr(process, "_spawn_capture", fake_spawn)

    for _ in range(2):
        rc, err, _pid = process.run_target_file("/bin/true", b"x", 1.0, str(tmp_path), None)
        assert (rc, err) == (0, "")

    # Not the old fixed name, and unique across calls.
    assert f"fuzz_{os.getpid()}" not in seen
    assert seen[0] != seen[1], "two execs reused one input file name"
    # Each path lives inside the caller's private dir, and the files are
    # unlinked once the exec is over.
    for p in seen:
        assert os.path.dirname(p) == str(tmp_path)
    assert not list(tmp_path.iterdir()), "input files must be unlinked after the exec"
