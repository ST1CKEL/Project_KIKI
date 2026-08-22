from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

import pytest

from kiki.runners.local import LocalWorkspaceRunner
from kiki.runners.process import RunnerError, assert_safe_argv, sanitized_env, spawn
from kiki.workspaces.models import Workspace


def test_sanitized_env_drops_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "leaked")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/ssh")
    env = sanitized_env(home="/tmp/ws", extra=dict(os.environ))
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert "OPENAI_API_KEY" not in env
    assert "SSH_AUTH_SOCK" not in env
    assert env["HOME"] == "/tmp/ws"
    assert env["GIT_TERMINAL_PROMPT"] == "0"


def test_free_shell_argv_rejected() -> None:
    with pytest.raises(RunnerError) as exc:
        assert_safe_argv(["/bin/sh", "-c", "rm -rf /"])
    assert exc.value.code == "free_shell"


def test_process_timeout_kills(tmp_path: Path) -> None:
    async def _run() -> None:
        handle = await spawn(["sleep", "30"], cwd=tmp_path, env=sanitized_env(home=str(tmp_path)))
        result = await handle.wait(timeout=0.4)
        assert result.timed_out is True
        assert result.killed is True
        time.sleep(0.1)
        if handle.pid:
            with pytest.raises(OSError):
                os.kill(handle.pid, 0)

    asyncio.run(_run())


def test_process_group_stop(tmp_path: Path) -> None:
    script = tmp_path / "group.py"
    marker = tmp_path / "pgid.txt"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import os, sys, time\n"
        "if '--version' in sys.argv:\n"
        "    print('ok')\n"
        "    raise SystemExit(0)\n"
        f"marker = {str(marker)!r}\n"
        "child = os.fork()\n"
        "if child == 0:\n"
        "    time.sleep(30)\n"
        "    os._exit(0)\n"
        "with open(marker, 'w', encoding='utf-8') as fh:\n"
        "    fh.write(f'{os.getpgrp()} {child}\\n')\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    script.chmod(0o755)

    async def _run() -> None:
        handle = await spawn([str(script)], cwd=tmp_path, env=sanitized_env(home=str(tmp_path)))
        for _ in range(50):
            if marker.is_file():
                break
            await asyncio.sleep(0.05)
        text = marker.read_text(encoding="utf-8").strip()
        _pgid, child = (int(part) for part in text.split())
        handle.stop()
        await handle.wait(timeout=3)
        dead = False
        for _ in range(40):
            try:
                # If process is reaped or does not exist
                os.kill(child, 0)
                # Check if it is a zombie (terminated but not reaped by init)
                status_path = Path(f"/proc/{child}/status")
                if status_path.is_file():
                    content = status_path.read_text(encoding="utf-8")
                    if "State:\tZ (zombie)" in content or "State:\tX" in content:
                        dead = True
                        break
                await asyncio.sleep(0.05)
            except OSError:
                dead = True
                break
        try:
            os.kill(child, 9)
        except OSError:
            pass
        assert dead, f"Child process {child} still alive after killpg"

    asyncio.run(_run())


def test_process_capture_drains_stderr_and_caps_output(tmp_path: Path) -> None:
    script = tmp_path / "output.py"
    script.write_text(
        "import sys, time\n"
        "print('stderr-line', file=sys.stderr, flush=True)\n"
        "time.sleep(0.1)\n"
        "print('stdout-line', flush=True)\n"
        "sys.stdout.write('x' * 200000)\n"
        "sys.stdout.flush()\n",
        encoding="utf-8",
    )

    async def _run() -> None:
        handle = await spawn(
            [sys.executable, str(script)],
            cwd=tmp_path,
            env=sanitized_env(home=str(tmp_path)),
        )
        result = await handle.capture(timeout=5, max_output_chars=1024)
        assert result.exit_code == 0
        assert len(result.output) == 1024
        assert "stdout-line" in result.output
        assert "stderr-line" in result.output
        assert result.output_truncated is True

    asyncio.run(_run())


def test_cancelling_capture_terminates_process(tmp_path: Path) -> None:
    script = tmp_path / "long.py"
    marker = tmp_path / "pid.txt"
    script.write_text(
        "import os, pathlib, time\n"
        f"pathlib.Path({str(marker)!r}).write_text(str(os.getpid()), encoding='utf-8')\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )

    async def _run() -> int:
        handle = await spawn(
            [sys.executable, str(script)],
            cwd=tmp_path,
            env=sanitized_env(home=str(tmp_path)),
        )
        task = asyncio.create_task(handle.capture(timeout=30))
        for _ in range(100):
            if marker.exists():
                break
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return int(marker.read_text(encoding="utf-8"))

    pid = asyncio.run(_run())
    with pytest.raises(OSError):
        os.kill(pid, 0)


def test_unknown_profile() -> None:
    runner = LocalWorkspaceRunner()
    with pytest.raises(RunnerError) as exc:
        runner.profile_argv("rm_rf")
    assert exc.value.code == "unknown_profile"


def test_run_profile_argv_is_fixed() -> None:
    runner = LocalWorkspaceRunner()
    assert runner.profile_argv("python_pytest") == ["python3", "-m", "pytest", "-q"]
    dummy = Workspace(
        id="w",
        display_name="w",
        canonical_path="/tmp",
        remote_url=None,
        active_branch="main",
        git_head=None,
        risk_profile="observe",
        created_at="",
        last_used_at="",
    )
    assert dummy.canonical_path == "/tmp"
