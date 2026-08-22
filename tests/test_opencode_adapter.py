from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from kiki.agents.models import AgentStartRequest, PermissionProfile, SessionKind
from kiki.agents.opencode import OpenCodeAdapter, resolve_binary

_FAKE = """#!/usr/bin/env python3
import sys
args = sys.argv[1:]
if "--version" in args or args[:1] == ["version"]:
    print("opencode 0.0-test")
    raise SystemExit(0)
if args[:1] == ["run"]:
    print("status: running")
    print("1. Inspect the repo")
    print("2. Write tests")
    raise SystemExit(0)
raise SystemExit(2)
"""


def _fake_bin(tmp_path: Path) -> Path:
    path = tmp_path / "opencode"
    path.write_text(_FAKE, encoding="utf-8")
    path.chmod(0o755)
    return path


def test_availability_and_plan_stream(tmp_path: Path) -> None:
    binary = _fake_bin(tmp_path)
    adapter = OpenCodeAdapter(str(binary))

    async def _run() -> None:
        health = await adapter.check_availability()
        assert health.ok is True
        assert "0.0-test" in (health.version or "")
        workspace = tmp_path / "ws"
        workspace.mkdir()
        request = AgentStartRequest(
            workspace_id="w1",
            workspace_path=str(workspace),
            task="Add a button",
            kind=SessionKind.PLAN,
            permission_profile=PermissionProfile.OBSERVE,
            session_id="s1",
        )
        session = await adapter.start_session(request)
        events = [event async for event in adapter.stream_events(session.id)]
        types = [event.type.value for event in events]
        assert "session_started" in types
        assert "message" in types
        assert "plan" in types
        assert "session_finished" in types

    asyncio.run(_run())


def test_missing_binary() -> None:
    adapter = OpenCodeAdapter("/no/such/opencode")

    async def _run() -> None:
        health = await adapter.check_availability()
        assert health.ok is False

    asyncio.run(_run())


def test_resolve_binary_finds_official_user_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / ".opencode/bin/opencode"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setattr("kiki.agents.opencode.Path.home", lambda: tmp_path)
    monkeypatch.setattr("kiki.agents.opencode.shutil.which", lambda _name: None)

    assert resolve_binary("opencode") == str(binary.resolve())


def test_argv_does_not_use_shell(tmp_path: Path) -> None:
    binary = _fake_bin(tmp_path)
    adapter = OpenCodeAdapter(str(binary))
    assert os.access(adapter.binary_path() or "", os.X_OK)
