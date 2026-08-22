"""OpenCode adapter. argv is a fixed template; task text is one argument."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

from kiki.agents.models import (
    AgentAvailability,
    AgentError,
    AgentEvent,
    AgentEventType,
    AgentSession,
    AgentSessionStatus,
    AgentStartRequest,
    SessionKind,
    parse_plan_text,
)
from kiki.runners.process import ProcessHandle, RunnerError, sanitized_env, spawn

log = logging.getLogger(__name__)

_MODEL_RE = re.compile(r"^[A-Za-z0-9_./:+-]+$")
_PLAN_PREFIX = (
    "Create a plan only. Do not modify files. Do not run commands that change state. "
    "List steps, likely files, tests, risks, and open questions.\n\nTask:\n"
)
_IMPLEMENT_PREFIX = (
    "Implement the approved task. Stay inside this Git repository. "
    "Do not use sudo, do not push, do not access files outside the workspace.\n\nTask:\n"
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def resolve_binary(configured: str) -> str | None:
    text = (configured or "opencode").strip() or "opencode"
    if "/" in text:
        path = Path(text).expanduser()
        if path.is_file() and os_access_ok(path):
            return str(path.resolve())
        return None
    found = shutil.which(text)
    if found:
        return found
    if text == "opencode":
        for candidate in (
            Path.home() / ".local/bin/opencode",
            Path.home() / ".opencode/bin/opencode",
        ):
            if os_access_ok(candidate):
                return str(candidate.resolve())
    return None


def os_access_ok(path: Path) -> bool:
    return path.is_file() and os_executable(path)


def os_executable(path: Path) -> bool:
    import os

    return os.access(path, os.X_OK)


class OpenCodeAdapter:
    name = "opencode"

    def __init__(self, binary: str = "opencode", *, stop_grace_seconds: float = 2.0) -> None:
        self._configured = binary
        self._stop_grace_seconds = max(0.05, float(stop_grace_seconds))
        self._live: dict[str, ProcessHandle] = {}
        self._meta: dict[str, AgentSession] = {}

    def binary_path(self) -> str | None:
        return resolve_binary(self._configured)

    async def check_availability(self) -> AgentAvailability:
        path = self.binary_path()
        if path is None:
            return AgentAvailability(
                ok=False,
                name=self.name,
                binary=self._configured,
                detail="OpenCode-Binary nicht gefunden.",
            )
        cwd = Path("/tmp") if Path("/tmp").is_dir() else Path.cwd()
        version = await self._capture_version(path, cwd, ["--version"])
        if version is None:
            version = await self._capture_version(path, cwd, ["version"])
        if version is None:
            return AgentAvailability(
                ok=False,
                name=self.name,
                binary=path,
                detail="OpenCode --version schlug fehl.",
            )
        return AgentAvailability(
            ok=True,
            name=self.name,
            binary=path,
            version=version,
            detail="OpenCode verfügbar.",
        )

    async def _capture_version(self, path: str, cwd: Path, args: list[str]) -> str | None:
        handle = None
        try:
            handle = await spawn([path, *args], cwd=cwd, env=sanitized_env())

            async def _read() -> tuple[list[str], int | None]:
                chunks: list[str] = []
                async for _, line in handle.iter_lines():
                    if line.strip():
                        chunks.append(line.strip())
                result = await handle.wait()
                return chunks, result.exit_code

            chunks, code = await asyncio.wait_for(_read(), timeout=8.0)
        except (RunnerError, TimeoutError, OSError):
            if handle is not None:
                handle.stop(force=True)
            return None
        if code not in {0, None}:
            return None
        return " ".join(chunks)[:80] or "ok"

    async def start_session(self, request: AgentStartRequest) -> AgentSession:
        path = self.binary_path()
        if path is None:
            raise AgentError("not_available", "OpenCode ist nicht installiert.")
        workspace = Path(request.workspace_path)
        if not workspace.is_dir():
            raise AgentError("not_a_directory", f"Workspace fehlt: {workspace}")
        task = request.task.strip()
        if not task:
            raise AgentError("empty_task", "Aufgabe ist leer.")
        prefix = _PLAN_PREFIX if request.kind is SessionKind.PLAN else _IMPLEMENT_PREFIX
        argv = [path, "run"]
        model = (request.model or "").strip()
        if model:
            if not _MODEL_RE.fullmatch(model):
                raise AgentError("invalid_model", "Modellname enthält unzulässige Zeichen.")
            argv.extend(["-m", model])
        argv.append(prefix + task)
        env = sanitized_env(home=str(workspace))
        handle = await spawn(argv, cwd=workspace, env=env)
        session = AgentSession(
            id=request.session_id,
            workspace_id=request.workspace_id,
            agent_name=self.name,
            agent_version=None,
            model_name=model or None,
            task_text=task,
            status=AgentSessionStatus.RUNNING,
            permission_profile=request.permission_profile.value,
            kind=request.kind,
            git_branch_before=None,
            git_head_before=None,
            started_at=_now(),
        )
        self._live[session.id] = handle
        self._meta[session.id] = session
        return session

    async def stream_events(self, session_id: str) -> AsyncIterator[AgentEvent]:
        handle = self._live.get(session_id)
        session = self._meta.get(session_id)
        if handle is None or session is None:
            yield AgentEvent(type=AgentEventType.ERROR, text="Unbekannte Session.", ts=_now())
            return
        yield AgentEvent(type=AgentEventType.SESSION_STARTED, text="OpenCode gestartet.", ts=_now())
        yield AgentEvent(
            type=AgentEventType.STATUS_CHANGED,
            text=AgentSessionStatus.RUNNING.value,
            ts=_now(),
        )
        collected: list[str] = []
        async for stream, line in handle.iter_lines():
            event = _normalize_line(stream, line)
            if event.type is AgentEventType.MESSAGE and event.text:
                collected.append(event.text)
            yield event
        result = await handle.wait()
        if session.kind is SessionKind.PLAN and collected:
            raw = "\n".join(collected)
            plan = parse_plan_text(raw)
            yield AgentEvent(type=AgentEventType.PLAN, text=plan.raw, data={"steps": list(plan.steps)}, ts=_now())
        if result.timed_out:
            yield AgentEvent(type=AgentEventType.ERROR, text="Timeout — Prozessgruppe beendet.", ts=_now())
            status = AgentSessionStatus.FAILED
        elif result.killed:
            yield AgentEvent(type=AgentEventType.STATUS_CHANGED, text="stopped", ts=_now())
            status = AgentSessionStatus.FAILED
        elif result.exit_code not in {0, None}:
            yield AgentEvent(
                type=AgentEventType.ERROR,
                text=f"OpenCode exit {result.exit_code}",
                ts=_now(),
            )
            status = AgentSessionStatus.FAILED
        else:
            status = AgentSessionStatus.FINISHED
        session.status = status
        session.exit_code = result.exit_code
        session.finished_at = _now()
        yield AgentEvent(
            type=AgentEventType.SESSION_FINISHED,
            text=status.value,
            data={"exit_code": result.exit_code, "timed_out": result.timed_out},
            ts=_now(),
        )
        self._live.pop(session_id, None)
        self._meta.pop(session_id, None)

    async def stop_session(self, session_id: str) -> None:
        handle = self._live.get(session_id)
        if handle is None:
            return
        await handle.terminate(grace_seconds=self._stop_grace_seconds)


def _normalize_line(stream: str, line: str) -> AgentEvent:
    text = line.rstrip()
    ts = _now()
    if stream == "stderr" and text:
        lowered = text.lower()
        if "error" in lowered or "fatal" in lowered:
            return AgentEvent(type=AgentEventType.ERROR, text=text[:2000], ts=ts)
        return AgentEvent(type=AgentEventType.MESSAGE, text=text[:2000], data={"stream": "stderr"}, ts=ts)
    if not text:
        return AgentEvent(type=AgentEventType.MESSAGE, text="", ts=ts)
    if text[:1] in "{[":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            kind = str(payload.get("type") or payload.get("event") or "message").lower()
            body = str(payload.get("text") or payload.get("message") or payload.get("part") or text)
            mapping = {
                "plan": AgentEventType.PLAN,
                "error": AgentEventType.ERROR,
                "tool": AgentEventType.TOOL_REQUEST,
                "tool_request": AgentEventType.TOOL_REQUEST,
                "file": AgentEventType.FILE_CHANGE,
                "diff": AgentEventType.DIFF_AVAILABLE,
                "test": AgentEventType.TEST_OUTPUT,
            }
            return AgentEvent(type=mapping.get(kind, AgentEventType.MESSAGE), text=body[:2000], data=payload, ts=ts)
    return AgentEvent(type=AgentEventType.MESSAGE, text=text[:2000], ts=ts)
