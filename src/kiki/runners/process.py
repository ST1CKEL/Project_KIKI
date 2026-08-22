"""Process-group spawn with env allowlist, timeout and kill. No shell."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

_SHELL_NAMES = frozenset({"sh", "bash", "zsh", "dash", "fish", "csh", "tcsh", "ksh"})
_SECRET_KEYS = frozenset(
    {
        "SSH_AUTH_SOCK",
        "SSH_AGENT_PID",
        "GPG_AGENT_INFO",
        "GNUPGHOME",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_ACCESS_KEY_ID",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GITHUB_TOKEN",
        "GH_TOKEN",
        "XAI_API_KEY",
        "GROK_API_KEY",
    }
)
_SECRET_FRAGMENTS = ("SECRET", "TOKEN", "PASSWORD", "API_KEY", "PRIVATE_KEY", "PASSWD")
_DESKTOP_KEYS = (
    "WAYLAND_DISPLAY",
    "DISPLAY",
    "XDG_RUNTIME_DIR",
    "XDG_SESSION_TYPE",
    "XDG_CURRENT_DESKTOP",
    "DESKTOP_SESSION",
    "DBUS_SESSION_BUS_ADDRESS",
)


class RunnerError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def sanitized_env(*, home: str | None = None, extra: dict[str, str] | None = None) -> dict[str, str]:
    """Build an environment from scratch. Caller secrets are never copied."""
    env = {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TERM": "dumb",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_PAGER": "cat",
        "NO_COLOR": "1",
    }
    locked = frozenset(env) | {"HOME"}
    if extra:
        for key, value in extra.items():
            if key in locked or _is_secret_key(key):
                continue
            env[key] = value
    if home:
        env["HOME"] = home
    return env


def desktop_env(*, home: str | None = None) -> dict[str, str]:
    """Session bus + compositor vars so xdg-open/terminals work. No secret keys."""
    extra: dict[str, str] = {}
    for key in _DESKTOP_KEYS:
        value = os.environ.get(key)
        if value:
            extra[key] = value
    env = sanitized_env(home=home, extra=extra)
    env["TERM"] = os.environ.get("TERM") or "xterm-256color"
    return env


def _is_secret_key(name: str) -> bool:
    upper = name.upper()
    if upper in _SECRET_KEYS:
        return True
    return any(part in upper for part in _SECRET_FRAGMENTS)


def assert_safe_argv(argv: list[str]) -> list[str]:
    if not argv or not str(argv[0]).strip():
        raise RunnerError("invalid_argv", "Leeres Kommando.")
    if any("\x00" in str(part) for part in argv):
        raise RunnerError("invalid_argv", "NUL im Kommando.")
    name = Path(str(argv[0])).name
    if name in _SHELL_NAMES and "-c" in argv:
        raise RunnerError("free_shell", "Freie Shell (-c) ist verboten.")
    return [str(part) for part in argv]


@dataclass
class ProcessResult:
    exit_code: int | None
    timed_out: bool
    killed: bool
    output: str = ""
    output_truncated: bool = False


class ProcessHandle:
    def __init__(self, proc: asyncio.subprocess.Process, pgid: int | None) -> None:
        self.proc = proc
        self.pgid = pgid
        self._killed = False

    @property
    def pid(self) -> int | None:
        return self.proc.pid

    async def iter_lines(self) -> AsyncIterator[tuple[str, str]]:
        queue: asyncio.Queue[tuple[str, str] | None] = asyncio.Queue()

        async def _pump(stream: asyncio.StreamReader | None, label: str) -> None:
            if stream is None:
                await queue.put(None)
                return
            pending = bytearray()
            while True:
                raw = await stream.read(8192)
                if not raw:
                    break
                pending.extend(raw)
                while True:
                    newline = pending.find(b"\n")
                    if newline < 0:
                        break
                    line = bytes(pending[:newline])
                    del pending[: newline + 1]
                    await queue.put((label, line.decode("utf-8", errors="replace")))
                # A tool may emit a very large JSON/log line. Flush bounded
                # chunks so StreamReader's line limit cannot deadlock capture.
                if len(pending) >= 65_536:
                    await queue.put((label, bytes(pending).decode("utf-8", errors="replace")))
                    pending.clear()
            if pending:
                await queue.put((label, bytes(pending).decode("utf-8", errors="replace")))
            await queue.put(None)

        tasks = [
            asyncio.create_task(_pump(self.proc.stdout, "stdout")),
            asyncio.create_task(_pump(self.proc.stderr, "stderr")),
        ]
        done = 0
        try:
            while done < len(tasks):
                item = await queue.get()
                if item is None:
                    done += 1
                    continue
                yield item
        finally:
            for task in tasks:
                task.cancel()

    async def wait(self, timeout: float | None = None) -> ProcessResult:
        try:
            if timeout is None:
                code = await self.proc.wait()
                return ProcessResult(exit_code=code, timed_out=False, killed=self._killed)
            code = await asyncio.wait_for(self.proc.wait(), timeout=timeout)
            return ProcessResult(exit_code=code, timed_out=False, killed=self._killed)
        except TimeoutError:
            self.stop()
            try:
                await asyncio.wait_for(self.proc.wait(), timeout=2.0)
            except TimeoutError:
                self.stop(force=True)
                await self.proc.wait()
            return ProcessResult(exit_code=self.proc.returncode, timed_out=True, killed=True)

    async def capture(
        self,
        *,
        timeout: float | None = None,
        max_output_chars: int = 65_536,
    ) -> ProcessResult:
        """Drain both pipes while waiting and retain a bounded, readable transcript."""
        limit = max(1, int(max_output_chars))
        chunks: list[str] = []
        size = 0
        truncated = False

        async def _drain() -> None:
            nonlocal size, truncated
            async for stream, line in self.iter_lines():
                rendered = f"[stderr] {line}" if stream == "stderr" else line
                rendered += "\n"
                remaining = limit - size
                if remaining > 0:
                    part = rendered[:remaining]
                    chunks.append(part)
                    size += len(part)
                if len(rendered) > remaining:
                    truncated = True

        drain = asyncio.create_task(_drain())
        try:
            result = await self.wait(timeout=timeout)
            try:
                await asyncio.wait_for(drain, timeout=2.0)
            except TimeoutError:
                drain.cancel()
                truncated = True
        except BaseException:
            await self.terminate()
            raise
        finally:
            if not drain.done():
                drain.cancel()
        return ProcessResult(
            exit_code=result.exit_code,
            timed_out=result.timed_out,
            killed=result.killed,
            output="".join(chunks),
            output_truncated=truncated,
        )

    async def terminate(self, *, grace_seconds: float = 2.0) -> ProcessResult:
        """Stop the whole process group and wait until it is actually gone."""
        self.stop()
        try:
            code = await asyncio.wait_for(
                self.proc.wait(),
                timeout=max(0.05, float(grace_seconds)),
            )
        except TimeoutError:
            self.stop(force=True)
            code = await self.proc.wait()
        return ProcessResult(exit_code=code, timed_out=False, killed=True)

    def stop(self, *, force: bool = False) -> None:
        self._killed = True
        sig = signal.SIGKILL if force else signal.SIGTERM
        if self.pgid is not None and self.pgid > 0:
            try:
                os.killpg(self.pgid, sig)
                return
            except ProcessLookupError:
                return
            except PermissionError:
                log.debug("killpg permission denied, falling back to pid")
        if self.proc.returncode is None and self.proc.pid:
            try:
                self.proc.send_signal(sig)
            except ProcessLookupError:
                return


async def spawn(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
) -> ProcessHandle:
    cleaned = assert_safe_argv(argv)
    if not cwd.is_dir():
        raise RunnerError("not_a_directory", f"cwd ist kein Verzeichnis: {cwd}")
    try:
        proc = await asyncio.create_subprocess_exec(
            *cleaned,
            cwd=str(cwd),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
    except FileNotFoundError as exc:
        raise RunnerError("not_found", f"Binary fehlt: {cleaned[0]}") from exc
    except OSError as exc:
        raise RunnerError("spawn_failed", str(exc)) from exc
    pgid = None
    if proc.pid:
        try:
            pgid = os.getpgid(proc.pid)
        except OSError:
            pgid = proc.pid
    return ProcessHandle(proc, pgid)
