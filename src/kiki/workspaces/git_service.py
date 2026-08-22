"""Read-only Git inspection. Fixed argv, reduced environment, no hooks."""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import stat as stat_module
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from kiki.workspaces.models import GitDiff, GitSnapshot, WorkspaceError

log = logging.getLogger(__name__)

_GIT_TIMEOUT_S = 8
_DIFF_TIMEOUT_S = 20
_MAX_REMOTE_LEN = 500
_MAX_DIFF_CHARS = 120_000
_MAX_UNTRACKED_FINGERPRINT_BYTES = 128 * 1024 * 1024


def git_binary() -> str:
    found = shutil.which("git")
    if not found:
        raise WorkspaceError("git_missing", "git ist nicht installiert (Pfad).")
    return found


def git_env() -> dict[str, str]:
    """Minimal environment. Caller secrets are never copied in."""
    return {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "LC_ALL": "C",
        "LANG": "C",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "GIT_PAGER": "cat",
        "GIT_OPTIONAL_LOCKS": "0",
    }


def _argv(repo: Path, *args: str) -> list[str]:
    return [
        git_binary(),
        "--no-pager",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.untrackedCache=false",
        "-C",
        str(repo),
        *args,
    ]


def _run(repo: Path, *args: str, timeout: int = _GIT_TIMEOUT_S) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        _argv(repo, *args),
        check=False,
        capture_output=True,
        text=True,
        env=git_env(),
        timeout=timeout,
    )


def inspect_git(path: Path) -> GitSnapshot:
    """Read branch, HEAD, dirtiness and origin URL. Never mutates the repo."""
    repo = path.resolve(strict=True)
    if not repo.is_dir():
        raise WorkspaceError("not_a_directory", f"Kein Verzeichnis: {repo}")
    try:
        top = _run(repo, "rev-parse", "--show-toplevel")
        branch_p = _run(repo, "symbolic-ref", "--quiet", "--short", "HEAD")
        head_p = _run(repo, "rev-parse", "HEAD")
        porcelain = _run(repo, "status", "--porcelain=v1", "-unormal")
        remote_p = _run(repo, "remote", "get-url", "origin")
    except subprocess.TimeoutExpired as exc:
        raise WorkspaceError("git_timeout", "Git-Inspektion hat das Zeitlimit überschritten.") from exc
    except WorkspaceError:
        raise
    except OSError as exc:
        raise WorkspaceError("git_failed", str(exc)) from exc
    if top.returncode != 0:
        detail = (top.stderr or top.stdout or "kein Repository").strip()
        raise WorkspaceError("not_git", f"{repo} ist kein Git-Repository ({detail[:180]}).")
    toplevel = Path(top.stdout.strip()).resolve(strict=False)

    branch_name = (branch_p.stdout or "").strip() or None
    detached = branch_p.returncode != 0
    if detached:
        branch_name = None
    head = (head_p.stdout or "").strip() or None
    if head_p.returncode != 0:
        head = None

    if porcelain.returncode != 0:
        detail = (porcelain.stderr or porcelain.stdout or "unbekannter Fehler").strip()
        raise WorkspaceError("git_status_failed", f"git status fehlgeschlagen ({detail[:180]}).")
    dirty = False
    untracked = False
    for line in porcelain.stdout.splitlines():
        if not line.strip():
            continue
        dirty = True
        if line.startswith("??"):
            untracked = True

    remote_url = None
    if remote_p.returncode == 0:
        remote_url = sanitize_remote_url(remote_p.stdout or "")

    return GitSnapshot(
        toplevel=str(toplevel),
        branch=branch_name,
        head=head,
        detached=detached,
        dirty=dirty,
        untracked=untracked,
        remote_url=remote_url,
    )


def read_diff(path: Path) -> GitDiff:
    """Working-tree diff against HEAD. Read-only, truncated."""
    repo = path.resolve(strict=True)
    try:
        stat = _run(
            repo,
            "diff",
            "--stat",
            "--no-ext-diff",
            "--no-textconv",
            "HEAD",
            timeout=_DIFF_TIMEOUT_S,
        )
        patch = _run(
            repo,
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "HEAD",
            timeout=_DIFF_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise WorkspaceError("git_timeout", "git diff Timeout.") from exc
    except WorkspaceError:
        raise
    except OSError as exc:
        raise WorkspaceError("git_failed", str(exc)) from exc
    if stat.returncode != 0 or patch.returncode != 0:
        failed = stat if stat.returncode != 0 else patch
        detail = (failed.stderr or failed.stdout or "unbekannter Fehler").strip()
        raise WorkspaceError("git_diff_failed", f"git diff fehlgeschlagen ({detail[:180]}).")
    stat_text = stat.stdout or ""
    patch_text = patch.stdout or ""
    truncated = False
    if len(patch_text) > _MAX_DIFF_CHARS:
        patch_text = patch_text[:_MAX_DIFF_CHARS] + "\n\n[Diff gekürzt]\n"
        truncated = True
    return GitDiff(stat=stat_text[:8000], patch=patch_text, truncated=truncated)


def worktree_fingerprint(path: Path) -> str:
    """Fingerprint tracked changes and untracked contents without hooks."""
    repo = path.resolve(strict=True)
    try:
        status = _run(repo, "status", "--porcelain=v1", "-z", "-unormal")
        untracked = _run(repo, "ls-files", "--others", "--exclude-standard", "-z")
        diff = _run(
            repo,
            "diff",
            "--binary",
            "--no-ext-diff",
            "--no-textconv",
            "HEAD",
            timeout=_DIFF_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise WorkspaceError("git_timeout", "Git-Fingerprint hat das Zeitlimit überschritten.") from exc
    if status.returncode != 0 or untracked.returncode != 0 or diff.returncode != 0:
        failed = status if status.returncode != 0 else (untracked if untracked.returncode != 0 else diff)
        detail = (failed.stderr or failed.stdout or "unbekannter Fehler").strip()
        raise WorkspaceError("git_status_failed", f"Git-Fingerprint fehlgeschlagen ({detail[:180]}).")
    digest = hashlib.sha256()
    digest.update((status.stdout or "").encode("utf-8", errors="replace"))
    digest.update(b"\0")
    digest.update((diff.stdout or "").encode("utf-8", errors="replace"))
    try:
        _hash_untracked_files(digest, repo, untracked.stdout or "")
    except OSError as exc:
        raise WorkspaceError("git_fingerprint_failed", f"Untracked-Datei nicht lesbar: {exc}") from exc
    return digest.hexdigest()


def _hash_untracked_files(digest: Any, repo: Path, listing: str) -> None:
    total_bytes = 0
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    for rel_text in sorted(item for item in listing.split("\0") if item):
        rel = Path(rel_text)
        if rel.is_absolute() or ".." in rel.parts:
            raise OSError(f"unsicherer Git-Pfad: {rel_text}")
        target = repo / rel
        info = target.lstat()
        digest.update(b"\0untracked\0")
        digest.update(rel_text.encode("utf-8", errors="surrogateescape"))
        digest.update(f"\0{info.st_mode:o}\0{info.st_size}\0".encode("ascii"))
        if stat_module.S_ISLNK(info.st_mode):
            digest.update(os.readlink(target).encode("utf-8", errors="surrogateescape"))
            continue
        if not stat_module.S_ISREG(info.st_mode):
            continue
        total_bytes += info.st_size
        if total_bytes > _MAX_UNTRACKED_FINGERPRINT_BYTES:
            raise WorkspaceError(
                "git_fingerprint_too_large",
                "Untracked-Dateien überschreiten 128 MiB; Observe-Prüfung verweigert den Start.",
            )
        fd = os.open(target, os.O_RDONLY | os.O_CLOEXEC | nofollow)
        with os.fdopen(fd, "rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)


def sanitize_remote_url(raw: str) -> str | None:
    """Return display-safe remote metadata without credentials or URL tokens."""
    text = str(raw).strip()
    if not text or any(ord(char) < 32 or ord(char) == 127 for char in text):
        return None
    if "://" in text:
        try:
            parsed = urlsplit(text)
        except ValueError:
            return None
        netloc = parsed.netloc.rsplit("@", 1)[-1]
        cleaned = urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
    else:
        cleaned = text.split("?", 1)[0].split("#", 1)[0]
        if "@" in cleaned:
            _userinfo, suffix = cleaned.rsplit("@", 1)
            if ":" in suffix:
                cleaned = suffix
    cleaned = cleaned.strip()
    return cleaned[:_MAX_REMOTE_LEN] or None
