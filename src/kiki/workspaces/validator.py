"""Allowlist, canonical paths, Git-root check. Fail closed on symlink escape."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

from kiki.workspaces.models import VALID_RISK_PROFILES, WorkspaceError

_FORBIDDEN_ROOT_NAMES = frozenset({"/", ""})


def expand_user_path(raw: str) -> Path:
    text = str(raw).strip()
    if not text:
        raise WorkspaceError("empty_path", "Pfad ist leer.")
    if "\x00" in text or "\n" in text:
        raise WorkspaceError("invalid_path", "Pfad enthält unzulässige Zeichen.")
    return Path(text).expanduser()


def canonical_path(path: Path, *, must_exist: bool) -> Path:
    expanded = path if path.is_absolute() else Path.cwd() / path
    try:
        resolved = expanded.resolve(strict=must_exist)
    except (OSError, FileNotFoundError) as exc:
        raise WorkspaceError("not_found", f"Pfad existiert nicht: {expanded}") from exc
    return resolved


def expand_allowed_roots(raw_roots: Sequence[str]) -> tuple[Path, ...]:
    """Resolve configured roots. Missing directories stay as resolved paths."""
    roots: list[Path] = []
    seen: set[str] = set()
    home = Path.home().resolve()
    for raw in raw_roots:
        text = str(raw).strip()
        if not text:
            continue
        try:
            resolved = canonical_path(expand_user_path(text), must_exist=False)
        except WorkspaceError:
            continue
        if str(resolved) in _FORBIDDEN_ROOT_NAMES or resolved == Path("/"):
            raise WorkspaceError("forbidden_root", "allowed_roots darf nicht / sein.")
        if resolved == home:
            raise WorkspaceError(
                "forbidden_root",
                "allowed_roots darf nicht das Home-Verzeichnis selbst sein.",
            )
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            roots.append(resolved)
    return tuple(roots)


def is_within(inner: Path, outer: Path) -> bool:
    try:
        inner.resolve(strict=False).relative_to(outer.resolve(strict=False))
        return True
    except (ValueError, OSError):
        return False


def lexical_is_within(inner: Path, outer: Path) -> bool:
    """Containment without following the final symlink (blocks /tmp/link → jail)."""
    try:
        left = Path(os.path.normpath(str(inner)))
        right = Path(os.path.normpath(str(outer)))
        left.relative_to(right)
        return True
    except (ValueError, OSError):
        return False


def within_allowed_roots(path: Path, roots: Sequence[Path]) -> Path | None:
    resolved = path.resolve(strict=False)
    for root in roots:
        if is_within(resolved, root):
            return root
    return None


def _reject_symlink_escape(user_path: Path, roots: Sequence[Path]) -> None:
    """Deny if any symlink component resolves outside the allowlist."""
    absolute = user_path if user_path.is_absolute() else Path.cwd() / user_path
    acc = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        acc = acc / part
        try:
            if not acc.is_symlink():
                continue
            target = acc.resolve(strict=False)
        except OSError as exc:
            raise WorkspaceError("symlink_escape", f"Symlink nicht auflösbar: {acc}") from exc
        if within_allowed_roots(target, roots) is None:
            raise WorkspaceError(
                "symlink_escape",
                f"Symlink {acc} zeigt aus den erlaubten Roots hinaus nach {target}.",
            )


def validate_risk_profile(value: str) -> str:
    profile = (value or "").strip() or "observe"
    if profile not in VALID_RISK_PROFILES:
        raise WorkspaceError("invalid_profile", f"Unbekanntes Risikoprofil: {profile}")
    return profile


def validate_candidate(
    raw_path: str,
    *,
    allowed_roots: Sequence[str] | Sequence[Path],
    require_git_root: bool = True,
    git_toplevel: str | None = None,
) -> Path:
    """Return the canonical directory if it is a legal coding workspace.

    ``git_toplevel`` is the resolved worktree root from git, if already queried.
    When ``require_git_root`` is true the path must equal that toplevel.
    """
    roots = expand_allowed_roots([str(r) for r in allowed_roots])
    if not roots:
        raise WorkspaceError("no_roots", "Keine erlaubten Workspace-Roots konfiguriert.")

    expanded = expand_user_path(raw_path)
    absolute = expanded if expanded.is_absolute() else Path.cwd() / expanded
    if not any(lexical_is_within(absolute, root) for root in roots):
        raise WorkspaceError(
            "outside_root",
            f"{absolute} liegt lexikalisch in keinem erlaubten Root.",
        )
    _reject_symlink_escape(expanded, roots)
    if not expanded.exists():
        raise WorkspaceError("not_found", f"Pfad existiert nicht: {expanded}")
    resolved = canonical_path(expanded, must_exist=True)
    if not resolved.is_dir():
        raise WorkspaceError("not_a_directory", f"Kein Verzeichnis: {resolved}")
    if within_allowed_roots(resolved, roots) is None:
        raise WorkspaceError(
            "outside_root",
            f"{resolved} liegt in keinem erlaubten Root.",
        )
    if require_git_root:
        if not git_toplevel:
            raise WorkspaceError("not_git", f"{resolved} ist kein Git-Repository.")
        top = Path(git_toplevel).resolve(strict=False)
        if top != resolved:
            raise WorkspaceError(
                "not_repo_root",
                f"{resolved} ist nicht die Repository-Wurzel (gefunden: {top}).",
            )
        if within_allowed_roots(top, roots) is None:
            raise WorkspaceError("symlink_escape", "Git-Toplevel liegt außerhalb der Roots.")
    return resolved


def resolve_inside_workspace(
    raw_path: str,
    workspace: Path,
    *,
    must_be_file: bool = True,
) -> Path:
    """Resolve a user path that must stay inside an already registered workspace."""
    root = workspace.resolve(strict=True)
    if not root.is_dir():
        raise WorkspaceError("not_a_directory", f"Kein Verzeichnis: {root}")
    text = str(raw_path).strip()
    if not text:
        raise WorkspaceError("empty_path", "Pfad ist leer.")
    expanded = expand_user_path(text)
    if not expanded.is_absolute():
        expanded = root / expanded
    if not lexical_is_within(expanded, root):
        raise WorkspaceError("outside_root", f"{expanded} liegt nicht im Workspace.")
    _reject_symlink_escape(expanded, [root])
    resolved = canonical_path(expanded, must_exist=True)
    if not is_within(resolved, root):
        raise WorkspaceError("outside_root", f"{resolved} liegt nicht im Workspace.")
    if must_be_file and not resolved.is_file():
        raise WorkspaceError("not_a_file", f"Keine Datei: {resolved}")
    return resolved
