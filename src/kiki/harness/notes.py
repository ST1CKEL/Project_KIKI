"""`create_note`: the one thing this harness may change on disk.

It creates a new markdown file inside one directory it was given, and that is
the whole of its power. It cannot delete, cannot overwrite, cannot read anything
back, cannot follow a link out of the workspace, and never touches a shell.

The path rules are enforced by construction rather than by inspection: a name is
rebuilt from a small alphabet instead of being checked for bad characters, and
the file is created with `O_CREAT|O_EXCL` so "does not overwrite" is the kernel's
answer, not a race between a check and a write.
"""

from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path
from typing import Any

from kiki.harness.models import ToolResult

MAX_TITLE_CHARS = 80
MAX_CONTENT_CHARS = 4000
SUFFIX = ".md"
# What a stored name may consist of. Everything else is folded away rather than
# rejected, so a perfectly reasonable German title still produces a file.
_ALLOWED = re.compile(r"[^a-z0-9]+")
_UMLAUTS = {"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"}


class NotesWorkspaceError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def slugify(title: str) -> str:
    """A file name built from scratch out of a title.

    Nothing from the input survives except letters and digits, so `..`, `/`, a
    leading dot, a NUL byte or a Windows device name cannot come out of here —
    there is no path for them to travel.
    """
    lowered = title.strip().lower()
    for umlaut, replacement in _UMLAUTS.items():
        lowered = lowered.replace(umlaut, replacement)
    folded = unicodedata.normalize("NFKD", lowered)
    ascii_only = folded.encode("ascii", "ignore").decode("ascii")
    slug = _ALLOWED.sub("-", ascii_only).strip("-")
    return slug[:MAX_TITLE_CHARS]


class NotesWorkspace:
    """One directory, given explicitly, never guessed."""

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    def relative(self, name: str) -> str:
        """What the user is shown: the name inside the workspace, no more."""
        return f"{name}{SUFFIX}"

    def create(self, name: str, content: str) -> str:
        """Write one new note atomically, or fail without leaving a trace.

        The content goes to a temporary file in the same directory and is
        renamed into place, so a crash halfway through leaves either nothing or
        a complete note — never half of one.
        """
        if not name:
            raise NotesWorkspaceError("invalid_arguments")
        target = self._root / f"{name}{SUFFIX}"
        # Belt and braces: even though the name was rebuilt, the resolved path
        # must still sit directly inside the workspace.
        try:
            self._root.mkdir(parents=True, exist_ok=True)
            resolved_root = self._root.resolve(strict=True)
        except OSError as exc:
            raise NotesWorkspaceError("tool_unavailable") from exc
        if target.parent.resolve() != resolved_root:
            raise NotesWorkspaceError("invalid_arguments")
        if target.exists() or target.is_symlink():
            # is_symlink first: a dangling link is not "exists" but must still
            # never be written through.
            raise NotesWorkspaceError("note_exists")

        temporary = self._root / f".{name}{SUFFIX}.part"
        try:
            handle = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            raise NotesWorkspaceError("note_exists") from exc
        except OSError as exc:
            raise NotesWorkspaceError("tool_unavailable") from exc
        try:
            with open(handle, "w", encoding="utf-8") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            # O_EXCL on the final name: the kernel decides whether it existed,
            # not a check we made a moment ago.
            os.link(temporary, target)
        except FileExistsError as exc:
            _unlink(temporary)
            raise NotesWorkspaceError("note_exists") from exc
        except OSError as exc:
            _unlink(temporary)
            raise NotesWorkspaceError("tool_failed") from exc
        _unlink(temporary)
        return self.relative(name)


def _unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def spec(workspace: NotesWorkspace) -> Any:
    """`create_note` as the production registry defines tools.

    WRITE, so the policy asks for a confirmation on every autonomy level below
    jarvis — the harness no longer decides that for itself.
    """
    from kiki.tools.policy import RiskLevel
    from kiki.tools.registry import ToolSpec

    tool = CreateNoteTool(workspace)

    def _handler(params: dict[str, Any]) -> dict[str, Any]:
        name, content = tool.normalise(params)
        return {"created": True, "note": workspace.create(name, content)}

    return ToolSpec(
        name=CreateNoteTool.name,
        title="Notiz anlegen",
        description=CreateNoteTool.description,
        risk=RiskLevel.WRITE,
        parameters=dict(CreateNoteTool.input_schema),
        handler=_handler,
        effect="Legt eine neue Notiz im KIKI-Notizbereich an.",
        auto_allow=True,
        requires_integration=False,
        model_callable=True,
        # The note text is the payload; it has no business in a security log.
        sensitive_parameters=("content",),
        audit_parameters=("title",),
    )


class CreateNoteTool:
    """`create_note`. Proposes; the harness will not run it unconfirmed."""

    name = "create_note"
    description = (
        "Legt eine neue Notiz als Markdown-Datei im KIKI-Notizbereich an. "
        "Überschreibt nichts und braucht eine Bestätigung."
    )
    read_only = False
    confirmation_required = True
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["title", "content"],
        "additionalProperties": False,
    }

    def __init__(self, workspace: NotesWorkspace) -> None:
        self._workspace = workspace

    def preview(self, arguments: dict[str, Any]) -> tuple[str, str]:
        """Target and content, exactly as they will be written.

        The target is workspace-relative on purpose: a dialog, a screenshot of
        one and a trace all get the same string, and none of them learns where
        the workspace lives.
        """
        name, content = self._normalise(arguments)
        return self._workspace.relative(name), content

    def normalise(self, arguments: dict[str, Any]) -> tuple[str, str]:
        """Validated title and content, or a category. Shared with the ToolSpec."""
        return self._normalise(arguments)

    def _normalise(self, arguments: dict[str, Any]) -> tuple[str, str]:
        title = arguments.get("title")
        content = arguments.get("content")
        if not isinstance(title, str) or not isinstance(content, str):
            raise NotesWorkspaceError("invalid_arguments")
        name = slugify(title)
        if not name:
            raise NotesWorkspaceError("invalid_arguments")
        if not content.strip() or len(content) > MAX_CONTENT_CHARS:
            raise NotesWorkspaceError("invalid_arguments")
        return name, content

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            name, content = self._normalise(arguments)
            relative = self._workspace.create(name, content)
        except NotesWorkspaceError as exc:
            code = exc.code if exc.code in _CODES else "tool_failed"
            return ToolResult(call_id="", name=self.name, ok=False, error_code=code)
        return ToolResult(
            call_id="",
            name=self.name,
            ok=True,
            data={"created": True, "note": relative},
        )


# Every category this tool is allowed to report. Missing "note_exists" here
# once flattened the overwrite and symlink refusals into a generic failure,
# which hides exactly the two protections that matter most.
_CODES = {"invalid_arguments", "tool_unavailable", "tool_failed", "note_exists"}
