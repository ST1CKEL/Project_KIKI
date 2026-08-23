"""`create_note`: the only thing that touches disk, and everything it refuses."""

from __future__ import annotations

import asyncio
import os

import pytest

from kiki.harness.notes import (
    MAX_CONTENT_CHARS,
    CreateNoteTool,
    NotesWorkspace,
    NotesWorkspaceError,
    slugify,
)


def _tool(tmp_path) -> tuple[CreateNoteTool, NotesWorkspace]:
    workspace = NotesWorkspace(tmp_path / "notes")
    return CreateNoteTool(workspace), workspace


def _run(tool, **arguments):
    return asyncio.run(tool.execute(arguments))


# --- the name is rebuilt, never inspected -----------------------------------


@pytest.mark.parametrize(
    ("title", "slug"),
    [
        ("Milch kaufen", "milch-kaufen"),
        ("Größe prüfen", "groesse-pruefen"),
        ("Straße & Öl", "strasse-oel"),
        ("../../etc/passwd", "etc-passwd"),
        ("/absolut/weg", "absolut-weg"),
        ("..", ""),
        ("....//....", ""),
        ("C:\\Windows\\system32", "c-windows-system32"),
        ("note\x00null", "note-null"),
        ("  ...  ", ""),
        ("CON", "con"),
        # Bounded at MAX_TITLE_CHARS, cut wherever that lands.
        ("Ein sehr " + "langer " * 40 + "Titel", "ein-sehr-" + "langer-" * 10 + "l"),
    ],
)
def test_a_name_is_built_from_scratch(title, slug) -> None:
    """Nothing from the title survives except letters and digits, so traversal,
    absolute paths and NUL bytes have no route out of here."""
    assert slugify(title) == slug


def test_a_name_can_never_escape_the_workspace() -> None:
    for title in ("../../etc/passwd", "/etc/shadow", "..\\..\\win.ini", "a/b/c"):
        slug = slugify(title)
        assert "/" not in slug
        assert "\\" not in slug
        assert not slug.startswith(".")


# --- the happy path ---------------------------------------------------------


def test_a_note_is_written_where_it_was_promised(tmp_path) -> None:
    tool, workspace = _tool(tmp_path)
    target, content = tool.preview({"title": "Milch kaufen", "content": "Milch kaufen"})

    assert target == "milch-kaufen.md"
    assert content == "Milch kaufen"

    result = _run(tool, title="Milch kaufen", content="Milch kaufen")
    assert result.ok is True
    assert result.data == {"created": True, "note": "milch-kaufen.md"}
    written = workspace.root / "milch-kaufen.md"
    assert written.read_text(encoding="utf-8") == "Milch kaufen"


def test_the_preview_is_exactly_what_gets_written(tmp_path) -> None:
    """A user can only approve what they were shown."""
    tool, workspace = _tool(tmp_path)
    arguments = {"title": "Größe prüfen", "content": "Zeile eins\nZeile zwei"}
    target, content = tool.preview(arguments)
    _run(tool, **arguments)

    assert (workspace.root / target).read_text(encoding="utf-8") == content


def test_the_note_is_private(tmp_path) -> None:
    import stat

    tool, workspace = _tool(tmp_path)
    _run(tool, title="Geheim", content="nur für mich")
    mode = stat.S_IMODE((workspace.root / "geheim.md").stat().st_mode)
    assert mode == 0o600


def test_only_the_note_appears_on_disk(tmp_path) -> None:
    """No temporary file is left over next to it."""
    tool, workspace = _tool(tmp_path)
    _run(tool, title="Sauber", content="x")
    assert sorted(path.name for path in workspace.root.iterdir()) == ["sauber.md"]


# --- everything it refuses --------------------------------------------------


def test_an_existing_note_is_never_overwritten(tmp_path) -> None:
    tool, workspace = _tool(tmp_path)
    _run(tool, title="Einmal", content="original")
    result = _run(tool, title="Einmal", content="ÜBERSCHRIEBEN")

    assert result.ok is False
    assert result.error_code == "note_exists"
    assert (workspace.root / "einmal.md").read_text(encoding="utf-8") == "original"


def test_a_symlink_is_never_written_through(tmp_path) -> None:
    """A link planted under the name must not become a way out of the box."""
    tool, workspace = _tool(tmp_path)
    workspace.root.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside.md"
    (workspace.root / "ziel.md").symlink_to(outside)

    result = _run(tool, title="Ziel", content="darf nicht rausgehen")

    assert result.ok is False
    assert result.error_code == "note_exists"
    assert not outside.exists()


def test_a_dangling_symlink_is_refused_too(tmp_path) -> None:
    tool, workspace = _tool(tmp_path)
    workspace.root.mkdir(parents=True, exist_ok=True)
    (workspace.root / "weg.md").symlink_to(tmp_path / "gibtsnicht.md")

    result = _run(tool, title="Weg", content="x")
    assert result.error_code == "note_exists"
    assert not (tmp_path / "gibtsnicht.md").exists()


@pytest.mark.parametrize(
    "arguments",
    [
        {"title": "", "content": "x"},
        {"title": "   ", "content": "x"},
        {"title": "...", "content": "x"},
        {"title": "Gut", "content": ""},
        {"title": "Gut", "content": "   "},
        {"title": "Gut", "content": "x" * (MAX_CONTENT_CHARS + 1)},
        {"title": 42, "content": "x"},
        {"title": "Gut", "content": ["nicht", "text"]},
    ],
)
def test_bad_arguments_write_nothing(tmp_path, arguments) -> None:
    tool, workspace = _tool(tmp_path)
    result = _run(tool, **arguments)

    assert result.ok is False
    assert result.error_code == "invalid_arguments"
    assert not workspace.root.exists() or list(workspace.root.iterdir()) == []


def test_a_failure_leaves_no_partial_file(tmp_path) -> None:
    """Atomic: either a whole note or nothing, never half of one."""
    tool, workspace = _tool(tmp_path)
    workspace.root.mkdir(parents=True, exist_ok=True)
    real_link = os.link

    def _boom(src, dst):
        raise OSError("Platte voll")

    os.link = _boom
    try:
        result = _run(tool, title="Halb", content="x" * 1000)
    finally:
        os.link = real_link

    assert result.ok is False
    assert result.error_code == "tool_failed"
    assert list(workspace.root.iterdir()) == [], "kein Teilstück blieb liegen"


def test_the_tool_says_it_needs_confirmation() -> None:
    tool = CreateNoteTool(NotesWorkspace("/nirgendwo"))
    assert tool.confirmation_required is True
    assert tool.read_only is False
    assert tool.input_schema["additionalProperties"] is False


def test_the_workspace_refuses_a_name_it_did_not_build(tmp_path) -> None:
    workspace = NotesWorkspace(tmp_path / "notes")
    with pytest.raises(NotesWorkspaceError):
        workspace.create("", "x")


def test_the_source_runs_no_shell() -> None:
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "src" / "kiki" / "harness" / "notes.py"
    ).read_text(encoding="utf-8")
    for forbidden in ("subprocess", "shell=True", "os.system", "eval(", "exec("):
        assert forbidden not in source, forbidden
