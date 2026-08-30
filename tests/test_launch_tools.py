"""Openers KIKI runs herself: what the `trusted` level does and does not permit."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import pytest

from kiki.tools.launch_tools import DesktopLaunchSkill
from kiki.tools.policy import AutonomyLevel, DecisionKind, Origin, RiskLevel, ToolPolicy
from kiki.tools.workspace_tools import clipboard_copy_spec, desktop_notification_spec
from kiki.workspaces.models import WorkspaceError


@dataclass
class FakeWorkspace:
    id: str
    display_name: str
    canonical_path: str


class FakeRegistry:
    """Stands in for WorkspaceRegistry, recording that require() was called."""

    def __init__(self, workspaces: list[FakeWorkspace]) -> None:
        self._by_id = {w.id: w for w in workspaces}
        self.required: list[str] = []

    def list(self) -> list[FakeWorkspace]:
        return list(self._by_id.values())

    def require(self, workspace_id: str) -> FakeWorkspace:
        self.required.append(workspace_id)
        found = self._by_id.get(workspace_id)
        if found is None:
            raise WorkspaceError("not_registered", f"Unbekannt: {workspace_id}")
        return found


@pytest.fixture(autouse=True)
def no_real_launches(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Never spawn a real desktop application from the test suite.

    The openers call `desktop_tools._launch`, which runs xdg-open. Without this
    the suite pops file-manager windows on whoever runs it.
    """
    import kiki.tools.desktop_tools as desktop_tools

    launched: list[list[str]] = []

    def _record(argv: list[str], *, cwd: Path) -> None:
        launched.append(list(argv))

    monkeypatch.setattr(desktop_tools, "_launch", _record)
    return launched


@pytest.fixture
def workspace(tmp_path: Path) -> FakeWorkspace:
    root = tmp_path / "projekt"
    (root / "src").mkdir(parents=True)
    (root / "src" / "main.py").write_text("print('hi')\n", encoding="utf-8")
    (tmp_path / "geheim.txt").write_text("nicht im workspace\n", encoding="utf-8")
    return FakeWorkspace(id="ws-1", display_name="Projekt", canonical_path=str(root))


@pytest.fixture
def skill(workspace: FakeWorkspace) -> DesktopLaunchSkill:
    return DesktopLaunchSkill(FakeRegistry([workspace]))


def _spec(skill: DesktopLaunchSkill, name: str):
    for spec in skill.tools():
        if spec.name == name:
            return spec
    raise AssertionError(f"kein Spec {name}")


# --- classification ---------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "workspace.open_in_file_manager",
        "workspace.open_file",
        "terminal.open_workspace",
        "workspace.open_in_editor",
    ],
)
def test_openers_are_launch_and_model_callable(skill, name) -> None:
    spec = _spec(skill, name)
    assert spec.risk is RiskLevel.LAUNCH
    assert spec.model_callable is True
    assert spec.auto_allow is True


def test_browser_stays_external_and_confirmation_only(skill) -> None:
    spec = _spec(skill, "browser.open_url")
    assert spec.risk is RiskLevel.EXTERNAL
    assert spec.model_callable is True
    assert spec.auto_allow is True
    for level in AutonomyLevel:
        decision = ToolPolicy(level.value).evaluate(
            name=spec.name,
            params={"url": "https://example.com"},
            spec=spec,
            panic=False,
            integrations_enabled=True,
            origin=Origin.MODEL,
        )
        assert decision.kind is DecisionKind.CONFIRM


def test_listing_workspaces_is_read_only(skill) -> None:
    spec = _spec(skill, "workspace.list")
    assert spec.risk is RiskLevel.READ
    assert spec.model_callable is True


def test_clipboard_and_notification_stay_out_of_the_launch_set(skill) -> None:
    """Replacing the clipboard changes the user's data, so it keeps its card."""
    names = {spec.name for spec in skill.tools()}
    assert "desktop.copy_text" not in names
    assert "desktop.show_notification" not in names
    # And the declarations they still use are not model-callable.
    assert clipboard_copy_spec().model_callable is False
    assert desktop_notification_spec().model_callable is False


def test_the_shared_declarations_are_left_untouched(skill) -> None:
    """The control window's previews must not inherit auto_allow."""
    from kiki.tools.workspace_tools import workspace_open_spec

    declaration = workspace_open_spec()
    assert declaration.auto_allow is False
    assert declaration.model_callable is False
    assert declaration.risk is RiskLevel.EXTERNAL
    # The executable copy is a separate object.
    assert _spec(skill, "workspace.open_in_file_manager") is not declaration


# --- what each autonomy level permits ---------------------------------------


@pytest.mark.parametrize(
    ("level", "expected"),
    [
        (AutonomyLevel.STRICT, DecisionKind.CONFIRM),
        (AutonomyLevel.BALANCED, DecisionKind.CONFIRM),
        (AutonomyLevel.TRUSTED, DecisionKind.ALLOW),
    ],
)
def test_only_trusted_opens_without_a_card(skill, level, expected) -> None:
    policy = ToolPolicy(level.value)
    decision = policy.evaluate(
        name="workspace.open_in_file_manager",
        params={"workspace_id": "ws-1"},
        spec=_spec(skill, "workspace.open_in_file_manager"),
        panic=False,
        integrations_enabled=True,
        origin=Origin.MODEL,
    )
    assert decision.kind is expected


def test_trusted_still_confirms_writes_and_external(skill) -> None:
    policy = ToolPolicy("trusted")
    for spec, params in (
        (clipboard_copy_spec(), {"text": "hallo"}),
        (desktop_notification_spec(), {"title": "t", "body": "b"}),
    ):
        # These are not model-callable at all, so they are denied outright …
        decision = policy.evaluate(
            name=spec.name,
            params=params,
            spec=spec,
            panic=False,
            integrations_enabled=True,
            origin=Origin.MODEL,
        )
        assert decision.kind is DecisionKind.DENY

    # … and a hypothetical model-callable WRITE still only reaches CONFIRM.
    import dataclasses

    writable = dataclasses.replace(
        clipboard_copy_spec(), model_callable=True, auto_allow=True
    )
    decision = policy.evaluate(
        name=writable.name,
        params={"text": "hallo"},
        spec=writable,
        panic=False,
        integrations_enabled=True,
        origin=Origin.MODEL,
    )
    assert decision.kind is DecisionKind.CONFIRM


def test_an_explicit_user_launch_needs_no_second_card(skill) -> None:
    """The trusted parser already binds the person's command to this target."""
    decision = ToolPolicy("trusted").evaluate(
        name="workspace.open_in_file_manager",
        params={"workspace_id": "ws-1"},
        spec=_spec(skill, "workspace.open_in_file_manager"),
        panic=False,
        integrations_enabled=True,
        origin=Origin.USER,
    )
    assert decision.kind is DecisionKind.ALLOW


def test_panic_blocks_opening_at_every_level(skill) -> None:
    decision = ToolPolicy("trusted").evaluate(
        name="workspace.open_in_file_manager",
        params={"workspace_id": "ws-1"},
        spec=_spec(skill, "workspace.open_in_file_manager"),
        panic=True,
        integrations_enabled=True,
        origin=Origin.MODEL,
    )
    assert decision.kind is DecisionKind.DENY


# --- path confinement -------------------------------------------------------


def test_an_unregistered_workspace_is_refused(skill) -> None:
    result = _spec(skill, "workspace.open_in_file_manager").handler({"workspace_id": "fremd"})
    assert result["ok"] is False
    assert "Unbekannt" in result["error"]


def test_a_file_outside_the_workspace_is_refused(skill, workspace) -> None:
    handler = _spec(skill, "workspace.open_file").handler
    for escape in ("../geheim.txt", "/etc/passwd", "src/../../geheim.txt"):
        result = handler({"workspace_id": "ws-1", "path": escape})
        assert result["ok"] is False, escape


def test_every_open_revalidates_the_workspace(skill, no_real_launches) -> None:
    """A path redirected after registration must not stay trusted."""
    registry = skill._workspaces
    folder = _spec(skill, "workspace.open_in_file_manager").handler({"workspace_id": "ws-1"})
    opened = _spec(skill, "workspace.open_file").handler(
        {"workspace_id": "ws-1", "path": "src/main.py"}
    )
    assert folder["ok"] is True and opened["ok"] is True
    assert registry.required == ["ws-1", "ws-1"]
    # Both went through xdg-open with a resolved absolute path, not model text.
    assert len(no_real_launches) == 2
    assert all(argv[0].endswith("xdg-open") for argv in no_real_launches)
    assert no_real_launches[1][1].endswith("src/main.py")


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "javascript:alert(1)",
        "data:text/html,<h1>x",
        "https://user:pass@example.com/",
        "nicht mal eine url",
    ],
)
def test_dangerous_urls_are_refused(skill, url) -> None:
    result = _spec(skill, "browser.open_url").handler({"url": url})
    assert result["ok"] is False


def test_listing_reports_the_registered_workspaces(skill) -> None:
    result = _spec(skill, "workspace.list").handler({})
    assert result["count"] == 1
    assert result["workspaces"][0]["id"] == "ws-1"


# --- exposure ---------------------------------------------------------------


def test_openers_reach_the_model_only_through_the_registry(skill, tools_env) -> None:
    from kiki.tools.exposure import exposed_specs

    registry, executor = tools_env
    for spec in skill.tools():
        registry.register(spec)
    executor.policy.set_autonomy("trusted")

    names = {s.name for s in exposed_specs(registry, executor.policy, panic=False, integrations_enabled=True)}
    assert "workspace.open_in_file_manager" in names
    assert "workspace.list" in names

    assert exposed_specs(registry, executor.policy, panic=True, integrations_enabled=True) == []


def test_an_open_runs_through_the_executor_and_is_audited(skill, tools_env, db) -> None:
    registry, executor = tools_env
    for spec in skill.tools():
        registry.register(spec)
    executor.policy.set_autonomy("trusted")

    result = asyncio.run(
        executor.run(
            "workspace.list",
            {},
            panic=False,
            integrations_enabled=True,
            origin=Origin.MODEL,
        )
    )
    assert result.ok is True
    rows = db.conn.execute("SELECT tool, decision, result FROM audit_log ORDER BY id").fetchall()
    assert any(r["tool"] == "workspace.list" and r["decision"] == "allow" for r in rows)
    # The origin is recorded, so a model-initiated open stays distinguishable.
    assert any("[model]" in (r["result"] or "") for r in rows)
