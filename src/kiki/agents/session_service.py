"""Plan-first coding sessions. Default deny; develop needs a bound approval."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any

from kiki.agents.broker import AgentBroker
from kiki.agents.models import (
    AgentAvailability,
    AgentError,
    AgentEvent,
    AgentEventType,
    AgentSession,
    AgentSessionStatus,
    AgentStartRequest,
    PermissionProfile,
    SessionKind,
    arguments_hash,
)
from kiki.runners.local import LocalWorkspaceRunner
from kiki.storage.agent_session_repository import AgentSessionRepository
from kiki.storage.approval_repository import ApprovalRepository, ApprovalRequest
from kiki.storage.audit_repository import AgentAuditRepository
from kiki.tools.agent_tools import (
    agent_availability_spec,
    agent_plan_spec,
    agent_start_spec,
    agent_stop_spec,
    git_status_spec,
)
from kiki.tools.desktop_tools import (
    copy_text_to_clipboard,
    open_editor_at,
    open_file_with_default_app,
    open_http_url,
    open_path_in_file_manager,
    open_terminal_at,
    show_desktop_notification,
    validate_http_url,
)
from kiki.tools.policy import DecisionKind, ToolPolicy
from kiki.tools.test_tools import tests_run_profile_spec
from kiki.tools.workspace_tools import (
    browser_open_spec,
    clipboard_copy_spec,
    desktop_notification_spec,
    git_diff_spec,
    terminal_open_spec,
    workspace_open_editor_spec,
    workspace_open_file_spec,
    workspace_open_spec,
)
from kiki.workspaces.git_service import inspect_git, read_diff, worktree_fingerprint
from kiki.workspaces.models import GitDiff, Workspace
from kiki.workspaces.registry import WorkspaceRegistry
from kiki.workspaces.validator import resolve_inside_workspace

_SPECS = {
    spec.name: spec
    for spec in (
        agent_plan_spec(),
        agent_start_spec(),
        agent_stop_spec(),
        agent_availability_spec(),
        git_status_spec(),
        git_diff_spec(),
        workspace_open_spec(),
        workspace_open_file_spec(),
        terminal_open_spec(),
        workspace_open_editor_spec(),
        browser_open_spec(),
        clipboard_copy_spec(),
        desktop_notification_spec(),
        tests_run_profile_spec(),
    )
}


class SessionService:
    def __init__(
        self,
        workspaces: WorkspaceRegistry,
        sessions: AgentSessionRepository,
        approvals: ApprovalRepository,
        audit: AgentAuditRepository,
        broker: AgentBroker,
        runner: LocalWorkspaceRunner,
        policy: ToolPolicy | None = None,
        *,
        plan_first: bool = True,
    ) -> None:
        self._workspaces = workspaces
        self._sessions = sessions
        self._approvals = approvals
        self._audit = audit
        self._broker = broker
        self._runner = runner
        self._policy = policy or ToolPolicy()
        self._plan_first = bool(plan_first)
        self._starting_workspaces: set[str] = set()

    @property
    def workspaces(self) -> WorkspaceRegistry:
        return self._workspaces

    def list_workspaces(self) -> list[Workspace]:
        return self._workspaces.list()

    def register_workspace(self, path: str, *, display_name: str | None = None) -> Workspace:
        return self._workspaces.register(path, display_name=display_name)

    def remove_workspace(self, workspace_id: str) -> Path:
        return self._workspaces.remove(workspace_id)

    def list_events(self, session_id: str):
        return self._sessions.list_events(session_id)

    def list_sessions(self, workspace_id: str) -> list[AgentSession]:
        return self._sessions.list_for_workspace(workspace_id)

    def get_session(self, session_id: str) -> AgentSession | None:
        return self._sessions.get(session_id)

    def recent_audit(self, limit: int = 80):
        return self._audit.recent(limit)

    def set_plan_first(self, enabled: bool) -> None:
        self._plan_first = bool(enabled)

    def _decide(self, tool: str, params: dict[str, Any], *, profile: str, panic: bool):
        spec = _SPECS.get(tool)
        return self._policy.evaluate(
            name=tool,
            params=params,
            spec=spec,
            panic=panic,
            integrations_enabled=True,
            profile=profile,
        )

    def _audit_decision(
        self,
        tool: str,
        params: dict[str, Any],
        decision,
        *,
        session_id: str | None = None,
        approval_id: str | None = None,
        result_status: str | None = None,
        result_summary: str | None = None,
    ) -> None:
        self._audit.record(
            actor="user",
            event_type="policy",
            risk_class=decision.risk.value,
            requested_action=tool,
            arguments_hash=arguments_hash(params),
            policy_decision=decision.kind.value,
            session_id=session_id,
            approval_id=approval_id,
            result_status=result_status,
            result_summary=result_summary or decision.reason,
        )

    def request_approval(
        self,
        tool: str,
        params: dict[str, Any],
        *,
        profile: str,
        panic: bool = False,
    ) -> ApprovalRequest:
        decision = self._decide(tool, params, profile=profile, panic=panic)
        self._audit_decision(tool, params, decision)
        if decision.kind is DecisionKind.DENY:
            raise AgentError("denied", decision.reason)
        cleaned = decision.params or params
        if tool == "agent.start_implementation":
            self._require_completed_plan(cleaned)
        spec = _SPECS.get(tool)
        stored_params = dict(cleaned)
        for name in tuple(getattr(spec, "sensitive_parameters", ())):
            if name in stored_params:
                value = stored_params[name]
                size = len(value) if isinstance(value, str) else 0
                stored_params[name] = f"<redacted:{size}>"
        return self._approvals.create(
            tool,
            cleaned,
            risk_class=decision.risk.value,
            stored_params=stored_params,
        )

    def _require_completed_plan(self, params: dict[str, Any]) -> AgentSession | None:
        plan_id = str(params.get("plan_session_id") or "").strip()
        if not plan_id:
            if not self._plan_first:
                return None
            raise AgentError("plan_required", "Plan-First ist aktiv. Erstelle zuerst einen Plan.")
        plan = self._sessions.get(plan_id)
        if plan is None:
            raise AgentError("invalid_plan", "Die angegebene Plan-Session existiert nicht.")
        if plan.kind is not SessionKind.PLAN or plan.status is not AgentSessionStatus.FINISHED:
            raise AgentError("invalid_plan", "Nur eine erfolgreich abgeschlossene Plan-Session ist gültig.")
        if plan.workspace_id != str(params.get("workspace_id") or ""):
            raise AgentError("invalid_plan", "Der Plan gehört zu einem anderen Workspace.")
        if plan.task_text.strip() != str(params.get("task") or "").strip():
            raise AgentError("plan_stale", "Die Aufgabe wurde seit dem Plan verändert. Bitte neu planen.")
        return plan

    def decide_approval(self, approval_id: str, *, approved: bool) -> None:
        found = self._approvals.get(approval_id)
        if found is None:
            raise AgentError("no_approval", "Freigabe unbekannt.")
        self._approvals.decide(approval_id, approved=approved, actor="user")
        self._audit.record(
            actor="user",
            event_type="approval",
            risk_class=found.risk_class,
            requested_action=found.tool,
            arguments_hash=found.params_hash,
            policy_decision="confirmed" if approved else "cancelled",
            approval_id=approval_id,
            result_status="approved" if approved else "denied",
        )

    async def availability(self, *, panic: bool = False) -> AgentAvailability:
        params: dict[str, Any] = {}
        decision = self._decide("agent.availability_check", params, profile="observe", panic=panic)
        self._audit_decision("agent.availability_check", params, decision)
        if decision.kind is DecisionKind.DENY:
            raise AgentError("denied", decision.reason)
        return await self._broker.get("opencode").check_availability()

    async def git_status(self, workspace_id: str, *, profile: str = "observe", panic: bool = False):
        params = {"workspace_id": workspace_id}
        decision = self._decide("git.status", params, profile=profile, panic=panic)
        self._audit_decision("git.status", params, decision)
        if decision.kind is DecisionKind.DENY:
            raise AgentError("denied", decision.reason)
        workspace = self._workspaces.require(workspace_id)
        return inspect_git(Path(workspace.canonical_path))

    async def git_diff(self, workspace_id: str, *, profile: str = "observe", panic: bool = False) -> GitDiff:
        params = {"workspace_id": workspace_id}
        decision = self._decide("git.diff", params, profile=profile, panic=panic)
        self._audit_decision("git.diff", params, decision)
        if decision.kind is DecisionKind.DENY:
            raise AgentError("denied", decision.reason)
        workspace = self._workspaces.require(workspace_id)
        return read_diff(Path(workspace.canonical_path))

    def open_workspace_folder(
        self,
        workspace_id: str,
        *,
        profile: str = "observe",
        panic: bool = False,
        approval_id: str | None = None,
    ) -> str:
        params = {"workspace_id": workspace_id}
        decision = self._decide("workspace.open_in_file_manager", params, profile=profile, panic=panic)
        self._audit_decision("workspace.open_in_file_manager", params, decision, approval_id=approval_id)
        if decision.kind is DecisionKind.DENY:
            raise AgentError("denied", decision.reason)
        cleaned = decision.params or params
        if decision.kind is DecisionKind.CONFIRM:
            if not approval_id:
                raise AgentError("no_approval", "Ordner öffnen braucht eine konkrete Freigabe.")
            if not self._approvals.consume_if_valid(approval_id, "workspace.open_in_file_manager", cleaned):
                raise AgentError("no_approval", "Freigabe passt nicht oder wurde verbraucht.")
        workspace = self._workspaces.require(workspace_id)
        open_path_in_file_manager(Path(workspace.canonical_path))
        self._audit.record(
            actor="user",
            event_type="execution",
            risk_class=decision.risk.value,
            requested_action="workspace.open_in_file_manager",
            arguments_hash=arguments_hash(cleaned),
            policy_decision="executed",
            approval_id=approval_id,
            result_status="opened",
            result_summary=workspace.canonical_path,
        )
        return workspace.canonical_path

    def open_terminal(
        self,
        workspace_id: str,
        *,
        profile: str = "observe",
        panic: bool = False,
        approval_id: str | None = None,
    ) -> list[str]:
        params = {"workspace_id": workspace_id}
        decision = self._decide("terminal.open_workspace", params, profile=profile, panic=panic)
        self._audit_decision("terminal.open_workspace", params, decision, approval_id=approval_id)
        if decision.kind is DecisionKind.DENY:
            raise AgentError("denied", decision.reason)
        cleaned = decision.params or params
        if decision.kind is DecisionKind.CONFIRM:
            if not approval_id:
                raise AgentError("no_approval", "Terminal braucht eine konkrete Freigabe.")
            if not self._approvals.consume_if_valid(approval_id, "terminal.open_workspace", cleaned):
                raise AgentError("no_approval", "Freigabe passt nicht oder wurde verbraucht.")
        workspace = self._workspaces.require(workspace_id)
        argv = open_terminal_at(Path(workspace.canonical_path))
        self._audit.record(
            actor="user",
            event_type="execution",
            risk_class=decision.risk.value,
            requested_action="terminal.open_workspace",
            arguments_hash=arguments_hash(cleaned),
            policy_decision="executed",
            approval_id=approval_id,
            result_status="opened",
            result_summary=" ".join(argv),
        )
        return argv

    def open_editor(
        self,
        workspace_id: str,
        *,
        profile: str = "observe",
        panic: bool = False,
        approval_id: str | None = None,
    ) -> list[str]:
        params = {"workspace_id": workspace_id}
        decision = self._decide("workspace.open_in_editor", params, profile=profile, panic=panic)
        self._audit_decision("workspace.open_in_editor", params, decision, approval_id=approval_id)
        if decision.kind is DecisionKind.DENY:
            raise AgentError("denied", decision.reason)
        cleaned = decision.params or params
        if decision.kind is DecisionKind.CONFIRM:
            if not approval_id:
                raise AgentError("no_approval", "Editor braucht eine konkrete Freigabe.")
            if not self._approvals.consume_if_valid(approval_id, "workspace.open_in_editor", cleaned):
                raise AgentError("no_approval", "Freigabe passt nicht oder wurde verbraucht.")
        workspace = self._workspaces.require(workspace_id)
        argv = open_editor_at(Path(workspace.canonical_path))
        self._audit.record(
            actor="user",
            event_type="execution",
            risk_class=decision.risk.value,
            requested_action="workspace.open_in_editor",
            arguments_hash=arguments_hash(cleaned),
            policy_decision="executed",
            approval_id=approval_id,
            result_status="opened",
            result_summary=" ".join(argv),
        )
        return argv

    def open_workspace_file(
        self,
        workspace_id: str,
        rel_path: str,
        *,
        profile: str = "observe",
        panic: bool = False,
        approval_id: str | None = None,
    ) -> str:
        params = {"workspace_id": workspace_id, "path": rel_path}
        decision = self._decide("workspace.open_file", params, profile=profile, panic=panic)
        self._audit_decision("workspace.open_file", params, decision, approval_id=approval_id)
        if decision.kind is DecisionKind.DENY:
            raise AgentError("denied", decision.reason)
        cleaned = decision.params or params
        if decision.kind is DecisionKind.CONFIRM:
            if not approval_id:
                raise AgentError("no_approval", "Datei öffnen braucht eine konkrete Freigabe.")
            if not self._approvals.consume_if_valid(approval_id, "workspace.open_file", cleaned):
                raise AgentError("no_approval", "Freigabe passt nicht oder wurde verbraucht.")
        workspace = self._workspaces.require(workspace_id)
        try:
            target = resolve_inside_workspace(str(cleaned["path"]), Path(workspace.canonical_path))
        except Exception as exc:
            raise AgentError(getattr(exc, "code", "outside_root"), str(exc)) from exc
        open_file_with_default_app(target)
        self._audit.record(
            actor="user",
            event_type="execution",
            risk_class=decision.risk.value,
            requested_action="workspace.open_file",
            arguments_hash=arguments_hash(cleaned),
            policy_decision="executed",
            approval_id=approval_id,
            result_status="opened",
            result_summary=str(target),
        )
        return str(target)

    def open_browser_url(
        self,
        url: str,
        *,
        profile: str = "observe",
        panic: bool = False,
        approval_id: str | None = None,
    ) -> str:
        try:
            cleaned_url = validate_http_url(url)
        except Exception as exc:
            raise AgentError("invalid_url", str(exc)) from exc
        params = {"url": cleaned_url}
        decision = self._decide("browser.open_url", params, profile=profile, panic=panic)
        self._audit_decision("browser.open_url", params, decision, approval_id=approval_id)
        if decision.kind is DecisionKind.DENY:
            raise AgentError("denied", decision.reason)
        cleaned = decision.params or params
        if decision.kind is DecisionKind.CONFIRM:
            if not approval_id:
                raise AgentError("no_approval", "Browser braucht eine konkrete Freigabe.")
            if not self._approvals.consume_if_valid(approval_id, "browser.open_url", cleaned):
                raise AgentError("no_approval", "Freigabe passt nicht oder wurde verbraucht.")
        opened = open_http_url(str(cleaned["url"]))
        self._audit.record(
            actor="user",
            event_type="execution",
            risk_class=decision.risk.value,
            requested_action="browser.open_url",
            arguments_hash=arguments_hash(cleaned),
            policy_decision="executed",
            approval_id=approval_id,
            result_status="opened",
            result_summary=opened,
        )
        return opened

    def copy_desktop_text(
        self,
        text: str,
        *,
        profile: str = "observe",
        panic: bool = False,
        approval_id: str | None = None,
    ) -> int:
        params = {"text": text}
        decision = self._decide("desktop.copy_text", params, profile=profile, panic=panic)
        self._audit_decision("desktop.copy_text", params, decision, approval_id=approval_id)
        if decision.kind is DecisionKind.DENY:
            raise AgentError("denied", decision.reason)
        cleaned = decision.params or params
        if decision.kind is DecisionKind.CONFIRM:
            if not approval_id:
                raise AgentError("no_approval", "Zwischenablage braucht eine konkrete Freigabe.")
            if not self._approvals.consume_if_valid(approval_id, "desktop.copy_text", cleaned):
                raise AgentError("no_approval", "Freigabe passt nicht oder wurde verbraucht.")
        try:
            count = copy_text_to_clipboard(str(cleaned["text"]))
        except Exception as exc:
            raise AgentError(getattr(exc, "code", "desktop_error"), str(exc)) from exc
        self._audit.record(
            actor="user",
            event_type="execution",
            risk_class=decision.risk.value,
            requested_action="desktop.copy_text",
            arguments_hash=arguments_hash(cleaned),
            policy_decision="executed",
            approval_id=approval_id,
            result_status="copied",
            result_summary=f"{count} Zeichen kopiert; Inhalt nicht protokolliert",
        )
        return count

    def show_notification(
        self,
        title: str,
        body: str,
        *,
        profile: str = "observe",
        panic: bool = False,
        approval_id: str | None = None,
    ) -> tuple[str, str]:
        params = {"title": title, "body": body}
        decision = self._decide("desktop.show_notification", params, profile=profile, panic=panic)
        self._audit_decision("desktop.show_notification", params, decision, approval_id=approval_id)
        if decision.kind is DecisionKind.DENY:
            raise AgentError("denied", decision.reason)
        cleaned = decision.params or params
        if decision.kind is DecisionKind.CONFIRM:
            if not approval_id:
                raise AgentError("no_approval", "Benachrichtigung braucht eine konkrete Freigabe.")
            if not self._approvals.consume_if_valid(
                approval_id, "desktop.show_notification", cleaned
            ):
                raise AgentError("no_approval", "Freigabe passt nicht oder wurde verbraucht.")
        try:
            rendered = show_desktop_notification(
                str(cleaned["title"]),
                str(cleaned["body"]),
            )
        except Exception as exc:
            raise AgentError(getattr(exc, "code", "desktop_error"), str(exc)) from exc
        self._audit.record(
            actor="user",
            event_type="execution",
            risk_class=decision.risk.value,
            requested_action="desktop.show_notification",
            arguments_hash=arguments_hash(cleaned),
            policy_decision="executed",
            approval_id=approval_id,
            result_status="shown",
            result_summary="Lokale Benachrichtigung angezeigt; Inhalt nicht protokolliert",
        )
        return rendered

    async def run_plan(
        self,
        workspace_id: str,
        task: str,
        *,
        profile: str = "observe",
        model: str = "",
        panic: bool = False,
        agent_timeout: float = 120.0,
    ) -> AgentSession:
        params = {"workspace_id": workspace_id, "task": task, "model": model, "profile": profile}
        decision = self._decide("agent.plan", params, profile=profile, panic=panic)
        self._audit_decision("agent.plan", params, decision)
        if decision.kind is DecisionKind.DENY:
            raise AgentError("denied", decision.reason)
        return await self._run_agent(
            workspace_id,
            task,
            kind=SessionKind.PLAN,
            profile=profile,
            model=model,
            timeout=agent_timeout,
        )

    async def start_implementation(
        self,
        workspace_id: str,
        task: str,
        *,
        profile: str = "develop",
        model: str = "",
        plan_session_id: str | None = None,
        approval_id: str | None = None,
        panic: bool = False,
        agent_timeout: float = 120.0,
    ) -> AgentSession:
        params = {
            "workspace_id": workspace_id,
            "task": task,
            "model": model,
            "profile": profile,
        }
        if plan_session_id:
            params["plan_session_id"] = plan_session_id
        decision = self._decide("agent.start_implementation", params, profile=profile, panic=panic)
        self._audit_decision("agent.start_implementation", params, decision, approval_id=approval_id)
        if decision.kind is DecisionKind.DENY:
            raise AgentError("denied", decision.reason)
        self._require_completed_plan(decision.params or params)
        if decision.kind is DecisionKind.CONFIRM:
            if not approval_id:
                raise AgentError("no_approval", "Umsetzung braucht eine konkrete Freigabe.")
            bound = dict(decision.params or params)
            if not self._approvals.consume_if_valid(approval_id, "agent.start_implementation", bound):
                raise AgentError(
                    "no_approval",
                    "Freigabe passt nicht zu Tool und Parametern oder wurde bereits verbraucht.",
                )
        return await self._run_agent(
            workspace_id,
            task,
            kind=SessionKind.IMPLEMENT,
            profile=profile,
            model=model,
            timeout=agent_timeout,
            approval_id=approval_id,
            plan_session_id=plan_session_id,
        )

    async def stop(self, session_id: str, *, profile: str = "observe", panic: bool = False) -> None:
        params = {"session_id": session_id}
        decision = self._decide("agent.stop", params, profile=profile, panic=panic)
        self._audit_decision("agent.stop", params, decision, session_id=session_id)
        if decision.kind is DecisionKind.DENY:
            raise AgentError("denied", decision.reason)
        stored = self._sessions.get(session_id)
        if stored is None:
            raise AgentError("unknown_session", "Session ist unbekannt.")
        if stored.status is not AgentSessionStatus.RUNNING:
            raise AgentError("not_running", "Session läuft nicht mehr und wird nicht verändert.")
        self._sessions.update_status(session_id, AgentSessionStatus.STOPPING)
        await self._broker.get("opencode").stop_session(session_id)
        for _ in range(100):
            stored = self._sessions.get(session_id)
            if stored is None or stored.status is not AgentSessionStatus.STOPPING:
                break
            await asyncio.sleep(0.05)
        if stored is not None and stored.status is AgentSessionStatus.STOPPING:
            self._sessions.update_status(
                session_id,
                AgentSessionStatus.FAILED,
                summary="stopped",
                finished=True,
            )
        self._audit.record(
            actor="user",
            event_type="execution",
            risk_class=decision.risk.value,
            requested_action="agent.stop",
            arguments_hash=arguments_hash(params),
            policy_decision="executed",
            session_id=session_id,
            result_status="stopped",
        )

    async def run_tests(
        self,
        workspace_id: str,
        profile_name: str,
        *,
        profile: str = "develop",
        approval_id: str | None = None,
        timeout_seconds: int | None = None,
        panic: bool = False,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"workspace_id": workspace_id, "profile": profile_name}
        if timeout_seconds is not None:
            params["timeout_seconds"] = int(timeout_seconds)
        decision = self._decide("tests.run_profile", params, profile=profile, panic=panic)
        self._audit_decision("tests.run_profile", params, decision, approval_id=approval_id)
        if decision.kind is DecisionKind.DENY:
            raise AgentError("denied", decision.reason)
        cleaned = decision.params or params
        if decision.kind is DecisionKind.CONFIRM:
            if not approval_id:
                raise AgentError("no_approval", "Tests brauchen eine konkrete Freigabe.")
            if not self._approvals.consume_if_valid(approval_id, "tests.run_profile", cleaned):
                raise AgentError("no_approval", "Freigabe passt nicht oder wurde verbraucht.")
        workspace = self._workspaces.require(workspace_id)
        if session_id:
            parent = self._sessions.get(session_id)
            if parent is None or parent.workspace_id != workspace_id:
                raise AgentError("invalid_session", "Testlauf und Session gehören nicht zusammen.")
        argv = self._runner.profile_argv(profile_name)
        timeout = self._runner.clamp_timeout(cleaned.get("timeout_seconds"))
        test_id = str(uuid.uuid4())
        self._sessions.insert_test_run(
            test_id=test_id,
            workspace_id=workspace_id,
            profile=profile_name,
            argv=argv,
            session_id=session_id,
        )
        if session_id:
            self._sessions.add_event(
                session_id,
                AgentEvent(
                    type=AgentEventType.TEST_STARTED,
                    text=profile_name,
                    data={"argv": argv, "timeout": timeout},
                ),
            )

        def _finish_test(
            *,
            status: str,
            exit_code: int | None,
            summary: str,
            output: str,
            output_truncated: bool,
        ) -> None:
            self._sessions.finish_test_run(
                test_id,
                status=status,
                exit_code=exit_code,
                summary=summary,
                output=output,
                output_truncated=output_truncated,
            )
            if session_id:
                if output:
                    self._sessions.add_event(
                        session_id,
                        AgentEvent(
                            type=AgentEventType.TEST_OUTPUT,
                            text=output,
                            data={"truncated": output_truncated},
                        ),
                    )
                self._sessions.add_event(
                    session_id,
                    AgentEvent(
                        type=AgentEventType.TEST_FINISHED,
                        text=summary,
                        data={"status": status, "exit_code": exit_code},
                    ),
                )
            self._audit.record(
                actor="user",
                event_type="execution",
                risk_class=decision.risk.value,
                requested_action="tests.run_profile",
                arguments_hash=arguments_hash(cleaned),
                policy_decision="executed",
                approval_id=approval_id,
                result_status=status,
                result_summary=summary,
            )

        try:
            handle = await self._runner.run_argv(argv, workspace=workspace)
            result = await handle.capture(timeout=timeout)
        except BaseException as exc:
            status = "cancelled" if isinstance(exc, asyncio.CancelledError) else "error"
            output = f"{type(exc).__name__}: {exc}"
            summary = f"{status}: {str(exc)[:300]}"
            _finish_test(
                status=status,
                exit_code=None,
                summary=summary,
                output=output,
                output_truncated=False,
            )
            raise
        status = "timeout" if result.timed_out else ("ok" if result.exit_code == 0 else "failed")
        summary = f"{status} exit={result.exit_code}"
        _finish_test(
            status=status,
            exit_code=result.exit_code,
            summary=summary,
            output=result.output,
            output_truncated=result.output_truncated,
        )
        return {
            "ok": status == "ok",
            "status": status,
            "exit_code": result.exit_code,
            "argv": argv,
            "timeout": timeout,
            "timed_out": result.timed_out,
            "output": result.output,
            "output_truncated": result.output_truncated,
            "test_id": test_id,
        }

    async def _run_agent(
        self,
        workspace_id: str,
        task: str,
        *,
        kind: SessionKind,
        profile: str,
        model: str,
        timeout: float,
        approval_id: str | None = None,
        plan_session_id: str | None = None,
    ) -> AgentSession:
        workspace = self._workspaces.require(workspace_id)
        if workspace_id in self._starting_workspaces:
            raise AgentError("session_active", "In diesem Workspace startet bereits eine Agent-Session.")
        self._starting_workspaces.add(workspace_id)
        adapter = self._broker.get("opencode")
        session_id = ""
        try:
            active = {
                AgentSessionStatus.STARTING,
                AgentSessionStatus.RUNNING,
                AgentSessionStatus.STOPPING,
            }
            if any(session.status in active for session in self._sessions.list_for_workspace(workspace_id)):
                raise AgentError("session_active", "In diesem Workspace läuft bereits eine Agent-Session.")
            snap = inspect_git(Path(workspace.canonical_path))
            plan_fingerprint = (
                worktree_fingerprint(Path(workspace.canonical_path))
                if kind is SessionKind.PLAN
                else None
            )
            available = await adapter.check_availability()
            if not available.ok:
                raise AgentError("not_available", available.detail)
            session_id = str(uuid.uuid4())
            request = AgentStartRequest(
                workspace_id=workspace.id,
                workspace_path=workspace.canonical_path,
                task=task,
                kind=kind,
                permission_profile=PermissionProfile(profile),
                model=model,
                session_id=session_id,
            )
            live = await adapter.start_session(request)
            live.git_branch_before = snap.branch
            live.git_head_before = snap.head
            live.agent_version = available.version
            live.plan_session_id = plan_session_id
            try:
                self._sessions.insert(live)
            except BaseException:
                await adapter.stop_session(session_id)
                raise
        finally:
            self._starting_workspaces.discard(workspace_id)
        self._workspaces.touch(workspace.id)
        summary_bits: list[str] = []

        async def _consume() -> None:
            async for event in adapter.stream_events(session_id):
                self._sessions.add_event(session_id, event)
                if event.type is AgentEventType.MESSAGE and event.text:
                    summary_bits.append(event.text)
                if event.type is AgentEventType.SESSION_FINISHED:
                    stored = self._sessions.get(session_id)
                    stopping = stored is not None and stored.status is AgentSessionStatus.STOPPING
                    try:
                        status = AgentSessionStatus(event.text)
                    except ValueError:
                        status = AgentSessionStatus.FINISHED
                    if event.data.get("timed_out"):
                        status = AgentSessionStatus.FAILED
                    if stopping:
                        status = AgentSessionStatus.FAILED
                    self._sessions.update_status(
                        session_id,
                        status,
                        exit_code=event.data.get("exit_code") if isinstance(event.data.get("exit_code"), int) else None,
                        summary=("stopped" if stopping else ("\n".join(summary_bits)[:500] or status.value)),
                        finished=True,
                    )

        try:
            await asyncio.wait_for(_consume(), timeout=timeout)
        except TimeoutError:
            await adapter.stop_session(session_id)
            self._sessions.update_status(
                session_id,
                AgentSessionStatus.FAILED,
                summary="timeout",
                finished=True,
            )
        except asyncio.CancelledError:
            await adapter.stop_session(session_id)
            self._sessions.update_status(
                session_id,
                AgentSessionStatus.FAILED,
                summary="cancelled",
                finished=True,
            )
            raise
        except Exception as exc:
            await adapter.stop_session(session_id)
            self._sessions.add_event(
                session_id,
                AgentEvent(type=AgentEventType.ERROR, text=f"Agent-Stream fehlgeschlagen: {exc}"),
            )
            self._sessions.update_status(
                session_id,
                AgentSessionStatus.FAILED,
                summary=f"stream-error: {str(exc)[:300]}",
                finished=True,
            )
        current = self._sessions.get(session_id)
        if current and current.status is AgentSessionStatus.RUNNING:
            await adapter.stop_session(session_id)
            self._sessions.update_status(session_id, AgentSessionStatus.FAILED, summary="forced-stop", finished=True)
        current = self._sessions.get(session_id)
        if kind is SessionKind.PLAN and current is not None and current.status is AgentSessionStatus.FINISHED:
            try:
                changed = plan_fingerprint != worktree_fingerprint(Path(workspace.canonical_path))
            except Exception as exc:
                message = f"Observe-Prüfung fehlgeschlagen: {exc}"
                self._sessions.add_event(
                    session_id,
                    AgentEvent(type=AgentEventType.ERROR, text=message),
                )
                self._sessions.update_status(
                    session_id,
                    AgentSessionStatus.FAILED,
                    summary="observe-check-failed",
                    finished=True,
                )
            else:
                if changed:
                    message = "Observe-Verstoß: Der Plan-Agent hat den Git-Arbeitsbaum verändert."
                    self._sessions.add_event(
                        session_id,
                        AgentEvent(type=AgentEventType.ERROR, text=message),
                    )
                    self._sessions.update_status(
                        session_id,
                        AgentSessionStatus.FAILED,
                        summary="observe-worktree-changed",
                        finished=True,
                    )
        finished = self._sessions.get(session_id)
        assert finished is not None
        self._audit.record(
            actor="user",
            event_type="execution",
            risk_class="read" if kind is SessionKind.PLAN else "write",
            requested_action="agent.plan" if kind is SessionKind.PLAN else "agent.start_implementation",
            arguments_hash=arguments_hash(
                {
                    "workspace_id": workspace_id,
                    "task": task,
                    "model": model,
                    "profile": profile,
                    **({"plan_session_id": plan_session_id} if plan_session_id else {}),
                }
            ),
            policy_decision="executed",
            session_id=session_id,
            approval_id=approval_id,
            result_status=finished.status.value,
            result_summary=finished.summary,
        )
        return finished
