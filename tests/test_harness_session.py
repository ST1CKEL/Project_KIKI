"""The vertical slice: model → tool → confirmation → write → UI → voice.

Fakes throughout: no model, no GTK, no PipeWire, no network, and every file
under `tmp_path`. What is under test is the product behaviour — what the user
sees, what KIKI says, and above all what does and does not get written.
"""

from __future__ import annotations

import asyncio

import pytest

from kiki.harness.confirmation import ConfirmationRequest
from kiki.harness.models import ActionKind, ModelAction, RunStatus
from kiki.harness.notes import CreateNoteTool, NotesWorkspace
from kiki.harness.runner import AgentRunner, RunBusyError
from kiki.harness.session import (
    SPEECH_NEEDS_CONFIRMATION,
    STATUS_CANCELLED,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_NEEDS_CONFIRMATION,
    STATUS_WORKING,
    HarnessSession,
    SessionCallbacks,
)
from kiki.harness.system_status import SystemStatusTool
from kiki.harness.tools import ToolRegistry

NOTE_ARGS = {"title": "Milch kaufen", "content": "Milch kaufen"}


class ScriptedModel:
    """Replays a fixed list of actions, one per step."""

    def __init__(self, actions) -> None:
        self._actions = list(actions)
        self.calls = 0

    async def next_action(self, *, user_text, tool_schemas, observations, cancel_token):
        del user_text, tool_schemas, observations, cancel_token
        index = min(self.calls, len(self._actions) - 1)
        self.calls += 1
        return self._actions[index]


class Recorder:
    """Stands in for the UI and the voice path."""

    def __init__(self) -> None:
        self.status: list[str] = []
        self.answers: list[str] = []
        self.spoken: list[str] = []
        self.confirmations: list[ConfirmationRequest] = []

    def callbacks(self) -> SessionCallbacks:
        return SessionCallbacks(
            on_status=self.status.append,
            on_answer=self.answers.append,
            on_confirmation=self.confirmations.append,
            on_speak=self.spoken.append,
        )


def _build(tmp_path, actions, *, auto=None):
    """A session wired to a real registry, a real workspace and fake edges."""
    workspace = NotesWorkspace(tmp_path / "notes")
    registry = ToolRegistry()
    registry.register(SystemStatusTool(uptime=lambda: 5.0))
    registry.register(CreateNoteTool(workspace))
    recorder = Recorder()
    holder: dict = {}

    def _on_confirmation(request):
        holder["session"].announce_confirmation(request)
        if auto is not None:
            auto(holder["session"], request)

    runner = AgentRunner(
        ScriptedModel(actions),
        registry,
        trace_dir=tmp_path / "traces",
        on_confirmation_required=_on_confirmation,
    )
    session = HarnessSession(runner, recorder.callbacks())
    holder["session"] = session
    return session, runner, recorder, workspace


def _approve(session, request) -> None:
    session.confirm(request.run_id, request.call_id, request.fingerprint)


def _reject(session, request) -> None:
    session.reject(request.run_id, request.call_id)


def _notes(workspace) -> list[str]:
    root = workspace.root
    return sorted(path.name for path in root.iterdir()) if root.exists() else []


# --- case 1: a direct answer ------------------------------------------------


def test_a_direct_answer_reaches_ui_and_voice_once(tmp_path) -> None:
    session, _runner, recorder, workspace = _build(
        tmp_path, [ModelAction.answer("Mir geht es gut.")]
    )
    run = asyncio.run(session.ask("Wie geht es dir?"))

    assert run.status is RunStatus.COMPLETED
    assert recorder.status == [STATUS_WORKING, STATUS_DONE]
    assert recorder.answers == ["Mir geht es gut."]
    assert recorder.spoken == ["Mir geht es gut."]
    assert _notes(workspace) == []


# --- case 2: the status question --------------------------------------------


def test_the_status_question_uses_the_tool_and_speaks_once(tmp_path) -> None:
    session, _runner, recorder, _workspace = _build(
        tmp_path,
        [ModelAction.call("system_status"), ModelAction.answer("Der Harness ist erreichbar.")],
    )
    run = asyncio.run(session.ask("Wie ist dein Status?"))

    assert run.status is RunStatus.COMPLETED
    assert run.tool_calls == 1
    assert recorder.answers == ["Der Harness ist erreichbar."]
    assert len(recorder.spoken) == 1


# --- case 3: proposal and approval ------------------------------------------


def test_an_approved_note_is_written_exactly_once(tmp_path) -> None:
    session, _runner, recorder, workspace = _build(
        tmp_path,
        [ModelAction.call("create_note", NOTE_ARGS), ModelAction.answer("Die Notiz ist angelegt.")],
        auto=_approve,
    )
    run = asyncio.run(session.ask("Lege eine Notiz an."))

    assert run.status is RunStatus.COMPLETED
    assert _notes(workspace) == ["milch-kaufen.md"]
    assert (workspace.root / "milch-kaufen.md").read_text(encoding="utf-8") == "Milch kaufen"
    assert STATUS_NEEDS_CONFIRMATION in recorder.status
    assert recorder.status[-1] == STATUS_DONE
    assert recorder.spoken == [SPEECH_NEEDS_CONFIRMATION, "Die Notiz ist angelegt."]


def test_the_confirmation_shows_the_exact_target_and_content(tmp_path) -> None:
    session, _runner, recorder, _workspace = _build(
        tmp_path, [ModelAction.call("create_note", NOTE_ARGS), ModelAction.answer("fertig")],
        auto=_approve,
    )
    asyncio.run(session.ask("Lege eine Notiz an."))

    request = recorder.confirmations[0]
    assert request.tool_name == "create_note"
    assert request.target == "milch-kaufen.md"
    assert request.content == "Milch kaufen"
    assert "/" not in request.target
    assert str(tmp_path) not in repr(request)


# --- case 4: proposal and rejection -----------------------------------------


def test_a_rejected_note_is_never_written_and_no_success_is_spoken(tmp_path) -> None:
    session, _runner, recorder, workspace = _build(
        tmp_path, [ModelAction.call("create_note", NOTE_ARGS)], auto=_reject
    )
    run = asyncio.run(session.ask("Lege eine Notiz an."))

    assert run.status is RunStatus.FAILED
    assert run.error_code == "confirmation_rejected"
    assert _notes(workspace) == []
    assert recorder.status[-1] == STATUS_CANCELLED
    assert not any("angelegt" in line for line in recorder.spoken)


# --- case 5 and 6: cancel ---------------------------------------------------


def test_a_cancel_before_confirmation_writes_nothing(tmp_path) -> None:
    def _cancel(session, _request):
        session.cancel()

    session, _runner, recorder, workspace = _build(
        tmp_path, [ModelAction.call("create_note", NOTE_ARGS)], auto=_cancel
    )
    run = asyncio.run(session.ask("Lege eine Notiz an."))

    assert run.status is RunStatus.CANCELLED
    assert _notes(workspace) == []
    assert recorder.status[-1] == STATUS_CANCELLED
    assert recorder.answers == []
    # A cancel says nothing: a spoken "done" after an interruption is the worst
    # possible answer.
    assert recorder.spoken == [SPEECH_NEEDS_CONFIRMATION]


def test_a_cancel_during_the_model_step_ends_the_run(tmp_path) -> None:
    class _Waiting:
        def __init__(self) -> None:
            self.calls = 0

        async def next_action(self, *, user_text, tool_schemas, observations, cancel_token):
            del user_text, tool_schemas, observations
            self.calls += 1
            for _ in range(5000):
                if cancel_token.cancelled:
                    return ModelAction.answer("zu spät")
                await asyncio.sleep(0.001)
            raise AssertionError("nie freigegeben")

    workspace = NotesWorkspace(tmp_path / "notes")
    registry = ToolRegistry()
    registry.register(CreateNoteTool(workspace))
    recorder = Recorder()
    runner = AgentRunner(_Waiting(), registry, trace_dir=tmp_path / "traces")
    session = HarnessSession(runner, recorder.callbacks())

    async def go():
        task = session.start("Lege eine Notiz an.")
        for _ in range(2000):
            if runner.active_run_id:
                break
            await asyncio.sleep(0.001)
        session.cancel()
        return await task

    run = asyncio.run(go())
    assert run.status is RunStatus.CANCELLED
    assert _notes(workspace) == []
    assert recorder.spoken == []


# --- case 7: a tampered confirmation ----------------------------------------


def test_a_tampered_confirmation_is_refused_and_writes_nothing(tmp_path) -> None:
    """Change one character after the dialog was shown and the approval is void."""
    outcome: list[bool] = []

    def _tamper(session, request):
        forged = ConfirmationRequest.build(
            request.run_id, request.call, title=request.tool_name,
            target=request.target, content="ETWAS ANDERES",
        )
        # Collected, not asserted: the runner guards this listener, so a failing
        # assertion in here would be swallowed and hide the very acceptance the
        # test is looking for.
        outcome.append(session.confirm(request.run_id, request.call_id, forged.fingerprint))
        session.cancel()

    session, _runner, recorder, workspace = _build(
        tmp_path, [ModelAction.call("create_note", NOTE_ARGS)], auto=_tamper
    )
    run = asyncio.run(session.ask("Lege eine Notiz an."))

    assert outcome == [False], "eine gefälschte Bestätigung darf nicht zählen"
    assert run.status is RunStatus.CANCELLED
    assert _notes(workspace) == []


def test_an_approval_for_another_run_is_refused(tmp_path) -> None:
    outcome: list[bool] = []

    def _wrong_run(session, request):
        outcome.append(session.confirm("run-gibtsnicht", request.call_id, request.fingerprint))
        session.cancel()

    session, _runner, _recorder, workspace = _build(
        tmp_path, [ModelAction.call("create_note", NOTE_ARGS)], auto=_wrong_run
    )
    asyncio.run(session.ask("Lege eine Notiz an."))
    assert outcome == [False]
    assert _notes(workspace) == []


# --- case 8: a double confirmation ------------------------------------------


def test_a_second_confirmation_does_not_write_a_second_note(tmp_path) -> None:
    seen: list[bool] = []

    def _twice(session, request):
        seen.append(session.confirm(request.run_id, request.call_id, request.fingerprint))
        seen.append(session.confirm(request.run_id, request.call_id, request.fingerprint))

    session, _runner, _recorder, workspace = _build(
        tmp_path,
        [ModelAction.call("create_note", NOTE_ARGS), ModelAction.answer("fertig")],
        auto=_twice,
    )
    run = asyncio.run(session.ask("Lege eine Notiz an."))

    assert seen == [True, False], "die zweite Bestätigung darf nicht zählen"
    assert run.status is RunStatus.COMPLETED
    assert _notes(workspace) == ["milch-kaufen.md"]


# --- case 9: a tool failure -------------------------------------------------


def test_a_note_that_already_exists_fails_without_overwriting(tmp_path) -> None:
    session, _runner, recorder, workspace = _build(
        tmp_path,
        [ModelAction.call("create_note", NOTE_ARGS), ModelAction.answer("fertig")],
        auto=_approve,
    )
    workspace.root.mkdir(parents=True, exist_ok=True)
    (workspace.root / "milch-kaufen.md").write_text("original", encoding="utf-8")

    run = asyncio.run(session.ask("Lege eine Notiz an."))

    assert run.status is RunStatus.FAILED
    assert run.error_code == "note_exists"
    assert (workspace.root / "milch-kaufen.md").read_text(encoding="utf-8") == "original"
    assert recorder.status[-1] == STATUS_FAILED
    assert recorder.answers == ["Eine Notiz mit diesem Namen gibt es schon."]


# --- case 10: shutdown during confirmation ----------------------------------


def test_a_shutdown_while_waiting_can_never_still_write(tmp_path) -> None:
    outcome: list[bool] = []

    def _shutdown(session, request):
        session.shutdown()
        # Even a perfectly valid approval afterwards must be refused.
        outcome.append(session.confirm(request.run_id, request.call_id, request.fingerprint))

    session, _runner, _recorder, workspace = _build(
        tmp_path, [ModelAction.call("create_note", NOTE_ARGS)], auto=_shutdown
    )
    run = asyncio.run(session.ask("Lege eine Notiz an."))

    assert outcome == [False], "nach dem Shutdown darf nichts mehr eingelöst werden"
    assert run.status is RunStatus.CANCELLED
    assert _notes(workspace) == []


def test_a_closed_session_starts_nothing_new(tmp_path) -> None:
    session, _runner, _recorder, workspace = _build(
        tmp_path, [ModelAction.answer("hallo")]
    )
    session.shutdown()
    with pytest.raises(RunBusyError):
        asyncio.run(session.ask("noch etwas"))
    assert _notes(workspace) == []


# --- what never leaks -------------------------------------------------------


def test_nothing_internal_reaches_the_user(tmp_path) -> None:
    session, _runner, recorder, _workspace = _build(
        tmp_path, [ModelAction(ActionKind.FINAL)]  # malformed on purpose
    )
    run = asyncio.run(session.ask("was auch immer"))

    assert run.status is RunStatus.FAILED
    assert run.error_code == "model_protocol_error"
    for line in recorder.answers + recorder.spoken + recorder.status:
        assert str(tmp_path) not in line
        assert "Traceback" not in line
        assert "model_protocol_error" not in line


def test_the_trace_proves_proposal_approval_and_one_write(tmp_path) -> None:
    import json

    session, _runner, _recorder, _workspace = _build(
        tmp_path,
        [ModelAction.call("create_note", NOTE_ARGS), ModelAction.answer("fertig")],
        auto=_approve,
    )
    run = asyncio.run(session.ask("Lege eine Notiz an."))

    records = [
        json.loads(line)
        for line in (tmp_path / "traces" / f"{run.id}.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line
    ]
    events = [record["event"] for record in records]
    assert events.count("confirmation_requested") == 1
    assert events.count("confirmation_approved") == 1
    assert events.count("write_executed") == 1
    assert events.index("confirmation_requested") < events.index("confirmation_approved")
    assert events.index("confirmation_approved") < events.index("write_executed")

    blob = json.dumps(records, ensure_ascii=False)
    assert "Milch kaufen" not in blob, "weder Nutzertext noch Notizinhalt gehören hinein"
    assert str(tmp_path) not in blob
    # The target name is allowed; the absolute path is not.
    assert "milch-kaufen.md" in blob


def test_a_rejected_proposal_shows_no_write_in_the_trace(tmp_path) -> None:
    import json

    session, _runner, _recorder, _workspace = _build(
        tmp_path, [ModelAction.call("create_note", NOTE_ARGS)], auto=_reject
    )
    run = asyncio.run(session.ask("Lege eine Notiz an."))

    events = [
        json.loads(line)["event"]
        for line in (tmp_path / "traces" / f"{run.id}.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line
    ]
    assert "confirmation_rejected" in events
    assert "write_executed" not in events
    assert "confirmation_approved" not in events


# --- the application wiring, without GTK ------------------------------------


class _AppStub:
    """Only what `_build_harness` and the callbacks touch."""

    def __init__(self, tmp_path) -> None:
        self._harness = None
        self._chat = None
        self._pet = None
        self._speech = None
        self.toasts: list[str] = []
        self.spoken: list[str] = []
        self.submitted: list[object] = []
        outer = self

        class _Bridge:
            def submit(self, coro, **_kwargs):
                outer.submitted.append(coro)
                coro.close()
                return None

        self._bridge = _Bridge()

    def _toast(self, text: str) -> None:
        self.toasts.append(text)

    def notify_status(self, _title: str, text: str) -> None:
        self.toasts.append(text)


def _app_method(name: str):
    from kiki.application import KikiApplication

    return getattr(KikiApplication, name)


def test_the_ui_entry_point_refuses_empty_text(tmp_path) -> None:
    stub = _AppStub(tmp_path)
    assert _app_method("ask_harness")(stub, "   ") is False
    assert stub.submitted == []


def test_the_ui_entry_point_says_so_when_the_harness_is_missing(tmp_path) -> None:
    stub = _AppStub(tmp_path)
    stub._build_harness = lambda: None
    assert _app_method("ask_harness")(stub, "Wie ist dein Status?") is False
    assert stub.toasts == ["Der Agent steht gerade nicht bereit."]


def test_a_busy_harness_is_not_asked_twice(tmp_path) -> None:
    stub = _AppStub(tmp_path)

    class _Busy:
        busy = True

    stub._harness = _Busy()
    assert _app_method("ask_harness")(stub, "noch etwas") is False
    assert stub.toasts == ["KIKI arbeitet noch an der letzten Aufgabe."]
    assert stub.submitted == []


def test_the_run_goes_to_the_bridge_not_the_gtk_thread(tmp_path) -> None:
    """Model and tool work must never happen on the UI thread."""
    session, _runner, _recorder, _workspace = _build(
        tmp_path, [ModelAction.answer("fertig")]
    )
    stub = _AppStub(tmp_path)
    stub._harness = session

    assert _app_method("ask_harness")(stub, "Wie ist dein Status?") is True
    assert len(stub.submitted) == 1, "die Arbeit ging über die Bridge"


def test_every_callback_hops_to_the_gtk_thread(monkeypatch, tmp_path) -> None:
    from gi.repository import GLib

    scheduled: list[str] = []
    monkeypatch.setattr(
        GLib, "idle_add", lambda callback, *args, **kw: scheduled.append(callback.__name__)
    )
    stub = _AppStub(tmp_path)
    # The real handlers, so what gets scheduled is what the app would schedule.
    for name in ("_apply_harness_status", "_apply_harness_answer",
                 "_apply_harness_speak", "_apply_harness_confirmation"):
        setattr(type(stub), name, _app_method(name))
    for name, argument in (
        ("_on_harness_status", "KIKI arbeitet …"),
        ("_on_harness_answer", "fertig"),
        ("_on_harness_speak", "fertig"),
        ("_on_harness_confirmation", object()),
    ):
        _app_method(name)(stub, argument)

    assert scheduled == [
        "_apply_harness_status",
        "_apply_harness_answer",
        "_apply_harness_speak",
        "_apply_harness_confirmation",
    ]


def test_the_shutdown_closes_a_waiting_proposal(tmp_path) -> None:
    session, _runner, _recorder, _workspace = _build(
        tmp_path, [ModelAction.answer("fertig")]
    )
    stub = _AppStub(tmp_path)
    stub._harness = session

    _app_method("_close_harness")(stub)

    assert stub._harness is None
    assert session._closed is True


def test_speech_only_happens_when_tts_is_allowed(tmp_path) -> None:
    spoken: list[str] = []

    class _Speech:
        def say(self, text: str) -> None:
            spoken.append(text)

    class _Settings:
        def tts_allowed(self) -> bool:
            return False

    stub = _AppStub(tmp_path)
    stub._speech = _Speech()
    stub._settings = _Settings()
    _app_method("_apply_harness_speak")(stub, "fertig")
    assert spoken == []

    class _Allowed(_Settings):
        def tts_allowed(self) -> bool:
            return True

    stub._settings = _Allowed()
    _app_method("_apply_harness_speak")(stub, "fertig")
    assert spoken == ["fertig"]
