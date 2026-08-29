"""The run service: runner events in, sanitised German out.

The service is the only thing between the runner's categories and what a
person reads and hears. These tests hold the four exits -- status, answer,
confirmation, speech -- to the harness session's old promises, plus the one
the event world adds: a question nobody can present is refused, never waited
for. Every scenario runs under `wait_for`, so a broken promise surfaces as a
timeout, not as a suite that never ends.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from kiki.assistant.adapter import StepEvent
from kiki.assistant.run_service import (
    LIMIT_TEXT,
    SPEECH_FAILED,
    SPEECH_NEEDS_CONFIRMATION,
    RunCallbacks,
    RunService,
    failure_text,
)
from kiki.assistant.runner import AssistantRunner
from kiki.harness.confirmation import ConfirmationRequest
from kiki.harness.models import (
    CancelToken,
    HarnessStatusEvent,
    ModelAction,
    RunBusyError,
    RunStatus,
    ToolResult,
)
from kiki.storage.database import Database
from kiki.tools.audit import AuditLog
from kiki.tools.executor import ToolExecutor
from kiki.tools.gateway import ToolGateway
from kiki.tools.policy import RiskLevel, ToolPolicy
from kiki.tools.registry import ToolRegistry, ToolSpec

READ_SCHEMA = {"type": "object", "properties": {}, "required": []}
TEXT_SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
    "additionalProperties": False,
}

# The user text never reaches a log or a trace; here it proves the service
# passes it only to the runner, never out through a callback by accident.
USER_TEXT = "Bitte schreib ghp_testtoken nicht in irgendwelche Ausgaben."


# -- fakes --------------------------------------------------------------------


class ScriptedSteps:
    """Plays scripted steps. Each *argument* is one step: StepEvents ending in
    exactly one action."""

    def __init__(self, *steps: list[StepEvent]) -> None:
        self._steps = [list(step) for step in steps]
        self.calls = 0

    async def next_action_stream(
        self,
        *,
        user_text: str,
        tool_schemas: list[dict[str, Any]],
        observations: list[ToolResult],
        cancel_token: CancelToken,
    ) -> Any:
        del user_text, tool_schemas, cancel_token
        events = self._steps[self.calls] if self.calls < len(self._steps) else []
        self.calls += 1
        for event in events:
            yield event


class BlockingSteps:
    """Blocks inside the model step until cancelled."""

    async def next_action_stream(
        self,
        *,
        user_text: str,
        tool_schemas: list[dict[str, Any]],
        observations: list[ToolResult],
        cancel_token: CancelToken,
    ) -> Any:
        del user_text, tool_schemas, observations
        while not cancel_token.cancelled:
            await asyncio.sleep(0.001)
        yield StepEvent(kind="action", action=ModelAction.answer("zu spät"))


class Recorder:
    """Stands in for the UI and the voice path. Records, never throws."""

    def __init__(self) -> None:
        self.status: list[HarnessStatusEvent] = []
        self.answers: list[str] = []
        self.spoken: list[str] = []
        self.confirmations: list[ConfirmationRequest] = []

    def callbacks(self) -> RunCallbacks:
        return RunCallbacks(
            on_status=self.status.append,
            on_answer=self.answers.append,
            on_confirmation=self.confirmations.append,
            on_speak=self.spoken.append,
        )

    @property
    def status_codes(self) -> list[str]:
        return [event.message_code for event in self.status]


# -- helpers ------------------------------------------------------------------


def _spec(name: str, risk: RiskLevel, *, handler: Any = None) -> ToolSpec:
    return ToolSpec(
        name=name,
        title=name,
        description=f"{name} description",
        risk=risk,
        parameters=READ_SCHEMA,
        handler=handler or (lambda params: {"ran": name}),
        effect=f"{name} effect",
        auto_allow=True,
        requires_integration=False,
        model_callable=True,
    )


def _write_spec(name: str, counter: dict[str, int]) -> ToolSpec:
    def _handler(params: dict[str, Any]) -> dict[str, Any]:
        counter["writes"] += 1
        return {"ran": name}

    return ToolSpec(
        name=name,
        title=name,
        description=f"{name} description",
        risk=RiskLevel.WRITE,
        parameters=TEXT_SCHEMA,
        handler=_handler,
        effect=f"{name} effect",
        auto_allow=False,
        requires_integration=False,
        model_callable=True,
        sensitive_parameters=("text",),
    )


class _Live:
    def __init__(self) -> None:
        self.panic = False
        self.integrations = True


def _service(
    tmp_path: Path,
    model: Any,
    recorder: Recorder,
    *specs: ToolSpec,
) -> RunService:
    registry = ToolRegistry()
    for spec in specs:
        registry.register(spec)
    world = _Live()
    executor = ToolExecutor(registry, ToolPolicy("balanced"), AuditLog(Database(tmp_path / "kiki.db")))
    gateway = ToolGateway(
        executor,
        panic_check=lambda: world.panic,
        integrations_check=lambda: world.integrations,
    )
    runner = AssistantRunner(
        model,
        gateway,
        trace_dir=tmp_path / "trace",
        max_steps=4,
        max_tool_calls=6,
    )
    return RunService(runner, recorder.callbacks())


def delta(text: str) -> StepEvent:
    return StepEvent(kind="delta", text=text)


def answer(text: str) -> StepEvent:
    return StepEvent(kind="action", action=ModelAction.answer(text))


def calls(name: str, **arguments: Any) -> StepEvent:
    return StepEvent(kind="action", action=ModelAction.call(name, arguments))


def _ask(service: RunService, text: str = USER_TEXT) -> Any:
    # The insurance policy: a promise broken into a hang fails as a timeout
    # instead of taking the whole suite down with it.
    return asyncio.wait_for(service.ask(text), timeout=10)


# --- the settled answer ------------------------------------------------------


def test_a_completed_run_delivers_answer_and_speech_once(tmp_path):
    recorder = Recorder()
    service = _service(
        tmp_path,
        ScriptedSteps([delta("Gut "), delta("geht."), answer("Gut geht.")]),
        recorder,
    )

    async def scenario():
        return await _ask(service)

    run = asyncio.run(scenario())
    assert run.status is RunStatus.COMPLETED
    assert recorder.status_codes == ["working", "completed"]
    assert all(event.run_id == run.id for event in recorder.status)
    assert recorder.status[-1].terminal is True
    # Deltas are not answers: the settled text is what arrives, exactly once.
    assert recorder.answers == ["Gut geht."]
    assert recorder.spoken == ["Gut geht."]
    assert USER_TEXT not in recorder.answers[0]


def test_a_tool_run_reports_tool_running_and_settles(tmp_path):
    recorder = Recorder()
    service = _service(
        tmp_path,
        ScriptedSteps(
            [calls("status_tool")],
            [answer("Der Status steht.")],
        ),
        recorder,
        _spec("status_tool", RiskLevel.READ),
    )

    async def scenario():
        return await _ask(service)

    run = asyncio.run(scenario())
    assert run.status is RunStatus.COMPLETED
    # A read tool needs no pause, so there is no "working" after it: the run
    # goes straight from the tool to its terminal state.
    assert recorder.status_codes == ["working", "tool_running", "completed"]
    assert recorder.answers == ["Der Status steht."]


# --- failure and limit lines -------------------------------------------------


def test_a_limit_run_says_so_once(tmp_path):
    recorder = Recorder()
    service = _service(
        tmp_path,
        ScriptedSteps(*([[calls("status_tool")]] * 10)),
        recorder,
        _spec("status_tool", RiskLevel.READ),
    )

    async def scenario():
        return await _ask(service)

    run = asyncio.run(scenario())
    assert run.status is RunStatus.LIMIT_REACHED
    assert recorder.status_codes[-1] == "limit_reached"
    assert recorder.answers == [LIMIT_TEXT]
    assert recorder.spoken == [LIMIT_TEXT]


def test_a_provider_failure_gets_its_fixed_line(tmp_path):
    from kiki.harness.adapter import ProviderError

    class _Broken:
        async def next_action_stream(self, **_kwargs: Any) -> Any:
            raise ProviderError("provider_error")
            yield  # pragma: no cover - makes this an async generator

    recorder = Recorder()
    service = _service(tmp_path, _Broken(), recorder)

    async def scenario():
        return await _ask(service)

    run = asyncio.run(scenario())
    assert run.status is RunStatus.FAILED
    assert run.error_code == "provider_error"
    assert recorder.answers == ["Ich konnte das Modell nicht erreichen."]
    assert recorder.spoken == recorder.answers


def test_an_unknown_category_gets_the_generic_line():
    assert failure_text("kein_bekannter_code") == SPEECH_FAILED
    assert failure_text(None) == SPEECH_FAILED


# --- cancellation ------------------------------------------------------------


def test_a_cancelled_run_stays_silent(tmp_path):
    recorder = Recorder()
    service = _service(tmp_path, BlockingSteps(), recorder)

    async def scenario():
        task = asyncio.create_task(_ask(service))
        await asyncio.sleep(0.01)
        assert service.busy is True
        assert service.cancel() is True
        return await task

    asyncio.run(scenario())
    assert recorder.answers == []
    assert recorder.spoken == []
    assert recorder.status_codes[-1] == "cancelled"


def test_busy_is_true_while_a_run_is_active_and_free_after(tmp_path):
    recorder = Recorder()
    service = _service(tmp_path, BlockingSteps(), recorder)

    async def scenario():
        assert service.busy is False
        task = asyncio.create_task(_ask(service))
        await asyncio.sleep(0.01)
        mid = service.busy
        service.cancel()
        return mid, await task

    mid, run = asyncio.run(scenario())
    assert mid is True
    assert run.status is RunStatus.CANCELLED
    assert service.busy is False
    assert service.active_run_id is None


def test_a_second_ask_is_refused_and_the_first_survives(tmp_path):
    recorder = Recorder()
    # BlockingSteps keeps the first run genuinely active: a scripted quick
    # answer would be settled before the second ask arrives.
    service = _service(tmp_path, BlockingSteps(), recorder)

    async def scenario():
        first = asyncio.create_task(_ask(service))
        await asyncio.sleep(0.01)
        with pytest.raises(RunBusyError):
            await _ask(service)
        service.cancel()
        return await first

    run = asyncio.run(scenario())
    assert run.status is RunStatus.CANCELLED


# --- confirmation ------------------------------------------------------------


def test_an_approval_runs_the_write_and_completes(tmp_path):
    counter = {"writes": 0}
    recorder = Recorder()
    service = _service(
        tmp_path,
        ScriptedSteps(
            [calls("write_tool", text="meine Notiz")],
            [answer("Angelegt.")],
        ),
        recorder,
        _write_spec("write_tool", counter),
    )

    async def scenario():
        task = asyncio.create_task(_ask(service))
        # Wait for the card, then approve it with the id we were handed.
        for _ in range(500):
            if service.pending_confirmation is not None:
                break
            await asyncio.sleep(0.01)
        request = service.pending_confirmation
        assert request is not None
        assert request.request_id != ""
        assert service.confirm(request.run_id, request.call_id, request.request_id) is True
        return await task, request

    run, request = asyncio.run(scenario())
    assert counter["writes"] == 1
    assert run.status is RunStatus.COMPLETED
    assert len(recorder.confirmations) == 1
    assert recorder.confirmations[0].request_id == request.request_id
    assert "needs_confirmation" in recorder.status_codes
    assert SPEECH_NEEDS_CONFIRMATION in recorder.spoken
    assert recorder.answers == ["Angelegt."]


def test_a_wrong_request_id_is_refused_the_right_one_still_works(tmp_path):
    counter = {"writes": 0}
    recorder = Recorder()
    service = _service(
        tmp_path,
        ScriptedSteps(
            [calls("write_tool", text="meine Notiz")],
            [answer("Angelegt.")],
        ),
        recorder,
        _write_spec("write_tool", counter),
    )

    async def scenario():
        task = asyncio.create_task(_ask(service))
        for _ in range(500):
            if service.pending_confirmation is not None:
                break
            await asyncio.sleep(0.01)
        request = service.pending_confirmation
        assert request is not None
        assert service.confirm(request.run_id, request.call_id, "erfundene-id") is False
        assert service.confirm(request.run_id, request.call_id, "") is False
        assert service.confirm("run-falsch", request.call_id, request.request_id) is False
        assert service.confirm(request.run_id, request.call_id, request.request_id) is True
        return await task

    run = asyncio.run(scenario())
    assert counter["writes"] == 1
    assert run.status is RunStatus.COMPLETED


def test_a_rejection_runs_nothing_and_the_run_continues(tmp_path):
    counter = {"writes": 0}
    recorder = Recorder()
    service = _service(
        tmp_path,
        ScriptedSteps(
            [calls("write_tool", text="meine Notiz")],
            [answer("Dann eben nicht.")],
        ),
        recorder,
        _write_spec("write_tool", counter),
    )

    async def scenario():
        task = asyncio.create_task(_ask(service))
        for _ in range(500):
            if service.pending_confirmation is not None:
                break
            await asyncio.sleep(0.01)
        request = service.pending_confirmation
        assert request is not None
        assert service.reject(request.run_id, request.call_id) is True
        return await task

    run = asyncio.run(scenario())
    assert counter["writes"] == 0
    assert run.status is RunStatus.COMPLETED
    assert recorder.answers == ["Dann eben nicht."]


def test_a_question_nobody_can_present_is_refused_not_waited_for(tmp_path):
    """The slice-7 promise, carried into the event world.

    No `on_confirmation` callback: nobody is asked, nobody will answer. The
    service must refuse the question so the run settles -- not hang.
    """
    counter = {"writes": 0}
    recorder = Recorder()

    class _NoCard(Recorder):
        def callbacks(self) -> RunCallbacks:
            return RunCallbacks(
                on_status=self.status.append,
                on_answer=self.answers.append,
                on_speak=self.spoken.append,
            )

    rig = _NoCard()
    service = _service(
        tmp_path,
        ScriptedSteps(
            [calls("write_tool", text="meine Notiz")],
            [answer("Ohne Karte.")],
        ),
        rig,
        _write_spec("write_tool", counter),
    )
    del recorder

    async def scenario():
        return await _ask(service)

    run = asyncio.run(scenario())
    assert run.status is RunStatus.COMPLETED
    assert counter["writes"] == 0
    assert rig.confirmations == []
    # The run continues with the refusal as its observation and answers.
    assert rig.answers == ["Ohne Karte."]


def test_a_throwing_presentation_is_refused_not_waited_for(tmp_path):
    counter = {"writes": 0}

    def _broken(_request: ConfirmationRequest) -> None:
        raise RuntimeError("UI ist weg")

    service = _service(
        tmp_path,
        ScriptedSteps(
            [calls("write_tool", text="meine Notiz")],
            [answer("Trotzdem fertig.")],
        ),
        _ThrowingRecorder(_broken),
        _write_spec("write_tool", counter),
    )

    async def scenario():
        return await _ask(service)

    run = asyncio.run(scenario())
    assert run.status is RunStatus.COMPLETED
    assert counter["writes"] == 0


class _ThrowingRecorder(Recorder):
    """Records everything but lets the confirmation card explode."""

    def __init__(self, on_confirmation: Any) -> None:
        super().__init__()
        self._on_confirmation = on_confirmation

    def callbacks(self) -> RunCallbacks:
        return RunCallbacks(
            on_status=self.status.append,
            on_answer=self.answers.append,
            on_confirmation=self._on_confirmation,
            on_speak=self.spoken.append,
        )


def test_cancelling_a_pending_card_cancels_the_run(tmp_path):
    counter = {"writes": 0}
    recorder = Recorder()
    service = _service(
        tmp_path,
        ScriptedSteps(
            [calls("write_tool", text="meine Notiz")],
            [answer("nie erreicht")],
        ),
        recorder,
        _write_spec("write_tool", counter),
    )

    async def scenario():
        task = asyncio.create_task(_ask(service))
        for _ in range(500):
            if service.pending_confirmation is not None:
                break
            await asyncio.sleep(0.01)
        assert service.cancel() is True
        return await task

    run = asyncio.run(scenario())
    assert counter["writes"] == 0
    assert run.status is RunStatus.CANCELLED
    # The card was announced before the cancel, so that one line was spoken;
    # the cancel itself stays silent -- no answer, no spoken "done".
    assert recorder.answers == []
    assert recorder.spoken == [SPEECH_NEEDS_CONFIRMATION]


# --- shutdown ----------------------------------------------------------------


def test_shutdown_cancels_and_closes_the_service(tmp_path):
    recorder = Recorder()
    # BlockingSteps: the run is still inside its first model step when
    # shutdown arrives, so there is something to cancel.
    service = _service(tmp_path, BlockingSteps(), recorder)

    async def scenario():
        task = asyncio.create_task(_ask(service))
        await asyncio.sleep(0.01)
        service.shutdown()
        return await task

    run = asyncio.run(scenario())
    assert run.status is RunStatus.CANCELLED
    assert recorder.answers == []
    with pytest.raises(RunBusyError):
        asyncio.run(_ask(service))


# --- listener safety ---------------------------------------------------------


def test_a_throwing_listener_never_takes_the_run_down(tmp_path):
    recorder = Recorder()

    class _AngryStatus:
        def append(self, event: HarnessStatusEvent) -> None:
            recorder.status.append(event)
            if event.message_code == "working":
                raise RuntimeError("status listener kaputt")

    class _AngryRecorder(Recorder):
        def callbacks(self) -> RunCallbacks:
            return RunCallbacks(
                on_status=_AngryStatus().append,
                on_answer=self.answers.append,
                on_speak=self.spoken.append,
            )

    service = _service(
        tmp_path,
        ScriptedSteps([answer("Trotzdem fertig.")]),
        _AngryRecorder(),
    )

    async def scenario():
        return await _ask(service)

    run = asyncio.run(scenario())
    assert run.status is RunStatus.COMPLETED
