"""The assistant runtime: one runner for model/tool turns, no GTK.

This package unifies what were two orchestration paths -- the chat agent
loop and the harness runner, both removed once this package replaced them --
into one `AssistantRunner` on one `ToolGateway`. Not to be confused with
`kiki.agents` (plural), which adapts external coding agents like opencode.

Nothing here imports GTK, torch, CUDA or a model runtime. The model is behind
the step-adapter protocol with fakes only; tools run only through the gateway.
"""

from kiki.assistant.adapter import (
    ChatStepAdapter,
    ProviderStepAdapter,
    StepAdapter,
    StepEvent,
    as_step_adapter,
)
from kiki.assistant.run_service import (
    CORRELATION_MEMORY,
    LIMIT_TEXT,
    SPEECH_FAILED,
    SPEECH_NEEDS_CONFIRMATION,
    DuplicateCorrelationError,
    RunCallbacks,
    RunPausedError,
    RunService,
    failure_text,
)
from kiki.assistant.runner import (
    MAX_STEPS,
    RESULT_LIMIT,
    AssistantRunner,
    ModelProtocolFault,
    RunnerEvent,
)
from kiki.harness.confirmation import ConfirmationError, ConfirmationRequest
from kiki.harness.models import AgentRun, RunBusyError, RunStatus

__all__ = [
    "CORRELATION_MEMORY",
    "LIMIT_TEXT",
    "MAX_STEPS",
    "RESULT_LIMIT",
    "AgentRun",
    "AssistantRunner",
    "ChatStepAdapter",
    "ConfirmationError",
    "ConfirmationRequest",
    "DuplicateCorrelationError",
    "ModelProtocolFault",
    "ProviderStepAdapter",
    "RunBusyError",
    "RunCallbacks",
    "RunPausedError",
    "RunService",
    "RunnerEvent",
    "RunStatus",
    "SPEECH_FAILED",
    "SPEECH_NEEDS_CONFIRMATION",
    "StepAdapter",
    "StepEvent",
    "as_step_adapter",
    "failure_text",
]
