"""A small, self-contained agent harness: one run, one loop, one read-only tool.

**Not to be confused with `kiki.agents`** (plural), which adapts external coding
agents like opencode. This package is KIKI's own harness: a controlled
user-text → model → tool → answer loop with limits, cancellation and a local
structured trace, deliberately kept out of the UI, voice and production tool
policy so it can be evaluated on its own.

Nothing here imports GTK, torch, CUDA, a model runtime or the network. The model
is behind a protocol with fakes only; the real binding comes in a later slice.
"""

from kiki.harness.confirmation import (
    ConfirmationError,
    ConfirmationRequest,
    PendingConfirmation,
    fingerprint,
)
from kiki.harness.models import (
    ERROR_CODES,
    HARNESS_MESSAGE_CODES,
    ActionKind,
    AgentRun,
    CancelToken,
    HarnessStatusEvent,
    ModelAction,
    RunStatus,
    ToolCall,
    ToolResult,
    validate_action,
)
from kiki.harness.notes import CreateNoteTool, NotesWorkspace, slugify
from kiki.harness.runner import AgentRunner, RunBusyError
from kiki.harness.session import HarnessSession, SessionCallbacks
from kiki.harness.system_status import SystemStatusTool
from kiki.harness.tools import Tool, ToolRegistry
from kiki.harness.trace import TraceRecorder, TraceWriteError

__all__ = [
    "ERROR_CODES",
    "HARNESS_MESSAGE_CODES",
    "ConfirmationError",
    "ConfirmationRequest",
    "CreateNoteTool",
    "HarnessSession",
    "HarnessStatusEvent",
    "NotesWorkspace",
    "PendingConfirmation",
    "SessionCallbacks",
    "fingerprint",
    "slugify",
    "ActionKind",
    "AgentRun",
    "AgentRunner",
    "CancelToken",
    "ModelAction",
    "RunBusyError",
    "RunStatus",
    "SystemStatusTool",
    "Tool",
    "ToolCall",
    "ToolRegistry",
    "ToolResult",
    "TraceRecorder",
    "TraceWriteError",
    "validate_action",
]
