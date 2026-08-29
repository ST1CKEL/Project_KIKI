"""What KIKI is doing and just did — one bounded, content-free view.

Three things happen in the background: runs (agent and voice), routine
fires, and watch notices. Each has its own delivery path -- callbacks,
summaries, toasts -- and none of them can answer the questions the next
slices ask: is a run active right now (assistant pause), what did KIKI do
while I was away (run bar, pet menu). This module is the small shared
registry behind those views, and deliberately not more:

* it is not a bus and not a log. Producers keep their own paths; this is an
  observation sink they also feed;
* it is not persistent and not an audit. The audit and the trace are the
  revision-safe records with their own privacy rules; activity is ephemeral
  UI state that dies with the process;
* it is bounded. A ring, not a history: beyond the limit the oldest entry
  goes, because the question "what happened recently" has no interest in
  last week;
* it is content-free by construction. Entries carry kinds, codes and
  identifiers -- run ids, tool names, notice keys -- and a screened subject.
  No user text, no tool arguments, no answers, no notice prose. German
  sentences belong to the UI that displays this, not to the data.

Thread context: producers call from the asyncio thread, readers from GTK.
One lock keeps the ring consistent; the payloads are frozen and shared.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace

from kiki.harness.models import HarnessStatusEvent
from kiki.harness.trace import sanitize

DEFAULT_LIMIT = 100

# Kinds one activity record can carry. A new producer adds a kind here, in
# the open, not by inventing a string somewhere.
KIND_RUN = "run"
KIND_ROUTINE = "routine"
KIND_NOTICE = "notice"
# The assistant itself: pause and resume are activity too -- the feed should
# explain why nothing else happened in that window.
KIND_ASSISTANT = "assistant"
_KINDS: frozenset[str] = frozenset({KIND_RUN, KIND_ROUTINE, KIND_NOTICE, KIND_ASSISTANT})


@dataclass(frozen=True)
class Activity:
    """One observable thing that happened. Identifiers and codes, never content.

    `subject` is an identifier -- a tool name, a notice key -- and is screened
    on the way in: anything that looks like a path, a URL or a credential
    becomes `[entfernt]`, because a bounded UI feed must not become the place
    a slipped-in value feels at home.
    """

    kind: str
    code: str
    at: float
    run_id: str = ""
    subject: str = ""
    terminal: bool = False

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(f"unbekannte Aktivitätsart: {self.kind}")
        if not self.code:
            raise ValueError("eine Aktivität braucht einen Code")

    @classmethod
    def from_status(cls, event: HarnessStatusEvent) -> Activity:
        """A run's status transition, as activity. Codes and ids only."""
        return cls(
            kind=KIND_RUN,
            code=event.message_code,
            at=time.time(),
            run_id=event.run_id,
            terminal=event.terminal,
        )


Listener = Callable[[Activity], None]


class ActivityService:
    """A bounded ring of recent activity, and who is active right now."""

    def __init__(self, *, limit: int = DEFAULT_LIMIT, clock: Callable[[], float] = time.time) -> None:
        if int(limit) < 1:
            raise ValueError("das Aktivitätsfenster muss mindestens 1 sein")
        self._limit = int(limit)
        self._clock = clock
        self._ring: deque[Activity] = deque(maxlen=self._limit)
        self._listeners: list[Listener] = []
        self._lock = threading.Lock()

    @property
    def limit(self) -> int:
        return self._limit

    def record(self, activity: Activity) -> None:
        """One thing happened. Bounded, screened, announced to whoever listens."""
        entry = activity
        if entry.at <= 0:
            # A producer that cannot tell time still gets an honest order.
            entry = replace(entry, at=self._clock())
        entry = replace(entry, subject=str(sanitize(entry.subject)))
        with self._lock:
            self._ring.append(entry)
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(entry)
            except Exception:
                # A view that breaks must not take the recording down with it.
                pass

    def recent(self, limit: int = 20) -> list[Activity]:
        """The newest entries first. Never more than the ring holds."""
        with self._lock:
            take = max(0, min(int(limit), len(self._ring)))
            return list(self._ring)[-take:][::-1]

    def active_run(self) -> Activity | None:
        """The newest run entry that has not settled, or None.

        One run at a time is the runner's own promise, so the newest
        unsettled entry *is* the active run; a terminal entry closes it.
        """
        with self._lock:
            for activity in reversed(self._ring):
                if activity.kind == KIND_RUN:
                    return None if activity.terminal else activity
        return None

    def subscribe(self, listener: Listener) -> None:
        """Be told about every entry. For views that show activity live."""
        with self._lock:
            if listener not in self._listeners:
                self._listeners.append(listener)

    def unsubscribe(self, listener: Listener) -> None:
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    # --- convenience constructors for the three producers -------------------

    def record_status(self, event: HarnessStatusEvent) -> None:
        self.record(Activity.from_status(event))

    def record_routine(self, *, code: str, tool: str, run_id: str = "") -> None:
        self.record(
            Activity(
                kind=KIND_ROUTINE,
                code=code,
                at=self._clock(),
                run_id=run_id,
                subject=tool,
            )
        )

    def record_notice(self, *, key: str, severity: str) -> None:
        self.record(
            Activity(
                kind=KIND_NOTICE,
                code=severity,
                at=self._clock(),
                subject=key,
            )
        )

    def record_assistant(self, code: str) -> None:
        """The assistant's own state changes: paused, resumed."""
        self.record(
            Activity(
                kind=KIND_ASSISTANT,
                code=code,
                at=self._clock(),
            )
        )


def first_of_kind(entries: list[Activity], kind: str) -> Activity | None:
    """Newest entry of one kind, or None. For views that ask one question."""
    for entry in entries:
        if entry.kind == kind:
            return entry
    return None
