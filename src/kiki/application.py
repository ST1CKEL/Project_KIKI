from __future__ import annotations

import asyncio
import logging
import sys
import time
import uuid
from pathlib import Path

# This import is intentionally isolated: it selects and loads GI versions
# before any other KIKI module can import a GTK widget.
# isort: off
from kiki.ui.gi_bootstrap import Adw, Gdk, Gio, GLib, Gtk
# isort: on

from kiki import APP_ID, APP_NAME
from kiki.agents.broker import AgentBroker
from kiki.agents.session_service import SessionService
from kiki.ai.chat_service import ChatService
from kiki.character.assets import ensure_character_pack
from kiki.character.state_machine import CharacterState, CharacterStateMachine
from kiki.config.settings import Settings, load_settings, save_settings
from kiki.harness.confirmation import ConfirmationRequest
from kiki.harness.models import HarnessStatusEvent
from kiki.harness.notes import NotesWorkspace
from kiki.harness.notes import spec as harness_note_spec
from kiki.harness.system_status import spec as harness_status_spec
from kiki.integrations.datetime import DateTimeIntegration
from kiki.integrations.disk import DiskIntegration
from kiki.integrations.networkmanager import NetworkManagerIntegration
from kiki.integrations.upower import UPowerIntegration
from kiki.paths import cache_dir, database_path, icon_search_path, state_dir, user_data_dir
from kiki.platform.autostart import set_enabled as set_autostart
from kiki.platform.capabilities import detect_capabilities
from kiki.routines.engine import RoutineEngine
from kiki.routines.metrics import IntegrationMetrics
from kiki.routines.repository import RoutineRepository
from kiki.routines.service import RoutineService
from kiki.routines.skill import RoutinesSkill
from kiki.runners.local import LocalWorkspaceRunner
from kiki.runtime.activity import ActivityService
from kiki.runtime.async_bridge import AsyncBridge
from kiki.runtime.event_bus import EventBus
from kiki.runtime.pause import AssistantPause
from kiki.skills.desktop import DesktopPerceptionSkill
from kiki.skills.registry import SkillRegistry
from kiki.skills.system_status import SystemStatusSkill
from kiki.storage.agent_session_repository import AgentSessionRepository
from kiki.storage.approval_repository import ApprovalRepository
from kiki.storage.audit_repository import AgentAuditRepository
from kiki.storage.chat_repository import ChatRepository
from kiki.storage.database import Database
from kiki.storage.memory_repository import MemoryRepository
from kiki.storage.secrets import create_secret_store
from kiki.storage.workspace_repository import WorkspaceRepository
from kiki.tools.app_launch_tools import AppLaunchSkill
from kiki.tools.audio_tools import AudioControlSkill
from kiki.tools.audit import AuditLog
from kiki.tools.display_tools import DisplayControlSkill
from kiki.tools.executor import ToolExecutor
from kiki.tools.gateway import ToolGateway
from kiki.tools.launch_tools import DesktopLaunchSkill
from kiki.tools.media_tools import MediaControlSkill
from kiki.tools.memory_tools import MemorySkill
from kiki.tools.network_tools import NetworkControlSkill
from kiki.tools.policy import RiskLevel, ToolPolicy
from kiki.tools.power_tools import PowerControlSkill
from kiki.tools.registry import ActionPreview, ToolRegistry
from kiki.tools.session_tools import SessionControlSkill
from kiki.ui.chat_window import ChatWindow
from kiki.ui.coding_session_window import CodingSessionWindow
from kiki.ui.confirmation_dialog import present_confirmation
from kiki.ui.css import APP_CSS
from kiki.ui.desktop_control_model import is_desktop_control_intent
from kiki.ui.desktop_control_window import DesktopControlWindow
from kiki.ui.pet_window import PetWindow
from kiki.ui.preferences_window import PreferencesWindow
from kiki.ui.run_bar_model import text_for as run_status_text
from kiki.ui.workspace_manager_window import WorkspaceManagerWindow
from kiki.voice.answer import VoiceAnswerDelivery, plan_voice_answer
from kiki.voice.director import SpeechDirector
from kiki.voice.follow_up import FollowUpTurn
from kiki.voice.recorder import AudioRecorder, RecorderError
from kiki.voice.stt import SpeechError, ensure_vosk_model, transcribe_wav, vosk_model_ready
from kiki.voice.system_tts import synthesize_system_wav
from kiki.voice.tts.policy import VoicePolicyConfig, VoiceResponsePolicy
from kiki.voice.tts_client import TtsError, synthesize_wav
from kiki.voice.tts_player import PipeWirePlayer
from kiki.voice.wake import (
    MicrophoneStream,
    UtteranceStream,
    WakeError,
    WakeWordListener,
    wake_word_supported,
)
from kiki.watch.models import Notice
from kiki.watch.notifier import Notifier, NotifierPolicy, parse_clock
from kiki.watch.service import WatchService
from kiki.watch.watchers import BatteryWatcher, DiskWatcher
from kiki.workspaces.registry import WorkspaceRegistry

log = logging.getLogger(__name__)


class KikiApplication(Adw.Application):
    def __init__(self) -> None:
        super().__init__(application_id=APP_ID, flags=Gio.ApplicationFlags.DEFAULT_FLAGS)
        self.set_application_id(APP_ID)
        self._settings = load_settings()
        self._capabilities = detect_capabilities()
        self._bridge = AsyncBridge()
        self._bus = EventBus()
        self._machine = CharacterStateMachine()
        self._db: Database | None = None
        # Only built when tts.use_controller_route is on; needs closing at exit.
        self._voice_controller: object | None = None
        # Built on first use; None means the harness is simply not available.
        self._harness: object | None = None
        self._pet: PetWindow | None = None
        self._chat: ChatWindow | None = None
        self._prefs: PreferencesWindow | None = None
        self._service: ChatService | None = None
        self._chats: ChatRepository | None = None
        self._pack = None
        self._executor: ToolExecutor | None = None
        self._memories: MemoryRepository | None = None
        self._recorder = AudioRecorder()
        self._voice_busy = False
        self._wake: WakeWordListener | None = None
        self._wake_starting = False
        self._follow_up = FollowUpTurn()
        self._watch: WatchService | None = None
        self._notifier: Notifier | None = None
        self._routines: RoutineRepository | None = None
        self._routines_service: RoutineService | None = None
        # The bounded, content-free view of what KIKI is doing and just did.
        self._activity = ActivityService()
        # The assistant pause: no new runs, no routine fires, no notices.
        # Session state, separate from the character pause and from panic.
        self._assistant_pause = AssistantPause()
        self._speech: SpeechDirector | None = None
        self._tts_remote_retry_after = 0.0
        self._tts_remote_error = ""
        self._session_service: SessionService | None = None
        self._coding: CodingSessionWindow | None = None
        self._ws_manager: WorkspaceManagerWindow | None = None
        self._desktop_control: DesktopControlWindow | None = None
        self.connect("startup", self._on_startup)
        self.connect("activate", self._on_activate)
        self.connect("shutdown", self._on_shutdown)

    def _on_startup(self, *_args: object) -> None:
        Adw.StyleManager.get_default().set_color_scheme(Adw.ColorScheme.DEFAULT)
        self._install_css()
        self._install_icons()
        self._bridge.start()
        self._db = Database(database_path())
        chats = ChatRepository(self._db)
        self._chats = chats
        secrets = create_secret_store()
        self._secrets = secrets
        memories = MemoryRepository(self._db)
        self._memories = memories
        workspaces = WorkspaceRegistry(
            WorkspaceRepository(self._db),
            allowed_roots=self._settings.workspaces.allowed_roots,
        )
        self._session_service = SessionService(
            workspaces,
            AgentSessionRepository(self._db),
            ApprovalRepository(self._db),
            AgentAuditRepository(self._db),
            AgentBroker(opencode_binary=self._settings.agents.opencode_binary),
            LocalWorkspaceRunner(),
            plan_first=self._settings.agents.plan_first,
        )
        tools = ToolRegistry()
        skills = SkillRegistry()
        disk_path = (self._settings.integrations.disk.extra or {}).get("path") or None
        upower = UPowerIntegration()
        disk = DiskIntegration(disk_path)
        skills.register(
            SystemStatusSkill(
                [
                    DateTimeIntegration(),
                    upower,
                    NetworkManagerIntegration(),
                    disk,
                ]
            )
        )
        self._build_watchers(upower, disk)
        skills.register(DesktopPerceptionSkill())
        skills.register(MemorySkill(memories))
        skills.register(DesktopLaunchSkill(workspaces))
        skills.register(MediaControlSkill())
        skills.register(AudioControlSkill())
        skills.register(DisplayControlSkill())
        skills.register(AppLaunchSkill())
        skills.register(SessionControlSkill())
        skills.register(NetworkControlSkill())
        skills.register(PowerControlSkill())
        routines_repo = RoutineRepository(self._db)
        skills.register(RoutinesSkill(routines_repo, tools))
        skills.install_into(tools)
        # The harness once carried these two itself. They live here now, so
        # there is one list of what exists and one policy that gates it.
        tools.register(harness_status_spec())
        tools.register(harness_note_spec(NotesWorkspace(user_data_dir() / "notes")))
        executor = ToolExecutor(
            tools, ToolPolicy(self._settings.tools.autonomy), AuditLog(self._db)
        )
        self._executor = executor
        self._routines = routines_repo
        # Routine fires go through the same door as everything else: the
        # gateway's live switches instead of the engine's snapshots, and only
        # for a recipe the repository holds exactly as confirmed. The engine
        # itself is untouched -- it asked for an executor, it got one shape,
        # a stricter one.
        from kiki.tools.routine_gateway import RoutineToolGateway

        self._routines_service = RoutineService(
            RoutineEngine(
                routines_repo,
                RoutineToolGateway(
                    ToolGateway(
                        executor,
                        panic_check=lambda: self._settings.app.privacy_panic,
                        integrations_check=self._settings.integrations_active,
                    ),
                    routines_repo,
                    activity=self._activity,
                    pause=self._assistant_pause,
                ),
                IntegrationMetrics(upower, disk).snapshot,
                panic_check=lambda: self._settings.app.privacy_panic,
                integrations_check=self._settings.integrations_active,
            )
        )
        self._service = ChatService(
            self._settings,
            chats,
            secrets,
            self._bus,
            executor,
            confirm=self._confirm_model_action,
            memories=memories,
            trace_dir=state_dir() / "assistant",
        )
        self._pack = ensure_character_pack(self._settings.character.id)
        wav_dir = cache_dir() / "tts"
        wav_dir.mkdir(parents=True, exist_ok=True)
        self._speech = SpeechDirector(
            synthesize=self._synthesize_tts,
            player=PipeWirePlayer(),
            submit=self._bridge.submit,
            wav_dir=wav_dir,
            on_speaking=self._on_tts_speaking,
            on_audio_started=self._on_tts_audio_started,
            on_idle=self._on_tts_idle,
            on_error=self._on_tts_error,
            controller=self._build_voice_controller(),
            use_controller_route=self._settings.tts.use_controller_route,
            policy=self._voice_policy,
        )
        self._register_actions()
        self._subscribe_ui_event("chat.stream.start", self._on_stream_start)
        self._subscribe_ui_event("chat.stream.delta", self._on_stream_delta)
        self._subscribe_ui_event("chat.stream.speaking", self._on_stream_speaking)
        self._subscribe_ui_event("chat.stream.tool_start", self._on_stream_tool)
        self._subscribe_ui_event("chat.stream.done", self._on_stream_done)
        self._subscribe_ui_event("chat.stream.error", self._on_stream_error)
        if self._settings.app.autostart:
            set_autostart(True)

    # --- agent harness ------------------------------------------------------

    def _build_harness(self):
        """The agent path, or nothing. Never fatal at startup.

        `/agent` is the legacy and developer path since the chat itself runs
        on the unified runner. It uses the same stack now — step adapter,
        `AssistantRunner`, `ToolGateway`, `RunService` — so there is no second
        policy, no second confirmation system and no second executor behind
        it. Built lazily and defensively: a missing provider, an unusable
        executor or any import problem leaves KIKI exactly as she was.
        """
        try:
            from kiki.ai.factory import active_model, create_provider
            from kiki.assistant import (
                AssistantRunner,
                ProviderStepAdapter,
                RunCallbacks,
                RunService,
            )

            provider = create_provider(self._settings, self._secrets)
            if not hasattr(provider, "stream_chat_tools"):
                log.info("agent path disabled: the provider cannot call tools")
                return None
            executor = self._executor
            if executor is None:
                log.info("agent path disabled: the tool executor is not up yet")
                return None
            # Same gateway contract as the chat path: live sources, so the
            # panic switch reaches an agent run that is already under way.
            gateway = ToolGateway(
                executor,
                panic_check=lambda: self._settings.app.privacy_panic,
                integrations_check=self._settings.integrations_active,
            )
            adapter = ProviderStepAdapter(
                provider,
                model=active_model(self._settings),
                system_prompt=self._settings.compose_prompt(),
                # The old agent path never set a window and ran on the
                # provider's 4096 default -- the exact trap the architecture
                # documents for thinking models. One configuration now.
                num_ctx=self._settings.ai.ollama.num_ctx,
            )
            runner = AssistantRunner(
                adapter,
                gateway,
                profile="observe",
                trace_dir=state_dir() / "assistant",
                max_steps=self._settings.tools.max_steps,
                max_tool_calls=self._settings.tools.max_tool_calls,
            )
            service = RunService(
                runner,
                RunCallbacks(
                    on_status=self._on_harness_status,
                    on_answer=self._on_harness_answer,
                    on_confirmation=self._on_harness_confirmation,
                    on_speak=self._on_harness_speak,
                ),
                paused=self._assistant_pause,
            )
            self._harness = service
            return service
        except Exception:
            log.warning("agent path unavailable", exc_info=False)
            return None

    def _on_harness_bridge_error(self, _exc: BaseException) -> None:
        from kiki.assistant import RunPausedError

        self._follow_up.cancel()
        if isinstance(_exc, RunPausedError):
            GLib.idle_add(self._toast, "KIKI macht gerade Pause.")
            return
        log.warning("harness bridge task failed", exc_info=False)
        GLib.idle_add(self._toast, "KIKI konnte die Aufgabe nicht starten.")

    def ask_harness(self, user_text: str) -> bool:
        """Start one agent run from the UI. Returns False when it cannot."""
        text = (user_text or "").strip()
        if not text:
            return False
        session = self._harness or self._build_harness()
        if session is None:
            self._toast("Der Agent steht gerade nicht bereit.")
            return False
        if session.busy:
            self._toast("KIKI arbeitet noch an der letzten Aufgabe.")
            return False
        try:
            self._bridge.submit(
                session.ask(text),
                on_error=self._on_harness_bridge_error,
            )
        except Exception:
            log.debug("could not hand the harness run to the bridge", exc_info=True)
            self._toast("KIKI konnte die Aufgabe nicht starten.")
            return False
        return True

    def cancel_harness(self, run_id: str | None = None) -> bool:
        """Cancel the active harness run. Returns False when there is none."""
        if self._harness is None:
            return False
        return self._harness.cancel(run_id=run_id)

    # Every callback below arrives on the asyncio thread and hops to GTK, the
    # same way the wake word and the watcher already report.

    def toggle_assistant_pause(self) -> None:
        """The assistant pause: no new runs, fires or notices.

        Separate from the character pause (an animation) and from panic (a
        privacy emergency). The active run is left to settle; an open
        approval card stays answerable; typed chat keeps working, because
        someone typing is asking, not background work.
        """
        paused = self._assistant_pause.toggle()
        self._activity.record_assistant("paused" if paused else "resumed")
        if paused:
            self._toast("KIKI macht Pause. Keine neuen Aufgaben bis zum Fortsetzen.")
        else:
            self._toast("KIKI ist wieder bereit.")

    def _on_harness_status(self, event: HarnessStatusEvent) -> None:
        # Runs feed the shared activity view: ids and codes only, never text.
        self._activity.record_status(event)
        GLib.idle_add(self._apply_harness_status, event)

    def _apply_harness_status(self, event: HarnessStatusEvent) -> bool:
        if self._harness is None:
            self._follow_up.cancel()
            return False
        if event.terminal:
            self._follow_up.mark_terminal(
                cancelled=event.message_code == "cancelled"
            )
        try:
            if self._chat is not None and hasattr(self._chat, "set_run_status"):
                self._chat.set_run_status(
                    run_id=event.run_id,
                    message_code=event.message_code,
                    terminal=event.terminal,
                )
            elif event.message_code != "completed":
                # No chat window: the same fixed sentences, from the same
                # single vocabulary the run bar uses.
                try:
                    self._toast(run_status_text(event.message_code))
                except ValueError:
                    log.debug("unknown run message_code: %s", event.message_code)
        except Exception:
            log.warning("run status could not be delivered")
            self._harness_delivery_failed()
        return False

    def _on_harness_answer(self, text: str) -> None:
        GLib.idle_add(self._apply_harness_answer, text)

    def _apply_harness_answer(self, text: str) -> bool:
        """Put one finished answer into the chat the user already has.

        Through `append_note`, the same call the coding summary uses: it creates
        the conversation if needed, stores the text as an assistant message and
        adds one bubble. Not through the EventBus — `ChatWindow` does not listen
        to it at all, it drives itself from `ChatService.send()`. The bus is a
        side channel for the character state machine and for voice, so an event
        published there would have had no receiver whichever name it carried.

        No fake token stream either: the harness produces one complete answer,
        and pretending it arrived in pieces would only invent work.
        """
        if self._harness is None:
            # Shutdown happened while this sat in the idle queue.
            return False
        try:
            if self._follow_up.active and self._follow_up.terminal:
                delivery = self._plan_voice_answer(text)
                if delivery.open_chat:
                    self.open_chat()
            if self._chat is not None:
                self._chat.append_note(text, toast=None)
            else:
                self.notify_status("KIKI", text)
        except Exception:
            # A delivery failure is a category, never a traceback into GTK, and
            # it must leave any pending confirmation exactly as it was.
            log.warning("harness answer could not be delivered")
            self._harness_delivery_failed()
        return False

    def _harness_delivery_failed(self) -> None:
        try:
            self._toast("KIKI konnte die Antwort nicht anzeigen.")
        except Exception:
            log.debug("could not report the delivery failure either", exc_info=True)

    def _on_harness_speak(self, text: str) -> None:
        GLib.idle_add(self._apply_harness_speak, text)

    def _apply_harness_speak(self, text: str) -> bool:
        """Speak one finished answer through the existing one-shot path.

        `SpeechDirector.say` is what the greeting, the watcher notices and the
        timer notifications use: it routes to the active player, obeys the mute
        switch and stays out of the token loop.
        """
        if self._harness is None:
            self._follow_up.cancel()
            return False
        spoken = text
        if self._follow_up.active and self._follow_up.terminal:
            delivery = self._plan_voice_answer(text)
            spoken = delivery.spoken_text
            if delivery.open_chat:
                self.open_chat()
        speech = self._speech
        if self._settings.tts_allowed() and speech is not None:
            try:
                # `say()` first stops an older intermediate prompt. Marking
                # delivery afterwards prevents that prompt's idle callback
                # from opening follow-up before the final answer starts.
                speech.say(spoken)
            except Exception:
                self._follow_up.cancel()
                log.warning("harness speech could not be delivered")
                self._harness_delivery_failed()
                return False
            delivered = self._follow_up.mark_response_delivered()
            if delivered and not speech.active:
                self._try_arm_follow_up()
            return False
        if self._follow_up.mark_response_delivered():
            self._try_arm_follow_up()
        return False

    def _on_harness_confirmation(self, request: ConfirmationRequest) -> None:
        # The service has already announced the question (status event and
        # speech); this callback only puts the card on screen. A failure here
        # is not survivable silence: the service refuses what nobody can
        # answer, so nothing runs unasked.
        GLib.idle_add(self._apply_harness_confirmation, request)

    def _apply_harness_confirmation(self, request: ConfirmationRequest) -> bool:
        """Show the write confirmation on screen, bound to this exact request."""
        session = self._harness
        if session is None:
            return False
        preview = ActionPreview(
            tool=request.tool_name,
            title=request.title,
            params={"Datei": request.target},
            target=request.target,
            effect=request.content,
            risk=RiskLevel.WRITE,
            reason="KIKI hat diese Aktion vorgeschlagen.",
            request_id=request.request_id,
        )

        def _settle(approved: bool) -> None:
            # The exact run and call that were on screen, answered with the id
            # this dialog was handed. The dialog computes no authorisation of
            # its own; the broker behind the gateway mints the grant.
            if approved:
                session.confirm(request.run_id, request.call_id, request.request_id)
            else:
                session.reject(request.run_id, request.call_id)

        present_confirmation(self._chat or self._pet, preview, _settle)
        return False

    def _close_harness(self) -> None:
        """Shutdown: nothing pending may still be written."""
        session, self._harness = self._harness, None
        if session is not None:
            session.shutdown()
        if self._chat is not None and hasattr(self._chat, "clear_run_status"):
            self._chat.clear_run_status()

    def _voice_policy(self) -> VoiceResponsePolicy:
        """What KIKI may read out loud, from the user's own config.

        Built here rather than left to the director's defaults so the settings
        file actually governs speech. Passed to the director uncalled, so it is
        re-read per utterance: a category switched off in preferences applies to
        the answer already being spoken, not only after a restart.
        """
        chosen = self._settings.voice.response_policy
        return VoiceResponsePolicy(
            VoicePolicyConfig(
                speak_code=chosen.speak_code,
                speak_logs=chosen.speak_logs,
                speak_urls=chosen.speak_urls,
                speak_paths=chosen.speak_paths,
                speak_tables=chosen.speak_tables,
                speak_secrets=chosen.speak_secrets,
            )
        )

    def _plan_voice_answer(self, text: str) -> VoiceAnswerDelivery:
        chosen = self._settings.voice.response_policy
        return plan_voice_answer(
            text,
            policy=self._voice_policy(),
            concise=chosen.concise_answers,
            open_chat_for_details=chosen.open_chat_for_details,
        )

    def _build_voice_controller(self):
        """The opt-in route, built only when the flag asks for it.

        Constructed lazily so the adapters — and httpx with them — stay out of
        the process on the default path. Returns None on any failure: the
        director then keeps the file-based route rather than losing speech.
        """
        if not self._settings.tts.use_controller_route:
            return None
        try:
            from kiki.voice.tts.composition import build_controller_route
            from kiki.voice.tts.controller import VoicePlaybackController

            tts = self._settings.tts
            # Which pair, and how much to hold back, is decided in one place.
            # Nothing about streaming is chosen here.
            route = build_controller_route(
                base_url=tts.base_url, speaker=tts.speaker, language=tts.language
            )
            # synthesize() refuses to run before load(), and load() is a health
            # probe against the service — so it belongs on the bridge, not here
            # on the GTK thread during startup.
            self._bridge.submit(
                route.provider.load(),
                on_error=self._on_voice_route_unavailable,
            )
            self._voice_controller = VoicePlaybackController(
                route.provider,
                route.sink,
                on_audio_started=self._on_voice_audio_started,
                prebuffer_chunks=route.prebuffer_chunks,
            )
            return self._voice_controller
        except Exception:
            log.exception("could not build the controller voice route — staying on the old one")
            return None

    def _on_voice_audio_started(self, event: object) -> None:
        """Runs on the asyncio thread — hand the event to GTK, touch nothing.

        Same route as the wake word and the watcher use: GLib.idle_add. The
        character state machine may only be moved on the main thread, and this
        must not block the audio path while it happens.
        """
        GLib.idle_add(self._apply_audio_started, event)

    def _apply_audio_started(self, event: object) -> bool:
        if self._speech is not None:
            self._speech.audio_started(event)
        return False

    def _on_voice_route_unavailable(self, exc: BaseException) -> None:
        """The opt-in route could not be brought up. Go back to the old one.

        Without this the flag would stay on and every single sentence would fail
        with `not_ready` — speech off, rather than speech the old way. The
        message comes from the adapter, which already strips URLs; no
        configuration value is logged here.
        """
        log.warning("controller voice route unavailable, using the file route: %s", exc)
        if self._speech is not None:
            self._speech.disable_controller_route()

    def _install_css(self) -> None:
        provider = Gtk.CssProvider()
        provider.load_from_string(APP_CSS)
        display = Gdk.Display.get_default()
        if display is not None:
            Gtk.StyleContext.add_provider_for_display(
                display, provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
            )

    def _subscribe_ui_event(self, event: str, listener) -> None:
        def _queue(**payload: object) -> None:
            def _run() -> bool:
                listener(**payload)
                return False

            GLib.idle_add(_run)

        self._bus.subscribe(event, _queue)

    def _install_icons(self) -> None:
        display = Gdk.Display.get_default()
        if display is None:
            return
        path = icon_search_path()
        if path.is_dir():
            Gtk.IconTheme.get_for_display(display).add_search_path(str(path))

    def _register_actions(self) -> None:
        def _add(name: str, handler) -> None:
            action = Gio.SimpleAction.new(name, None)
            action.connect("activate", handler)
            self.add_action(action)

        _add("chat", lambda *_: self.open_chat())
        _add("pause", lambda *_: self._machine.pause())
        _add("resume", lambda *_: self._machine.resume())
        _add("assistant-pause-toggle", lambda *_: self.toggle_assistant_pause())
        _add("preferences", lambda *_: self.open_preferences())
        _add("reload-character", lambda *_: self.reload_character())
        _add("window-menu", lambda *_: self._pet.show_window_menu() if self._pet else None)
        _add("quit", lambda *_: self.quit())
        _add("privacy-panic", lambda *_: self._toggle_panic())
        _add("screenshot", lambda *_: self.request_screenshot())
        _add("voice-toggle", lambda *_: self.toggle_voice())
        _add("tts-stop", lambda *_: self.stop_speech())
        _add("coding", lambda *_: self.open_coding())
        _add("workspaces", lambda *_: self.open_workspaces())
        _add("desktop-control", lambda *_: self.open_desktop_control())
        self.set_accels_for_action("app.quit", ["<Control>q"])
        self.set_accels_for_action("app.chat", ["<Control>period"])
        self.set_accels_for_action("app.preferences", ["<Control>comma"])
        self.set_accels_for_action("app.coding", ["<Control><Shift>c"])
        self.set_accels_for_action("app.desktop-control", ["<Control><Shift>p"])
        self.set_accels_for_action("app.screenshot", ["<Control><Shift>s"])
        self.set_accels_for_action("app.voice-toggle", ["<Control><Shift>v"])

    def _on_activate(self, *_args: object) -> None:
        if self._pack is None:
            self._pack = ensure_character_pack(self._settings.character.id)
        if self._pet is None:
            self._pet = PetWindow(
                application=self,
                pack=self._pack,
                machine=self._machine,
                settings=self._settings,
                capabilities=self._capabilities,
            )
        self._pet.present()
        if self._settings.app.greet_on_start:
            self._machine.set(CharacterState.GREET)
        self._sync_wake()
        self._sync_watch()
        self._sync_routines()

    def _show_missing_assets(self) -> None:
        win = Adw.ApplicationWindow(application=self, title=APP_NAME)
        win.set_default_size(420, 200)
        bar = Adw.HeaderBar()
        status = Adw.StatusPage(
            title="Charakterpaket fehlt",
            description="Lege data/character/<id>/ mit manifest.toml und PNG-Frames an.",
            icon_name="dialog-error-symbolic",
        )
        view = Adw.ToolbarView()
        view.add_top_bar(bar)
        view.set_content(status)
        win.set_content(view)
        win.present()

    def open_chat(self) -> None:
        if self._service is None or self._chats is None:
            log.error("chat service not ready")
            return
        try:
            if self._chat is None:
                self._chat = ChatWindow(
                    application=self,
                    chats=self._chats,
                    service=self._service,
                    bridge=self._bridge,
                    settings=self._settings,
                )
            self._chat.present_chat()
            if self.is_listening():
                self._chat.set_listening(True)
        except Exception:
            log.exception("failed to open chat")

    def open_preferences(self) -> None:
        try:
            dialog = PreferencesWindow(
                settings=self._settings,
                secrets=self._secrets,
                bridge=self._bridge,
                on_change=self._apply_settings,
                on_tts_test=self.test_tts_voice,
                memories=self._memories,
                routines=self._routines,
            )
            parent = self._chat if (self._chat and self._chat.is_visible()) else self._pet
            dialog.present(parent)
        except Exception:
            log.exception("failed to open preferences")

    def open_coding(self) -> None:
        if self._session_service is None:
            log.error("session service not ready")
            return
        try:
            if self._coding is None:
                self._coding = CodingSessionWindow(
                    application=self,
                    service=self._session_service,
                    bridge=self._bridge,
                    settings=self._settings,
                )
            self._coding.present_coding()
        except Exception:
            log.exception("failed to open coding session")

    def open_coding_with_task(self, text: str) -> None:
        self.open_coding()
        if self._coding is not None and self._coding.set_task(text):
            self.speak_status("Aufgabe in die Coding-Session übernommen.")
            self.notify_status("KIKI", "Aufgabe in die Coding-Session übernommen.")

    def post_coding_summary_to_chat(self, text: str) -> None:
        self.open_chat()
        if self._chat is not None:
            self._chat.append_note(text)

    def open_workspaces(self) -> None:
        if self._session_service is None:
            return
        try:
            if self._ws_manager is None:
                self._ws_manager = WorkspaceManagerWindow(
                    application=self,
                    service=self._session_service,
                    bridge=self._bridge,
                    on_change=self._reload_workspace_views,
                )
            self._ws_manager.present()
            self._ws_manager.reload()
        except Exception:
            log.exception("failed to open workspace manager")

    def _reload_workspace_views(self) -> None:
        if self._coding is not None:
            self._coding.reload_workspaces()
        if self._desktop_control is not None:
            self._desktop_control.reload_workspaces()

    def open_desktop_control(self) -> None:
        if self._session_service is None:
            log.error("session service not ready")
            return
        try:
            if self._desktop_control is None:
                self._desktop_control = DesktopControlWindow(
                    application=self,
                    service=self._session_service,
                    settings=self._settings,
                )
            self._desktop_control.present_control()
        except Exception:
            log.exception("failed to open desktop control")

    def speak_status(self, text: str) -> None:
        if not self._settings.tts_allowed() or self._speech is None:
            return
        cleaned = " ".join(text.split())[:160]
        if cleaned:
            self._speech.say(cleaned)

    def notify_status(self, title: str, body: str) -> None:
        note = Gio.Notification.new((title or "KIKI")[:80])
        note.set_body(" ".join((body or "").split())[:200])
        self.send_notification("kiki-status", note)

    def _apply_settings(self, settings: Settings) -> None:
        self._settings = settings
        self._tts_remote_retry_after = 0.0
        self._tts_remote_error = ""
        if self._service is not None:
            self._service.update_settings(settings)
        if self._pet is not None:
            self._pet.update_settings(settings)
        if self._chat is not None:
            self._chat.update_settings(settings)
        if self._coding is not None:
            self._coding.update_settings(settings)
        if self._desktop_control is not None:
            self._desktop_control.update_settings(settings)
        if self._session_service is not None:
            self._session_service.set_plan_first(settings.agents.plan_first)
        if not settings.tts_allowed():
            self.stop_speech()
        # Covers the panic switch too: voice_allowed() folds it in.
        self._sync_wake()
        self._sync_watch()
        self._sync_routines()

    def reload_character(self) -> None:
        pack = ensure_character_pack(self._settings.character.id)
        self._pack = pack
        if self._pet is not None:
            self._pet.reload_pack(pack)
        self._machine.set(CharacterState.GREET)

    def _toggle_panic(self) -> None:
        self._settings.app.privacy_panic = not self._settings.app.privacy_panic
        save_settings(self._settings)
        self._apply_settings(self._settings)
        if self._settings.app.privacy_panic:
            self.stop_speech()
            if self.is_listening():
                self._stop_voice(discard=True)

    def is_listening(self) -> bool:
        return self._recorder.recording

    def is_speaking(self) -> bool:
        return bool(self._speech and self._speech.active)

    def assistant_paused(self) -> bool:
        return self._assistant_pause.paused

    def stop_speech(self) -> None:
        if self._speech is not None:
            self._speech.stop()

    def test_tts_voice(self) -> None:
        if not self._settings.tts_allowed():
            self._toast("Sprachausgabe ist deaktiviert (Privatsphäre oder Einstellungen).")
            return
        if self._speech is None:
            return
        self._speech.say("Hallo, ich bin KIKI. Meine Sprachausgabe funktioniert lokal.")

    def _toast(self, text: str) -> None:
        if self._chat is not None:
            self._chat.show_toast(text)
        else:
            log.info("%s", text)
            self.notify_status("KIKI", text)

    async def _confirm_model_action(self, request: ConfirmationRequest) -> bool:
        """Approval card for a tool KIKI asked for herself. Runs on the asyncio thread.

        The card is built from the display record the runner armed; the answer
        that comes back is a verdict, not an authorisation. The service spends
        it through the request id, and the broker behind the gateway decides
        whether it still buys anything.
        """
        preview = ActionPreview(
            tool=request.tool_name,
            title=request.title,
            params={"Ziel": request.target},
            target=request.target,
            effect=request.content,
            risk=RiskLevel.WRITE,
            reason="KIKI hat diese Aktion vorgeschlagen.",
            request_id=request.request_id,
        )

        def _present(settle) -> None:
            present_confirmation(self._chat or self._pet, preview, settle)

        return await self._bridge.ask_ui(_present)

    def request_screenshot(self) -> None:
        if not self._settings.screenshot_allowed():
            self._toast("Bildschirmfoto deaktiviert (Privatsphäre oder Einstellungen).")
            return
        preview = ActionPreview(
            tool="capture_screen",
            title="Bildschirmfoto",
            params={"interactive": self._settings.screenshot.interactive},
            target="aktueller Bildschirm",
            effect="KIKI erhält ein Bild des Desktops und sendet es an das lokale Modell. "
            "Ohne deine Freigabe passiert nichts. Der Systemdialog kann zusätzlich erscheinen.",
            risk=RiskLevel.READ,
            reason="Bildschirminhalt ist privat — ausdrückliche Zustimmung nötig.",
        )
        present_confirmation(self._chat or self._pet, preview, self._on_screenshot_confirmed)

    def _on_screenshot_confirmed(self, ok: bool) -> None:
        executor = self._executor
        if executor is None:
            return
        if not ok:
            executor.audit.record("capture_screen", {}, "cancelled")
            return
        executor.audit.record("capture_screen", {}, "confirmed")
        if self._pet is not None:
            self._pet.set_visible(False)

        def _go() -> bool:
            from kiki.integrations.screenshot import capture_screenshot

            capture_screenshot(
                self._on_screenshot_ready,
                interactive=self._settings.screenshot.interactive,
            )
            return False

        GLib.timeout_add(220, _go)

    def _on_screenshot_ready(self, path, error: str | None) -> None:
        if self._pet is not None:
            self._pet.set_visible(True)
            self._pet.present()
        executor = self._executor
        if error == "cancelled":
            if executor:
                executor.audit.record("capture_screen", {}, "cancelled")
            self._toast("Bildschirmfoto abgebrochen.")
            return
        if error or path is None:
            if executor:
                executor.audit.record("capture_screen", {}, "error", error=error)
            self._toast(error or "Kein Bildschirmfoto.")
            return
        if executor:
            executor.audit.record("capture_screen", {"path": str(path)}, "executed")
        self.open_chat()
        if self._chat is None:
            return
        self._chat.submit_vision(
            "Was siehst du auf meinem Bildschirm? Antworte auf Deutsch und beschreibe nur das Foto.",
            path,
            label="Bildschirmfoto",
        )

    def toggle_voice(self) -> None:
        if self._recorder.recording:
            self._stop_voice(discard=False)
            return
        self._start_voice()

    def _start_voice(self) -> None:
        if not self._settings.voice_allowed():
            self._toast("Spracheingabe ist deaktiviert (Privatsphäre oder Einstellungen).")
            return
        if self._voice_busy:
            return
        self._follow_up.cancel()
        self.stop_speech()
        if not vosk_model_ready():
            self._voice_busy = True
            self._toast("Lade deutsches Sprachmodell (~45 MB, einmalig) …")

            async def _prep():
                return await asyncio.to_thread(ensure_vosk_model)

            self._bridge.submit(_prep(), on_success=lambda _p: self._arm_recorder(), on_error=self._voice_failed)
            return
        self._arm_recorder()

    def _arm_recorder(self) -> None:
        self._voice_busy = False
        from kiki.paths import cache_dir

        wav = cache_dir() / "voice" / "take.wav"
        try:
            self._recorder.start(wav)
        except RecorderError as exc:
            self._toast(str(exc))
            return
        # Push-to-talk owns the microphone for its own pipeline; the wake
        # listener must not race it or transcribe the same dictation.
        self._pause_wake()
        self._machine.set(CharacterState.LISTENING, hold_ms=0)
        if self._chat is not None:
            self._chat.set_listening(True)
        self._toast("KIKI hört zu. Nochmal klicken zum Stoppen.")

    # --- Proactive notices -----------------------------------------------

    def _build_watchers(self, upower: object, disk: object) -> None:
        watch = self._settings.watch
        watchers = []
        if watch.battery_enabled:
            watchers.append(BatteryWatcher(upower, threshold_percent=watch.battery_percent))
        if watch.disk_enabled:
            watchers.append(DiskWatcher(disk, threshold_percent=watch.disk_percent))
        self._notifier = Notifier(self._watch_policy())
        self._watch = WatchService(
            watchers,
            on_notice=lambda notice: GLib.idle_add(self._on_notice, notice),
            interval_s=watch.interval_s,
            # Re-read every tick so the panic switch takes effect without a restart.
            enabled=lambda: self._settings.watch.enabled
            and not self._settings.app.privacy_panic,
        )

    def _watch_policy(self) -> NotifierPolicy:
        from datetime import time as _time

        watch = self._settings.watch
        return NotifierPolicy(
            speak=watch.speak,
            quiet_start=parse_clock(watch.quiet_start, _time(22, 0)),
            quiet_end=parse_clock(watch.quiet_end, _time(8, 0)),
            cooldown_s=float(watch.cooldown_s),
            max_per_hour=watch.max_per_hour,
        )

    def _busy_with_user(self) -> bool:
        """True while a conversation is in flight — KIKI must not talk over it."""
        if self._recorder.recording or self._voice_busy:
            return True
        if self._speech is not None and self._speech.active:
            return True
        wake = self._wake
        return wake is not None and wake.state.value == "capturing"

    def _on_notice(self, notice: Notice) -> bool:
        notifier = self._notifier
        if notifier is None:
            return False
        if self._assistant_pause.paused:
            # A paused assistant does not speak up. The watcher's condition
            # will re-evaluate on a later tick once she is resumed.
            log.debug("notice %s suppressed: assistant paused", notice.key)
            return False
        delivery = notifier.decide(
            notice,
            panic=self._settings.app.privacy_panic,
            busy=self._busy_with_user(),
        )
        if delivery.silent:
            log.debug("notice %s suppressed: %s", notice.key, delivery.reason)
            return False
        log.info("notice %s delivered (speak=%s)", notice.key, delivery.speak)
        # What the user actually saw becomes activity; suppressed notices
        # were not delivered and are not part of what happened on screen.
        self._activity.record_notice(key=notice.key, severity=notice.severity.value)
        self.notify_status(notice.title, notice.detail or notice.spoken)
        if self._machine.state not in {CharacterState.PAUSED, CharacterState.LISTENING}:
            self._machine.set(CharacterState.NOTIFICATION)
        if delivery.speak and self._settings.tts_allowed() and self._speech is not None:
            self._speech.say(notice.spoken)
        return False

    def _sync_watch(self) -> None:
        if self._notifier is not None:
            self._notifier.update_policy(self._watch_policy())
        if self._watch is None:
            return
        # The service checks `enabled` itself each tick; only the loop needs
        # starting once the app is up.
        if not self._watch.running:
            self._watch.start(self._bridge)

    def _sync_routines(self) -> None:
        if self._routines_service is None:
            return
        # The engine re-reads panic and integration flags on every tick;
        # only the loop needs starting once the app is up.
        if not self._routines_service.running:
            self._routines_service.start(self._bridge)

    # --- Wake word -------------------------------------------------------

    def _wake_allowed(self) -> bool:
        return self._settings.voice_allowed() and self._settings.voice.wake.enabled

    def _sync_wake(self) -> None:
        """Bring the listener in line with the current settings."""
        if not self._wake_allowed():
            self._stop_wake()
            return
        if self._wake is not None or self._wake_starting:
            return
        if not wake_word_supported():
            self._toast("Weckwort braucht eine neuere Vosk-Laufzeit.")
            return
        self._wake_starting = True
        wake = self._settings.voice.wake

        async def _arm() -> WakeWordListener:
            # Downloading and loading the model must not block GTK.
            model_dir = await asyncio.to_thread(ensure_vosk_model)
            listener = WakeWordListener(
                stream=UtteranceStream(model_dir=model_dir),
                microphone=MicrophoneStream(),
                phrases=wake.phrases,
                cooldown_ms=wake.cooldown_ms,
                command_timeout_s=wake.command_timeout_s,
                on_detect=lambda: GLib.idle_add(self._on_wake_detected),
                on_command=lambda text: GLib.idle_add(self._on_wake_command, text),
                on_timeout=lambda: GLib.idle_add(self._on_wake_timeout),
                on_error=lambda exc: GLib.idle_add(self._on_wake_error, exc),
            )
            await asyncio.to_thread(listener.start)
            return listener

        self._bridge.submit(
            _arm(), on_success=self._on_wake_started, on_error=self._on_wake_error
        )

    def _on_wake_started(self, listener: WakeWordListener) -> None:
        self._wake_starting = False
        if not self._wake_allowed():
            # Settings changed while the model was loading.
            listener.stop()
            return
        self._wake = listener
        if self._speech is not None and self._speech.active:
            listener.pause()
        if self._recorder.recording:
            listener.pause()
        self._toast(f"Weckwort aktiv. Sag „{self._settings.voice.wake.phrases[0]}“.")

    def _on_wake_error(self, exc: BaseException) -> None:
        self._wake_starting = False
        listener, self._wake = self._wake, None
        if listener is not None:
            listener.stop()
        message = str(exc) if isinstance(exc, WakeError | RecorderError) else f"Weckwort: {exc}"
        log.warning("wake word unavailable: %s", exc)
        self._toast(message)

    def _stop_wake(self) -> None:
        self._follow_up.cancel()
        listener, self._wake = self._wake, None
        if listener is None:
            return
        listener.stop()

    def _pause_wake(self) -> None:
        if self._wake is not None:
            self._wake.pause()

    def _resume_wake(self) -> None:
        if self._wake is not None and self._wake_allowed():
            self._wake.resume()

    def _on_wake_detected(self) -> bool:
        self._follow_up.cancel()
        self.stop_speech()
        self._machine.set(CharacterState.LISTENING, hold_ms=0)
        if self._chat is not None:
            self._chat.set_listening(True)
        self._toast("KIKI hört. Sag deine Frage.")
        return False

    def _on_wake_command(self, text: str) -> bool:
        if self._chat is not None:
            self._chat.set_listening(False)
        if self._machine.state is CharacterState.LISTENING:
            self._machine.set(CharacterState.IDLE, hold_ms=0)
        self._follow_up.begin(enabled=self._follow_up_allowed())
        # One captured command is one spoken event, minted at the source.
        started = self._route_spoken_text(
            text, correlation_id=f"wake-{uuid.uuid4().hex[:12]}"
        )
        if not started:
            self._follow_up.cancel()
        return False

    def _on_wake_timeout(self) -> bool:
        """Close the visible listening state after wake or follow-up silence."""
        if self._chat is not None:
            self._chat.set_listening(False)
        if self._machine.state is CharacterState.LISTENING:
            self._machine.set(CharacterState.IDLE, hold_ms=0)
        self._toast("Zuhören beendet. Sag „KIKI“ für eine neue Frage.")
        return False

    def _follow_up_allowed(self) -> bool:
        wake = self._settings.voice.wake
        return bool(
            self._settings.voice_allowed()
            and self._settings.voice.auto_send
            and self._settings.tts_allowed()
            and wake.enabled
            and wake.follow_up
            and self._wake is not None
            and not self._assistant_pause.paused
        )

    def _try_arm_follow_up(self) -> bool:
        """Consume a finished voice turn and open one local listening window."""
        if not self._follow_up.consume_ready():
            return False
        if not self._follow_up_allowed():
            self._resume_wake()
            return False
        self._resume_wake()
        listener = self._wake
        if listener is None or not listener.capture_next():
            return False
        self._machine.set(CharacterState.LISTENING, hold_ms=0)
        if self._chat is not None:
            self._chat.set_listening(True)
        self._toast("KIKI hört noch kurz zu. Du kannst direkt weiterreden.")
        return True

    def _stop_voice(self, *, discard: bool) -> None:
        path = self._recorder.stop()
        self._resume_wake()
        if self._chat is not None:
            self._chat.set_listening(False)
        if self._machine.state is CharacterState.LISTENING:
            self._machine.set(CharacterState.IDLE, hold_ms=0)
        if discard or path is None:
            return
        if not path.is_file() or path.stat().st_size < 64:
            self._toast("Keine Aufnahme.")
            return
        self._voice_busy = True
        self._toast("Erkenne Sprache …")
        self._machine.set(CharacterState.THINKING, hold_ms=0)

        async def _run():
            return await asyncio.to_thread(transcribe_wav, path)

        # One recording is one spoken event: the id is minted where the event
        # is born and travels with its transcript, so a delivery that fires
        # twice can never start two runs for one utterance.
        take_id = f"take-{uuid.uuid4().hex[:12]}"
        self._bridge.submit(
            _run(),
            on_success=lambda text: self._on_transcript(text, take_id),
            on_error=self._voice_failed,
        )

    def _on_transcript(self, text: str, correlation_id: str = "") -> None:
        self._voice_busy = False
        if self._machine.state is CharacterState.THINKING:
            self._machine.set(CharacterState.IDLE, hold_ms=0)
        self._follow_up.begin(enabled=False)
        if not self._route_spoken_text(text, correlation_id=correlation_id):
            self._follow_up.cancel()

    def _route_spoken_text(self, text: str, *, correlation_id: str = "") -> bool:
        """One path for recognized speech, whether push-to-talk or wake word.

        One spoken event becomes exactly one run: the correlation id binds the
        utterance to its run and refuses a second delivery of the same event.
        With `voice.auto_send` off, the person reviews first -- the text lands
        in the input box and nothing runs until Enter.
        """
        if not text.strip():
            self._toast("Nichts verstanden. Bitte nochmal.")
            return False
        if is_desktop_control_intent(text):
            self.open_desktop_control()
            self._toast("PC-Steuerung geöffnet. Aktionen brauchen weiterhin einen Klick und Freigabe.")
            return False
        if not self._settings.voice.auto_send:
            self.open_chat()
            if self._chat is not None:
                self._chat.submit_transcript(text.strip(), send=False)
            return False
        if self._assistant_pause.paused:
            # Spoken to, but paused: the toast alone would be invisible to
            # someone talking. Say it, once, and drop the utterance.
            self._toast("KIKI macht gerade Pause.")
            self._say_paused()
            return False
        service = self._harness or self._build_harness()
        if service is None:
            # No run service (no tool-capable provider): the chat still
            # answers, voice does not die with the agent path.
            self.open_chat()
            if self._chat is not None:
                self._chat.submit_transcript(text.strip(), send=True)
                return True
            return False
        if service.busy:
            self._toast("KIKI arbeitet noch an der letzten Aufgabe.")
            return False
        coro = service.ask(text.strip(), correlation_id=correlation_id)
        try:
            self._bridge.submit(coro, on_error=self._on_voice_run_error)
        except Exception as exc:
            coro.close()
            self._on_voice_run_error(exc)
            return False
        return True

    def _on_voice_run_error(self, exc: BaseException) -> None:
        from kiki.assistant import DuplicateCorrelationError, RunPausedError

        if isinstance(exc, DuplicateCorrelationError):
            # The same utterance already has its run. Silence is correct:
            # a second toast or answer would be the duplicate, not this.
            log.info("duplicate spoken event dropped")
            return
        if isinstance(exc, RunPausedError):
            self._follow_up.cancel()
            self._say_paused()
            return
        self._follow_up.cancel()
        self._on_harness_bridge_error(exc)

    def _say_paused(self) -> None:
        if self._settings.tts_allowed() and self._speech is not None:
            self._speech.say("Ich mache gerade Pause. Bitte später nochmal.")

    def _voice_failed(self, exc: BaseException) -> None:
        self._follow_up.cancel()
        self._voice_busy = False
        if self._recorder.recording:
            self._recorder.stop()
        if self._chat is not None:
            self._chat.set_listening(False)
        self._machine.set(CharacterState.ERROR)
        self._toast(str(exc) if not isinstance(exc, SpeechError) else str(exc))

    def _on_stream_start(self, **_payload: object) -> None:
        self.stop_speech()
        if (
            not self._follow_up.active
            and self._settings.tts_allowed()
            and self._speech is not None
        ):
            self._speech.begin()
        self._machine.set(CharacterState.THINKING, hold_ms=0)

    def _on_stream_delta(self, **payload: object) -> None:
        if self._follow_up.active:
            # A microphone-started answer is planned once as a whole at done;
            # speaking deltas here would bypass its global sentence budget.
            return
        if not self._settings.tts_allowed() or self._speech is None:
            return
        if not self._settings.tts.stream_sentences:
            return
        text = payload.get("text")
        if isinstance(text, str) and text:
            self._speech.feed(text)

    def _on_stream_tool(self, **_payload: object) -> None:
        # Working on a tool is thinking, not talking — keep the figure in step.
        if self._machine.state is not CharacterState.PAUSED:
            self._machine.set(CharacterState.THINKING, hold_ms=0)

    def _on_stream_speaking(self, **_payload: object) -> None:
        if self._settings.tts_allowed():
            return
        self._machine.set(CharacterState.SPEAKING, hold_ms=0)

    def _on_stream_done(self, **payload: object) -> None:
        ok = bool(payload.get("ok", True))
        if not ok:
            self._follow_up.cancel()
            return
        voice_turn = self._follow_up.active
        self._follow_up.mark_terminal()
        text = payload.get("text")
        if self._settings.tts_allowed() and self._speech is not None:
            if voice_turn:
                answer = text if isinstance(text, str) else ""
                delivery = self._plan_voice_answer(answer)
                if delivery.open_chat:
                    self.open_chat()
                self._speech.say(delivery.spoken_text)
                delivered = self._follow_up.mark_response_delivered()
                if delivered and not self._speech.active:
                    self._try_arm_follow_up()
                return
            if not self._settings.tts.stream_sentences and isinstance(text, str) and text.strip():
                self._speech.feed(text)
            self._speech.flush()
            return
        delivered = self._follow_up.mark_response_delivered()
        if delivered and self._try_arm_follow_up():
            return
        if self._machine.state is not CharacterState.PAUSED:
            self._machine.set(CharacterState.IDLE, hold_ms=0)

    def _on_stream_error(self, **_payload: object) -> None:
        self._follow_up.cancel()
        self.stop_speech()
        self._machine.set(CharacterState.ERROR)

    async def _synthesize_tts(self, text: str, dest: Path):
        tts = self._settings.tts
        now = time.monotonic()
        if now >= self._tts_remote_retry_after:
            try:
                rendered = await synthesize_wav(
                    tts.base_url,
                    text,
                    dest=dest,
                    language=tts.language,
                    speaker=tts.speaker,
                )
            except TtsError as exc:
                self._tts_remote_retry_after = time.monotonic() + 30.0
                self._tts_remote_error = str(exc)
                log.info("GPU TTS unavailable, considering system fallback: %s", exc)
            else:
                self._tts_remote_retry_after = 0.0
                self._tts_remote_error = ""
                return rendered
        if tts.fallback_to_system:
            return await synthesize_system_wav(text, dest=dest)
        detail = self._tts_remote_error or "TTS-Dienst vorübergehend nicht erreichbar"
        raise TtsError(detail)

    def _on_tts_speaking(self) -> None:
        """The request was accepted. On the controller route KIKI is still
        silent at this point, so only the microphone is dealt with here."""
        # KIKI's own voice must not wake her: the microphone hears the speakers.
        # Muted on acceptance rather than on first audio — early is harmless,
        # late would let her hear herself.
        self._pause_wake()

    def _on_tts_audio_started(self) -> None:
        """First audible chunk. Both routes emit it; only now does she speak."""
        if self._machine.state is CharacterState.PAUSED:
            return
        self._machine.set(CharacterState.SPEAKING, hold_ms=0)

    def _on_tts_idle(self) -> None:
        if self._try_arm_follow_up():
            return
        self._resume_wake()
        if self._machine.state in {CharacterState.SPEAKING, CharacterState.THINKING}:
            self._machine.set(CharacterState.IDLE, hold_ms=0)

    def _close_voice_controller(self) -> None:
        """Release provider and sink at shutdown, without stalling GTK.

        Handed to the bridge rather than awaited: the remaining teardown steps
        give the loop its turns, and `bridge.stop()` afterwards cancels and
        gathers whatever is left. `shutdown()` is idempotent, so a second call
        from anywhere costs nothing.
        """
        controller = self._voice_controller
        if controller is None:
            return
        self._voice_controller = None
        coro = controller.shutdown()
        try:
            self._bridge.submit(
                coro,
                on_error=lambda exc: log.debug("voice controller shutdown failed: %s", exc),
            )
        except Exception:
            # Creating a coroutine transfers ownership only once submit()
            # accepts it.  A stopped bridge rejects it synchronously, so the
            # caller must close it instead of leaving a RuntimeWarning behind.
            coro.close()
            log.debug("could not hand the voice shutdown to the bridge", exc_info=True)

    def _on_tts_error(self, exc: BaseException) -> None:
        message = str(exc) if not isinstance(exc, TtsError) else str(exc)
        log.warning("tts: %s", message)
        self._toast(message)

    def _on_shutdown(self, *_args: object) -> None:
        self._close_harness()
        self._follow_up.cancel()
        self.stop_speech()
        self._close_voice_controller()
        # Release the microphone and the poll loop before the bridge goes away.
        self._stop_wake()
        if self._watch is not None:
            self._watch.stop()
        if self._routines_service is not None:
            self._routines_service.stop()
        if self._recorder.recording:
            self._recorder.stop()
        if self._pet is not None:
            self._pet.destroy()
            self._pet = None
        self._bridge.stop()
        if self._db is not None:
            self._db.close()



def run_application(argv: list[str] | None = None) -> int:
    app = KikiApplication()
    return int(app.run(argv if argv is not None else sys.argv))
