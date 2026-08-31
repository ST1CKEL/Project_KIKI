"""Voice-first pet process: sprites, state from the orchestrator, no chat stage.

Left click starts listening. Chat is an emergency toolbox (`python3 -m kiki`).
"""

from __future__ import annotations

import argparse
import json
import logging
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from kiki import APP_NAME
from kiki.character.assets import ensure_character_pack
from kiki.character.state_machine import CharacterState, CharacterStateMachine
from kiki.config.runtime import load_runtime
from kiki.config.settings import load_settings
from kiki.ipc.paths import runtime_dir, socket_path
from kiki.platform.capabilities import detect_capabilities

log = logging.getLogger("kiki-pet")

PET_APP_ID = "io.github.projectkiki.Kiki.Pet"

_STATE = {
    "idle": CharacterState.IDLE,
    "listening": CharacterState.LISTENING,
    "thinking": CharacterState.THINKING,
    "speaking": CharacterState.SPEAKING,
    "error": CharacterState.ERROR,
    "notification": CharacterState.NOTIFICATION,
}


def _send(path: Path, payload: dict) -> None:
    if not path.exists():
        log.warning("socket missing: %s", path)
        return
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(0.4)
        sock.connect(str(path))
        sock.sendall(json.dumps(payload).encode("utf-8") + b"\n")
    except Exception as exc:
        log.warning("send to %s failed: %s", path, exc)
    finally:
        sock.close()


class OrchestratorWatch:
    def __init__(self, ui_socket: Path, on_payload) -> None:
        self.ui_socket = ui_socket
        self.on_payload = on_payload
        self._running = True

    def start(self) -> None:
        threading.Thread(target=self._loop, name="kiki-pet-ipc", daemon=True).start()

    def stop(self) -> None:
        self._running = False

    def _loop(self) -> None:
        while self._running:
            if not self.ui_socket.exists():
                time.sleep(0.8)
                continue
            try:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.connect(str(self.ui_socket))
                log.info("pet connected to orchestrator")
                with sock.makefile("r", encoding="utf-8") as fh:
                    for line in fh:
                        if not self._running:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            msg = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        self.on_payload(msg)
            except Exception:
                time.sleep(1.0)


def run_pet(ui_socket: Path, audio_socket: Path) -> int:
    from kiki.ui.gi_bootstrap import Adw, Gio, GLib
    from kiki.ui.pet_window import PetWindow

    settings = load_settings()
    machine = CharacterStateMachine()
    pack = ensure_character_pack(settings.character.id)
    capabilities = detect_capabilities()

    app = Adw.Application(application_id=PET_APP_ID, flags=Gio.ApplicationFlags.FLAGS_NONE)

    def trigger(*_args: object) -> None:
        _send(audio_socket, {"command": "trigger_turn", "source": "pet"})
        _send(ui_socket, {"command": "trigger"})

    def emergency_chat(*_args: object) -> None:
        # Chat is the toolbox, not the stage. Spawn the full app if needed.
        subprocess.Popen(
            [sys.executable, "-m", "kiki"],
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def panic(*_args: object) -> None:
        _send(ui_socket, {"command": "panic"})
        machine.set(CharacterState.ERROR)

    def apply_payload(msg: dict) -> None:
        def _ui() -> bool:
            raw = str(msg.get("state") or "idle")
            health = str(msg.get("health") or "healthy")
            target = _STATE.get(raw, CharacterState.IDLE)
            if health not in {"healthy", "degraded_wake"} and target is CharacterState.IDLE:
                target = CharacterState.NOTIFICATION
            if health in {"fatal", "degraded_stt", "degraded_tts", "degraded_audio"}:
                if target is CharacterState.IDLE:
                    target = CharacterState.ERROR
            machine.set(target)
            return False

        GLib.idle_add(_ui)

    def on_activate(_app: Adw.Application) -> None:
        for name, callback in (
            ("chat", emergency_chat),
            ("voice-toggle", trigger),
            ("quit", lambda *_: app.quit()),
            ("pause", lambda *_: machine.pause()),
            ("resume", lambda *_: machine.set(CharacterState.IDLE)),
            ("preferences", emergency_chat),
            ("desktop-control", emergency_chat),
            ("coding", emergency_chat),
            ("workspaces", emergency_chat),
            ("screenshot", lambda *_: None),
            ("tts-stop", lambda *_: _send(ui_socket, {"command": "panic"})),
            ("assistant-pause-toggle", lambda *_: None),
            ("reload-character", lambda *_: None),
            ("window-menu", lambda *_: None),
        ):
            action = Gio.SimpleAction.new(name, None)
            action.connect("activate", callback)
            app.add_action(action)

        pet = PetWindow(
            application=app,
            pack=pack,
            machine=machine,
            settings=settings,
            capabilities=capabilities,
            on_primary_click=trigger,
        )
        pet.present()
        app.remove_action("window-menu")
        window_menu = Gio.SimpleAction.new("window-menu", None)
        window_menu.connect("activate", lambda *_: pet.show_window_menu())
        app.add_action(window_menu)
        watch = OrchestratorWatch(ui_socket, apply_payload)
        watch.start()
        app.connect("shutdown", lambda *_: watch.stop())

    app.connect("activate", on_activate)
    return int(app.run([]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"{APP_NAME} voice-first pet")
    parser.add_argument("--socket", default="")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] kiki-pet: %(message)s",
    )
    cfg = load_runtime()
    ui = Path(args.socket) if args.socket else cfg.socket("ui")
    audio = socket_path("audio", runtime=runtime_dir(ui.parent) if args.socket else cfg.socket_dir)
    return run_pet(ui, audio)


if __name__ == "__main__":
    raise SystemExit(main())
