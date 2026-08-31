"""Async vision agent — the slow path that must never stall the reflex.

Takes a screenshot, asks a local vision model for the next click/type, drives
ydotool. Bounded steps. No free shell. VRAM is checked first; if the GPU is
full we say so instead of swapping the LLM out from under a live turn.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kiki.paths import cache_dir
from kiki.tools.policy import RiskLevel
from kiki.tools.registry import ToolSpec

log = logging.getLogger("kiki.vision")

MAX_STEPS = 8
# Linux evdev key codes. ydotool does not take the names "enter"/"esc".
# SECURITY: only this map is reachable — no free key strings from the model.
_KEYCODES: dict[str, str] = {
    "enter": "28:1 28:0",
    "esc": "1:1 1:0",
    "tab": "15:1 15:0",
    "backspace": "14:1 14:0",
    "space": "57:1 57:0",
    "up": "103:1 103:0",
    "down": "108:1 108:0",
    "left": "105:1 105:0",
    "right": "106:1 106:0",
    "delete": "111:1 111:0",
    "home": "102:1 102:0",
    "end": "107:1 107:0",
}
_ALLOWED_KEYS = frozenset(_KEYCODES)
_SECRETISH = re.compile(r"(password|passwd|secret|token|api[_-]?key|sudo)", re.I)

SpeakFn = Callable[[str], Awaitable[None]]


class VisionError(RuntimeError):
    """The slow path could not run. Callers must speak this, not invent clicks."""


@dataclass
class VisionJob:
    job_id: str
    instruction: str
    status: str = "queued"
    steps: int = 0
    log: list[str] = field(default_factory=list)
    error: str = ""


def _which(*names: str) -> str | None:
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    return None


def capture_screenshot() -> Path:
    dest = cache_dir() / "vision" / f"{uuid.uuid4().hex}.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    grim = _which("grim")
    if grim:
        proc = subprocess.run([grim, str(dest)], capture_output=True, timeout=8, check=False)
        if proc.returncode == 0 and dest.is_file():
            return dest
    spectacle = _which("spectacle")
    if spectacle:
        proc = subprocess.run(
            [spectacle, "--fullscreen", "--background", "--nonotify", "-o", str(dest)],
            capture_output=True,
            timeout=8,
            check=False,
        )
        if proc.returncode == 0 and dest.is_file():
            return dest
    gnome = _which("gnome-screenshot")
    if gnome:
        proc = subprocess.run([gnome, "-f", str(dest)], capture_output=True, timeout=8, check=False)
        if proc.returncode == 0 and dest.is_file():
            return dest
    raise VisionError(
        "Kein Screenshot-Werkzeug (grim, spectacle, gnome-screenshot). "
        "Der Vision-Agent rät keine Pixel."
    )


def _ydotool(*args: str) -> None:
    binary = _which("ydotool")
    if binary is None:
        raise VisionError("ydotool fehlt. Ohne uinput klicke ich nicht ins Leere.")
    proc = subprocess.run([binary, *args], capture_output=True, timeout=5, check=False)
    if proc.returncode != 0:
        raise VisionError((proc.stderr or proc.stdout).decode("utf-8", "replace")[:240] or "ydotool failed")


def apply_action(action: dict[str, Any]) -> str:
    kind = str(action.get("action") or "").strip().lower()
    if kind == "click":
        x, y = int(action["x"]), int(action["y"])
        if not (0 <= x <= 7680 and 0 <= y <= 4320):
            raise VisionError("Klickkoordinate außerhalb des Bildschirms.")
        _ydotool("mousemove", "--absolute", str(x), str(y))
        _ydotool("click", "0xC0")
        return f"klick {x},{y}"
    if kind == "type":
        text = str(action.get("text") or "")
        if _SECRETISH.search(text):
            raise VisionError("Ich tippe keine Passwörter oder Tokens.")
        if len(text) > 200:
            raise VisionError("Texteingabe zu lang.")
        _ydotool("type", text)
        return "getippt"
    if kind == "key":
        key = str(action.get("key") or "").strip().lower()
        if key not in _ALLOWED_KEYS:
            raise VisionError(f"Taste {key!r} ist nicht erlaubt.")
        _ydotool("key", *_KEYCODES[key].split())
        return f"taste {key}"
    if kind == "done":
        return "done"
    raise VisionError(f"Unbekannte Vision-Aktion {kind!r}.")


def vision_tool_spec(handler) -> ToolSpec:
    return ToolSpec(
        name="desktop_vision_task",
        title="Bildschirm bedienen",
        description=(
            "Startet den asynchronen Vision-Agenten, der den Bildschirm liest und "
            "über ydotool klickt/tippt. Nur wenn keine API existiert. "
            "Gibt sofort zurück; die Arbeit läuft im Hintergrund."
        ),
        risk=RiskLevel.WRITE,
        parameters={
            "type": "object",
            "properties": {
                "instruction": {
                    "type": "string",
                    "description": "Was auf dem Bildschirm erledigt werden soll.",
                }
            },
            "required": ["instruction"],
            "additionalProperties": False,
        },
        handler=handler,
        effect="Steuert Maus und Tastatur anhand von Screenshots.",
        target="Desktop",
        auto_allow=False,
        model_callable=True,
        audit_parameters=("instruction",),
    )


class VisionAgent:
    def __init__(
        self,
        *,
        llm_url: str,
        vision_model: str,
        can_allocate,
        speak: SpeakFn | None = None,
    ) -> None:
        self.llm_url = llm_url.rstrip("/")
        self.vision_model = vision_model
        self.can_allocate = can_allocate
        self.speak = speak
        self.jobs: dict[str, VisionJob] = {}
        self.busy = False

    def queue(self, instruction: str) -> VisionJob:
        job = VisionJob(job_id=f"vis-{uuid.uuid4().hex[:8]}", instruction=instruction.strip())
        self.jobs[job.job_id] = job
        return job

    async def run_job(self, job: VisionJob) -> None:
        """Slow path. The orchestrator must keep listening while this runs."""
        self.busy = True
        job.status = "running"
        try:
            ok, reason = self.can_allocate()
            if not ok:
                raise VisionError(reason)
            if self.speak:
                await self.speak("Ich mach das, gib mir einen Moment.")
            for step in range(MAX_STEPS):
                job.steps = step + 1
                path = capture_screenshot()
                action = await self._ask_model(job.instruction, path, step)
                note = apply_action(action)
                job.log.append(note)
                if str(action.get("action") or "").lower() == "done":
                    job.status = "done"
                    if self.speak:
                        await self.speak("Fertig.")
                    return
            job.status = "limit"
            job.error = "Schrittlimit erreicht."
            if self.speak:
                await self.speak("Ich komme hier nicht weiter, das Schrittlimit ist erreicht.")
        except Exception as exc:
            job.status = "error"
            job.error = str(exc)
            log.warning("vision job failed: %s", exc)
            if self.speak:
                await self.speak(str(exc) if isinstance(exc, VisionError) else "Der Bildschirm-Agent ist gescheitert.")
        finally:
            self.busy = False

    async def _ask_model(self, instruction: str, image: Path, step: int) -> dict[str, Any]:
        import base64

        import httpx

        raw = image.read_bytes()
        b64 = base64.b64encode(raw).decode("ascii")
        prompt = (
            "Du steuerst einen Linux-Desktop. Aufgabe: "
            f"{instruction}\nSchritt {step + 1}/{MAX_STEPS}. "
            "Antworte NUR mit JSON: "
            '{"action":"click","x":0,"y":0} oder '
            '{"action":"type","text":"..."} oder '
            '{"action":"key","key":"enter"} oder {"action":"done"}. '
            "Keine Prosa."
        )
        payload = {
            "model": self.vision_model,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [b64],
                }
            ],
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(f"{self.llm_url}/api/chat", json=payload)
            response.raise_for_status()
            body = response.json()
        text = str((body.get("message") or {}).get("content") or "")
        match = re.search(r"\{.*\}", text, re.S)
        if match is None:
            raise VisionError("Das Vision-Modell lieferte kein JSON.")
        data = json.loads(match.group(0))
        if not isinstance(data, dict):
            raise VisionError("Vision-JSON ist kein Objekt.")
        return data
