"""Volume over pactl. Fixed argv, numbers only — no mixer strings from models.

pactl runs with a scrubbed environment and `LC_ALL=C` so the German locale
cannot turn `Mute: yes` into `Mute: ja` between parse and action. Only the
default sink is addressed; per-device routing stays out on purpose.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from collections.abc import Callable
from typing import Any

from kiki.runners.process import desktop_env
from kiki.tools.policy import RiskLevel
from kiki.tools.registry import ToolSpec

log = logging.getLogger(__name__)

_PACTL_TIMEOUT_S = 10
_PERCENT_RE = re.compile(r"(\d+)\s*%")
_DEFAULT_SINK = "@DEFAULT_SINK@"

PactlRunner = Callable[[list[str]], str]


class AudioError(RuntimeError):
    """pactl is missing, failed, or answered something unparsable."""


def run_pactl(argv: list[str]) -> str:
    binary = shutil.which("pactl")
    if binary is None:
        raise AudioError("pactl nicht gefunden — PipeWire/PulseAudio-CLI fehlt.")
    env = desktop_env(home=os.environ.get("HOME") or "/tmp")
    env["LC_ALL"] = "C"
    try:
        proc = subprocess.run(
            [binary, *argv],
            capture_output=True,
            text=True,
            timeout=_PACTL_TIMEOUT_S,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioError("pactl hat nicht rechtzeitig geantwortet.") from exc
    except OSError as exc:
        raise AudioError(f"pactl konnte nicht gestartet werden: {exc}") from exc
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "Unbekannter pactl-Fehler").strip()
        raise AudioError(detail[:200])
    return proc.stdout


def parse_volume_percent(output: str) -> int | None:
    match = _PERCENT_RE.search(output)
    return int(match.group(1)) if match else None


def parse_muted(output: str) -> bool | None:
    lowered = output.lower()
    for marker in ("muted:", "mute:"):
        pos = lowered.find(marker)
        if pos < 0:
            continue
        value = lowered[pos + len(marker) :].strip().split(maxsplit=1)
        if not value:
            return None
        word = value[0].strip(":")
        if word in {"yes", "true", "on", "1", "an", "ja"}:
            return True
        if word in {"no", "false", "off", "0", "aus", "nein"}:
            return False
    return None


class AudioControlSkill:
    id = "audio_control"
    name = "Lautstärke"
    description = "Lautstärke der Standard-Ausgabe lesen, setzen und stumm schalten."

    def __init__(self, runner: PactlRunner | None = None) -> None:
        self._run = runner or run_pactl

    def tools(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="audio.volume_get",
                title="Lautstärke abfragen",
                description="Nennt die aktuelle Lautstärke der Standard-Ausgabe in Prozent und ob sie stumm ist.",
                risk=RiskLevel.READ,
                parameters={"type": "object", "properties": {}, "additionalProperties": False},
                handler=self._volume_get,
                effect="Liest pactl-Status. Keine Änderung.",
                target="Standard-Ausgabe",
                auto_allow=True,
                model_callable=True,
            ),
            ToolSpec(
                name="audio.volume_set",
                title="Lautstärke setzen",
                description=(
                    "Setzt die Lautstärke der Standard-Ausgabe auf 0–100 Prozent. "
                    "Werte außerhalb werden auf den Bereich begrenzt."
                ),
                risk=RiskLevel.CONTROL,
                parameters={
                    "type": "object",
                    "properties": {
                        "percent": {"type": "integer"}
                    },
                    "required": ["percent"],
                    "additionalProperties": False,
                },
                handler=self._volume_set,
                effect="Ändert die Lautstärke der Standard-Ausgabe.",
                target="Standard-Ausgabe",
                auto_allow=True,
                model_callable=True,
            ),
            ToolSpec(
                name="audio.mute",
                title="Stumm schalten",
                description=(
                    "Schaltet die Standard-Ausgabe stumm (muted=true) oder hebt die "
                    "Stummschaltung auf (muted=false)."
                ),
                risk=RiskLevel.CONTROL,
                parameters={
                    "type": "object",
                    "properties": {
                        "muted": {"type": "boolean"}
                    },
                    "required": ["muted"],
                    "additionalProperties": False,
                },
                handler=self._mute,
                effect="Schaltet die Standard-Ausgabe stumm oder frei.",
                target="Standard-Ausgabe",
                auto_allow=True,
                model_callable=True,
            ),
        ]

    def _volume_get(self, _params: dict[str, Any]) -> dict[str, Any]:
        try:
            volume_out = self._run(["get-sink-volume", _DEFAULT_SINK])
            mute_out = self._run(["get-sink-mute", _DEFAULT_SINK])
        except AudioError as exc:
            return {"ok": False, "error": str(exc)}
        percent = parse_volume_percent(volume_out)
        muted = parse_muted(mute_out)
        if percent is None or muted is None:
            return {"ok": False, "error": "pactl-Antwort nicht lesbar."}
        return {"ok": True, "percent": percent, "muted": muted}

    def _volume_set(self, params: dict[str, Any]) -> dict[str, Any]:
        # The schema has no numeric bounds; clamping happens here so a model
        # typo becomes 100, never an error or something surprising.
        percent = max(0, min(100, int(params["percent"])))
        try:
            self._run(["set-sink-volume", _DEFAULT_SINK, f"{percent}%"])
        except AudioError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "percent": percent}

    def _mute(self, params: dict[str, Any]) -> dict[str, Any]:
        muted = bool(params["muted"])
        try:
            self._run(["set-sink-mute", _DEFAULT_SINK, "1" if muted else "0"])
        except AudioError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "muted": muted}
